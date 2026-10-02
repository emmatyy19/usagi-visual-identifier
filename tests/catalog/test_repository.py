"""Tests for personal catalog persistence operations."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from usagi_visual_identifier.catalog.database import CatalogDatabase
from usagi_visual_identifier.catalog.models import (
    CatalogRecordCreate,
    CatalogRecordUpdate,
    VerificationStatus,
)
from usagi_visual_identifier.catalog.repository import (
    CatalogConflictError,
    CatalogRecordNotFoundError,
    CatalogRepository,
)


@pytest.fixture
def repository(tmp_path: Path) -> CatalogRepository:
    """Return a repository backed by a migrated temporary database."""
    database = CatalogDatabase(tmp_path / "catalog.sqlite3")
    database.initialize()
    return CatalogRepository(database)


def test_repository_creates_lists_gets_and_updates(
    repository: CatalogRepository,
) -> None:
    """The main record lifecycle preserves stable IDs and creation time."""
    second = repository.create_record(CatalogRecordCreate(preferred_name="Z Item"))
    first = repository.create_record(
        CatalogRecordCreate(
            preferred_name="Magical Usagi",
            owned=True,
            linked_identity_id="usagi-0004",
        )
    )

    assert repository.list_records() == (first, second)
    assert repository.get_record(first.record_id) == first

    updated = repository.update_record(
        record_id=first.record_id,
        changes=CatalogRecordUpdate(
            wanted=True,
            owned=False,
            verification_status=VerificationStatus.MANUALLY_VERIFIED,
            notes="Confirmed from the product tag.",
        ),
    )
    assert updated.record_id == first.record_id
    assert updated.created_at == first.created_at
    assert updated.updated_at >= first.updated_at
    assert updated.wanted is True
    assert updated.owned is False
    assert repository.get_record(first.record_id) == updated


def test_repository_rejects_duplicate_visual_identity(
    repository: CatalogRepository,
) -> None:
    """One model identity can link to only one personal catalog record."""
    repository.create_record(
        CatalogRecordCreate(preferred_name="First", linked_identity_id="usagi-0001")
    )
    with pytest.raises(CatalogConflictError, match="visual identity"):
        repository.create_record(
            CatalogRecordCreate(
                preferred_name="Second", linked_identity_id="usagi-0001"
            )
        )


def test_repository_validates_merged_update(repository: CatalogRepository) -> None:
    """A partial patch cannot create a contradictory complete record."""
    record = repository.create_record(
        CatalogRecordCreate(preferred_name="Example", wanted=True)
    )
    with pytest.raises(ValidationError, match="wanted and not wanted"):
        repository.update_record(
            record_id=record.record_id,
            changes=CatalogRecordUpdate(not_wanted=True),
        )


def test_repository_reports_missing_record(repository: CatalogRepository) -> None:
    """Read and update use the same domain specific missing error."""
    missing = "item-0123456789abcdef0123456789abcdef"
    with pytest.raises(CatalogRecordNotFoundError):
        repository.get_record(missing)
    with pytest.raises(CatalogRecordNotFoundError):
        repository.update_record(
            record_id=missing,
            changes=CatalogRecordUpdate(notes="Missing"),
        )
