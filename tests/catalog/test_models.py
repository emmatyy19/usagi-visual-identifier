"""Tests for personal catalog request and storage models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from usagi_visual_identifier.catalog.models import (
    CatalogRecord,
    CatalogRecordCreate,
    CatalogRecordUpdate,
)


def test_catalog_record_accepts_independent_personal_statuses() -> None:
    """Ownership and interest are separate catalog decisions."""
    record = CatalogRecordCreate(
        preferred_name="Magical Usagi",
        owned=True,
        wanted=False,
        linked_identity_id="usagi-0004",
    )
    assert record.known_to_exist is True
    assert record.owned is True
    assert record.wanted is False


@pytest.mark.parametrize(
    "value",
    (
        {"preferred_name": "Example", "wanted": True, "not_wanted": True},
        {"preferred_name": "Example", "owned": True, "known_to_exist": False},
    ),
)
def test_catalog_record_rejects_contradictory_statuses(
    value: dict[str, object],
) -> None:
    """Impossible personal status combinations fail at validation."""
    with pytest.raises(ValidationError):
        CatalogRecordCreate.model_validate(value)


def test_catalog_update_requires_an_explicit_field() -> None:
    """An empty patch cannot create a meaningless database write."""
    with pytest.raises(ValidationError, match="at least one"):
        CatalogRecordUpdate()


def test_stored_record_uses_a_stable_item_identifier() -> None:
    """Stored catalog records validate their public ID and timestamps."""
    now = datetime.now(UTC)
    record = CatalogRecord(
        record_id="item-0123456789abcdef0123456789abcdef",
        preferred_name="Posture Pal",
        created_at=now,
        updated_at=now,
    )
    assert record.record_id.startswith("item-")
