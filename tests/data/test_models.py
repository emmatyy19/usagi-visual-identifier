"""Tests for catalog and image manifest records."""

import pytest
from pydantic import ValidationError

from usagi_visual_identifier.data.models import (
    Catalog,
    CatalogItem,
    DatasetSplit,
    ImageManifest,
    ImageRecord,
    ImageSource,
    LabelStatus,
)


def _catalog_item(*, catalog_id: str = "usagi-0001") -> CatalogItem:
    """Create one valid catalog item."""
    return CatalogItem(
        catalog_id=catalog_id,
        display_name="Yellow Bow Usagi",
        label_status=LabelStatus.VERIFIED,
        distinguishing_cues=("yellow bow", "white dress"),
        product_type="plush",
        included_in_model=True,
    )


def _image_record(
    *,
    image_id: str = "img-000001",
    catalog_id: str = "usagi-0001",
    session_id: str = "session-001",
    source_image_id: str = "src-000001",
    split: DatasetSplit = DatasetSplit.TRAIN,
) -> ImageRecord:
    """Create one valid known identity image record."""
    return ImageRecord(
        image_id=image_id,
        relative_path=f"images/{catalog_id}/{session_id}/{image_id}.jpg",
        catalog_id=catalog_id,
        capture_session_id=session_id,
        source_image_id=source_image_id,
        split=split,
        source=ImageSource.SYNTHETIC,
    )


def test_catalog_requires_verified_items_for_modeling() -> None:
    """A provisional identity cannot enter supervised modeling."""
    with pytest.raises(ValidationError, match="only verified"):
        CatalogItem(
            catalog_id="usagi-0001",
            display_name="Working Name",
            label_status=LabelStatus.PROVISIONAL,
            product_type="plush",
            included_in_model=True,
        )


def test_catalog_rejects_duplicate_ids() -> None:
    """Two catalog items cannot share a stable identifier."""
    with pytest.raises(ValidationError, match="catalog IDs must be unique"):
        Catalog(items=(_catalog_item(), _catalog_item()))


def test_catalog_rejects_repeated_cues() -> None:
    """Visual cues should not be repeated with different capitalization."""
    with pytest.raises(ValidationError, match="distinguishing cues must be unique"):
        CatalogItem(
            catalog_id="usagi-0001",
            display_name="Yellow Bow Usagi",
            label_status=LabelStatus.VERIFIED,
            distinguishing_cues=("Yellow bow", "yellow bow"),
            product_type="plush",
            included_in_model=True,
        )


def test_catalog_indexes_items_by_id() -> None:
    """A catalog exposes its items through stable identifiers."""
    item = _catalog_item()

    assert Catalog(items=(item,)).by_id() == {"usagi-0001": item}


@pytest.mark.parametrize(
    "relative_path",
    (
        "/absolute/image.jpg",
        "../outside/image.jpg",
        "images\\usagi-0001\\image.jpg",
        "./images/image.jpg",
    ),
)
def test_image_record_rejects_unsafe_paths(relative_path: str) -> None:
    """Private image paths must remain normalized beneath their root."""
    with pytest.raises(ValidationError, match="path"):
        ImageRecord(
            image_id="img-000001",
            relative_path=relative_path,
            catalog_id="usagi-0001",
            capture_session_id="session-001",
            source_image_id="src-000001",
            split=DatasetSplit.TRAIN,
            source=ImageSource.SYNTHETIC,
        )


def test_image_record_requires_exactly_one_identity() -> None:
    """An image cannot be both known and unknown."""
    with pytest.raises(ValidationError, match="one known or unknown"):
        ImageRecord(
            image_id="img-000001",
            relative_path="images/example.jpg",
            catalog_id="usagi-0001",
            unknown_identity_id="unknown-0001",
            capture_session_id="session-001",
            source_image_id="src-000001",
            split=DatasetSplit.TRAIN,
            source=ImageSource.SYNTHETIC,
        )


def test_unknown_image_exposes_unknown_identity() -> None:
    """An unknown image uses its unknown identity as the lookup key."""
    record = ImageRecord(
        image_id="img-000001",
        relative_path="images/unknown/example.jpg",
        unknown_identity_id="unknown-0001",
        capture_session_id="session-001",
        source_image_id="src-000001",
        split=DatasetSplit.VALIDATION,
        source=ImageSource.SYNTHETIC,
    )

    assert record.identity_key == "unknown-0001"


def test_manifest_rejects_duplicate_image_ids() -> None:
    """Every manifest image needs one unique identifier."""
    first = _image_record()
    second = _image_record(source_image_id="src-000002")

    with pytest.raises(ValidationError, match="duplicate image ID"):
        ImageManifest(records=(first, second))


def test_manifest_rejects_session_split_leakage() -> None:
    """One capture session cannot appear in two dataset splits."""
    train = _image_record()
    validation = _image_record(
        image_id="img-000002",
        source_image_id="src-000002",
        split=DatasetSplit.VALIDATION,
    )

    with pytest.raises(ValidationError, match="capture session crosses splits"):
        ImageManifest(records=(train, validation))


def test_manifest_rejects_source_family_leakage() -> None:
    """Derived images from one source cannot cross splits."""
    train = _image_record()
    validation = _image_record(
        image_id="img-000002",
        session_id="session-002",
        split=DatasetSplit.VALIDATION,
    )

    with pytest.raises(ValidationError, match="source image family"):
        ImageManifest(records=(train, validation))


def test_manifest_rejects_source_family_identity_change() -> None:
    """One source family cannot be assigned to two identities."""
    first = _image_record()
    second = _image_record(image_id="img-000002", catalog_id="usagi-0002")

    with pytest.raises(ValidationError, match="source image family"):
        ImageManifest(records=(first, second))


def test_manifest_accepts_shared_sessions_within_one_split() -> None:
    """Different dolls should share a session setup within one split."""
    first = _image_record()
    second = _image_record(
        image_id="img-000002",
        catalog_id="usagi-0002",
        source_image_id="src-000002",
    )

    manifest = ImageManifest(records=(first, second))

    assert len(manifest.records) == 2
