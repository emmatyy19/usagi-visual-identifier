"""Tests for deterministic dataset fingerprints."""

from usagi_visual_identifier.data.fingerprint import dataset_fingerprint
from usagi_visual_identifier.data.models import (
    Catalog,
    CatalogItem,
    DatasetSplit,
    ImageManifest,
    ImageRecord,
    ImageSource,
    LabelStatus,
)


def _item(*, catalog_id: str, display_name: str) -> CatalogItem:
    """Create a verified test catalog item."""
    return CatalogItem(
        catalog_id=catalog_id,
        display_name=display_name,
        label_status=LabelStatus.VERIFIED,
        distinguishing_cues=(display_name.casefold(),),
        product_type="plush",
        included_in_model=True,
    )


def _record(*, image_id: str, catalog_id: str, source_id: str) -> ImageRecord:
    """Create a synthetic image record."""
    return ImageRecord(
        image_id=image_id,
        relative_path=f"images/{catalog_id}/{image_id}.jpg",
        catalog_id=catalog_id,
        capture_session_id="session-001",
        source_image_id=source_id,
        split=DatasetSplit.TRAIN,
        source=ImageSource.SYNTHETIC,
    )


def test_fingerprint_is_independent_of_row_order() -> None:
    """Reordering equivalent manifest rows keeps the same fingerprint."""
    first_item = _item(catalog_id="usagi-0001", display_name="Yellow Bow")
    second_item = _item(catalog_id="usagi-0002", display_name="Gray Hat")
    first_record = _record(
        image_id="img-000001", catalog_id="usagi-0001", source_id="src-000001"
    )
    second_record = _record(
        image_id="img-000002", catalog_id="usagi-0002", source_id="src-000002"
    )

    forward = dataset_fingerprint(
        catalog=Catalog(items=(first_item, second_item)),
        manifest=ImageManifest(records=(first_record, second_record)),
    )
    reverse = dataset_fingerprint(
        catalog=Catalog(items=(second_item, first_item)),
        manifest=ImageManifest(records=(second_record, first_record)),
    )

    assert forward == reverse
    assert len(forward) == 64


def test_fingerprint_changes_with_catalog_content() -> None:
    """Changing a catalog value changes the dataset identity."""
    record = _record(
        image_id="img-000001", catalog_id="usagi-0001", source_id="src-000001"
    )
    manifest = ImageManifest(records=(record,))

    first = dataset_fingerprint(
        catalog=Catalog(
            items=(_item(catalog_id="usagi-0001", display_name="Yellow Bow"),)
        ),
        manifest=manifest,
    )
    second = dataset_fingerprint(
        catalog=Catalog(
            items=(_item(catalog_id="usagi-0001", display_name="Golden Bow"),)
        ),
        manifest=manifest,
    )

    assert first != second
