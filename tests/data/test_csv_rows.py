"""Tests for the typed CSV boundary models."""

import pytest
from pydantic import ValidationError

from usagi_visual_identifier.data.csv_rows import (
    CatalogCsvRow,
    ImageManifestCsvRow,
    catalog_csv_columns,
    image_manifest_csv_columns,
)
from usagi_visual_identifier.data.models import DatasetSplit, LabelStatus


def test_catalog_columns_come_from_boundary_model() -> None:
    """The catalog header follows the declared CSV row fields."""
    assert catalog_csv_columns() == (
        "catalog_id",
        "display_name",
        "label_status",
        "distinguishing_cues",
        "product_type",
        "included_in_model",
    )


def test_catalog_csv_row_converts_to_domain_item() -> None:
    """CSV text becomes the strict catalog representation."""
    row = CatalogCsvRow.model_validate(
        {
            "catalog_id": "usagi-0001",
            "display_name": "Yellow Bow Usagi",
            "label_status": "verified",
            "distinguishing_cues": "yellow bow | white dress",
            "product_type": "plush",
            "included_in_model": "true",
        }
    )

    item = row.to_catalog_item()

    assert item.label_status is LabelStatus.VERIFIED
    assert item.distinguishing_cues == ("yellow bow", "white dress")
    assert item.included_in_model is True


@pytest.mark.parametrize("value", ("yes", "TRUE", "1", True))
def test_catalog_csv_row_rejects_implicit_booleans(value: object) -> None:
    """The CSV boundary does not guess boolean meanings."""
    with pytest.raises(ValidationError, match="must be true or false"):
        CatalogCsvRow.model_validate(
            {
                "catalog_id": "usagi-0001",
                "display_name": "Yellow Bow Usagi",
                "label_status": "verified",
                "distinguishing_cues": "yellow bow",
                "product_type": "plush",
                "included_in_model": value,
            }
        )


def test_catalog_csv_row_requires_text_cues() -> None:
    """The CSV cue field must arrive as pipe separated text."""
    with pytest.raises(ValidationError, match="pipe separated text"):
        CatalogCsvRow.model_validate(
            {
                "catalog_id": "usagi-0001",
                "display_name": "Yellow Bow Usagi",
                "label_status": "verified",
                "distinguishing_cues": ["yellow bow"],
                "product_type": "plush",
                "included_in_model": "true",
            }
        )


def test_image_columns_come_from_boundary_model() -> None:
    """The image header follows the declared CSV row fields."""
    assert image_manifest_csv_columns() == (
        "image_id",
        "relative_path",
        "catalog_id",
        "unknown_identity_id",
        "capture_session_id",
        "source_image_id",
        "split",
        "source",
    )


def test_image_csv_row_converts_blank_identity_to_none() -> None:
    """A blank known identity becomes an explicit unknown record."""
    row = ImageManifestCsvRow.model_validate(
        {
            "image_id": "img-000001",
            "relative_path": "images/unknown/example.jpg",
            "catalog_id": "",
            "unknown_identity_id": "unknown-0001",
            "capture_session_id": "session-001",
            "source_image_id": "src-000001",
            "split": "validation",
            "source": "synthetic",
        }
    )

    record = row.to_image_record()

    assert record.catalog_id is None
    assert record.unknown_identity_id == "unknown-0001"
    assert record.split is DatasetSplit.VALIDATION
