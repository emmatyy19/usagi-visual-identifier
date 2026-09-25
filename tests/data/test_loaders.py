"""Tests for strict private CSV loading."""

from pathlib import Path

import pytest

from usagi_visual_identifier.data.loaders import (
    DataLoadError,
    load_catalog_csv,
    load_image_manifest_csv,
)
from usagi_visual_identifier.data.models import DatasetSplit, LabelStatus

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "data"


def test_load_catalog_csv() -> None:
    """The invented catalog fixture loads into typed records."""
    catalog = load_catalog_csv(path=FIXTURE_ROOT / "catalog.csv")

    assert len(catalog.items) == 3
    assert catalog.items[0].label_status is LabelStatus.VERIFIED
    assert catalog.items[2].included_in_model is False


def test_load_image_manifest_csv() -> None:
    """The invented image fixture preserves splits and identities."""
    manifest = load_image_manifest_csv(path=FIXTURE_ROOT / "images.csv")

    assert len(manifest.records) == 7
    assert manifest.records[0].split is DatasetSplit.TRAIN
    assert manifest.records[-1].identity_key == "unknown-0001"


def test_catalog_loader_rejects_unexpected_columns(tmp_path: Path) -> None:
    """A changed CSV header cannot be silently accepted."""
    path = tmp_path / "catalog.csv"
    path.write_text("catalog_id,name\nusagi-0001,Example\n", encoding="utf-8")

    with pytest.raises(DataLoadError, match="unexpected columns"):
        load_catalog_csv(path=path)


def test_catalog_loader_reports_missing_file(tmp_path: Path) -> None:
    """A missing private data file produces a domain error."""
    with pytest.raises(DataLoadError, match="could not open"):
        load_catalog_csv(path=tmp_path / "missing.csv")


def test_catalog_loader_rejects_malformed_row(tmp_path: Path) -> None:
    """A row with more values than its header is rejected."""
    path = tmp_path / "catalog.csv"
    path.write_text(
        "catalog_id,display_name,label_status,distinguishing_cues,"
        "product_type,included_in_model\n"
        "usagi-0001,Example,verified,hat,plush,true,extra\n",
        encoding="utf-8",
    )

    with pytest.raises(DataLoadError, match="malformed CSV row"):
        load_catalog_csv(path=path)


def test_catalog_loader_rejects_invalid_boolean(tmp_path: Path) -> None:
    """Boolean CSV fields accept only explicit lowercase values."""
    path = tmp_path / "catalog.csv"
    path.write_text(
        "catalog_id,display_name,label_status,distinguishing_cues,"
        "product_type,included_in_model\n"
        "usagi-0001,Example,verified,hat,plush,yes\n",
        encoding="utf-8",
    )

    with pytest.raises(DataLoadError, match="expected true or false"):
        load_catalog_csv(path=path)


def test_catalog_loader_reports_invalid_row(tmp_path: Path) -> None:
    """A bad row reports its file and line number."""
    path = tmp_path / "catalog.csv"
    path.write_text(
        "catalog_id,display_name,label_status,distinguishing_cues,"
        "product_type,included_in_model\n"
        "bad-id,Example,verified,hat,plush,true\n",
        encoding="utf-8",
    )

    with pytest.raises(DataLoadError, match=r"catalog\.csv:2"):
        load_catalog_csv(path=path)


def test_catalog_loader_rejects_empty_catalog(tmp_path: Path) -> None:
    """A catalog must contain at least one item."""
    path = tmp_path / "catalog.csv"
    path.write_text(
        "catalog_id,display_name,label_status,distinguishing_cues,"
        "product_type,included_in_model\n",
        encoding="utf-8",
    )

    with pytest.raises(DataLoadError, match="invalid catalog"):
        load_catalog_csv(path=path)


def test_image_loader_reports_invalid_row(tmp_path: Path) -> None:
    """An invalid image row reports its file and line number."""
    path = tmp_path / "images.csv"
    path.write_text(
        "image_id,relative_path,catalog_id,unknown_identity_id,"
        "capture_session_id,source_image_id,split,source\n"
        "bad-id,images/example.jpg,usagi-0001,,session-001,src-000001,"
        "train,synthetic\n",
        encoding="utf-8",
    )

    with pytest.raises(DataLoadError, match=r"images\.csv:2"):
        load_image_manifest_csv(path=path)


def test_image_loader_rejects_empty_manifest(tmp_path: Path) -> None:
    """An image manifest must contain at least one record."""
    path = tmp_path / "images.csv"
    path.write_text(
        "image_id,relative_path,catalog_id,unknown_identity_id,"
        "capture_session_id,source_image_id,split,source\n",
        encoding="utf-8",
    )

    with pytest.raises(DataLoadError, match="invalid image manifest"):
        load_image_manifest_csv(path=path)
