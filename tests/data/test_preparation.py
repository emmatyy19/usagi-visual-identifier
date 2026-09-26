"""Tests for reproducible private image preparation."""

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from usagi_visual_identifier.data.loaders import load_image_manifest_csv
from usagi_visual_identifier.data.models import DatasetSplit
from usagi_visual_identifier.data.preparation import (
    PreparationError,
    prepare_session,
)


def _write_catalog(private_root: Path, *, count: int = 2) -> None:
    """Create a small invented catalog."""
    rows = [
        "catalog_id,display_name,label_status,distinguishing_cues,"
        "product_type,included_in_model"
    ]
    for number in range(1, count + 1):
        rows.append(
            f"usagi-{number:04d},Example {number},verified,cue {number},plush,true"
        )
    private_root.mkdir(parents=True, exist_ok=True)
    (private_root / "catalog.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_image(
    private_root: Path,
    *,
    catalog_id: str,
    session_id: str,
    filename: str,
    color: tuple[int, int, int],
    size: tuple[int, int] = (48, 32),
) -> Path:
    """Create one tiny owner captured stand in image."""
    path = private_root / "images" / catalog_id / session_id / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _create_session(
    private_root: Path, *, session_id: str = "session-001"
) -> tuple[Path, ...]:
    """Create two identities with two photos each."""
    return (
        _write_image(
            private_root,
            catalog_id="usagi-0001",
            session_id=session_id,
            filename="001.png",
            color=(255, 0, 0),
        ),
        _write_image(
            private_root,
            catalog_id="usagi-0001",
            session_id=session_id,
            filename="002.png",
            color=(220, 0, 0),
        ),
        _write_image(
            private_root,
            catalog_id="usagi-0002",
            session_id=session_id,
            filename="001.png",
            color=(0, 255, 0),
        ),
        _write_image(
            private_root,
            catalog_id="usagi-0002",
            session_id=session_id,
            filename="002.png",
            color=(0, 220, 0),
        ),
    )


def test_prepare_session_creates_manifests_jpegs_and_audit(tmp_path: Path) -> None:
    """A complete run preserves originals and creates reviewable outputs."""
    private_root = tmp_path / "private"
    processed_root = tmp_path / "processed"
    _write_catalog(private_root)
    sources = _create_session(private_root)
    source_bytes = {path: path.read_bytes() for path in sources}

    result = prepare_session(
        private_root=private_root,
        processed_root=processed_root,
        session_id="session-001",
        split=DatasetSplit.TRAIN,
    )

    assert result.image_count == 4
    assert result.identity_count == 2
    assert result.contact_sheet_count == 2
    assert len(result.fingerprint) == 64
    assert all(path.read_bytes() == source_bytes[path] for path in sources)

    manifest = load_image_manifest_csv(path=result.manifest_path)
    assert [record.image_id for record in manifest.records] == [
        "img-000001",
        "img-000002",
        "img-000003",
        "img-000004",
    ]
    assert all(record.split is DatasetSplit.TRAIN for record in manifest.records)
    assert (processed_root / "images/usagi-0001/session-001/001.jpg").is_file()
    assert (processed_root / "audit/session-001/usagi-0001.jpg").is_file()

    with result.prepared_manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["width"] == "48"
    assert rows[0]["height"] == "32"
    assert len(rows[0]["source_sha256"]) == 64

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["counts_by_session"] == {
        "session-001": {"usagi-0001": 2, "usagi-0002": 2}
    }


def test_prepare_session_is_idempotent_and_appends_stable_ids(tmp_path: Path) -> None:
    """Repeated work keeps IDs while a later session receives new IDs."""
    private_root = tmp_path / "private"
    processed_root = tmp_path / "processed"
    _write_catalog(private_root)
    _create_session(private_root)

    first = prepare_session(
        private_root=private_root,
        processed_root=processed_root,
        session_id="session-001",
        split=DatasetSplit.TRAIN,
    )
    repeated = prepare_session(
        private_root=private_root,
        processed_root=processed_root,
        session_id="session-001",
        split=DatasetSplit.TRAIN,
    )
    assert repeated.fingerprint == first.fingerprint

    _write_image(
        private_root,
        catalog_id="usagi-0001",
        session_id="session-002",
        filename="001.png",
        color=(0, 0, 255),
    )
    appended = prepare_session(
        private_root=private_root,
        processed_root=processed_root,
        session_id="session-002",
        split=DatasetSplit.VALIDATION,
    )
    manifest = load_image_manifest_csv(path=appended.manifest_path)

    assert len(manifest.records) == 5
    assert manifest.records[0].image_id == "img-000001"
    assert manifest.records[-1].image_id == "img-000005"
    assert manifest.records[-1].source_image_id == "src-000005"
    assert manifest.records[-1].split is DatasetSplit.VALIDATION
    assert appended.contact_sheet_count == 3


def test_prepare_session_rejects_a_changed_split(tmp_path: Path) -> None:
    """A session cannot silently change its declared split."""
    private_root = tmp_path / "private"
    processed_root = tmp_path / "processed"
    _write_catalog(private_root)
    _create_session(private_root)
    prepare_session(
        private_root=private_root,
        processed_root=processed_root,
        session_id="session-001",
        split=DatasetSplit.TRAIN,
    )

    with pytest.raises(PreparationError, match="assignment conflicts"):
        prepare_session(
            private_root=private_root,
            processed_root=processed_root,
            session_id="session-001",
            split=DatasetSplit.TEST,
        )


@pytest.mark.parametrize("session_id", ["session-1", "first-session"])
def test_prepare_session_rejects_invalid_session_ids(
    tmp_path: Path, session_id: str
) -> None:
    """Session names must follow the stable ID contract."""
    private_root = tmp_path / "private"
    _write_catalog(private_root)
    (private_root / "images").mkdir()

    with pytest.raises(PreparationError, match="invalid capture session ID"):
        prepare_session(
            private_root=private_root,
            processed_root=tmp_path / "processed",
            session_id=session_id,
            split=DatasetSplit.TRAIN,
        )


def test_prepare_session_rejects_unknown_identity_directory(tmp_path: Path) -> None:
    """A directory cannot invent a catalog identity by accident."""
    private_root = tmp_path / "private"
    _write_catalog(private_root, count=1)
    _write_image(
        private_root,
        catalog_id="usagi-9999",
        session_id="session-001",
        filename="001.png",
        color=(0, 0, 0),
    )

    with pytest.raises(PreparationError, match="no catalog record"):
        prepare_session(
            private_root=private_root,
            processed_root=tmp_path / "processed",
            session_id="session-001",
            split=DatasetSplit.TRAIN,
        )


def test_prepare_session_rejects_corrupt_images(tmp_path: Path) -> None:
    """A corrupt original stops the run before a manifest is published."""
    private_root = tmp_path / "private"
    _write_catalog(private_root, count=1)
    source = private_root / "images/usagi-0001/session-001/001.jpg"
    source.parent.mkdir(parents=True)
    source.write_text("not an image", encoding="utf-8")

    with pytest.raises(PreparationError, match="could not decode"):
        prepare_session(
            private_root=private_root,
            processed_root=tmp_path / "processed",
            session_id="session-001",
            split=DatasetSplit.TRAIN,
        )
    assert not (private_root / "images.csv").exists()


def test_prepare_session_rejects_missing_images_and_bad_quality(tmp_path: Path) -> None:
    """Empty sessions and invalid encoding settings fail clearly."""
    private_root = tmp_path / "private"
    _write_catalog(private_root)
    (private_root / "images").mkdir()

    with pytest.raises(PreparationError, match="no supported images"):
        prepare_session(
            private_root=private_root,
            processed_root=tmp_path / "processed",
            session_id="session-001",
            split=DatasetSplit.TRAIN,
        )
    with pytest.raises(PreparationError, match="quality"):
        prepare_session(
            private_root=private_root,
            processed_root=tmp_path / "processed",
            session_id="session-001",
            split=DatasetSplit.TRAIN,
            jpeg_quality=0,
        )
