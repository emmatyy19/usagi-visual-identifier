"""Tests for verified private transfer bundles."""

import csv
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from usagi_visual_identifier.data.csv_rows import (
    catalog_csv_columns,
    image_manifest_csv_columns,
)
from usagi_visual_identifier.embeddings.artifacts import PREPARED_COLUMNS
from usagi_visual_identifier.training import (
    TransferBundleError,
    export_result_bundle,
    export_result_collection,
    export_training_bundle,
    import_transfer_bundle,
)


def _write_csv(
    *, path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]
) -> None:
    """Write one temporary fixture CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _private_dataset(root: Path) -> tuple[Path, Path, Path]:
    """Create train, validation, and sealed test fixture files."""
    private_root = root / "data/private"
    processed_root = root / "data/processed"
    artifact_root = root / "artifacts"
    catalog_rows = [
        {
            "catalog_id": "usagi-0001",
            "display_name": "One",
            "label_status": "verified",
            "distinguishing_cues": "red",
            "product_type": "plush",
            "included_in_model": "true",
        },
        {
            "catalog_id": "usagi-0002",
            "display_name": "Two",
            "label_status": "verified",
            "distinguishing_cues": "blue",
            "product_type": "plush",
            "included_in_model": "true",
        },
    ]
    _write_csv(
        path=private_root / "catalog.csv",
        columns=catalog_csv_columns(),
        rows=catalog_rows,
    )
    manifest_rows: list[dict[str, str]] = []
    prepared_rows: list[dict[str, str]] = []
    assignments = (
        (1, "usagi-0001", "session-001", "train", "red"),
        (2, "usagi-0002", "session-002", "validation", "blue"),
        (3, "usagi-0001", "session-003", "test", "green"),
    )
    for number, identity, session, split, color in assignments:
        image_id = f"img-{number:06d}"
        relative = f"images/{identity}/{session}/{number}.heic"
        processed_relative = f"images/{identity}/{session}/{number}.jpg"
        manifest_rows.append(
            {
                "image_id": image_id,
                "relative_path": relative,
                "catalog_id": identity,
                "unknown_identity_id": "",
                "capture_session_id": session,
                "source_image_id": f"src-{number:06d}",
                "split": split,
                "source": "owner_capture",
            }
        )
        image_path = processed_root / processed_relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), color).save(image_path)
        prepared_rows.append(
            {
                "image_id": image_id,
                "source_relative_path": relative,
                "processed_relative_path": processed_relative,
                "source_sha256": f"source-{number}",
                "processed_sha256": f"processed-{number}",
                "width": "4",
                "height": "4",
            }
        )
    _write_csv(
        path=private_root / "images.csv",
        columns=image_manifest_csv_columns(),
        rows=manifest_rows,
    )
    _write_csv(
        path=processed_root / "images.csv",
        columns=PREPARED_COLUMNS,
        rows=prepared_rows,
    )
    linear_root = artifact_root / "models/linear_probe"
    run_root = linear_root / "linear-run"
    run_root.mkdir(parents=True)
    (linear_root / "latest.json").write_text(
        json.dumps({"schema_version": 1, "run_id": "linear-run"}),
        encoding="utf-8",
    )
    for name in (
        "checkpoint.pt",
        "metadata.json",
        "training_history.json",
        "validation_report.json",
    ):
        (run_root / name).write_text(name, encoding="utf-8")
    return private_root, processed_root, artifact_root


def test_training_bundle_excludes_test_and_imports_idempotently(tmp_path: Path) -> None:
    """The portable data archive never contains Session 3 photographs."""
    private_root, processed_root, artifact_root = _private_dataset(tmp_path / "source")
    bundle = export_training_bundle(
        private_root=private_root,
        processed_root=processed_root,
        artifact_root=artifact_root,
    )
    destination = tmp_path / "destination"
    manifest = import_transfer_bundle(
        archive_path=bundle,
        project_root=destination,
    )
    _ = import_transfer_bundle(archive_path=bundle, project_root=destination)

    assert manifest["kind"] == "partial_finetune_training_inputs"
    with (destination / "data/private/images.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert {row["split"] for row in rows} == {"train", "validation"}
    assert not (
        destination / "data/processed/images/usagi-0001/session-003/3.jpg"
    ).exists()


def test_result_bundle_round_trip_and_overwrite_protection(tmp_path: Path) -> None:
    """A result archive installs complete artifacts but preserves local differences."""
    artifact_root = tmp_path / "source/artifacts"
    run_root = artifact_root / "models/partial_finetune/run-123"
    run_root.mkdir(parents=True)
    for name in (
        "trained_delta.safetensors",
        "metadata.json",
        "training_history.json",
        "validation_report.json",
    ):
        (run_root / name).write_text(name, encoding="utf-8")
    bundle = export_result_bundle(artifact_root=artifact_root, run_id="run-123")
    destination = tmp_path / "destination"
    _ = import_transfer_bundle(archive_path=bundle, project_root=destination)

    metadata = destination / "artifacts/models/partial_finetune/run-123/metadata.json"
    assert metadata.is_file()
    metadata.write_text("different", encoding="utf-8")
    with pytest.raises(TransferBundleError, match="refusing to overwrite"):
        import_transfer_bundle(archive_path=bundle, project_root=destination)


def test_result_collection_includes_every_complete_run(tmp_path: Path) -> None:
    """One archive can return the complete predetermined experiment matrix."""
    artifact_root = tmp_path / "source/artifacts"
    model_root = artifact_root / "models/partial_finetune"
    for run_id in ("run-one", "run-two"):
        run_root = model_root / run_id
        run_root.mkdir(parents=True)
        for name in (
            "trained_delta.safetensors",
            "metadata.json",
            "training_history.json",
            "validation_report.json",
        ):
            (run_root / name).write_text(f"{run_id}:{name}", encoding="utf-8")
    (model_root / "latest.json").write_text(
        json.dumps({"schema_version": 1, "run_id": "run-two"}),
        encoding="utf-8",
    )

    bundle = export_result_collection(artifact_root=artifact_root)
    destination = tmp_path / "destination"
    manifest = import_transfer_bundle(
        archive_path=bundle,
        project_root=destination,
    )

    assert manifest["details"] == {
        "run_count": 2,
        "run_ids": ["run-one", "run-two"],
    }
    assert (
        destination
        / "artifacts/models/partial_finetune/run-one/trained_delta.safetensors"
    ).is_file()


def test_import_rejects_an_unlisted_member(tmp_path: Path) -> None:
    """A file added after bundle creation cannot be installed."""
    private_root, processed_root, artifact_root = _private_dataset(tmp_path / "source")
    bundle = export_training_bundle(
        private_root=private_root,
        processed_root=processed_root,
        artifact_root=artifact_root,
    )
    with zipfile.ZipFile(bundle, "a") as archive:
        archive.writestr("unexpected.txt", "unexpected")

    with pytest.raises(TransferBundleError, match="unlisted"):
        import_transfer_bundle(
            archive_path=bundle,
            project_root=tmp_path / "destination",
        )
