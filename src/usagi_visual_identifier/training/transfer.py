"""Verified private bundles for moving training inputs and results."""

import csv
import hashlib
import io
import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import cast

from usagi_visual_identifier.data.csv_rows import image_manifest_csv_columns
from usagi_visual_identifier.data.fingerprint import dataset_fingerprint
from usagi_visual_identifier.data.loaders import (
    load_catalog_csv,
    load_image_manifest_csv,
)
from usagi_visual_identifier.data.models import DatasetSplit, ImageRecord
from usagi_visual_identifier.embeddings.artifacts import PREPARED_COLUMNS

MAX_MEMBER_BYTES = 250 * 1024 * 1024
MAX_BUNDLE_BYTES = 2 * 1024 * 1024 * 1024


class TransferBundleError(ValueError):
    """Raised when a private transfer bundle is incomplete or unsafe."""


def _object_dict(value: object, *, name: str) -> dict[str, object]:
    """Validate an untyped JSON object and return string keyed values."""
    if not isinstance(value, dict):
        raise TransferBundleError(f"{name} must be a JSON object")
    result: dict[str, object] = {}
    for key, item in cast(dict[object, object], value).items():
        if not isinstance(key, str):
            raise TransferBundleError(f"{name} keys must be strings")
        result[key] = item
    return result


def _sha256_bytes(value: bytes) -> str:
    """Hash one in memory bundle member."""
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    """Hash one file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_bytes(
    *, columns: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> bytes:
    """Serialize a stable UTF-8 CSV member."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _manifest_row(record: ImageRecord) -> dict[str, str]:
    """Convert an image record to its CSV representation."""
    return {
        "image_id": record.image_id,
        "relative_path": record.relative_path,
        "catalog_id": record.catalog_id or "",
        "unknown_identity_id": record.unknown_identity_id or "",
        "capture_session_id": record.capture_session_id,
        "source_image_id": record.source_image_id,
        "split": record.split.value,
        "source": record.source.value,
    }


def _prepared_rows(path: Path) -> dict[str, dict[str, str]]:
    """Load the processed CSV under its exact public contract."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PREPARED_COLUMNS:
            raise TransferBundleError("processed image manifest columns are invalid")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            if None in row or any(
                not isinstance(row.get(column), str) for column in PREPARED_COLUMNS
            ):
                raise TransferBundleError("processed image manifest row is invalid")
            rows[row["image_id"]] = {column: row[column] for column in PREPARED_COLUMNS}
    return rows


def _write_bundle(
    *,
    path: Path,
    kind: str,
    values: Mapping[str, bytes | Path],
    details: Mapping[str, object],
) -> Path:
    """Write files and a signed member inventory into one ZIP archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    members: list[dict[str, object]] = []
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for member_name, source in sorted(values.items()):
            if isinstance(source, bytes):
                value = source
                archive.writestr(member_name, value)
                size = len(value)
                digest = _sha256_bytes(value)
            else:
                archive.write(source, member_name)
                size = source.stat().st_size
                digest = _sha256_path(source)
            members.append({"path": member_name, "size": size, "sha256": digest})
        manifest = {
            "schema_version": 1,
            "kind": kind,
            "details": dict(details),
            "members": members,
        }
        archive.writestr(
            "bundle_manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
    temporary.replace(path)
    return path


def export_training_bundle(
    *, private_root: Path, processed_root: Path, artifact_root: Path
) -> Path:
    """Export only training and validation photos plus the linear warm start."""
    catalog_path = private_root / "catalog.csv"
    catalog = load_catalog_csv(path=catalog_path)
    manifest = load_image_manifest_csv(path=private_root / "images.csv")
    selected = tuple(
        record
        for record in manifest.records
        if record.split in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}
        and record.catalog_id is not None
    )
    if not selected:
        raise TransferBundleError("no training or validation images were found")
    selected_ids = {record.image_id for record in selected}
    prepared = _prepared_rows(processed_root / "images.csv")
    missing = selected_ids - set(prepared)
    if missing:
        raise TransferBundleError(f"processed images are missing: {sorted(missing)}")
    filtered_prepared = [prepared[record.image_id] for record in selected]
    filtered_manifest = type(manifest)(records=selected)
    fingerprint = dataset_fingerprint(catalog=catalog, manifest=filtered_manifest)

    values: dict[str, bytes | Path] = {
        "data/private/catalog.csv": catalog_path,
        "data/private/images.csv": _csv_bytes(
            columns=image_manifest_csv_columns(),
            rows=[_manifest_row(record) for record in selected],
        ),
        "data/processed/images.csv": _csv_bytes(
            columns=PREPARED_COLUMNS,
            rows=filtered_prepared,
        ),
    }
    for row in filtered_prepared:
        relative_path = row["processed_relative_path"]
        source = processed_root / relative_path
        if not source.is_file():
            raise TransferBundleError(f"prepared image does not exist: {source}")
        values[f"data/processed/{relative_path}"] = source

    linear_root = artifact_root / "models/linear_probe"
    latest_raw: object = json.loads(
        (linear_root / "latest.json").read_text(encoding="utf-8")
    )
    latest = _object_dict(latest_raw, name="linear probe latest pointer")
    run_id_value = latest.get("run_id")
    if not isinstance(run_id_value, str):
        raise TransferBundleError("linear probe latest pointer is invalid")
    run_id = run_id_value
    run_root = linear_root / run_id
    for name in (
        "checkpoint.pt",
        "metadata.json",
        "training_history.json",
        "validation_report.json",
    ):
        source = run_root / name
        if not source.is_file():
            raise TransferBundleError(f"linear probe artifact is missing: {source}")
        values[f"artifacts/models/linear_probe/{run_id}/{name}"] = source
    values["artifacts/models/linear_probe/latest.json"] = linear_root / "latest.json"

    output = artifact_root / "transfers" / f"training-inputs-{fingerprint[:12]}.zip"
    return _write_bundle(
        path=output,
        kind="partial_finetune_training_inputs",
        values=values,
        details={
            "dataset_fingerprint": fingerprint,
            "image_count": len(selected),
            "splits": ["train", "validation"],
            "test_split_included": False,
            "linear_probe_run_id": run_id,
        },
    )


def export_result_bundle(*, artifact_root: Path, run_id: str | None = None) -> Path:
    """Export one completed fine tuning run for transfer to another machine."""
    model_root = artifact_root / "models/partial_finetune"
    if run_id is None:
        latest_raw: object = json.loads(
            (model_root / "latest.json").read_text(encoding="utf-8")
        )
        latest = _object_dict(latest_raw, name="fine tuning latest pointer")
        run_id_value = latest.get("run_id")
        if not isinstance(run_id_value, str):
            raise TransferBundleError("fine tuning latest pointer is invalid")
        run_id = run_id_value
    run_root = model_root / run_id
    values: dict[str, bytes | Path] = {}
    for name in (
        "trained_delta.safetensors",
        "metadata.json",
        "training_history.json",
        "validation_report.json",
    ):
        source = run_root / name
        if not source.is_file():
            raise TransferBundleError(f"fine tuning artifact is missing: {source}")
        values[f"artifacts/models/partial_finetune/{run_id}/{name}"] = source
    values["artifacts/models/partial_finetune/latest.json"] = json.dumps(
        {"schema_version": 1, "run_id": run_id},
        indent=2,
        sort_keys=True,
    ).encode()
    output = artifact_root / "transfers" / f"partial-finetune-{run_id}.zip"
    return _write_bundle(
        path=output,
        kind="partial_finetune_result",
        values=values,
        details={"run_id": run_id},
    )


def export_result_collection(*, artifact_root: Path) -> Path:
    """Export every complete fine tuning run as one experiment collection."""
    model_root = artifact_root / "models/partial_finetune"
    run_directories = sorted(
        path
        for path in model_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if not run_directories:
        raise TransferBundleError("no fine tuning runs were found")
    values: dict[str, bytes | Path] = {}
    run_ids: list[str] = []
    for run_root in run_directories:
        run_id = run_root.name
        for name in (
            "trained_delta.safetensors",
            "metadata.json",
            "training_history.json",
            "validation_report.json",
        ):
            source = run_root / name
            if not source.is_file():
                raise TransferBundleError(f"fine tuning artifact is missing: {source}")
            values[f"artifacts/models/partial_finetune/{run_id}/{name}"] = source
        run_ids.append(run_id)
    latest_path = model_root / "latest.json"
    if not latest_path.is_file():
        raise TransferBundleError("fine tuning latest pointer is missing")
    values["artifacts/models/partial_finetune/latest.json"] = latest_path
    return _write_bundle(
        path=artifact_root / "transfers/partial-finetune-experiment.zip",
        kind="partial_finetune_result",
        values=values,
        details={"run_ids": run_ids, "run_count": len(run_ids)},
    )


def _safe_member(name: str) -> bool:
    """Return whether a ZIP member is a normalized relative path."""
    path = PurePosixPath(name)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and name == path.as_posix()
        and name not in {"", "."}
    )


def import_transfer_bundle(
    *, archive_path: Path, project_root: Path
) -> dict[str, object]:
    """Verify every member before atomically installing a private bundle."""
    if archive_path.stat().st_size > MAX_BUNDLE_BYTES:
        raise TransferBundleError("transfer bundle is unexpectedly large")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "bundle_manifest.json" not in names:
            raise TransferBundleError("transfer bundle member inventory is invalid")
        if any(not _safe_member(name) for name in names):
            raise TransferBundleError("transfer bundle contains an unsafe path")
        manifest_raw: object = json.loads(
            archive.read("bundle_manifest.json").decode("utf-8")
        )
        manifest = _object_dict(manifest_raw, name="transfer bundle manifest")
        members_raw = manifest.get("members")
        if not isinstance(members_raw, list):
            raise TransferBundleError("transfer bundle member list is invalid")
        expected_names = {"bundle_manifest.json"}
        verified: list[tuple[str, bytes]] = []
        total_size = 0
        for raw_item in cast(list[object], members_raw):
            item = _object_dict(raw_item, name="transfer bundle member")
            name = item.get("path")
            size = item.get("size")
            digest = item.get("sha256")
            if (
                not isinstance(name, str)
                or not isinstance(size, int)
                or not isinstance(digest, str)
                or not _safe_member(name)
            ):
                raise TransferBundleError("transfer bundle member metadata is invalid")
            if size > MAX_MEMBER_BYTES:
                raise TransferBundleError(f"transfer member is too large: {name}")
            value = archive.read(name)
            if len(value) != size or _sha256_bytes(value) != digest:
                raise TransferBundleError(f"transfer member hash failed: {name}")
            total_size += size
            expected_names.add(name)
            verified.append((name, value))
        if total_size > MAX_BUNDLE_BYTES:
            raise TransferBundleError("expanded transfer bundle is unexpectedly large")
        if set(names) != expected_names:
            raise TransferBundleError("transfer bundle has unlisted members")

    kind = manifest.get("kind")
    if kind not in {"partial_finetune_training_inputs", "partial_finetune_result"}:
        raise TransferBundleError("transfer bundle kind is unsupported")
    if kind == "partial_finetune_training_inputs":
        image_manifest = next(
            value for name, value in verified if name == "data/private/images.csv"
        )
        rows = csv.DictReader(io.StringIO(image_manifest.decode("utf-8")))
        splits = {row["split"] for row in rows}
        if not splits or splits - {"train", "validation"}:
            raise TransferBundleError("training bundle contains a forbidden split")

    for name, value in verified:
        destination = project_root / PurePosixPath(name)
        if destination.exists():
            if _sha256_path(destination) == _sha256_bytes(value):
                continue
            raise TransferBundleError(
                f"refusing to overwrite a different local file: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_bytes(value)
        temporary.replace(destination)
    return manifest
