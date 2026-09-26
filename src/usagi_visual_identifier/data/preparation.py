"""Prepare private source photographs for modeling and review."""

import csv
import hashlib
import importlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageDraw, ImageOps

from usagi_visual_identifier.data.csv_rows import image_manifest_csv_columns
from usagi_visual_identifier.data.fingerprint import dataset_fingerprint
from usagi_visual_identifier.data.loaders import (
    load_catalog_csv,
    load_image_manifest_csv,
)
from usagi_visual_identifier.data.models import (
    Catalog,
    DatasetSplit,
    ImageManifest,
    ImageRecord,
    ImageSource,
)

SUPPORTED_SOURCE_SUFFIXES = frozenset({".heic", ".heif", ".jpeg", ".jpg", ".png"})
SESSION_PATTERN = re.compile(r"^session-\d{3}$")
PREPARED_COLUMNS = (
    "image_id",
    "source_relative_path",
    "processed_relative_path",
    "source_sha256",
    "processed_sha256",
    "width",
    "height",
)


class PreparationError(ValueError):
    """Raised when private images cannot be prepared safely."""


def _register_heif_support() -> None:
    """Register the HEIF Pillow plugin behind a typed boundary."""
    module = importlib.import_module("pillow_heif")
    register = cast(Callable[[], None], module.register_heif_opener)
    register()


@dataclass(frozen=True)
class PreparedImage:
    """One source image and its reproducible JPEG derivative."""

    image_id: str
    source_relative_path: str
    processed_relative_path: str
    source_sha256: str
    processed_sha256: str
    width: int
    height: int

    def csv_row(self) -> dict[str, str | int]:
        """Return values in the prepared image CSV representation."""
        return {
            "image_id": self.image_id,
            "source_relative_path": self.source_relative_path,
            "processed_relative_path": self.processed_relative_path,
            "source_sha256": self.source_sha256,
            "processed_sha256": self.processed_sha256,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class PreparationResult:
    """Summary returned after a complete preparation run."""

    manifest_path: Path
    prepared_manifest_path: Path
    summary_path: Path
    image_count: int
    identity_count: int
    contact_sheet_count: int
    fingerprint: str


def _sha256(path: Path) -> str:
    """Hash one file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_suffix(identifier: str) -> int:
    """Return the numeric portion of a validated stable identifier."""
    return int(identifier.rsplit("-", maxsplit=1)[1])


def _discover_session_sources(
    *, private_root: Path, session_id: str, catalog: Catalog
) -> tuple[tuple[str, Path], ...]:
    """Find supported originals for one session in catalog order."""
    if SESSION_PATTERN.fullmatch(session_id) is None:
        raise PreparationError(f"invalid capture session ID: {session_id}")

    images_root = private_root / "images"
    catalog_ids = catalog.by_id()
    discovered: list[tuple[str, Path]] = []

    if not images_root.is_dir():
        raise PreparationError(f"private image directory does not exist: {images_root}")

    for identity_directory in sorted(images_root.iterdir()):
        if not identity_directory.is_dir() or identity_directory.name.startswith("_"):
            continue
        catalog_id = identity_directory.name
        if catalog_id not in catalog_ids:
            raise PreparationError(
                f"image directory has no catalog record: {catalog_id}"
            )
        session_directory = identity_directory / session_id
        if not session_directory.exists():
            continue
        if not session_directory.is_dir():
            raise PreparationError(
                f"session path is not a directory: {session_directory}"
            )

        for source_path in sorted(session_directory.iterdir()):
            if source_path.is_file() and source_path.suffix.casefold() in (
                SUPPORTED_SOURCE_SUFFIXES
            ):
                discovered.append((catalog_id, source_path))

    if not discovered:
        raise PreparationError(f"no supported images found for {session_id}")
    return tuple(discovered)


def _load_existing_records(*, manifest_path: Path) -> tuple[ImageRecord, ...]:
    """Load an existing manifest when one has already been created."""
    if not manifest_path.exists():
        return ()
    return load_image_manifest_csv(path=manifest_path).records


def _updated_manifest(
    *,
    private_root: Path,
    session_id: str,
    split: DatasetSplit,
    catalog: Catalog,
) -> ImageManifest:
    """Add one session while preserving every existing stable ID."""
    manifest_path = private_root / "images.csv"
    existing_records = _load_existing_records(manifest_path=manifest_path)
    existing_by_path = {record.relative_path: record for record in existing_records}
    next_image_number = max(
        (_numeric_suffix(record.image_id) for record in existing_records), default=0
    )
    next_source_number = max(
        (_numeric_suffix(record.source_image_id) for record in existing_records),
        default=0,
    )
    records = list(existing_records)

    for catalog_id, source_path in _discover_session_sources(
        private_root=private_root,
        session_id=session_id,
        catalog=catalog,
    ):
        relative_path = source_path.relative_to(private_root).as_posix()
        existing = existing_by_path.get(relative_path)
        if existing is not None:
            expected = (catalog_id, session_id, split, ImageSource.OWNER_CAPTURE)
            actual = (
                existing.catalog_id,
                existing.capture_session_id,
                existing.split,
                existing.source,
            )
            if actual != expected:
                raise PreparationError(
                    f"existing manifest assignment conflicts with {relative_path}"
                )
            continue

        next_image_number += 1
        next_source_number += 1
        records.append(
            ImageRecord(
                image_id=f"img-{next_image_number:06d}",
                relative_path=relative_path,
                catalog_id=catalog_id,
                capture_session_id=session_id,
                source_image_id=f"src-{next_source_number:06d}",
                split=split,
                source=ImageSource.OWNER_CAPTURE,
            )
        )

    catalog_ids = catalog.by_id()
    for record in records:
        if record.catalog_id is not None and record.catalog_id not in catalog_ids:
            raise PreparationError(
                f"manifest references an unknown catalog ID: {record.catalog_id}"
            )
        source_path = private_root / record.relative_path
        if not source_path.is_file():
            raise PreparationError(
                f"manifest source image does not exist: {source_path}"
            )

    return ImageManifest(records=tuple(records))


def _atomic_csv(
    *, path: Path, columns: Sequence[str], rows: Iterable[dict[str, Any]]
) -> None:
    """Write a CSV through a neighboring temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _write_image_manifest(*, path: Path, manifest: ImageManifest) -> None:
    """Write validated image records in the declared column order."""
    rows: list[dict[str, str]] = []
    for record in manifest.records:
        rows.append(
            {
                "image_id": record.image_id,
                "relative_path": record.relative_path,
                "catalog_id": record.catalog_id or "",
                "unknown_identity_id": record.unknown_identity_id or "",
                "capture_session_id": record.capture_session_id,
                "source_image_id": record.source_image_id,
                "split": record.split.value,
                "source": record.source.value,
            }
        )
    _atomic_csv(path=path, columns=image_manifest_csv_columns(), rows=rows)


def _processed_relative_path(record: ImageRecord) -> Path:
    """Map an original path to the corresponding JPEG path."""
    return Path(record.relative_path).with_suffix(".jpg")


def _prepare_image(
    *, record: ImageRecord, private_root: Path, processed_root: Path, quality: int
) -> PreparedImage:
    """Decode, orient, and convert one original without modifying it."""
    source_path = private_root / record.relative_path
    processed_relative_path = _processed_relative_path(record)
    output_path = processed_root / processed_relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")

    try:
        with Image.open(source_path) as opened:
            oriented = ImageOps.exif_transpose(opened)
            image = oriented.convert("RGB")
            width, height = image.size
            image.save(
                temporary_path,
                format="JPEG",
                quality=quality,
                optimize=True,
            )
    except (OSError, ValueError) as error:
        temporary_path.unlink(missing_ok=True)
        raise PreparationError(f"could not decode {source_path}: {error}") from error

    temporary_path.replace(output_path)
    return PreparedImage(
        image_id=record.image_id,
        source_relative_path=record.relative_path,
        processed_relative_path=processed_relative_path.as_posix(),
        source_sha256=_sha256(source_path),
        processed_sha256=_sha256(output_path),
        width=width,
        height=height,
    )


def _contact_sheet(
    *,
    prepared_images: Sequence[PreparedImage],
    processed_root: Path,
    output_path: Path,
) -> None:
    """Build one compact visual audit sheet for an identity and session."""
    columns = 4
    cell_width = 260
    image_height = 210
    label_height = 34
    rows = (len(prepared_images) + columns - 1) // columns
    canvas = Image.new(
        "RGB", (columns * cell_width, rows * (image_height + label_height)), "white"
    )
    draw = ImageDraw.Draw(canvas)

    for index, prepared in enumerate(prepared_images):
        with Image.open(processed_root / prepared.processed_relative_path) as opened:
            thumbnail = opened.copy()
        thumbnail.thumbnail((cell_width - 16, image_height - 8))
        column = index % columns
        row = index // columns
        left = column * cell_width + (cell_width - thumbnail.width) // 2
        top = row * (image_height + label_height) + 4
        canvas.paste(thumbnail, (left, top))
        source_number = Path(prepared.source_relative_path).stem
        label = f"{prepared.image_id}  {source_number}"
        draw.text(
            (column * cell_width + 8, top + image_height + 2),
            label,
            fill="black",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    canvas.save(temporary_path, format="JPEG", quality=90, optimize=True)
    temporary_path.replace(output_path)


def _write_audit_outputs(
    *,
    catalog: Catalog,
    manifest: ImageManifest,
    prepared_images: Sequence[PreparedImage],
    processed_root: Path,
    updated_session_id: str,
) -> tuple[Path, int, str]:
    """Write contact sheets and a machine readable audit summary."""
    prepared_by_id = {image.image_id: image for image in prepared_images}
    grouped: dict[tuple[str, str], list[PreparedImage]] = defaultdict(list)
    for record in manifest.records:
        if record.catalog_id is None:
            continue
        grouped[(record.capture_session_id, record.catalog_id)].append(
            prepared_by_id[record.image_id]
        )

    for (session_id, catalog_id), images in grouped.items():
        _contact_sheet(
            prepared_images=images,
            processed_root=processed_root,
            output_path=(processed_root / "audit" / session_id / f"{catalog_id}.jpg"),
        )

    fingerprint = dataset_fingerprint(catalog=catalog, manifest=manifest)
    counts_by_session: dict[str, dict[str, int]] = defaultdict(dict)
    for (session_id, catalog_id), images in grouped.items():
        counts_by_session[session_id][catalog_id] = len(images)

    summary = {
        "schema_version": 1,
        "updated_session_id": updated_session_id,
        "dataset_fingerprint": fingerprint,
        "image_count": len(prepared_images),
        "identity_count": len({record.identity_key for record in manifest.records}),
        "counts_by_session": {
            session_id: dict(sorted(identity_counts.items()))
            for session_id, identity_counts in sorted(counts_by_session.items())
        },
    }
    summary_path = processed_root / "audit" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(summary_path)
    return summary_path, len(grouped), fingerprint


def prepare_session(
    *,
    private_root: Path,
    processed_root: Path,
    session_id: str,
    split: DatasetSplit,
    jpeg_quality: int = 95,
) -> PreparationResult:
    """Prepare one new session and refresh all derived dataset artifacts.

    Parameters
    ----------
    private_root
        Root containing the private catalog, manifest, and original images.
    processed_root
        Root for reproducible JPEG derivatives and audit artifacts.
    session_id
        Capture session to add or refresh.
    split
        Dataset split assigned to the complete session.
    jpeg_quality
        JPEG quality from 1 through 100.

    Returns
    -------
    PreparationResult
        Paths, counts, and the resulting dataset fingerprint.

    Raises
    ------
    PreparationError
        If discovery, validation, or image decoding fails.
    """
    if not 1 <= jpeg_quality <= 100:
        raise PreparationError("JPEG quality must be between 1 and 100")

    _register_heif_support()
    catalog_path = private_root / "catalog.csv"
    catalog = load_catalog_csv(path=catalog_path)
    manifest = _updated_manifest(
        private_root=private_root,
        session_id=session_id,
        split=split,
        catalog=catalog,
    )

    prepared_images = tuple(
        _prepare_image(
            record=record,
            private_root=private_root,
            processed_root=processed_root,
            quality=jpeg_quality,
        )
        for record in manifest.records
    )

    manifest_path = private_root / "images.csv"
    prepared_manifest_path = processed_root / "images.csv"
    _write_image_manifest(path=manifest_path, manifest=manifest)
    _atomic_csv(
        path=prepared_manifest_path,
        columns=PREPARED_COLUMNS,
        rows=(image.csv_row() for image in prepared_images),
    )
    summary_path, contact_sheet_count, fingerprint = _write_audit_outputs(
        catalog=catalog,
        manifest=manifest,
        prepared_images=prepared_images,
        processed_root=processed_root,
        updated_session_id=session_id,
    )

    return PreparationResult(
        manifest_path=manifest_path,
        prepared_manifest_path=prepared_manifest_path,
        summary_path=summary_path,
        image_count=len(prepared_images),
        identity_count=len({record.identity_key for record in manifest.records}),
        contact_sheet_count=contact_sheet_count,
        fingerprint=fingerprint,
    )
