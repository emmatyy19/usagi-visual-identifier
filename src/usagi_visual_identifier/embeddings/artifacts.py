"""Build and store reproducible frozen image embedding artifacts."""

import csv
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from usagi_visual_identifier.data.fingerprint import dataset_fingerprint
from usagi_visual_identifier.data.loaders import (
    load_catalog_csv,
    load_image_manifest_csv,
)
from usagi_visual_identifier.data.models import DatasetSplit

PREPARED_COLUMNS = (
    "image_id",
    "source_relative_path",
    "processed_relative_path",
    "source_sha256",
    "processed_sha256",
    "width",
    "height",
)


class EmbeddingError(ValueError):
    """Raised when an embedding artifact cannot be built or loaded safely."""


class ImageEncoder(Protocol):
    """Minimal interface shared by the real and test encoders."""

    @property
    def model_id(self) -> str:
        """Return the immutable model identifier."""
        ...

    @property
    def model_revision(self) -> str:
        """Return the pinned model revision."""
        ...

    @property
    def device_name(self) -> str:
        """Return the device used for inference."""
        ...

    def encode(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        """Encode a nonempty image batch as a two dimensional matrix."""
        ...


@dataclass(frozen=True)
class EmbeddingExample:
    """One validated processed image ready for an encoder."""

    image_id: str
    identity_id: str
    capture_session_id: str
    split: DatasetSplit
    processed_relative_path: str
    image_path: Path


@dataclass(frozen=True)
class EmbeddingArtifact:
    """Embeddings and aligned provenance loaded in memory."""

    image_ids: tuple[str, ...]
    identity_ids: tuple[str, ...]
    capture_session_ids: tuple[str, ...]
    splits: tuple[str, ...]
    processed_relative_paths: tuple[str, ...]
    embeddings: NDArray[np.float32]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class EmbeddingBuildResult:
    """Paths and dimensions produced by one embedding run."""

    artifact_path: Path
    metadata_path: Path
    image_count: int
    embedding_dimension: int
    device_name: str
    dataset_fingerprint: str
    elapsed_seconds: float
    images_per_second: float


def _read_prepared_rows(*, path: Path) -> dict[str, dict[str, str]]:
    """Load the processed manifest with an exact header contract."""
    try:
        handle = path.open(encoding="utf-8-sig", newline="")
    except OSError as error:
        raise EmbeddingError(f"could not open {path}: {error}") from error

    rows: dict[str, dict[str, str]] = {}
    with handle:
        reader = csv.DictReader(handle)
        actual_columns = tuple(reader.fieldnames or ())
        if actual_columns != PREPARED_COLUMNS:
            raise EmbeddingError(
                f"unexpected columns in {path}; expected {PREPARED_COLUMNS}, "
                f"received {actual_columns}"
            )
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(
                not isinstance(row.get(column), str) for column in PREPARED_COLUMNS
            ):
                raise EmbeddingError(f"malformed CSV row at {path}:{line_number}")
            image_id = row["image_id"]
            if image_id in rows:
                raise EmbeddingError(f"duplicate processed image ID: {image_id}")
            rows[image_id] = {column: row[column] for column in PREPARED_COLUMNS}
    return rows


def load_prepared_examples(
    *, private_root: Path, processed_root: Path
) -> tuple[tuple[EmbeddingExample, ...], str]:
    """Join private image records to their prepared JPEG derivatives."""
    catalog = load_catalog_csv(path=private_root / "catalog.csv")
    manifest = load_image_manifest_csv(path=private_root / "images.csv")
    prepared_rows = _read_prepared_rows(path=processed_root / "images.csv")
    examples: list[EmbeddingExample] = []

    manifest_ids = {record.image_id for record in manifest.records}
    extra_ids = set(prepared_rows) - manifest_ids
    if extra_ids:
        raise EmbeddingError(
            f"processed manifest has unknown image IDs: {sorted(extra_ids)}"
        )

    for record in manifest.records:
        row = prepared_rows.get(record.image_id)
        if row is None:
            raise EmbeddingError(
                f"processed manifest is missing image ID: {record.image_id}"
            )
        if row["source_relative_path"] != record.relative_path:
            raise EmbeddingError(
                f"source path conflicts for image ID: {record.image_id}"
            )
        image_path = processed_root / row["processed_relative_path"]
        if not image_path.is_file():
            raise EmbeddingError(f"processed image does not exist: {image_path}")
        examples.append(
            EmbeddingExample(
                image_id=record.image_id,
                identity_id=record.identity_key,
                capture_session_id=record.capture_session_id,
                split=record.split,
                processed_relative_path=row["processed_relative_path"],
                image_path=image_path,
            )
        )

    fingerprint = dataset_fingerprint(catalog=catalog, manifest=manifest)
    return tuple(examples), fingerprint


def _load_rgb_image(path: Path) -> Image.Image:
    """Load one prepared image and detach it from the file handle."""
    try:
        with Image.open(path) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")
    except OSError as error:
        raise EmbeddingError(
            f"could not decode prepared image {path}: {error}"
        ) from error


def _normalize_embeddings(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return finite unit length embeddings."""
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise EmbeddingError("encoder must return a nonempty two dimensional matrix")
    if not np.isfinite(values).all():
        raise EmbeddingError("encoder returned a nonfinite embedding")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise EmbeddingError("encoder returned a zero length embedding")
    return np.asarray(values / norms, dtype=np.float32)


def _slug(value: str) -> str:
    """Create a stable filesystem name from a model identifier."""
    return value.casefold().replace("/", "_").replace(" ", "_")


def _artifact_sha256(path: Path) -> str:
    """Hash the serialized embedding matrix and aligned labels."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_embedding_artifact(
    *,
    examples: Sequence[EmbeddingExample],
    dataset_fingerprint_value: str,
    encoder: ImageEncoder,
    artifact_root: Path,
    batch_size: int = 8,
) -> EmbeddingBuildResult:
    """Encode prepared images and atomically save aligned local artifacts."""
    if not examples:
        raise EmbeddingError("at least one prepared image is required")
    if batch_size < 1:
        raise EmbeddingError("batch size must be positive")

    batches: list[NDArray[np.float32]] = []
    started_at = time.perf_counter()
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        images = [_load_rgb_image(example.image_path) for example in batch_examples]
        try:
            encoded = np.asarray(encoder.encode(images), dtype=np.float32)
        finally:
            for image in images:
                image.close()
        if encoded.shape[0] != len(batch_examples):
            raise EmbeddingError("encoder row count does not match the image batch")
        batches.append(encoded)

    embeddings = _normalize_embeddings(np.concatenate(batches, axis=0))
    elapsed_seconds = time.perf_counter() - started_at
    images_per_second = len(examples) / elapsed_seconds
    output_directory = artifact_root / "embeddings" / _slug(encoder.model_id)
    artifact_path = output_directory / "embeddings.npz"
    metadata_path = output_directory / "metadata.json"
    output_directory.mkdir(parents=True, exist_ok=True)

    temporary_artifact = artifact_path.with_name(f".{artifact_path.name}.tmp")
    with temporary_artifact.open("wb") as handle:
        np.savez_compressed(
            handle,
            image_ids=np.asarray(
                [example.image_id for example in examples], dtype=np.str_
            ),
            identity_ids=np.asarray(
                [example.identity_id for example in examples], dtype=np.str_
            ),
            capture_session_ids=np.asarray(
                [example.capture_session_id for example in examples], dtype=np.str_
            ),
            splits=np.asarray(
                [example.split.value for example in examples], dtype=np.str_
            ),
            processed_relative_paths=np.asarray(
                [example.processed_relative_path for example in examples],
                dtype=np.str_,
            ),
            embeddings=embeddings,
        )
    temporary_artifact.replace(artifact_path)

    metadata: dict[str, object] = {
        "schema_version": 1,
        "model_id": encoder.model_id,
        "model_revision": encoder.model_revision,
        "device": encoder.device_name,
        "dataset_fingerprint": dataset_fingerprint_value,
        "image_count": len(examples),
        "embedding_dimension": embeddings.shape[1],
        "embedding_pooling": "cls_token",
        "l2_normalized": True,
        "elapsed_seconds": elapsed_seconds,
        "images_per_second": images_per_second,
        "artifact_sha256": _artifact_sha256(artifact_path),
    }
    temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_metadata.replace(metadata_path)

    return EmbeddingBuildResult(
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        image_count=len(examples),
        embedding_dimension=embeddings.shape[1],
        device_name=encoder.device_name,
        dataset_fingerprint=dataset_fingerprint_value,
        elapsed_seconds=elapsed_seconds,
        images_per_second=images_per_second,
    )


def _string_tuple(values: NDArray[np.str_]) -> tuple[str, ...]:
    """Convert a saved NumPy string vector to ordinary strings."""
    return tuple(str(value) for value in values.tolist())


def load_embedding_artifact(
    *, artifact_path: Path, metadata_path: Path
) -> EmbeddingArtifact:
    """Load and verify a saved embedding artifact without pickle support."""
    try:
        metadata_raw: object = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EmbeddingError(f"could not load embedding metadata: {error}") from error
    if not isinstance(metadata_raw, dict):
        raise EmbeddingError("embedding metadata must be a JSON object")
    metadata_object: dict[str, object] = {}
    for key, value in cast(dict[object, object], metadata_raw).items():
        if not isinstance(key, str):
            raise EmbeddingError("embedding metadata keys must be strings")
        metadata_object[key] = value
    if metadata_object.get("artifact_sha256") != _artifact_sha256(artifact_path):
        raise EmbeddingError("embedding artifact hash does not match its metadata")

    try:
        with np.load(artifact_path, allow_pickle=False) as saved:
            image_ids = _string_tuple(saved["image_ids"])
            identity_ids = _string_tuple(saved["identity_ids"])
            session_ids = _string_tuple(saved["capture_session_ids"])
            splits = _string_tuple(saved["splits"])
            paths = _string_tuple(saved["processed_relative_paths"])
            embeddings = np.asarray(saved["embeddings"], dtype=np.float32)
    except (OSError, KeyError, ValueError) as error:
        raise EmbeddingError(f"could not load embedding artifact: {error}") from error

    row_count = len(image_ids)
    aligned_lengths = {
        row_count,
        len(identity_ids),
        len(session_ids),
        len(splits),
        len(paths),
    }
    if len(aligned_lengths) != 1 or embeddings.shape[0] != row_count:
        raise EmbeddingError("embedding artifact arrays are not aligned")
    normalized = _normalize_embeddings(embeddings)
    if not np.allclose(normalized, embeddings, atol=1e-5):
        raise EmbeddingError("saved embeddings are not L2 normalized")

    return EmbeddingArtifact(
        image_ids=image_ids,
        identity_ids=identity_ids,
        capture_session_ids=session_ids,
        splits=splits,
        processed_relative_paths=paths,
        embeddings=embeddings,
        metadata=metadata_object,
    )
