"""Single image inference and local latency measurement."""

import importlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from usagi_visual_identifier.embeddings.aggregation import (
    DEFAULT_AGGREGATION_CONFIG,
    AggregationConfig,
    IdentityMatch,
    rank_gallery_identities,
)
from usagi_visual_identifier.embeddings.artifacts import (
    EmbeddingArtifact,
    EmbeddingError,
    ImageEncoder,
)


class _HeifModule(Protocol):
    """Typed boundary for the optional Pillow HEIF registration function."""

    register_heif_opener: Callable[[], None]


@dataclass(frozen=True)
class ImagePrediction:
    """Ranked identities and measured stages for one query image."""

    matches: tuple[IdentityMatch, ...]
    load_seconds: float
    encode_seconds: float
    retrieval_seconds: float

    @property
    def total_seconds(self) -> float:
        """Return file loading, encoding, and retrieval time combined."""
        return self.load_seconds + self.encode_seconds + self.retrieval_seconds


@dataclass(frozen=True)
class LatencySummary:
    """Summary statistics from repeated warm predictions."""

    warmup_runs: int
    measured_runs: int
    mean_seconds: float
    median_seconds: float
    p95_seconds: float
    minimum_seconds: float
    maximum_seconds: float


def _register_heif_support(path: Path) -> None:
    """Register HEIF decoding only when the query format requires it."""
    if path.suffix.casefold() not in {".heic", ".heif"}:
        return
    module = cast(_HeifModule, importlib.import_module("pillow_heif"))
    module.register_heif_opener()


def load_query_image(path: Path) -> Image.Image:
    """Load one query image with orientation applied and no open file handle."""
    if not path.is_file():
        raise EmbeddingError(f"query image does not exist: {path}")
    _register_heif_support(path)
    try:
        with Image.open(path) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")
    except OSError as error:
        raise EmbeddingError(f"could not decode query image {path}: {error}") from error


def _normalize_query(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Validate and normalize the single embedding returned by an encoder."""
    if values.ndim != 2 or values.shape[0] != 1 or values.shape[1] == 0:
        raise EmbeddingError(
            "single image inference requires exactly one nonempty embedding"
        )
    vector = np.asarray(values[0], dtype=np.float32)
    if not np.isfinite(vector).all():
        raise EmbeddingError("query encoder returned a nonfinite embedding")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise EmbeddingError("query encoder returned a zero length embedding")
    return np.asarray(vector / norm, dtype=np.float32)


class FrozenRetrievalPredictor:
    """Reusable frozen encoder and filtered reference gallery."""

    def __init__(
        self,
        *,
        artifact: EmbeddingArtifact,
        encoder: ImageEncoder,
        gallery_split: str = "train",
        aggregation: AggregationConfig = DEFAULT_AGGREGATION_CONFIG,
    ) -> None:
        """Keep only the requested gallery split and validate model provenance."""
        artifact_model_id = artifact.metadata.get("model_id")
        artifact_revision = artifact.metadata.get("model_revision")
        if artifact_model_id != encoder.model_id:
            raise EmbeddingError(
                "encoder model does not match the saved gallery artifact"
            )
        if artifact_revision != encoder.model_revision:
            raise EmbeddingError(
                "encoder revision does not match the saved gallery artifact"
            )

        indices = [
            index
            for index, split in enumerate(artifact.splits)
            if split == gallery_split
        ]
        if not indices:
            raise EmbeddingError(f"gallery split has no images: {gallery_split}")

        self._encoder = encoder
        self._aggregation = aggregation
        self._gallery_embeddings = np.asarray(
            artifact.embeddings[indices], dtype=np.float32
        )
        self._gallery_identity_ids = tuple(
            artifact.identity_ids[index] for index in indices
        )
        self._gallery_image_ids = tuple(artifact.image_ids[index] for index in indices)

    @property
    def device_name(self) -> str:
        """Return the encoder device used for prediction."""
        return self._encoder.device_name

    @property
    def gallery_size(self) -> int:
        """Return the number of reference images searched per query."""
        return len(self._gallery_image_ids)

    def predict(self, *, path: Path, top_k: int = 3) -> ImagePrediction:
        """Encode one file and return its strongest identity matches."""
        if top_k < 1:
            raise EmbeddingError("top k must be positive")

        started = time.perf_counter()
        image = load_query_image(path)
        loaded = time.perf_counter()
        query = _normalize_query(self._encoder.encode([image]))
        encoded = time.perf_counter()
        matches = rank_gallery_identities(
            query_embedding=query,
            gallery_embeddings=self._gallery_embeddings,
            gallery_identity_ids=self._gallery_identity_ids,
            gallery_image_ids=self._gallery_image_ids,
            config=self._aggregation,
        )
        retrieved = time.perf_counter()
        return ImagePrediction(
            matches=matches[:top_k],
            load_seconds=loaded - started,
            encode_seconds=encoded - loaded,
            retrieval_seconds=retrieved - encoded,
        )

    def benchmark(
        self,
        *,
        path: Path,
        warmup_runs: int = 3,
        measured_runs: int = 20,
    ) -> LatencySummary:
        """Measure repeated end to end prediction after explicit warmup runs."""
        if warmup_runs < 0:
            raise EmbeddingError("warmup runs cannot be negative")
        if measured_runs < 1:
            raise EmbeddingError("measured runs must be positive")
        for _ in range(warmup_runs):
            self.predict(path=path, top_k=1)
        timings = np.asarray(
            [
                self.predict(path=path, top_k=1).total_seconds
                for _ in range(measured_runs)
            ],
            dtype=np.float64,
        )
        return LatencySummary(
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            mean_seconds=float(np.mean(timings)),
            median_seconds=float(np.median(timings)),
            p95_seconds=float(np.quantile(timings, 0.95)),
            minimum_seconds=float(np.min(timings)),
            maximum_seconds=float(np.max(timings)),
        )
