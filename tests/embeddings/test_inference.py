"""Tests for single image frozen retrieval and latency summaries."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from usagi_visual_identifier.embeddings import (
    EmbeddingArtifact,
    FrozenRetrievalPredictor,
    load_query_image,
)
from usagi_visual_identifier.embeddings.artifacts import EmbeddingError


class FakeEncoder:
    """Return one deterministic embedding without loading a real model."""

    @property
    def model_id(self) -> str:
        """Return the invented model identifier."""
        return "example/model"

    @property
    def model_revision(self) -> str:
        """Return the invented model revision."""
        return "commit"

    @property
    def device_name(self) -> str:
        """Return the invented device name."""
        return "cpu"

    def encode(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        """Return a fixed nonnormalized query vector."""
        assert len(images) == 1
        return np.asarray([[3.0, 0.0]], dtype=np.float32)


def _artifact() -> EmbeddingArtifact:
    """Create a gallery with an excluded test row."""
    return EmbeddingArtifact(
        image_ids=("train-a", "train-b", "sealed-test"),
        identity_ids=("usagi-0001", "usagi-0002", "usagi-0003"),
        capture_session_ids=("session-001", "session-001", "session-003"),
        splits=("train", "train", "test"),
        processed_relative_paths=("a.jpg", "b.jpg", "test.jpg"),
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        metadata={"model_id": "example/model", "model_revision": "commit"},
    )


def test_predict_uses_only_train_gallery_and_records_latency(tmp_path: Path) -> None:
    """One image returns ranked references without exposing the test split."""
    query_path = tmp_path / "query.png"
    Image.new("RGB", (12, 10), "red").save(query_path)
    predictor = FrozenRetrievalPredictor(artifact=_artifact(), encoder=FakeEncoder())

    prediction = predictor.predict(path=query_path, top_k=2)

    assert predictor.device_name == "cpu"
    assert predictor.gallery_size == 2
    assert [match.identity_id for match in prediction.matches] == [
        "usagi-0001",
        "usagi-0002",
    ]
    assert prediction.matches[0].reference_image_ids == ("train-a",)
    assert prediction.total_seconds >= 0


def test_benchmark_summarizes_repeated_predictions(tmp_path: Path) -> None:
    """Warm prediction timings produce finite ordered summary statistics."""
    query_path = tmp_path / "query.jpg"
    Image.new("RGB", (8, 8), "blue").save(query_path)
    predictor = FrozenRetrievalPredictor(artifact=_artifact(), encoder=FakeEncoder())

    summary = predictor.benchmark(path=query_path, warmup_runs=1, measured_runs=3)

    assert summary.warmup_runs == 1
    assert summary.measured_runs == 3
    assert 0 <= summary.minimum_seconds <= summary.median_seconds
    assert summary.median_seconds <= summary.maximum_seconds
    assert summary.mean_seconds >= 0
    assert summary.p95_seconds >= 0


def test_inference_rejects_invalid_inputs(tmp_path: Path) -> None:
    """Missing files, bad counts, and model mismatches fail clearly."""
    missing = tmp_path / "missing.jpg"
    with pytest.raises(EmbeddingError, match="does not exist"):
        load_query_image(missing)

    mismatched = _artifact()
    mismatched = EmbeddingArtifact(
        image_ids=mismatched.image_ids,
        identity_ids=mismatched.identity_ids,
        capture_session_ids=mismatched.capture_session_ids,
        splits=mismatched.splits,
        processed_relative_paths=mismatched.processed_relative_paths,
        embeddings=mismatched.embeddings,
        metadata={"model_id": "different/model", "model_revision": "commit"},
    )
    with pytest.raises(EmbeddingError, match="does not match"):
        FrozenRetrievalPredictor(artifact=mismatched, encoder=FakeEncoder())

    query_path = tmp_path / "query.png"
    Image.new("RGB", (8, 8), "green").save(query_path)
    predictor = FrozenRetrievalPredictor(artifact=_artifact(), encoder=FakeEncoder())
    with pytest.raises(EmbeddingError, match="top k"):
        predictor.predict(path=query_path, top_k=0)
    with pytest.raises(EmbeddingError, match="measured runs"):
        predictor.benchmark(path=query_path, measured_runs=0)
