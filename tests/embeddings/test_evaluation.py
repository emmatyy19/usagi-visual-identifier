"""Tests for exact identity retrieval metrics."""

from pathlib import Path

import numpy as np
import pytest

from usagi_visual_identifier.data.models import DatasetSplit
from usagi_visual_identifier.embeddings.artifacts import (
    EmbeddingArtifact,
    EmbeddingError,
)
from usagi_visual_identifier.embeddings.evaluation import (
    evaluate_retrieval,
    write_retrieval_report,
)


def _artifact() -> EmbeddingArtifact:
    """Create two separable identities across train and validation."""
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
            [0.98, 0.02],
            [0.02, 0.98],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return EmbeddingArtifact(
        image_ids=tuple(f"img-{number:06d}" for number in range(1, 7)),
        identity_ids=(
            "usagi-0001",
            "usagi-0001",
            "usagi-0002",
            "usagi-0002",
            "usagi-0001",
            "usagi-0002",
        ),
        capture_session_ids=(
            "session-001",
            "session-001",
            "session-001",
            "session-001",
            "session-002",
            "session-002",
        ),
        splits=("train", "train", "train", "train", "validation", "validation"),
        processed_relative_paths=tuple(f"images/{number}.jpg" for number in range(6)),
        embeddings=embeddings,
        metadata={},
    )


def test_leave_one_out_and_cross_session_retrieval(tmp_path: Path) -> None:
    """Nearest image aggregation reports perfect invented retrieval."""
    artifact = _artifact()
    sanity = evaluate_retrieval(
        artifact=artifact,
        gallery_split=DatasetSplit.TRAIN,
        query_split=DatasetSplit.TRAIN,
    )
    validation = evaluate_retrieval(
        artifact=artifact,
        gallery_split=DatasetSplit.TRAIN,
        query_split=DatasetSplit.VALIDATION,
    )

    assert sanity.leave_one_out is True
    assert sanity.query_count == 4
    assert sanity.top1_accuracy == 1.0
    assert validation.leave_one_out is False
    assert validation.top3_accuracy == 1.0
    assert validation.mean_reciprocal_rank == 1.0
    assert validation.confusion_pairs == ()

    output_path = tmp_path / "report.json"
    write_retrieval_report(path=output_path, report=validation)
    assert '"top1_accuracy": 1.0' in output_path.read_text(encoding="utf-8")


def test_retrieval_records_confusion_pairs() -> None:
    """Wrong nearest identities appear in the diagnostic report."""
    artifact = _artifact()
    swapped = artifact.embeddings.copy()
    swapped[4] = np.asarray([0.0, 1.0], dtype=np.float32)
    confused = EmbeddingArtifact(
        image_ids=artifact.image_ids,
        identity_ids=artifact.identity_ids,
        capture_session_ids=artifact.capture_session_ids,
        splits=artifact.splits,
        processed_relative_paths=artifact.processed_relative_paths,
        embeddings=swapped,
        metadata={},
    )

    report = evaluate_retrieval(
        artifact=confused,
        gallery_split=DatasetSplit.TRAIN,
        query_split=DatasetSplit.VALIDATION,
    )

    assert report.top1_accuracy == 0.5
    assert report.macro_recall == 0.5
    assert report.confusion_pairs == (("usagi-0001", "usagi-0002", 1),)


def test_retrieval_requires_gallery_queries_and_peers() -> None:
    """Invalid split and leave one out configurations fail clearly."""
    artifact = _artifact()
    with pytest.raises(EmbeddingError, match="gallery split has no"):
        evaluate_retrieval(
            artifact=artifact,
            gallery_split=DatasetSplit.TEST,
            query_split=DatasetSplit.VALIDATION,
        )
    with pytest.raises(EmbeddingError, match="top_k"):
        evaluate_retrieval(
            artifact=artifact,
            gallery_split=DatasetSplit.TRAIN,
            query_split=DatasetSplit.TRAIN,
            top_k=0,
        )

    one_per_identity = EmbeddingArtifact(
        image_ids=("img-000001", "img-000002"),
        identity_ids=("usagi-0001", "usagi-0002"),
        capture_session_ids=("session-001", "session-001"),
        splits=("train", "train"),
        processed_relative_paths=("1.jpg", "2.jpg"),
        embeddings=np.eye(2, dtype=np.float32),
        metadata={},
    )
    with pytest.raises(EmbeddingError, match="requires two"):
        evaluate_retrieval(
            artifact=one_per_identity,
            gallery_split=DatasetSplit.TRAIN,
            query_split=DatasetSplit.TRAIN,
        )
