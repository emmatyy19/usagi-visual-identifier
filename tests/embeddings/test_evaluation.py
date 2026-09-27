"""Tests for exact identity retrieval metrics."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from usagi_visual_identifier.data.models import DatasetSplit
from usagi_visual_identifier.embeddings.aggregation import (
    AggregationConfig,
    AggregationMethod,
)
from usagi_visual_identifier.embeddings.artifacts import (
    EmbeddingArtifact,
    EmbeddingError,
)
from usagi_visual_identifier.embeddings.evaluation import (
    compare_retrieval_aggregations,
    evaluate_retrieval,
    write_aggregation_comparison,
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
    assert validation.aggregation.method is AggregationMethod.NEAREST_IMAGE
    first_ranking = validation.queries[0]["ranked_identities"]
    assert isinstance(first_ranking, list)
    assert first_ranking[0]["reference_image_ids"]

    output_path = tmp_path / "report.json"
    write_retrieval_report(path=output_path, report=validation)
    assert '"top1_accuracy": 1.0' in output_path.read_text(encoding="utf-8")


def test_aggregation_comparison_uses_identical_queries(tmp_path: Path) -> None:
    """A comparison records every configured method and same split warning."""
    configurations = (
        AggregationConfig(method=AggregationMethod.NEAREST_IMAGE),
        AggregationConfig(method=AggregationMethod.PROTOTYPE),
        AggregationConfig(
            method=AggregationMethod.TOP_K_MEAN,
            reference_count=2,
        ),
        AggregationConfig(
            method=AggregationMethod.WEIGHTED_NEIGHBORS,
            reference_count=3,
        ),
    )
    comparison = compare_retrieval_aggregations(
        artifact=_artifact(),
        gallery_split=DatasetSplit.TRAIN,
        query_split=DatasetSplit.TRAIN,
        configurations=configurations,
    )

    assert comparison.exploratory_same_split is True
    assert [report.aggregation for report in comparison.reports] == list(configurations)
    assert {report.query_count for report in comparison.reports} == {4}
    assert comparison.evaluation_role == "exploratory"
    assert comparison.selection_allowed is False
    assert comparison.as_dict()["selection_allowed"] is False

    output_path = tmp_path / "comparison.json"
    write_aggregation_comparison(path=output_path, comparison=comparison)
    saved = output_path.read_text(encoding="utf-8")
    assert '"method": "prototype"' in saved
    assert '"selection_allowed": false' in saved


def test_only_validation_comparisons_allow_model_selection() -> None:
    """Validation can guide choices while final test results cannot."""
    artifact = _artifact()
    configuration = AggregationConfig(method=AggregationMethod.NEAREST_IMAGE)
    validation = compare_retrieval_aggregations(
        artifact=artifact,
        gallery_split=DatasetSplit.TRAIN,
        query_split=DatasetSplit.VALIDATION,
        configurations=(configuration,),
    )
    final_test = compare_retrieval_aggregations(
        artifact=replace(
            artifact,
            splits=("train", "train", "train", "train", "test", "test"),
        ),
        gallery_split=DatasetSplit.TRAIN,
        query_split=DatasetSplit.TEST,
        configurations=(configuration,),
    )

    assert validation.evaluation_role == "validation"
    assert validation.selection_allowed is True
    assert final_test.evaluation_role == "final_test"
    assert final_test.selection_allowed is False


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
    with pytest.raises(EmbeddingError, match="at least one aggregation"):
        compare_retrieval_aggregations(
            artifact=artifact,
            gallery_split=DatasetSplit.TRAIN,
            query_split=DatasetSplit.VALIDATION,
            configurations=(),
        )
    duplicate = AggregationConfig(method=AggregationMethod.NEAREST_IMAGE)
    with pytest.raises(EmbeddingError, match="must be unique"):
        compare_retrieval_aggregations(
            artifact=artifact,
            gallery_split=DatasetSplit.TRAIN,
            query_split=DatasetSplit.VALIDATION,
            configurations=(duplicate, duplicate),
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
