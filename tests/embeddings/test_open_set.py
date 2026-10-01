"""Tests for validation only unknown rejection threshold selection."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from usagi_visual_identifier.data.models import DatasetSplit
from usagi_visual_identifier.embeddings.artifacts import (
    EmbeddingArtifact,
    EmbeddingError,
)
from usagi_visual_identifier.embeddings.open_set import (
    ConfidenceMethod,
    compare_confidence_methods,
    select_open_set_threshold,
    write_open_set_report,
)


def _open_set_artifact() -> EmbeddingArtifact:
    """Create known gallery, known validation, and unknown validation rows."""
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
            [0.98, 0.02],
            [0.02, 0.98],
            [0.70, 0.70],
            [-0.70, 0.70],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return EmbeddingArtifact(
        image_ids=tuple(f"img-{number:06d}" for number in range(1, 9)),
        identity_ids=(
            "usagi-0001",
            "usagi-0001",
            "usagi-0002",
            "usagi-0002",
            "usagi-0001",
            "usagi-0002",
            "unknown-0001",
            "unknown-0002",
        ),
        capture_session_ids=(
            "session-001",
            "session-001",
            "session-001",
            "session-001",
            "session-002",
            "session-002",
            "session-004",
            "session-005",
        ),
        splits=(
            "train",
            "train",
            "train",
            "train",
            "validation",
            "validation",
            "validation",
            "validation",
        ),
        processed_relative_paths=tuple(f"images/{number}.jpg" for number in range(8)),
        embeddings=embeddings,
        metadata={},
    )


def test_select_open_set_threshold_separates_invented_queries(tmp_path: Path) -> None:
    """The selected point retains known queries and rejects clear unknowns."""
    report = select_open_set_threshold(
        artifact=_open_set_artifact(),
        minimum_known_correct_acceptance=1.0,
    )

    assert report.known_query_count == 2
    assert report.unknown_query_count == 2
    assert report.selected.known_correct_acceptance_rate == 1.0
    assert report.selected.unknown_false_acceptance_rate == 0.0
    assert report.selected.accepted_known_accuracy == 1.0
    assert all(query.accepted for query in report.queries if query.is_known)
    assert not any(query.accepted for query in report.queries if not query.is_known)
    assert report.curve[0].known_accepted == 0

    output_path = tmp_path / "open_set.json"
    write_open_set_report(path=output_path, report=report)
    saved = output_path.read_text(encoding="utf-8")
    assert '"unknown_false_acceptance_rate": 0.0' in saved
    assert '"accepted": false' in saved


def test_threshold_selection_requires_validation_known_and_unknown_queries() -> None:
    """Selection rejects test tuning and one sided validation data."""
    artifact = _open_set_artifact()
    with pytest.raises(EmbeddingError, match="validation only"):
        select_open_set_threshold(
            artifact=replace(
                artifact,
                splits=("train", "train", "train", "train", *("test",) * 4),
            ),
            query_split=DatasetSplit.TEST,
        )
    with pytest.raises(EmbeddingError, match="known and unknown"):
        select_open_set_threshold(
            artifact=replace(
                artifact,
                splits=(
                    "train",
                    "train",
                    "train",
                    "train",
                    "validation",
                    "validation",
                    "test",
                    "test",
                ),
            )
        )
    with pytest.raises(EmbeddingError, match="must be in"):
        select_open_set_threshold(
            artifact=artifact,
            minimum_known_correct_acceptance=0.0,
        )


def test_open_set_gallery_cannot_contain_unknown_identities() -> None:
    """Unknown identities are queries and never gallery references."""
    artifact = _open_set_artifact()
    with pytest.raises(EmbeddingError, match="known gallery"):
        select_open_set_threshold(
            artifact=replace(
                artifact,
                identity_ids=(
                    "unknown-0003",
                    *artifact.identity_ids[1:],
                ),
            )
        )


def test_open_set_requires_independent_gallery_and_query_rows() -> None:
    """Missing splits and same split evaluation fail with clear errors."""
    artifact = _open_set_artifact()
    with pytest.raises(EmbeddingError, match="gallery split has no"):
        select_open_set_threshold(
            artifact=artifact,
            gallery_split=DatasetSplit.TEST,
        )
    with pytest.raises(EmbeddingError, match="query split has no"):
        select_open_set_threshold(
            artifact=replace(
                artifact,
                splits=("train",) * len(artifact.splits),
            )
        )


def test_open_set_reports_when_known_retention_is_impossible() -> None:
    """The retention requirement is not silently weakened."""
    artifact = _open_set_artifact()
    confused = artifact.embeddings.copy()
    confused[4] = np.asarray([0.0, 1.0], dtype=np.float32)
    with pytest.raises(EmbeddingError, match="no threshold retains"):
        select_open_set_threshold(
            artifact=replace(artifact, embeddings=confused),
            minimum_known_correct_acceptance=1.0,
        )


@pytest.mark.parametrize("method", tuple(ConfidenceMethod))
def test_each_confidence_method_uses_validation_only(
    method: ConfidenceMethod,
) -> None:
    """Every predeclared confidence signal produces a valid operating curve."""
    report = select_open_set_threshold(
        artifact=_open_set_artifact(),
        minimum_known_correct_acceptance=1.0,
        confidence_method=method,
    )
    assert report.confidence_method is method
    assert report.selected.known_correct_accepted == 2
    assert all(query.accepted for query in report.queries if query.is_known)


def test_confidence_comparison_prefers_simple_method_on_tie() -> None:
    """The declared complexity order makes tied selection deterministic."""
    comparison = compare_confidence_methods(
        artifact=_open_set_artifact(),
        minimum_known_correct_acceptance=1.0,
    )
    assert comparison.selected_method is ConfidenceMethod.MAX_SIMILARITY
    assert len(comparison.reports) == len(ConfidenceMethod)
