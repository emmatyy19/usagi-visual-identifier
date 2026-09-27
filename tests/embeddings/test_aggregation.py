"""Tests for identity aggregation over normalized reference embeddings."""

import numpy as np
import pytest

from usagi_visual_identifier.embeddings.aggregation import (
    AggregationConfig,
    AggregationMethod,
    rank_gallery_identities,
)
from usagi_visual_identifier.embeddings.artifacts import EmbeddingError


def _inputs() -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Create three invented identities with known cosine relationships."""
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    gallery = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.6, 0.8],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    identities = ("a", "a", "b", "b", "c")
    image_ids = ("a-1", "a-2", "b-1", "b-2", "c-1")
    return query, gallery, identities, image_ids


@pytest.mark.parametrize(
    ("config", "expected_score", "expected_references"),
    [
        (
            AggregationConfig(
                method=AggregationMethod.NEAREST_IMAGE,
                reference_count=1,
            ),
            1.0,
            ("a-1",),
        ),
        (
            AggregationConfig(
                method=AggregationMethod.PROTOTYPE,
                reference_count=1,
            ),
            0.948683,
            ("a-1", "a-2"),
        ),
        (
            AggregationConfig(
                method=AggregationMethod.TOP_K_MEAN,
                reference_count=2,
            ),
            0.9,
            ("a-1", "a-2"),
        ),
    ],
)
def test_cosine_aggregation_rules(
    config: AggregationConfig,
    expected_score: float,
    expected_references: tuple[str, ...],
) -> None:
    """Nearest, prototype, and mean rules preserve their exact definitions."""
    query, gallery, identities, image_ids = _inputs()
    matches = rank_gallery_identities(
        query_embedding=query,
        gallery_embeddings=gallery,
        gallery_identity_ids=identities,
        gallery_image_ids=image_ids,
        config=config,
    )

    assert [match.identity_id for match in matches] == ["a", "b", "c"]
    assert matches[0].score == pytest.approx(expected_score)
    assert matches[0].reference_image_ids == expected_references


def test_weighted_neighbors_use_global_softmax_vote() -> None:
    """Only the strongest global references vote with normalized weights."""
    query, gallery, identities, image_ids = _inputs()
    matches = rank_gallery_identities(
        query_embedding=query,
        gallery_embeddings=gallery,
        gallery_identity_ids=identities,
        gallery_image_ids=image_ids,
        config=AggregationConfig(
            method=AggregationMethod.WEIGHTED_NEIGHBORS,
            reference_count=3,
            temperature=1.0,
        ),
    )

    by_identity = {match.identity_id: match for match in matches}
    assert matches[0].identity_id == "a"
    assert by_identity["a"].reference_image_ids == ("a-1", "a-2")
    assert by_identity["b"].reference_image_ids == ("b-1",)
    assert by_identity["c"].reference_image_ids == ()
    assert sum(match.score for match in matches) == pytest.approx(1.0)


def test_aggregation_configuration_and_inputs_are_validated() -> None:
    """Invalid parameters and vector contracts fail before ranking."""
    with pytest.raises(EmbeddingError, match="reference count"):
        AggregationConfig(
            method=AggregationMethod.TOP_K_MEAN,
            reference_count=0,
        )
    with pytest.raises(EmbeddingError, match="temperature"):
        AggregationConfig(
            method=AggregationMethod.WEIGHTED_NEIGHBORS,
            temperature=float("nan"),
        )

    query, gallery, identities, image_ids = _inputs()
    with pytest.raises(EmbeddingError, match="L2 normalized"):
        rank_gallery_identities(
            query_embedding=query * 2,
            gallery_embeddings=gallery,
            gallery_identity_ids=identities,
            gallery_image_ids=image_ids,
            config=AggregationConfig(method=AggregationMethod.NEAREST_IMAGE),
        )
    with pytest.raises(EmbeddingError, match="identity IDs"):
        rank_gallery_identities(
            query_embedding=query,
            gallery_embeddings=gallery,
            gallery_identity_ids=identities[:-1],
            gallery_image_ids=image_ids,
            config=AggregationConfig(method=AggregationMethod.NEAREST_IMAGE),
        )


def test_zero_length_prototype_is_rejected() -> None:
    """Opposing references cannot silently create an undefined prototype."""
    with pytest.raises(EmbeddingError, match="prototype has zero length"):
        rank_gallery_identities(
            query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            gallery_embeddings=np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
            gallery_identity_ids=("a", "a"),
            gallery_image_ids=("a-1", "a-2"),
            config=AggregationConfig(method=AggregationMethod.PROTOTYPE),
        )
