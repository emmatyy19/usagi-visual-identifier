"""Identity aggregation rules for exact embedding retrieval."""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from usagi_visual_identifier.embeddings.artifacts import EmbeddingError


class AggregationMethod(StrEnum):
    """Supported ways to combine reference image evidence by identity."""

    NEAREST_IMAGE = "nearest_image"
    PROTOTYPE = "prototype"
    TOP_K_MEAN = "top_k_mean"
    WEIGHTED_NEIGHBORS = "weighted_neighbors"


@dataclass(frozen=True)
class AggregationConfig:
    """Parameters for one identity aggregation rule."""

    method: AggregationMethod
    reference_count: int = 3
    temperature: float = 0.07

    def __post_init__(self) -> None:
        """Reject parameters that would make a comparison undefined."""
        if self.reference_count < 1:
            raise EmbeddingError("aggregation reference count must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise EmbeddingError("aggregation temperature must be finite and positive")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON ready configuration."""
        return {
            "method": self.method.value,
            "reference_count": self.reference_count,
            "temperature": self.temperature,
        }


@dataclass(frozen=True)
class IdentityMatch:
    """One aggregated identity score and the references supporting it."""

    identity_id: str
    score: float
    reference_image_ids: tuple[str, ...]
    reference_similarities: tuple[float, ...]


DEFAULT_AGGREGATION_CONFIG = AggregationConfig(
    method=AggregationMethod.NEAREST_IMAGE,
    reference_count=1,
)


def _validate_inputs(
    *,
    query_embedding: NDArray[np.float32],
    gallery_embeddings: NDArray[np.float32],
    gallery_identity_ids: Sequence[str],
    gallery_image_ids: Sequence[str],
) -> None:
    """Validate the aligned unit length vectors used by every rule."""
    if query_embedding.ndim != 1 or query_embedding.size == 0:
        raise EmbeddingError("query embedding must be a nonempty vector")
    if gallery_embeddings.ndim != 2 or gallery_embeddings.shape[0] == 0:
        raise EmbeddingError("gallery embeddings must be a nonempty matrix")
    if gallery_embeddings.shape[1] != query_embedding.shape[0]:
        raise EmbeddingError("query and gallery embedding dimensions do not match")
    if len(gallery_identity_ids) != gallery_embeddings.shape[0]:
        raise EmbeddingError("gallery identity IDs are not aligned with embeddings")
    if len(gallery_image_ids) != gallery_embeddings.shape[0]:
        raise EmbeddingError("gallery image IDs are not aligned with embeddings")
    if (
        not np.isfinite(query_embedding).all()
        or not np.isfinite(gallery_embeddings).all()
    ):
        raise EmbeddingError("aggregation embeddings must be finite")
    if not np.isclose(np.linalg.norm(query_embedding), 1.0, atol=1e-5):
        raise EmbeddingError("query embedding must be L2 normalized")
    gallery_norms = np.linalg.norm(gallery_embeddings, axis=1)
    if not np.allclose(gallery_norms, 1.0, atol=1e-5):
        raise EmbeddingError("gallery embeddings must be L2 normalized")


def _ranked_indices(
    *,
    indices: Sequence[int],
    similarities: NDArray[np.float32],
    image_ids: Sequence[str],
) -> list[int]:
    """Return deterministic reference indices from strongest to weakest."""
    return sorted(
        indices,
        key=lambda index: (-float(similarities[index]), image_ids[index]),
    )


def _evidence(
    *,
    indices: Sequence[int],
    similarities: NDArray[np.float32],
    image_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Return aligned reference IDs and raw cosine similarities."""
    ordered = _ranked_indices(
        indices=indices,
        similarities=similarities,
        image_ids=image_ids,
    )
    return (
        tuple(image_ids[index] for index in ordered),
        tuple(float(similarities[index]) for index in ordered),
    )


def rank_gallery_identities(
    *,
    query_embedding: NDArray[np.float32],
    gallery_embeddings: NDArray[np.float32],
    gallery_identity_ids: Sequence[str],
    gallery_image_ids: Sequence[str],
    config: AggregationConfig,
) -> tuple[IdentityMatch, ...]:
    """Rank gallery identities using one explicit aggregation rule."""
    query = np.asarray(query_embedding, dtype=np.float32)
    gallery = np.asarray(gallery_embeddings, dtype=np.float32)
    _validate_inputs(
        query_embedding=query,
        gallery_embeddings=gallery,
        gallery_identity_ids=gallery_identity_ids,
        gallery_image_ids=gallery_image_ids,
    )
    similarities = np.asarray(gallery @ query, dtype=np.float32)
    indices_by_identity: dict[str, list[int]] = defaultdict(list)
    for index, identity_id in enumerate(gallery_identity_ids):
        indices_by_identity[identity_id].append(index)

    matches: list[IdentityMatch] = []
    if config.method is AggregationMethod.WEIGHTED_NEIGHBORS:
        all_indices = list(range(gallery.shape[0]))
        selected = _ranked_indices(
            indices=all_indices,
            similarities=similarities,
            image_ids=gallery_image_ids,
        )[: config.reference_count]
        selected_scores = similarities[selected]
        logits = (selected_scores - np.max(selected_scores)) / config.temperature
        unnormalized_weights = np.exp(logits)
        weights = unnormalized_weights / np.sum(unnormalized_weights)
        weight_by_index = {
            index: float(weight)
            for index, weight in zip(selected, weights, strict=True)
        }
        for identity_id in indices_by_identity:
            voting_indices = [
                index
                for index in selected
                if gallery_identity_ids[index] == identity_id
            ]
            reference_ids, reference_similarities = _evidence(
                indices=voting_indices,
                similarities=similarities,
                image_ids=gallery_image_ids,
            )
            matches.append(
                IdentityMatch(
                    identity_id=identity_id,
                    score=sum(weight_by_index[index] for index in voting_indices),
                    reference_image_ids=reference_ids,
                    reference_similarities=reference_similarities,
                )
            )
    else:
        for identity_id, identity_indices in indices_by_identity.items():
            ranked_indices = _ranked_indices(
                indices=identity_indices,
                similarities=similarities,
                image_ids=gallery_image_ids,
            )
            if config.method is AggregationMethod.NEAREST_IMAGE:
                contributing_indices = ranked_indices[:1]
                score = float(similarities[contributing_indices[0]])
            elif config.method is AggregationMethod.TOP_K_MEAN:
                contributing_indices = ranked_indices[: config.reference_count]
                score = float(np.mean(similarities[contributing_indices]))
            elif config.method is AggregationMethod.PROTOTYPE:
                contributing_indices = ranked_indices
                prototype = np.mean(gallery[identity_indices], axis=0)
                prototype_norm = float(np.linalg.norm(prototype))
                if prototype_norm <= 0:
                    raise EmbeddingError(
                        f"identity prototype has zero length: {identity_id}"
                    )
                score = float(np.dot(query, prototype / prototype_norm))
            else:
                raise EmbeddingError(
                    f"unsupported aggregation method: {config.method.value}"
                )
            reference_ids, reference_similarities = _evidence(
                indices=contributing_indices,
                similarities=similarities,
                image_ids=gallery_image_ids,
            )
            matches.append(
                IdentityMatch(
                    identity_id=identity_id,
                    score=score,
                    reference_image_ids=reference_ids,
                    reference_similarities=reference_similarities,
                )
            )

    return tuple(sorted(matches, key=lambda match: (-match.score, match.identity_id)))
