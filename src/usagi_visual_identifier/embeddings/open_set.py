"""Unknown item rejection for embedding retrieval."""

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

from usagi_visual_identifier.data.models import DatasetSplit
from usagi_visual_identifier.embeddings.aggregation import (
    DEFAULT_AGGREGATION_CONFIG,
    AggregationConfig,
    rank_gallery_identities,
)
from usagi_visual_identifier.embeddings.artifacts import (
    EmbeddingArtifact,
    EmbeddingError,
)


@dataclass(frozen=True)
class OpenSetQuery:
    """Best gallery match for one known or unknown query."""

    image_id: str
    true_identity: str
    predicted_identity: str
    score: float
    is_known: bool
    is_correct: bool
    reference_image_id: str | None
    accepted: bool = False

    def as_dict(self) -> dict[str, object]:
        """Return a JSON ready query record."""
        return {
            "image_id": self.image_id,
            "true_identity": self.true_identity,
            "predicted_identity": self.predicted_identity,
            "score": self.score,
            "is_known": self.is_known,
            "is_correct": self.is_correct,
            "reference_image_id": self.reference_image_id,
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class ThresholdMetrics:
    """Known and unknown behavior at one acceptance threshold."""

    threshold: float
    known_acceptance_rate: float
    known_correct_acceptance_rate: float
    accepted_known_accuracy: float
    unknown_false_acceptance_rate: float
    known_accepted: int
    known_correct_accepted: int
    unknown_accepted: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON ready threshold point."""
        return {
            "threshold": self.threshold,
            "known_acceptance_rate": self.known_acceptance_rate,
            "known_correct_acceptance_rate": self.known_correct_acceptance_rate,
            "accepted_known_accuracy": self.accepted_known_accuracy,
            "unknown_false_acceptance_rate": self.unknown_false_acceptance_rate,
            "known_accepted": self.known_accepted,
            "known_correct_accepted": self.known_correct_accepted,
            "unknown_accepted": self.unknown_accepted,
        }


@dataclass(frozen=True)
class OpenSetReport:
    """Selected validation threshold and the complete operating curve."""

    gallery_split: str
    query_split: str
    aggregation: AggregationConfig
    minimum_known_correct_acceptance: float
    known_query_count: int
    unknown_query_count: int
    selected: ThresholdMetrics
    curve: tuple[ThresholdMetrics, ...]
    queries: tuple[OpenSetQuery, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON ready open set report."""
        return {
            "schema_version": 1,
            "gallery_split": self.gallery_split,
            "query_split": self.query_split,
            "aggregation": self.aggregation.as_dict(),
            "minimum_known_correct_acceptance": (self.minimum_known_correct_acceptance),
            "known_query_count": self.known_query_count,
            "unknown_query_count": self.unknown_query_count,
            "selected": self.selected.as_dict(),
            "curve": [point.as_dict() for point in self.curve],
            "queries": [query.as_dict() for query in self.queries],
        }


def _split_indices(
    *, artifact: EmbeddingArtifact, split: DatasetSplit
) -> tuple[int, ...]:
    """Return artifact indices for one split."""
    return tuple(
        index for index, value in enumerate(artifact.splits) if value == split.value
    )


def _score_queries(
    *,
    artifact: EmbeddingArtifact,
    gallery_split: DatasetSplit,
    query_split: DatasetSplit,
    aggregation: AggregationConfig,
) -> tuple[OpenSetQuery, ...]:
    """Score every query against known gallery identities."""
    gallery_indices = _split_indices(artifact=artifact, split=gallery_split)
    query_indices = _split_indices(artifact=artifact, split=query_split)
    if not gallery_indices:
        raise EmbeddingError(f"gallery split has no embeddings: {gallery_split.value}")
    if not query_indices:
        raise EmbeddingError(f"query split has no embeddings: {query_split.value}")
    if gallery_split is query_split:
        raise EmbeddingError("open set evaluation requires independent query data")

    gallery_identity_ids = tuple(
        artifact.identity_ids[index] for index in gallery_indices
    )
    if any(identity_id.startswith("unknown-") for identity_id in gallery_identity_ids):
        raise EmbeddingError("the known gallery cannot contain unknown identities")
    known_identities = frozenset(gallery_identity_ids)
    gallery_embeddings = artifact.embeddings[list(gallery_indices)]
    gallery_image_ids = tuple(artifact.image_ids[index] for index in gallery_indices)

    queries: list[OpenSetQuery] = []
    for query_index in query_indices:
        matches = rank_gallery_identities(
            query_embedding=artifact.embeddings[query_index],
            gallery_embeddings=gallery_embeddings,
            gallery_identity_ids=gallery_identity_ids,
            gallery_image_ids=gallery_image_ids,
            config=aggregation,
        )
        best = matches[0]
        true_identity = artifact.identity_ids[query_index]
        is_known = true_identity in known_identities
        queries.append(
            OpenSetQuery(
                image_id=artifact.image_ids[query_index],
                true_identity=true_identity,
                predicted_identity=best.identity_id,
                score=best.score,
                is_known=is_known,
                is_correct=is_known and best.identity_id == true_identity,
                reference_image_id=(
                    best.reference_image_ids[0] if best.reference_image_ids else None
                ),
            )
        )
    return tuple(queries)


def _metrics_at_threshold(
    *, queries: tuple[OpenSetQuery, ...], threshold: float
) -> ThresholdMetrics:
    """Calculate acceptance metrics for one finite score threshold."""
    if not math.isfinite(threshold):
        raise EmbeddingError("open set threshold must be finite")
    known = tuple(query for query in queries if query.is_known)
    unknown = tuple(query for query in queries if not query.is_known)
    if not known or not unknown:
        raise EmbeddingError("open set evaluation requires known and unknown queries")

    accepted_known = tuple(query for query in known if query.score >= threshold)
    correct_accepted = sum(query.is_correct for query in accepted_known)
    accepted_unknown = sum(query.score >= threshold for query in unknown)
    return ThresholdMetrics(
        threshold=threshold,
        known_acceptance_rate=len(accepted_known) / len(known),
        known_correct_acceptance_rate=correct_accepted / len(known),
        accepted_known_accuracy=(
            correct_accepted / len(accepted_known) if accepted_known else 0.0
        ),
        unknown_false_acceptance_rate=accepted_unknown / len(unknown),
        known_accepted=len(accepted_known),
        known_correct_accepted=correct_accepted,
        unknown_accepted=accepted_unknown,
    )


def select_open_set_threshold(
    *,
    artifact: EmbeddingArtifact,
    gallery_split: DatasetSplit = DatasetSplit.TRAIN,
    query_split: DatasetSplit = DatasetSplit.VALIDATION,
    minimum_known_correct_acceptance: float = 0.95,
    aggregation: AggregationConfig = DEFAULT_AGGREGATION_CONFIG,
) -> OpenSetReport:
    """Select the safest validation threshold that retains enough known queries."""
    if query_split is not DatasetSplit.VALIDATION:
        raise EmbeddingError("threshold selection is allowed on validation only")
    if not 0 < minimum_known_correct_acceptance <= 1:
        raise EmbeddingError("minimum known correct acceptance must be in (0, 1]")

    queries = _score_queries(
        artifact=artifact,
        gallery_split=gallery_split,
        query_split=query_split,
        aggregation=aggregation,
    )
    scores = sorted({query.score for query in queries}, reverse=True)
    reject_all_threshold = math.nextafter(scores[0], math.inf)
    curve = tuple(
        _metrics_at_threshold(queries=queries, threshold=threshold)
        for threshold in (reject_all_threshold, *scores)
    )
    eligible = tuple(
        point
        for point in curve
        if point.known_correct_acceptance_rate >= minimum_known_correct_acceptance
    )
    if not eligible:
        raise EmbeddingError(
            "no threshold retains the required fraction of correctly identified "
            "known queries"
        )
    selected = min(
        eligible,
        key=lambda point: (
            point.unknown_false_acceptance_rate,
            -point.known_correct_acceptance_rate,
            -point.threshold,
        ),
    )
    accepted_queries = tuple(
        replace(query, accepted=query.score >= selected.threshold) for query in queries
    )
    return OpenSetReport(
        gallery_split=gallery_split.value,
        query_split=query_split.value,
        aggregation=aggregation,
        minimum_known_correct_acceptance=minimum_known_correct_acceptance,
        known_query_count=sum(query.is_known for query in queries),
        unknown_query_count=sum(not query.is_known for query in queries),
        selected=selected,
        curve=curve,
        queries=accepted_queries,
    )


def write_open_set_report(*, path: Path, report: OpenSetReport) -> None:
    """Atomically write one private threshold selection report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
