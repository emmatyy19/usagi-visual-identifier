"""Unknown item rejection for embedding retrieval."""

import json
import math
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

import numpy as np

from usagi_visual_identifier.data.models import DatasetSplit
from usagi_visual_identifier.embeddings.aggregation import (
    DEFAULT_AGGREGATION_CONFIG,
    AggregationConfig,
    AggregationMethod,
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


class ConfidenceMethod(StrEnum):
    """Validation confidence signals available for unknown rejection."""

    MAX_SIMILARITY = "max_similarity"
    IDENTITY_MARGIN = "identity_margin"
    NEIGHBOR_AGREEMENT = "neighbor_agreement"
    PROTOTYPE_SIMILARITY = "prototype_similarity"


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
    confidence_method: ConfidenceMethod = ConfidenceMethod.MAX_SIMILARITY

    def as_dict(self) -> dict[str, object]:
        """Return a JSON ready open set report."""
        return {
            "schema_version": 2,
            "confidence_method": self.confidence_method.value,
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


def _confidence_score(
    *,
    method: ConfidenceMethod,
    query_embedding: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_identity_ids: tuple[str, ...],
    predicted_identity: str,
    identity_scores: tuple[float, ...],
    neighbor_count: int,
) -> float:
    """Calculate one confidence score without using query labels."""
    if method is ConfidenceMethod.MAX_SIMILARITY:
        return identity_scores[0]
    if method is ConfidenceMethod.IDENTITY_MARGIN:
        if len(identity_scores) < 2:
            raise EmbeddingError("identity margin requires at least two identities")
        return identity_scores[0] - identity_scores[1]
    if method is ConfidenceMethod.NEIGHBOR_AGREEMENT:
        similarities = gallery_embeddings @ query_embedding
        order = np.argsort(-similarities, kind="stable")[:neighbor_count]
        matching = sum(
            gallery_identity_ids[int(index)] == predicted_identity for index in order
        )
        return matching / len(order)
    if method is ConfidenceMethod.PROTOTYPE_SIMILARITY:
        prototype_rows: list[np.ndarray] = []
        prototype_ids: list[str] = []
        for identity_id in sorted(set(gallery_identity_ids)):
            rows = gallery_embeddings[
                [value == identity_id for value in gallery_identity_ids]
            ]
            prototype = rows.mean(axis=0)
            norm = float(np.linalg.norm(prototype))
            if norm <= 0:
                raise EmbeddingError("identity prototype has zero length")
            prototype_rows.append(prototype / norm)
            prototype_ids.append(identity_id)
        scores = np.stack(prototype_rows) @ query_embedding
        best_index = int(np.argmax(scores))
        if prototype_ids[best_index] != predicted_identity:
            raise EmbeddingError("prototype prediction does not match scored identity")
        return float(scores[best_index])
    raise EmbeddingError(f"unsupported confidence method: {method}")


def _score_queries(
    *,
    artifact: EmbeddingArtifact,
    gallery_split: DatasetSplit,
    query_split: DatasetSplit,
    aggregation: AggregationConfig,
    confidence_method: ConfidenceMethod,
    neighbor_count: int,
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
        if confidence_method is ConfidenceMethod.PROTOTYPE_SIMILARITY:
            prototype_matches = rank_gallery_identities(
                query_embedding=artifact.embeddings[query_index],
                gallery_embeddings=gallery_embeddings,
                gallery_identity_ids=gallery_identity_ids,
                gallery_image_ids=gallery_image_ids,
                config=replace(aggregation, method=AggregationMethod.PROTOTYPE),
            )
            best = prototype_matches[0]
        true_identity = artifact.identity_ids[query_index]
        is_known = true_identity in known_identities
        score = _confidence_score(
            method=confidence_method,
            query_embedding=artifact.embeddings[query_index],
            gallery_embeddings=gallery_embeddings,
            gallery_identity_ids=gallery_identity_ids,
            predicted_identity=best.identity_id,
            identity_scores=tuple(match.score for match in matches),
            neighbor_count=min(neighbor_count, len(gallery_indices)),
        )
        queries.append(
            OpenSetQuery(
                image_id=artifact.image_ids[query_index],
                true_identity=true_identity,
                predicted_identity=best.identity_id,
                score=score,
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
    confidence_method: ConfidenceMethod = ConfidenceMethod.MAX_SIMILARITY,
    neighbor_count: int = 5,
) -> OpenSetReport:
    """Select the safest validation threshold that retains enough known queries."""
    if query_split is not DatasetSplit.VALIDATION:
        raise EmbeddingError("threshold selection is allowed on validation only")
    if not 0 < minimum_known_correct_acceptance <= 1:
        raise EmbeddingError("minimum known correct acceptance must be in (0, 1]")
    if neighbor_count < 1:
        raise EmbeddingError("neighbor count must be positive")

    queries = _score_queries(
        artifact=artifact,
        gallery_split=gallery_split,
        query_split=query_split,
        aggregation=aggregation,
        confidence_method=confidence_method,
        neighbor_count=neighbor_count,
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
        confidence_method=confidence_method,
    )


@dataclass(frozen=True)
class ConfidenceComparisonReport:
    """Validation comparison and a single frozen confidence rule."""

    reports: tuple[OpenSetReport, ...]
    selected_method: ConfidenceMethod

    def as_dict(self) -> dict[str, object]:
        """Return a JSON ready comparison."""
        selected = next(
            report
            for report in self.reports
            if report.confidence_method is self.selected_method
        )
        return {
            "schema_version": 1,
            "selection_data": "validation_only",
            "selection_order": [
                "fewest_unknown_false_acceptances",
                "most_correct_known_acceptances",
                "simplest_confidence_method",
            ],
            "selected_method": self.selected_method.value,
            "selected": selected.as_dict(),
            "candidates": [report.as_dict() for report in self.reports],
        }


def compare_confidence_methods(
    *,
    artifact: EmbeddingArtifact,
    minimum_known_correct_acceptance: float = 0.95,
    methods: tuple[ConfidenceMethod, ...] = tuple(ConfidenceMethod),
) -> ConfidenceComparisonReport:
    """Compare predeclared confidence rules and freeze the safest simple rule."""
    if not methods:
        raise EmbeddingError("at least one confidence method is required")
    if len(set(methods)) != len(methods):
        raise EmbeddingError("confidence methods must be unique")
    reports = tuple(
        select_open_set_threshold(
            artifact=artifact,
            minimum_known_correct_acceptance=minimum_known_correct_acceptance,
            confidence_method=method,
        )
        for method in methods
    )
    complexity = {method: index for index, method in enumerate(ConfidenceMethod)}
    selected = min(
        reports,
        key=lambda report: (
            report.selected.unknown_accepted,
            -report.selected.known_correct_accepted,
            complexity[report.confidence_method],
        ),
    )
    return ConfidenceComparisonReport(
        reports=reports,
        selected_method=selected.confidence_method,
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


def write_confidence_comparison(
    *, path: Path, report: ConfidenceComparisonReport
) -> None:
    """Atomically write one private validation comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
