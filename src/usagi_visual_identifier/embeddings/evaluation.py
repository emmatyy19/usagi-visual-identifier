"""Exact identity retrieval evaluation for saved embedding artifacts."""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from usagi_visual_identifier.data.models import DatasetSplit
from usagi_visual_identifier.embeddings.artifacts import (
    EmbeddingArtifact,
    EmbeddingError,
)


@dataclass(frozen=True)
class RetrievalReport:
    """Closed set retrieval metrics and per query rankings."""

    gallery_split: str
    query_split: str
    leave_one_out: bool
    query_count: int
    identity_count: int
    top1_accuracy: float
    top3_accuracy: float
    macro_recall: float
    mean_reciprocal_rank: float
    confusion_pairs: tuple[tuple[str, str, int], ...]
    queries: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON ready representation."""
        return {
            "schema_version": 1,
            "gallery_split": self.gallery_split,
            "query_split": self.query_split,
            "leave_one_out": self.leave_one_out,
            "query_count": self.query_count,
            "identity_count": self.identity_count,
            "top1_accuracy": self.top1_accuracy,
            "top3_accuracy": self.top3_accuracy,
            "macro_recall": self.macro_recall,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "confusion_pairs": [
                {
                    "true_identity": true_id,
                    "predicted_identity": predicted_id,
                    "count": count,
                }
                for true_id, predicted_id, count in self.confusion_pairs
            ],
            "queries": list(self.queries),
        }


def _indices_for_split(
    *, artifact: EmbeddingArtifact, split: DatasetSplit
) -> list[int]:
    """Return artifact rows belonging to one declared split."""
    return [
        index for index, value in enumerate(artifact.splits) if value == split.value
    ]


def evaluate_retrieval(
    *,
    artifact: EmbeddingArtifact,
    gallery_split: DatasetSplit,
    query_split: DatasetSplit,
    top_k: int = 3,
) -> RetrievalReport:
    """Rank identities by their nearest reference image cosine score."""
    if top_k < 1:
        raise EmbeddingError("top_k must be positive")
    gallery_indices = _indices_for_split(artifact=artifact, split=gallery_split)
    query_indices = _indices_for_split(artifact=artifact, split=query_split)
    if not gallery_indices:
        raise EmbeddingError(f"gallery split has no embeddings: {gallery_split.value}")
    if not query_indices:
        raise EmbeddingError(f"query split has no embeddings: {query_split.value}")

    leave_one_out = gallery_split is query_split
    gallery_identities = {artifact.identity_ids[index] for index in gallery_indices}
    if leave_one_out:
        counts = Counter(artifact.identity_ids[index] for index in gallery_indices)
        identities_without_peer = sorted(
            identity for identity, count in counts.items() if count < 2
        )
        if identities_without_peer:
            raise EmbeddingError(
                f"leave one out requires two gallery images per identity: {identities_without_peer}"
            )

    correct_by_identity: dict[str, list[bool]] = defaultdict(list)
    reciprocal_ranks: list[float] = []
    confusion_counts: Counter[tuple[str, str]] = Counter()
    query_results: list[dict[str, object]] = []
    top1_correct = 0
    top3_correct = 0

    for query_index in query_indices:
        eligible_gallery = [
            index
            for index in gallery_indices
            if not (leave_one_out and index == query_index)
        ]
        query_embedding = artifact.embeddings[query_index]
        identity_scores: dict[str, float] = {}
        reference_ids: dict[str, str] = {}
        for gallery_index in eligible_gallery:
            identity_id = artifact.identity_ids[gallery_index]
            score = float(np.dot(query_embedding, artifact.embeddings[gallery_index]))
            if (
                identity_id not in identity_scores
                or score > identity_scores[identity_id]
            ):
                identity_scores[identity_id] = score
                reference_ids[identity_id] = artifact.image_ids[gallery_index]

        ranked = sorted(identity_scores, key=identity_scores.__getitem__, reverse=True)
        true_identity = artifact.identity_ids[query_index]
        if true_identity not in ranked:
            raise EmbeddingError(
                f"query identity is absent from the gallery: {true_identity}"
            )
        rank = ranked.index(true_identity) + 1
        predicted_identity = ranked[0]
        is_correct = predicted_identity == true_identity
        top1_correct += int(is_correct)
        top3_correct += int(rank <= min(top_k, len(ranked)))
        reciprocal_ranks.append(1.0 / rank)
        correct_by_identity[true_identity].append(is_correct)
        if not is_correct:
            confusion_counts[(true_identity, predicted_identity)] += 1

        query_results.append(
            {
                "image_id": artifact.image_ids[query_index],
                "true_identity": true_identity,
                "true_identity_rank": rank,
                "ranked_identities": [
                    {
                        "identity_id": identity_id,
                        "score": identity_scores[identity_id],
                        "reference_image_id": reference_ids[identity_id],
                    }
                    for identity_id in ranked[:top_k]
                ],
            }
        )

    query_count = len(query_indices)
    macro_recall = float(
        np.mean(
            [sum(results) / len(results) for results in correct_by_identity.values()]
        )
    )
    confusion_pairs = tuple(
        (true_id, predicted_id, count)
        for (true_id, predicted_id), count in sorted(
            confusion_counts.items(), key=lambda item: (-item[1], item[0])
        )
    )
    return RetrievalReport(
        gallery_split=gallery_split.value,
        query_split=query_split.value,
        leave_one_out=leave_one_out,
        query_count=query_count,
        identity_count=len(gallery_identities),
        top1_accuracy=top1_correct / query_count,
        top3_accuracy=top3_correct / query_count,
        macro_recall=macro_recall,
        mean_reciprocal_rank=float(np.mean(reciprocal_ranks)),
        confusion_pairs=confusion_pairs,
        queries=tuple(query_results),
    )


def write_retrieval_report(*, path: Path, report: RetrievalReport) -> None:
    """Atomically write a private retrieval report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
