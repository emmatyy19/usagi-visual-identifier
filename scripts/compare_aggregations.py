"""Compare frozen retrieval aggregation rules on one shared split pair."""

import os
from pathlib import Path

from usagi_visual_identifier.data.models import DatasetSplit
from usagi_visual_identifier.embeddings import (
    AggregationConfig,
    AggregationMethod,
    compare_retrieval_aggregations,
    load_embedding_artifact,
    write_aggregation_comparison,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
EMBEDDING_ROOT = ARTIFACT_ROOT / "embeddings" / "facebook_dinov2-small"


def _environment_split(name: str, default: DatasetSplit) -> DatasetSplit:
    """Read one dataset split from the environment with a clear error."""
    value = os.environ.get(name, default.value)
    try:
        return DatasetSplit(value)
    except ValueError as error:
        allowed = ", ".join(split.value for split in DatasetSplit)
        raise ValueError(f"{name} must be one of: {allowed}") from error


def _positive_integer(name: str, default: int) -> int:
    """Read one positive integer comparison parameter."""
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float(name: str, default: float) -> float:
    """Read one positive finite floating point parameter."""
    value = float(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def main() -> None:
    """Evaluate every planned frozen aggregation without selecting a winner."""
    gallery_split = _environment_split("USAGI_GALLERY_SPLIT", DatasetSplit.TRAIN)
    query_split = _environment_split("USAGI_QUERY_SPLIT", DatasetSplit.TRAIN)
    mean_reference_count = _positive_integer("USAGI_TOP_K_MEAN", 3)
    voting_reference_count = _positive_integer("USAGI_WEIGHTED_NEIGHBORS", 5)
    temperature = _positive_float("USAGI_VOTE_TEMPERATURE", 0.07)

    artifact = load_embedding_artifact(
        artifact_path=EMBEDDING_ROOT / "embeddings.npz",
        metadata_path=EMBEDDING_ROOT / "metadata.json",
    )
    configurations = (
        AggregationConfig(
            method=AggregationMethod.NEAREST_IMAGE,
            reference_count=1,
        ),
        AggregationConfig(
            method=AggregationMethod.PROTOTYPE,
            reference_count=1,
        ),
        AggregationConfig(
            method=AggregationMethod.TOP_K_MEAN,
            reference_count=mean_reference_count,
        ),
        AggregationConfig(
            method=AggregationMethod.WEIGHTED_NEIGHBORS,
            reference_count=voting_reference_count,
            temperature=temperature,
        ),
    )
    comparison = compare_retrieval_aggregations(
        artifact=artifact,
        gallery_split=gallery_split,
        query_split=query_split,
        configurations=configurations,
    )
    report_path = (
        ARTIFACT_ROOT
        / "evaluations"
        / "dinov2_small"
        / f"aggregation_comparison_{gallery_split.value}_to_{query_split.value}.json"
    )
    write_aggregation_comparison(path=report_path, comparison=comparison)

    if comparison.exploratory_same_split:
        print(
            "Exploratory same split check only. "
            "Do not select an aggregation rule from these results."
        )
    print(f"Gallery split: {comparison.gallery_split}")
    print(f"Query split: {comparison.query_split}")
    print("Method                 Top 1   Top 3   Macro recall   MRR")
    for report in comparison.reports:
        print(
            f"{report.aggregation.method.value:<22} "
            f"{report.top1_accuracy:>5.3f}   "
            f"{report.top3_accuracy:>5.3f}   "
            f"{report.macro_recall:>12.3f}   "
            f"{report.mean_reciprocal_rank:>5.3f}"
        )
    print(f"Comparison report: {report_path}")


if __name__ == "__main__":
    main()
