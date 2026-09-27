"""Build frozen DINOv2 embeddings and run the Session 1 sanity check."""

import os
from pathlib import Path

from usagi_visual_identifier.data.models import DatasetSplit
from usagi_visual_identifier.embeddings import (
    DEFAULT_DINOV2_MODEL_ID,
    DEFAULT_DINOV2_REVISION,
    Dinov2Encoder,
    build_embedding_artifact,
    evaluate_retrieval,
    load_embedding_artifact,
    load_prepared_examples,
    select_embedding_examples,
    write_retrieval_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = PROJECT_ROOT / "data" / "private"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"


def _embedding_splits() -> frozenset[DatasetSplit]:
    """Read the explicitly selected artifact splits from the environment."""
    raw = os.environ.get("USAGI_EMBEDDING_SPLITS", "train,validation")
    values = [value.strip() for value in raw.split(",") if value.strip()]
    try:
        return frozenset(DatasetSplit(value) for value in values)
    except ValueError as error:
        allowed = ", ".join(split.value for split in DatasetSplit)
        raise ValueError(f"USAGI_EMBEDDING_SPLITS must use: {allowed}") from error


def _allows_final_test() -> bool:
    """Require a deliberate exact value before embedding the sealed test set."""
    return os.environ.get("USAGI_ALLOW_FINAL_TEST", "false").casefold() == "true"


def main() -> None:
    """Build the pinned baseline and save its private evaluation report."""
    requested_device = os.environ.get("USAGI_DEVICE", "auto")
    batch_size = int(os.environ.get("USAGI_BATCH_SIZE", "8"))

    all_examples, fingerprint = load_prepared_examples(
        private_root=PRIVATE_ROOT,
        processed_root=PROCESSED_ROOT,
    )
    included_splits = _embedding_splits()
    examples = select_embedding_examples(
        examples=all_examples,
        included_splits=included_splits,
        allow_test=_allows_final_test(),
    )
    encoder = Dinov2Encoder(
        model_id=DEFAULT_DINOV2_MODEL_ID,
        model_revision=DEFAULT_DINOV2_REVISION,
        requested_device=requested_device,
    )
    result = build_embedding_artifact(
        examples=examples,
        dataset_fingerprint_value=fingerprint,
        encoder=encoder,
        artifact_root=ARTIFACT_ROOT,
        batch_size=batch_size,
    )
    artifact = load_embedding_artifact(
        artifact_path=result.artifact_path,
        metadata_path=result.metadata_path,
    )
    report = evaluate_retrieval(
        artifact=artifact,
        gallery_split=DatasetSplit.TRAIN,
        query_split=DatasetSplit.TRAIN,
    )
    report_path = (
        ARTIFACT_ROOT
        / "evaluations"
        / "dinov2_small"
        / "session_001_leave_one_out.json"
    )
    write_retrieval_report(path=report_path, report=report)

    print(f"Device: {result.device_name}")
    print(
        "Included splits: "
        + ", ".join(sorted(split.value for split in included_splits))
    )
    print(f"Images: {result.image_count}")
    print(f"Embedding dimension: {result.embedding_dimension}")
    print(f"Embedding time: {result.elapsed_seconds:.2f} seconds")
    print(f"Throughput: {result.images_per_second:.2f} images per second")
    print(f"Dataset fingerprint: {result.dataset_fingerprint}")
    print(f"Top 1 sanity accuracy: {report.top1_accuracy:.3f}")
    print(f"Top 3 sanity accuracy: {report.top3_accuracy:.3f}")
    print(f"Macro recall: {report.macro_recall:.3f}")
    print(f"Mean reciprocal rank: {report.mean_reciprocal_rank:.3f}")
    print(f"Embedding artifact: {result.artifact_path}")
    print(f"Evaluation report: {report_path}")


if __name__ == "__main__":
    main()
