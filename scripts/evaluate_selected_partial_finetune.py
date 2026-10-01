"""Build tuned embeddings and compare unknown rejection on validation only."""

import json
import os
from pathlib import Path
from typing import cast

from usagi_visual_identifier.data.models import DatasetSplit
from usagi_visual_identifier.embeddings import (
    build_embedding_artifact,
    compare_confidence_methods,
    load_embedding_artifact,
    load_prepared_examples,
    select_embedding_examples,
    write_confidence_comparison,
)
from usagi_visual_identifier.embeddings.partial_finetune import (
    PartialFinetuneEncoder,
)
from usagi_visual_identifier.training.classifier_open_set import (
    ClassifierConfidence,
    evaluate_classifier_open_set,
    write_classifier_open_set_reports,
)
from usagi_visual_identifier.training.partial_finetune import load_partial_finetune

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = PROJECT_ROOT / "data" / "private"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
SELECTION_PATH = PROJECT_ROOT / "configs" / "partial_finetune" / "selected.json"


def _selection() -> dict[str, object]:
    """Load the public frozen checkpoint decision."""
    raw: object = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("selected checkpoint record must be a JSON object")
    return cast(dict[str, object], raw)


def main() -> None:
    """Evaluate the frozen checkpoint without reading the sealed test split."""
    selected = _selection()
    run_id = str(selected["run_id"])
    checkpoint_sha256 = str(selected["checkpoint_sha256"])
    model_root = ARTIFACT_ROOT / "models" / "partial_finetune" / run_id
    model, processor, metadata = load_partial_finetune(
        checkpoint_path=model_root / "trained_delta.safetensors",
        metadata_path=model_root / "metadata.json",
        requested_device=os.environ.get("USAGI_DEVICE", "auto"),
    )
    if metadata.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("loaded checkpoint does not match the frozen selection")

    all_examples, fingerprint = load_prepared_examples(
        private_root=PRIVATE_ROOT,
        processed_root=PROCESSED_ROOT,
    )
    examples = select_embedding_examples(
        examples=all_examples,
        included_splits=frozenset({DatasetSplit.TRAIN, DatasetSplit.VALIDATION}),
    )
    encoder = PartialFinetuneEncoder(
        model=model,
        processor=processor,
        run_id=run_id,
        checkpoint_sha256=checkpoint_sha256,
    )
    result = build_embedding_artifact(
        examples=examples,
        dataset_fingerprint_value=fingerprint,
        encoder=encoder,
        artifact_root=ARTIFACT_ROOT,
        batch_size=int(os.environ.get("USAGI_BATCH_SIZE", "8")),
    )
    artifact = load_embedding_artifact(
        artifact_path=result.artifact_path,
        metadata_path=result.metadata_path,
    )
    comparison = compare_confidence_methods(
        artifact=artifact,
        minimum_known_correct_acceptance=float(
            os.environ.get("USAGI_MINIMUM_KNOWN_ACCEPTANCE", "0.95")
        ),
    )
    report_path = (
        ARTIFACT_ROOT
        / "evaluations"
        / f"partial_finetune_{run_id}"
        / "confidence_comparison_validation.json"
    )
    write_confidence_comparison(path=report_path, report=comparison)

    raw_class_ids = metadata.get("class_ids")
    if not isinstance(raw_class_ids, list):
        raise ValueError("fine tuning class IDs are invalid")
    class_id_values = cast(list[object], raw_class_ids)
    if not all(isinstance(value, str) for value in class_id_values):
        raise ValueError("fine tuning class IDs are invalid")
    class_ids = tuple(cast(list[str], class_id_values))
    validation_examples = tuple(
        example for example in examples if example.split is DatasetSplit.VALIDATION
    )
    classifier_reports = tuple(
        evaluate_classifier_open_set(
            model=model,
            processor=processor,
            examples=validation_examples,
            class_ids=class_ids,
            method=method,
            minimum_known_correct_acceptance=float(
                os.environ.get("USAGI_MINIMUM_KNOWN_ACCEPTANCE", "0.95")
            ),
            batch_size=int(os.environ.get("USAGI_BATCH_SIZE", "8")),
        )
        for method in ClassifierConfidence
    )
    classifier_report_path = report_path.with_name(
        "classifier_confidence_validation.json"
    )
    write_classifier_open_set_reports(
        path=classifier_report_path, reports=classifier_reports
    )

    print(f"Frozen checkpoint: {run_id}")
    print(f"Device: {result.device_name}")
    for report in comparison.reports:
        point = report.selected
        print(
            f"{report.confidence_method.value}: known correct "
            f"{point.known_correct_accepted}/{report.known_query_count}, "
            f"unknown accepted {point.unknown_accepted}/{report.unknown_query_count}"
        )
    print(f"Selected tuned method: {comparison.selected_method.value}")
    for classifier_report in classifier_reports:
        point = classifier_report.selected
        print(
            f"classifier {classifier_report.method.value}: known correct "
            f"{point.known_correct_accepted}/{point.known_count}, unknown accepted "
            f"{point.unknown_accepted}/{point.unknown_count}"
        )
    print(f"Evaluation report: {report_path}")
    print(f"Classifier report: {classifier_report_path}")


if __name__ == "__main__":
    main()
