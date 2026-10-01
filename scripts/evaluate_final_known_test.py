"""Run the frozen one time Session 3 evaluation for known catalog dolls."""

import json
import os
from pathlib import Path
from typing import cast

from usagi_visual_identifier.data.models import DatasetSplit
from usagi_visual_identifier.embeddings import (
    DEFAULT_DINOV2_MODEL_ID,
    DEFAULT_DINOV2_REVISION,
    AggregationConfig,
    AggregationMethod,
    Dinov2Encoder,
    RetrievalReport,
    build_embedding_artifact,
    evaluate_retrieval,
    load_embedding_artifact,
    load_prepared_examples,
    select_embedding_examples,
    write_retrieval_report,
)
from usagi_visual_identifier.embeddings.partial_finetune import (
    PartialFinetuneEncoder,
)
from usagi_visual_identifier.training.classifier_test import (
    ClassifierTestReport,
    evaluate_classifier_test,
    write_classifier_test_report,
)
from usagi_visual_identifier.training.partial_finetune import load_partial_finetune

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = PROJECT_ROOT / "data" / "private"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "final_known_test"
MODEL_ROOT = PROJECT_ROOT / "artifacts" / "models" / "partial_finetune"
CONFIG_PATH = PROJECT_ROOT / "configs" / "open_set" / "final_known_test.json"
CHECKPOINT_SELECTION = PROJECT_ROOT / "configs" / "partial_finetune" / "selected.json"


def _json_object(path: Path) -> dict[str, object]:
    """Load a string keyed JSON object."""
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    values = cast(dict[object, object], raw)
    if not all(isinstance(key, str) for key in values):
        raise ValueError(f"{path} must use string keys")
    return cast(dict[str, object], values)


def _thresholds(config: dict[str, object]) -> dict[str, float]:
    """Read the thresholds frozen before Session 3 access."""
    raw = config.get("operating_points")
    if not isinstance(raw, dict):
        raise ValueError("final test operating points are missing")
    values = cast(dict[object, object], raw)
    result: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, (int, float)):
            raise ValueError("final test operating points are invalid")
        result[key] = float(value)
    return result


def _acceptance(report: RetrievalReport, *, threshold: float) -> dict[str, object]:
    """Apply one frozen score threshold without changing it from test data."""
    accepted = 0
    correct_accepted = 0
    for query in report.queries:
        ranked = query.get("ranked_identities")
        if not isinstance(ranked, list) or not ranked:
            raise ValueError("retrieval report is missing ranked identities")
        ranked_values = cast(list[object], ranked)
        best = ranked_values[0]
        if not isinstance(best, dict):
            raise ValueError("retrieval match must be an object")
        best_values = cast(dict[object, object], best)
        score = best_values.get("score")
        predicted = best_values.get("identity_id")
        true_identity = query.get("true_identity")
        if not isinstance(score, (int, float)):
            raise ValueError("retrieval score must be numeric")
        if not isinstance(predicted, str) or not isinstance(true_identity, str):
            raise ValueError("retrieval identities must be strings")
        is_accepted = float(score) >= threshold
        accepted += int(is_accepted)
        correct_accepted += int(is_accepted and predicted == true_identity)
    return {
        "threshold": threshold,
        "accepted_count": accepted,
        "known_acceptance_rate": accepted / report.query_count,
        "correct_accepted_count": correct_accepted,
        "known_correct_acceptance_rate": correct_accepted / report.query_count,
    }


def _classifier_summary(report: ClassifierTestReport) -> dict[str, object]:
    """Return compact classifier metrics for the combined report."""
    return {
        "top1_accuracy": report.top1_accuracy,
        "top3_accuracy": report.top3_accuracy,
        "macro_recall": report.macro_recall,
        "probability_margin_threshold": report.probability_margin_threshold,
        "accepted_count": report.accepted_count,
        "correct_accepted_count": report.correct_accepted_count,
        "known_acceptance_rate": report.accepted_count / report.query_count,
        "known_correct_acceptance_rate": (
            report.correct_accepted_count / report.query_count
        ),
    }


def main() -> None:
    """Evaluate frozen systems on Session 3 without permitting reselection."""
    config = _json_object(CONFIG_PATH)
    if config.get("post_test_tuning_allowed") is not False:
        raise ValueError("final test configuration must forbid post test tuning")
    thresholds = _thresholds(config)
    selection = _json_object(CHECKPOINT_SELECTION)
    run_id = str(selection.get("run_id"))
    if run_id != config.get("checkpoint_run_id"):
        raise ValueError("final test and checkpoint selections do not match")

    all_examples, fingerprint = load_prepared_examples(
        private_root=PRIVATE_ROOT,
        processed_root=PROCESSED_ROOT,
    )
    examples = select_embedding_examples(
        examples=all_examples,
        included_splits=frozenset({DatasetSplit.TRAIN, DatasetSplit.TEST}),
        allow_test=True,
    )
    test_examples = tuple(
        example for example in examples if example.split is DatasetSplit.TEST
    )
    if any(not example.identity_id.startswith("usagi-") for example in test_examples):
        raise ValueError("known final test cannot contain unknown identities")

    requested_device = os.environ.get("USAGI_DEVICE", "auto")
    batch_size = int(os.environ.get("USAGI_BATCH_SIZE", "8"))
    frozen_encoder = Dinov2Encoder(
        model_id=DEFAULT_DINOV2_MODEL_ID,
        model_revision=DEFAULT_DINOV2_REVISION,
        requested_device=requested_device,
    )
    frozen_result = build_embedding_artifact(
        examples=examples,
        dataset_fingerprint_value=fingerprint,
        encoder=frozen_encoder,
        artifact_root=ARTIFACT_ROOT / "frozen",
        batch_size=batch_size,
    )
    frozen_artifact = load_embedding_artifact(
        artifact_path=frozen_result.artifact_path,
        metadata_path=frozen_result.metadata_path,
    )
    frozen_report = evaluate_retrieval(
        artifact=frozen_artifact,
        gallery_split=DatasetSplit.TRAIN,
        query_split=DatasetSplit.TEST,
    )

    checkpoint_root = MODEL_ROOT / run_id
    model, processor, metadata = load_partial_finetune(
        checkpoint_path=checkpoint_root / "trained_delta.safetensors",
        metadata_path=checkpoint_root / "metadata.json",
        requested_device=requested_device,
    )
    if metadata.get("checkpoint_sha256") != selection.get("checkpoint_sha256"):
        raise ValueError("loaded checkpoint does not match the frozen selection")
    tuned_encoder = PartialFinetuneEncoder(
        model=model,
        processor=processor,
        run_id=run_id,
        checkpoint_sha256=str(selection.get("checkpoint_sha256")),
    )
    tuned_result = build_embedding_artifact(
        examples=examples,
        dataset_fingerprint_value=fingerprint,
        encoder=tuned_encoder,
        artifact_root=ARTIFACT_ROOT / "tuned",
        batch_size=batch_size,
    )
    tuned_artifact = load_embedding_artifact(
        artifact_path=tuned_result.artifact_path,
        metadata_path=tuned_result.metadata_path,
    )
    tuned_nearest = evaluate_retrieval(
        artifact=tuned_artifact,
        gallery_split=DatasetSplit.TRAIN,
        query_split=DatasetSplit.TEST,
    )
    tuned_prototype = evaluate_retrieval(
        artifact=tuned_artifact,
        gallery_split=DatasetSplit.TRAIN,
        query_split=DatasetSplit.TEST,
        aggregation=AggregationConfig(method=AggregationMethod.PROTOTYPE),
    )

    class_ids_raw = metadata.get("class_ids")
    if not isinstance(class_ids_raw, list):
        raise ValueError("fine tuning class IDs are invalid")
    class_id_values = cast(list[object], class_ids_raw)
    if not all(isinstance(value, str) for value in class_id_values):
        raise ValueError("fine tuning class IDs are invalid")
    class_ids = tuple(cast(list[str], class_id_values))
    classifier_report = evaluate_classifier_test(
        model=model,
        processor=processor,
        examples=test_examples,
        class_ids=class_ids,
        probability_margin_threshold=thresholds["classifier_probability_margin"],
        batch_size=batch_size,
    )

    report_root = ARTIFACT_ROOT / "reports"
    write_retrieval_report(
        path=report_root / "frozen_nearest.json", report=frozen_report
    )
    write_retrieval_report(
        path=report_root / "tuned_nearest.json", report=tuned_nearest
    )
    write_retrieval_report(
        path=report_root / "tuned_prototype.json", report=tuned_prototype
    )
    write_classifier_test_report(
        path=report_root / "tuned_classifier.json", report=classifier_report
    )

    summary = {
        "schema_version": 1,
        "evaluation_role": "final_known_test",
        "selection_allowed": False,
        "session": config.get("known_test_split"),
        "query_count": len(test_examples),
        "independent_unknown_test_available": False,
        "unknown_result_role": "validation_only",
        "systems": {
            "frozen_nearest": {
                "top1_accuracy": frozen_report.top1_accuracy,
                "top3_accuracy": frozen_report.top3_accuracy,
                "macro_recall": frozen_report.macro_recall,
                **_acceptance(
                    frozen_report,
                    threshold=thresholds["frozen_max_similarity"],
                ),
            },
            "tuned_nearest": {
                "top1_accuracy": tuned_nearest.top1_accuracy,
                "top3_accuracy": tuned_nearest.top3_accuracy,
                "macro_recall": tuned_nearest.macro_recall,
                **_acceptance(
                    tuned_nearest,
                    threshold=thresholds["tuned_max_similarity"],
                ),
            },
            "tuned_prototype": {
                "top1_accuracy": tuned_prototype.top1_accuracy,
                "top3_accuracy": tuned_prototype.top3_accuracy,
                "macro_recall": tuned_prototype.macro_recall,
                **_acceptance(
                    tuned_prototype,
                    threshold=thresholds["tuned_prototype_similarity"],
                ),
            },
            "tuned_classifier": _classifier_summary(classifier_report),
        },
    }
    summary_path = report_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("Session 3 final known test is complete. No reselection is allowed.")
    systems = cast(dict[str, dict[str, object]], summary["systems"])
    for name, values in systems.items():
        top1 = values.get("top1_accuracy")
        correct_accepted = values.get("correct_accepted_count")
        if not isinstance(top1, (int, float)) or not isinstance(correct_accepted, int):
            raise ValueError("final summary metrics are invalid")
        print(
            f"{name}: Top 1 {float(top1):.3f}, "
            f"correct accepted {correct_accepted}/{len(test_examples)}"
        )
    print("Independent unknown final test: not available")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
