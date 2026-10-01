"""Select the frozen retrieval unknown rejection threshold."""

import os
from pathlib import Path

from usagi_visual_identifier.embeddings import (
    compare_confidence_methods,
    load_embedding_artifact,
    write_confidence_comparison,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
EMBEDDING_ROOT = ARTIFACT_ROOT / "embeddings" / "facebook_dinov2-small"


def main() -> None:
    """Choose a threshold from known and unknown validation queries."""
    minimum_known_acceptance = float(
        os.environ.get("USAGI_MINIMUM_KNOWN_ACCEPTANCE", "0.95")
    )
    artifact = load_embedding_artifact(
        artifact_path=EMBEDDING_ROOT / "embeddings.npz",
        metadata_path=EMBEDDING_ROOT / "metadata.json",
    )
    comparison = compare_confidence_methods(
        artifact=artifact,
        minimum_known_correct_acceptance=minimum_known_acceptance,
    )
    report_path = (
        ARTIFACT_ROOT
        / "evaluations"
        / "dinov2_small"
        / "confidence_comparison_validation.json"
    )
    write_confidence_comparison(path=report_path, report=comparison)

    selected_report = next(
        report
        for report in comparison.reports
        if report.confidence_method is comparison.selected_method
    )
    selected = selected_report.selected
    for report in comparison.reports:
        point = report.selected
        print(
            f"{report.confidence_method.value}: threshold {point.threshold:.6f}, "
            f"known correct {point.known_correct_accepted}/"
            f"{report.known_query_count}, unknown accepted "
            f"{point.unknown_accepted}/{report.unknown_query_count}"
        )
    print(f"Selected confidence method: {comparison.selected_method.value}")
    print(f"Known validation queries: {selected_report.known_query_count}")
    print(f"Unknown validation queries: {selected_report.unknown_query_count}")
    print(f"Selected threshold: {selected.threshold:.6f}")
    print(f"Known acceptance: {selected.known_acceptance_rate:.3f}")
    print(f"Correct known acceptance: {selected.known_correct_acceptance_rate:.3f}")
    print(f"Accepted known accuracy: {selected.accepted_known_accuracy:.3f}")
    print(f"Unknown false acceptance: {selected.unknown_false_acceptance_rate:.3f}")
    print(f"Evaluation report: {report_path}")


if __name__ == "__main__":
    main()
