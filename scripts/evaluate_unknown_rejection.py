"""Select the frozen retrieval unknown rejection threshold."""

import os
from pathlib import Path

from usagi_visual_identifier.embeddings import (
    load_embedding_artifact,
    select_open_set_threshold,
    write_open_set_report,
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
    report = select_open_set_threshold(
        artifact=artifact,
        minimum_known_correct_acceptance=minimum_known_acceptance,
    )
    report_path = (
        ARTIFACT_ROOT
        / "evaluations"
        / "dinov2_small"
        / "unknown_rejection_validation.json"
    )
    write_open_set_report(path=report_path, report=report)

    selected = report.selected
    print(f"Known validation queries: {report.known_query_count}")
    print(f"Unknown validation queries: {report.unknown_query_count}")
    print(f"Selected cosine threshold: {selected.threshold:.6f}")
    print(f"Known acceptance: {selected.known_acceptance_rate:.3f}")
    print(f"Correct known acceptance: {selected.known_correct_acceptance_rate:.3f}")
    print(f"Accepted known accuracy: {selected.accepted_known_accuracy:.3f}")
    print(f"Unknown false acceptance: {selected.unknown_false_acceptance_rate:.3f}")
    print(f"Evaluation report: {report_path}")


if __name__ == "__main__":
    main()
