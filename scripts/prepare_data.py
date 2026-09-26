"""Run the local image preparation pipeline from the debugger."""

import os
from pathlib import Path

from usagi_visual_identifier.data.models import DatasetSplit
from usagi_visual_identifier.data.preparation import prepare_session

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = PROJECT_ROOT / "data" / "private"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    """Prepare the session selected by the active debug configuration."""
    session_id = os.environ.get("USAGI_SESSION", "session-001")
    split = DatasetSplit(os.environ.get("USAGI_SPLIT", DatasetSplit.TRAIN.value))

    result = prepare_session(
        private_root=PRIVATE_ROOT,
        processed_root=PROCESSED_ROOT,
        session_id=session_id,
        split=split,
    )

    print(f"Prepared {result.image_count} images.")
    print(f"Found {result.identity_count} identities.")
    print(f"Created {result.contact_sheet_count} contact sheets.")
    print(f"Dataset fingerprint: {result.fingerprint}")
    print(f"Audit summary: {result.summary_path}")


if __name__ == "__main__":
    main()
