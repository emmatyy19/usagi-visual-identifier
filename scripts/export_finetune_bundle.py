"""Create a verified archive for one completed fine tuning run."""

import argparse
from pathlib import Path

from usagi_visual_identifier.training.transfer import (
    export_result_bundle,
    export_result_collection,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Export the selected run or the latest completed run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--all", action="store_true")
    arguments = parser.parse_args()
    if arguments.all and arguments.run_id is not None:
        parser.error("choose --all or --run-id, not both")
    if arguments.all:
        path = export_result_collection(artifact_root=PROJECT_ROOT / "artifacts")
    else:
        path = export_result_bundle(
            artifact_root=PROJECT_ROOT / "artifacts",
            run_id=arguments.run_id,
        )
    print(f"Fine tuning result bundle: {path}")


if __name__ == "__main__":
    main()
