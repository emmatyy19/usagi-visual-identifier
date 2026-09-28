"""Verify and install a private training input or model result bundle."""

import argparse
from pathlib import Path

from usagi_visual_identifier.training.transfer import import_transfer_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Import only verified, path safe bundle members."""
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    arguments = parser.parse_args()
    manifest = import_transfer_bundle(
        archive_path=arguments.archive.resolve(),
        project_root=PROJECT_ROOT,
    )
    print(f"Imported bundle kind: {manifest['kind']}")
    print(f"Details: {manifest['details']}")


if __name__ == "__main__":
    main()
