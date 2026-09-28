"""Create a private train and validation bundle for the GPU computer."""

from pathlib import Path

from usagi_visual_identifier.training.transfer import export_training_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Export prepared inputs without the sealed test split."""
    path = export_training_bundle(
        private_root=PROJECT_ROOT / "data/private",
        processed_root=PROJECT_ROOT / "data/processed",
        artifact_root=PROJECT_ROOT / "artifacts",
    )
    print(f"Private training bundle: {path}")
    print("This archive contains owner photographs. Do not commit it.")


if __name__ == "__main__":
    main()
