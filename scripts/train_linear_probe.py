"""Train the deterministic linear probe from frozen embeddings."""

import os
from pathlib import Path

from usagi_visual_identifier.embeddings import load_embedding_artifact
from usagi_visual_identifier.training import LinearProbeConfig, train_linear_probe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
EMBEDDING_ROOT = ARTIFACT_ROOT / "embeddings/facebook_dinov2-small"


def _integer(name: str, default: int) -> int:
    """Read one integer training parameter."""
    return int(os.environ.get(name, str(default)))


def _float(name: str, default: float) -> float:
    """Read one floating point training parameter."""
    return float(os.environ.get(name, str(default)))


def main() -> None:
    """Train and publish a linear probe without accessing the test split."""
    config = LinearProbeConfig(
        seed=_integer("USAGI_SEED", 42),
        epochs=_integer("USAGI_EPOCHS", 300),
        batch_size=_integer("USAGI_BATCH_SIZE", 16),
        learning_rate=_float("USAGI_LEARNING_RATE", 0.01),
        weight_decay=_float("USAGI_WEIGHT_DECAY", 0.0001),
        patience=_integer("USAGI_PATIENCE", 40),
        minimum_delta=_float("USAGI_MINIMUM_DELTA", 0.00001),
        device=os.environ.get("USAGI_DEVICE", "cpu"),
    )
    artifact = load_embedding_artifact(
        artifact_path=EMBEDDING_ROOT / "embeddings.npz",
        metadata_path=EMBEDDING_ROOT / "metadata.json",
    )
    result = train_linear_probe(
        artifact=artifact,
        config=config,
        artifact_root=ARTIFACT_ROOT,
    )
    print(f"Run ID: {result.run_id}")
    print(f"Selected epoch: {result.selected_epoch}")
    print(f"Validation Top 1: {result.validation_top1_accuracy:.3f}")
    print(f"Validation Top 3: {result.validation_top3_accuracy:.3f}")
    print(f"Validation macro recall: {result.validation_macro_recall:.3f}")
    print(f"Artifacts: {result.output_directory}")


if __name__ == "__main__":
    main()
