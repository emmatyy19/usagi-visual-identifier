"""Train the final DINOv2 block from an explicit JSON configuration."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

from usagi_visual_identifier.embeddings import load_prepared_examples
from usagi_visual_identifier.training.partial_finetune import (
    PartialFinetuneConfig,
    load_partial_finetune_config,
    train_partial_finetune,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = PROJECT_ROOT / "data/private"
PROCESSED_ROOT = PROJECT_ROOT / "data/processed"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"


def _configuration(path: Path, seed: int | None) -> PartialFinetuneConfig:
    """Load a strict public experiment configuration."""
    return load_partial_finetune_config(path=path, seed_override=seed)


def _linear_probe_paths() -> tuple[Path, Path]:
    """Resolve the verified linear probe selected for warm start."""
    root = ARTIFACT_ROOT / "models/linear_probe"
    latest_raw: object = json.loads((root / "latest.json").read_text(encoding="utf-8"))
    if not isinstance(latest_raw, dict):
        raise ValueError("linear probe latest pointer is invalid")
    latest: dict[str, object] = {}
    for key, value in cast(dict[object, object], latest_raw).items():
        if not isinstance(key, str):
            raise ValueError("linear probe latest pointer is invalid")
        latest[key] = value
    run_id = latest.get("run_id")
    if not isinstance(run_id, str):
        raise ValueError("linear probe latest pointer is invalid")
    run_root = root / run_id
    return run_root / "checkpoint.pt", run_root / "metadata.json"


def main() -> None:
    """Load prepared images and publish one partial fine tuning run."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/partial_finetune/conservative.json"),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts"),
        help="Use artifacts/smoke for a disposable smoke test.",
    )
    arguments = parser.parse_args()
    config_path = (PROJECT_ROOT / arguments.config).resolve()
    artifact_root = (PROJECT_ROOT / arguments.artifact_root).resolve()
    config = _configuration(config_path, arguments.seed)
    if arguments.device is not None:
        config = replace(config, device=arguments.device)
    examples, fingerprint = load_prepared_examples(
        private_root=PRIVATE_ROOT,
        processed_root=PROCESSED_ROOT,
    )
    checkpoint_path, metadata_path = _linear_probe_paths()
    result = train_partial_finetune(
        examples=examples,
        dataset_fingerprint=fingerprint,
        config=config,
        artifact_root=artifact_root,
        linear_probe_checkpoint_path=checkpoint_path,
        linear_probe_metadata_path=metadata_path,
    )
    print(f"Run ID: {result.run_id}")
    print(f"Selected epoch: {result.selected_epoch}")
    print(f"Validation Top 1: {result.validation_top1_accuracy:.3f}")
    print(f"Validation Top 3: {result.validation_top3_accuracy:.3f}")
    print(f"Validation macro recall: {result.validation_macro_recall:.3f}")
    print(f"Artifacts: {result.output_directory}")


if __name__ == "__main__":
    main()
