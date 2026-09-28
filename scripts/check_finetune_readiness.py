"""Check the GPU computer before starting partial fine tuning."""

import argparse
import json
import shutil
from pathlib import Path
from typing import cast

import torch

from usagi_visual_identifier.embeddings import load_prepared_examples
from usagi_visual_identifier.training import (
    load_linear_probe,
    load_partial_finetune_config,
    prepare_partial_finetune_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"


def _linear_paths() -> tuple[Path, Path]:
    """Resolve the imported linear warm start artifacts."""
    root = ARTIFACT_ROOT / "models/linear_probe"
    raw: object = json.loads((root / "latest.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("linear probe latest pointer is invalid")
    values: dict[str, object] = {}
    for key, value in cast(dict[object, object], raw).items():
        if not isinstance(key, str):
            raise ValueError("linear probe latest pointer is invalid")
        values[key] = value
    run_id = values.get("run_id")
    if not isinstance(run_id, str):
        raise ValueError("linear probe latest pointer is invalid")
    run_root = root / run_id
    return run_root / "checkpoint.pt", run_root / "metadata.json"


def main() -> None:
    """Print actionable hardware, data, and warm start checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/partial_finetune/conservative.json"),
    )
    arguments = parser.parse_args()
    config = load_partial_finetune_config(path=PROJECT_ROOT / arguments.config)
    examples, fingerprint = load_prepared_examples(
        private_root=PROJECT_ROOT / "data/private",
        processed_root=PROJECT_ROOT / "data/processed",
    )
    selected = prepare_partial_finetune_data(examples=examples)
    checkpoint_path, metadata_path = _linear_paths()
    _, linear_metadata = load_linear_probe(
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
    )
    free_gib = shutil.disk_usage(PROJECT_ROOT).free / 1024**3

    print(f"Torch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Configured device: {config.device}")
    print(f"Training images: {len(selected.train_examples)}")
    print(f"Validation images: {len(selected.validation_examples)}")
    print(f"Classes: {len(selected.class_ids)}")
    print(f"Dataset fingerprint: {fingerprint}")
    print(f"Linear probe run: {linear_metadata['run_id']}")
    print(f"Free disk space: {free_gib:.1f} GiB")
    print("Test images selected: 0")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the configuration requires CUDA but CUDA is unavailable")
    if free_gib < 5:
        raise RuntimeError("at least 5 GiB of free disk space is recommended")
    print("Readiness check passed")


if __name__ == "__main__":
    main()
