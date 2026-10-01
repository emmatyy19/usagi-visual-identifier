"""Freeze one partial fine tuning checkpoint using known validation loss."""

from pathlib import Path

from usagi_visual_identifier.training.selection import (
    select_partial_finetune_checkpoint,
    write_selected_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "artifacts" / "models" / "partial_finetune"
OUTPUT_PATH = PROJECT_ROOT / "configs" / "partial_finetune" / "selected.json"


def main() -> None:
    """Select from completed runs before evaluating unknown photos."""
    metadata_paths = tuple(sorted(MODEL_ROOT.glob("*/metadata.json")))
    selected = select_partial_finetune_checkpoint(metadata_paths=metadata_paths)
    write_selected_checkpoint(path=OUTPUT_PATH, selected=selected)
    print(f"Selected run: {selected.run_id}")
    print(f"Selected epoch: {selected.selected_epoch}")
    print(f"Known validation loss: {selected.validation_loss:.6f}")
    print(f"Checkpoint SHA256: {selected.checkpoint_sha256}")
    print(f"Selection record: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
