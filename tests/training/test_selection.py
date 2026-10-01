"""Tests for validation only fine tuning checkpoint selection."""

import json
from pathlib import Path

import pytest

from usagi_visual_identifier.training.partial_finetune import PartialFinetuneError
from usagi_visual_identifier.training.selection import (
    select_partial_finetune_checkpoint,
)


def _metadata(
    path: Path, *, run_id: str, loss: float, seed: int, augmented: bool
) -> Path:
    """Write one compact test metadata record."""
    value = {
        "run_id": run_id,
        "checkpoint_sha256": f"sha-{run_id}",
        "selected_epoch": 3,
        "validation_loss": loss,
        "dataset_fingerprint": "dataset-sha",
        "model_id": "model",
        "model_revision": "revision",
        "test_split_accessed": False,
        "configuration": {"seed": seed, "augmentation": {"enabled": augmented}},
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_checkpoint_selection_uses_lowest_conservative_validation_loss(
    tmp_path: Path,
) -> None:
    """Nonaugmented runs are excluded before comparing known validation loss."""
    paths = (
        _metadata(tmp_path / "a.json", run_id="a", loss=0.2, seed=42, augmented=True),
        _metadata(tmp_path / "b.json", run_id="b", loss=0.1, seed=123, augmented=True),
        _metadata(tmp_path / "c.json", run_id="c", loss=0.01, seed=7, augmented=False),
    )
    selected = select_partial_finetune_checkpoint(metadata_paths=paths)
    assert selected.run_id == "b"
    assert selected.validation_loss == 0.1


def test_checkpoint_selection_rejects_test_access(tmp_path: Path) -> None:
    """A run that touched the final test split cannot be selected."""
    path = _metadata(
        tmp_path / "run.json", run_id="run", loss=0.1, seed=42, augmented=True
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value["test_split_accessed"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(PartialFinetuneError, match="test access"):
        select_partial_finetune_checkpoint(metadata_paths=(path,))
