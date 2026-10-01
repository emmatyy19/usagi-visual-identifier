"""Tests for validation only fine tuning checkpoint selection."""

import json
from pathlib import Path

import pytest

from usagi_visual_identifier.training.partial_finetune import PartialFinetuneError
from usagi_visual_identifier.training.selection import (
    select_partial_finetune_checkpoint,
    write_selected_checkpoint,
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


def test_checkpoint_selection_uses_seed_to_break_loss_tie(tmp_path: Path) -> None:
    """The lower seed makes an otherwise tied selection deterministic."""
    paths = (
        _metadata(tmp_path / "a.json", run_id="a", loss=0.1, seed=123, augmented=True),
        _metadata(tmp_path / "b.json", run_id="b", loss=0.1, seed=42, augmented=True),
    )
    selected = select_partial_finetune_checkpoint(metadata_paths=paths)
    assert selected.run_id == "b"

    output = tmp_path / "nested" / "selected.json"
    write_selected_checkpoint(path=output, selected=selected)
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["selection_data"] == "known_validation_only"
    assert value["run_id"] == "b"
    assert value["test_split_accessed"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"run_id": 1}, "run_id must be a string"),
        ({"selected_epoch": True}, "selected_epoch must be an integer"),
        ({"validation_loss": False}, "validation_loss must be numeric"),
    ],
)
def test_checkpoint_selection_validates_required_field_types(
    tmp_path: Path, mutation: dict[str, object], message: str
) -> None:
    """Malformed metadata fails at the field that breaks the public contract."""
    path = _metadata(
        tmp_path / "run.json", run_id="run", loss=0.1, seed=42, augmented=True
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(mutation)
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(PartialFinetuneError, match=message):
        select_partial_finetune_checkpoint(metadata_paths=(path,))


def test_checkpoint_selection_rejects_incompatible_or_missing_runs(
    tmp_path: Path,
) -> None:
    """Selection requires comparable conservative runs."""
    with pytest.raises(PartialFinetuneError, match="at least one"):
        select_partial_finetune_checkpoint(metadata_paths=())

    nonconservative = _metadata(
        tmp_path / "off.json", run_id="off", loss=0.1, seed=42, augmented=False
    )
    with pytest.raises(PartialFinetuneError, match="no conservative"):
        select_partial_finetune_checkpoint(metadata_paths=(nonconservative,))

    first = _metadata(
        tmp_path / "first.json", run_id="first", loss=0.2, seed=42, augmented=True
    )
    second = _metadata(
        tmp_path / "second.json", run_id="second", loss=0.1, seed=123, augmented=True
    )
    value = json.loads(second.read_text(encoding="utf-8"))
    value["dataset_fingerprint"] = "different"
    second.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(PartialFinetuneError, match="fingerprints do not match"):
        select_partial_finetune_checkpoint(metadata_paths=(first, second))


def test_checkpoint_selection_rejects_unreadable_metadata(tmp_path: Path) -> None:
    """Invalid JSON and nonobject JSON are reported as metadata errors."""
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")
    with pytest.raises(PartialFinetuneError, match="could not read"):
        select_partial_finetune_checkpoint(metadata_paths=(invalid,))

    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    with pytest.raises(PartialFinetuneError, match="must be a JSON object"):
        select_partial_finetune_checkpoint(metadata_paths=(array,))
