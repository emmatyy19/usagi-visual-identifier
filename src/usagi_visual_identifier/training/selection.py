"""Reproducible selection of one partial fine tuning checkpoint."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from usagi_visual_identifier.training.partial_finetune import PartialFinetuneError


@dataclass(frozen=True)
class SelectedCheckpoint:
    """Exact checkpoint chosen without consulting unknown validation images."""

    run_id: str
    checkpoint_sha256: str
    selected_epoch: int
    validation_loss: float
    seed: int
    dataset_fingerprint: str
    model_id: str
    model_revision: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON ready selection record."""
        return {
            "schema_version": 1,
            "selection_data": "known_validation_only",
            "selection_policy": {
                "required_augmentation": "conservative",
                "metric": "validation_loss",
                "direction": "minimum",
                "tie_breaker": "lowest_seed",
            },
            "run_id": self.run_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "selected_epoch": self.selected_epoch,
            "validation_loss": self.validation_loss,
            "seed": self.seed,
            "dataset_fingerprint": self.dataset_fingerprint,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "test_split_accessed": False,
        }


def _object(value: object, *, name: str) -> dict[str, object]:
    """Return a string keyed JSON object or fail clearly."""
    if not isinstance(value, dict):
        raise PartialFinetuneError(f"{name} must be a JSON object")
    raw = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in raw):
        raise PartialFinetuneError(f"{name} keys must be strings")
    return cast(dict[str, object], raw)


def _read_metadata(path: Path) -> dict[str, object]:
    """Read one fine tuning metadata file."""
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PartialFinetuneError(f"could not read {path}: {error}") from error
    return _object(value, name=str(path))


def _string(values: dict[str, object], key: str) -> str:
    """Read one required string field."""
    value = values.get(key)
    if not isinstance(value, str):
        raise PartialFinetuneError(f"{key} must be a string")
    return value


def _integer(values: dict[str, object], key: str) -> int:
    """Read one required integer field."""
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise PartialFinetuneError(f"{key} must be an integer")
    return value


def _number(values: dict[str, object], key: str) -> float:
    """Read one required numeric field."""
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PartialFinetuneError(f"{key} must be numeric")
    return float(value)


def select_partial_finetune_checkpoint(
    *, metadata_paths: tuple[Path, ...]
) -> SelectedCheckpoint:
    """Choose the lowest loss conservative run using known validation only."""
    if not metadata_paths:
        raise PartialFinetuneError("at least one fine tuning run is required")
    candidates: list[SelectedCheckpoint] = []
    expected_fingerprint: str | None = None
    expected_model: tuple[str, str] | None = None
    for path in metadata_paths:
        metadata = _read_metadata(path)
        configuration = _object(
            metadata.get("configuration"), name="fine tuning configuration"
        )
        augmentation = _object(
            configuration.get("augmentation"), name="augmentation configuration"
        )
        if augmentation.get("enabled") is not True:
            continue
        if metadata.get("test_split_accessed") is not False:
            raise PartialFinetuneError(
                "checkpoint selection cannot include test access"
            )
        candidate = SelectedCheckpoint(
            run_id=_string(metadata, "run_id"),
            checkpoint_sha256=_string(metadata, "checkpoint_sha256"),
            selected_epoch=_integer(metadata, "selected_epoch"),
            validation_loss=_number(metadata, "validation_loss"),
            seed=_integer(configuration, "seed"),
            dataset_fingerprint=_string(metadata, "dataset_fingerprint"),
            model_id=_string(metadata, "model_id"),
            model_revision=_string(metadata, "model_revision"),
        )
        if expected_fingerprint is None:
            expected_fingerprint = candidate.dataset_fingerprint
            expected_model = (candidate.model_id, candidate.model_revision)
        if candidate.dataset_fingerprint != expected_fingerprint:
            raise PartialFinetuneError("candidate dataset fingerprints do not match")
        if (candidate.model_id, candidate.model_revision) != expected_model:
            raise PartialFinetuneError("candidate base model revisions do not match")
        candidates.append(candidate)
    if not candidates:
        raise PartialFinetuneError("no conservative fine tuning runs were found")
    return min(candidates, key=lambda item: (item.validation_loss, item.seed))


def write_selected_checkpoint(*, path: Path, selected: SelectedCheckpoint) -> None:
    """Atomically write the public checkpoint selection record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(selected.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
