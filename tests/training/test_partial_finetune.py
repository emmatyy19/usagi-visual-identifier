"""Tests for partial DINOv2 fine tuning boundaries and artifacts."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import nn

from usagi_visual_identifier.data.models import DatasetSplit
from usagi_visual_identifier.embeddings import EmbeddingExample
from usagi_visual_identifier.training import (
    AugmentationConfig,
    PartialFinetuneConfig,
    PartialFinetuneError,
    PartialFinetuneModel,
    prepare_partial_finetune_data,
    train_partial_finetune,
)
from usagi_visual_identifier.training import partial_finetune as partial_finetune_module


class _TinyEncoder(nn.Module):
    """Expose transformer shaped layers under a typed attribute."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])


@dataclass(frozen=True)
class _TinyConfig:
    hidden_size: int = 4


class _TinyBackbone(nn.Module):
    """Small DINO shaped backbone used without downloading model weights."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _TinyConfig()
        self.input_projection = nn.Linear(3, 4)
        self.encoder = _TinyEncoder()
        self.layernorm = nn.LayerNorm(4)

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        """Return a CLS compatible hidden state."""
        pooled = pixel_values.mean(dim=(2, 3))
        hidden = self.input_projection(pooled).unsqueeze(1)
        for layer in self.encoder.layer:
            hidden = torch.tanh(layer(hidden))
        return SimpleNamespace(last_hidden_state=self.layernorm(hidden))


class _TinyProcessor:
    """Convert small PIL images to a tensor batch."""

    def __call__(
        self,
        *,
        images: Sequence[Image.Image],
        return_tensors: str,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Return normalized channel first pixels."""
        assert return_tensors == "pt"
        del kwargs
        arrays: list[NDArray[np.float32]] = [
            np.asarray(image.resize((4, 4)), dtype=np.float32).transpose(2, 0, 1)
            / 255.0
            for image in images
        ]
        return {
            "pixel_values": torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                np.stack(arrays)
            )
        }


def _load_tiny_components(
    *, config: PartialFinetuneConfig
) -> tuple[_TinyProcessor, _TinyBackbone]:
    """Return local test components through the production loader boundary."""
    del config
    return _TinyProcessor(), _TinyBackbone()


def _example(
    *, root: Path, number: int, identity: str, split: DatasetSplit, color: str
) -> EmbeddingExample:
    """Create one tiny prepared image example."""
    path = root / f"{number}.jpg"
    Image.new("RGB", (8, 8), color).save(path)
    return EmbeddingExample(
        image_id=f"img-{number:06d}",
        identity_id=identity,
        capture_session_id=(
            "session-001" if split is DatasetSplit.TRAIN else "session-002"
        ),
        split=split,
        processed_relative_path=path.name,
        image_path=path,
    )


def _examples(root: Path) -> tuple[EmbeddingExample, ...]:
    """Create two identities across independent sessions."""
    return (
        _example(
            root=root,
            number=1,
            identity="usagi-0001",
            split=DatasetSplit.TRAIN,
            color="red",
        ),
        _example(
            root=root,
            number=2,
            identity="usagi-0002",
            split=DatasetSplit.TRAIN,
            color="blue",
        ),
        _example(
            root=root,
            number=3,
            identity="usagi-0001",
            split=DatasetSplit.VALIDATION,
            color="red",
        ),
        _example(
            root=root,
            number=4,
            identity="usagi-0002",
            split=DatasetSplit.VALIDATION,
            color="blue",
        ),
    )


def test_model_freezes_everything_except_final_block_norm_and_head() -> None:
    """The trainable boundary stays narrow and inspectable."""
    model = PartialFinetuneModel(
        backbone=_TinyBackbone(),
        embedding_dimension=4,
        class_count=2,
    )
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert trainable == {
        "backbone.encoder.layer.1.weight",
        "backbone.encoder.layer.1.bias",
        "backbone.layernorm.weight",
        "backbone.layernorm.bias",
        "classifier.weight",
        "classifier.bias",
    }
    assert model(torch.ones((2, 3, 4, 4))).shape == (2, 2)


def test_data_selection_never_opens_the_test_split(tmp_path: Path) -> None:
    """Test metadata can exist while only train and validation enter a run."""
    examples = (*_examples(tmp_path),)
    sealed = EmbeddingExample(
        image_id="img-000005",
        identity_id="usagi-0001",
        capture_session_id="session-003",
        split=DatasetSplit.TEST,
        processed_relative_path="missing-test.jpg",
        image_path=tmp_path / "missing-test.jpg",
    )
    selected = prepare_partial_finetune_data(examples=(*examples, sealed))

    assert len(selected.train_examples) == 2
    assert len(selected.validation_examples) == 2
    assert all(
        example.split is not DatasetSplit.TEST for example in selected.train_examples
    )
    assert all(
        example.split is not DatasetSplit.TEST
        for example in selected.validation_examples
    )

    unknown_validation = dataclass_replace(
        examples[2],
        image_id="img-000006",
        identity_id="unknown-0001",
        capture_session_id="session-004",
    )
    selected_with_unknown = prepare_partial_finetune_data(
        examples=(*examples, unknown_validation)
    )
    assert selected_with_unknown.validation_examples == selected.validation_examples


def test_tiny_training_publishes_a_loadable_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The training loop selects and saves only trainable tensors."""
    image_root = tmp_path / "images"
    image_root.mkdir()
    config = PartialFinetuneConfig(
        seed=7,
        epochs=2,
        batch_size=2,
        backbone_learning_rate=0.001,
        head_learning_rate=0.01,
        patience=2,
        device="cpu",
        mixed_precision=False,
        warm_start_linear_probe=False,
        augmentation=AugmentationConfig(enabled=False),
        model_id="tiny/model",
        model_revision="revision",
    )
    result = train_partial_finetune(
        examples=_examples(image_root),
        dataset_fingerprint="fixture-fingerprint",
        config=config,
        artifact_root=tmp_path,
        processor=_TinyProcessor(),
        backbone=_TinyBackbone(),
    )

    assert result.checkpoint_path.suffix == ".safetensors"
    assert result.selected_epoch > 0
    assert result.validation_report_path.is_file()
    assert result.metadata_path.is_file()

    monkeypatch.setattr(
        "usagi_visual_identifier.training.partial_finetune._load_components",
        _load_tiny_components,
    )
    loaded, _, metadata = partial_finetune_module.load_partial_finetune(
        checkpoint_path=result.checkpoint_path,
        metadata_path=result.metadata_path,
        requested_device="cpu",
    )

    assert loaded(torch.ones((1, 3, 4, 4))).shape == (1, 2)
    assert metadata["run_id"] == result.run_id

    invalid_metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    invalid_metadata["class_ids"] = "not a list"
    invalid_path = tmp_path / "invalid-metadata.json"
    invalid_path.write_text(json.dumps(invalid_metadata), encoding="utf-8")
    with pytest.raises(PartialFinetuneError, match="class mapping"):
        partial_finetune_module.load_partial_finetune(
            checkpoint_path=result.checkpoint_path,
            metadata_path=invalid_path,
            requested_device="cpu",
        )

    invalid_metadata["class_ids"] = [1, "usagi-0002"]
    invalid_path.write_text(json.dumps(invalid_metadata), encoding="utf-8")
    with pytest.raises(PartialFinetuneError, match="class mapping"):
        partial_finetune_module.load_partial_finetune(
            checkpoint_path=result.checkpoint_path,
            metadata_path=invalid_path,
            requested_device="cpu",
        )


def test_config_rejects_an_aggressive_crop() -> None:
    """Identity details cannot be removed by an unsafe configured crop."""
    with pytest.raises(PartialFinetuneError, match="crop scale"):
        AugmentationConfig(crop_scale_minimum=0.5)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"crop_ratio_minimum": 0.7}, "ratio minimum"),
        ({"crop_ratio_maximum": 1.3}, "ratio maximum"),
        ({"rotation_degrees": 11.0}, "rotation"),
        ({"brightness": 0.3}, "brightness"),
        ({"contrast": 0.3}, "contrast"),
        ({"saturation": 0.3}, "saturation"),
        ({"hue": 0.1}, "hue"),
    ],
)
def test_augmentation_config_rejects_unsafe_values(
    values: dict[str, float], message: str
) -> None:
    """Every documented augmentation limit is enforced."""
    with pytest.raises(PartialFinetuneError, match=message):
        AugmentationConfig(**values)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"seed": -1}, "seed"),
        ({"epochs": 0}, "epochs"),
        ({"batch_size": 0}, "batch size"),
        ({"backbone_learning_rate": 0.0}, "backbone learning rate"),
        ({"head_learning_rate": float("nan")}, "head learning rate"),
        (
            {"backbone_learning_rate": 0.1, "head_learning_rate": 0.01},
            "lower than head",
        ),
        ({"weight_decay": -1.0}, "weight decay"),
        ({"gradient_clip_norm": 0.0}, "gradient clip"),
        ({"patience": 0}, "patience"),
        ({"minimum_delta": -1.0}, "minimum delta"),
        ({"device": "tpu"}, "device"),
        ({"model_id": ""}, "pinned model"),
        ({"model_revision": ""}, "pinned model"),
    ],
)
def test_training_config_rejects_invalid_values(
    values: dict[str, object], message: str
) -> None:
    """Training fails early when a controlled value is invalid."""
    with pytest.raises(PartialFinetuneError, match=message):
        PartialFinetuneConfig(**values)  # pyright: ignore[reportArgumentType]


def test_json_config_round_trip_and_seed_override(tmp_path: Path) -> None:
    """The checked in JSON format produces the expected typed configuration."""
    source = Path("configs/partial_finetune/conservative.json")
    path = tmp_path / "config.json"
    path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    config = partial_finetune_module.load_partial_finetune_config(
        path=path, seed_override=123
    )

    assert config.seed == 123
    assert config.augmentation.enabled
    assert config.as_dict()["model_revision"] == config.model_revision


@pytest.mark.parametrize(
    "content",
    ["not JSON", "[]"],
)
def test_json_config_rejects_invalid_documents(tmp_path: Path, content: str) -> None:
    """Malformed and incorrectly shaped configuration files are rejected."""
    path = tmp_path / "config.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(PartialFinetuneError):
        partial_finetune_module.load_partial_finetune_config(path=path)


def test_json_config_rejects_wrong_scalar_types() -> None:
    """Boolean, integer, numeric, and string fields remain distinct."""
    source = Path("configs/partial_finetune/conservative.json")
    values = json.loads(source.read_text(encoding="utf-8"))
    cases = (
        ("mixed_precision", 1, "Boolean"),
        ("epochs", True, "integer"),
        ("minimum_delta", True, "numeric"),
        ("device", 1, "string"),
    )
    for key, value, message in cases:
        changed = {**values, key: value}
        with pytest.raises(PartialFinetuneError, match=message):
            partial_finetune_module.partial_finetune_config_from_mapping(changed)


def test_data_selection_rejects_invalid_partitions(tmp_path: Path) -> None:
    """Training and validation must be complete, known, and session separated."""
    examples = _examples(tmp_path)
    with pytest.raises(PartialFinetuneError, match="both training"):
        prepare_partial_finetune_data(examples=examples[:2])

    unknown = dataclass_replace(examples[0], identity_id="unknown-0001")
    with pytest.raises(PartialFinetuneError, match="unknown identities"):
        prepare_partial_finetune_data(examples=(unknown, *examples[1:]))

    wrong_class = dataclass_replace(examples[2], identity_id="usagi-0003")
    with pytest.raises(PartialFinetuneError, match="must match"):
        prepare_partial_finetune_data(
            examples=(*examples[:2], wrong_class, examples[3])
        )

    shared_session = dataclass_replace(
        examples[2], capture_session_id=examples[0].capture_session_id
    )
    with pytest.raises(PartialFinetuneError, match="crosses"):
        prepare_partial_finetune_data(
            examples=(*examples[:2], shared_session, examples[3])
        )


def test_model_rejects_invalid_or_non_dino_backbones() -> None:
    """The trainable boundary cannot silently target the wrong architecture."""
    with pytest.raises(PartialFinetuneError, match="positive dimension"):
        PartialFinetuneModel(
            backbone=_TinyBackbone(), embedding_dimension=0, class_count=2
        )
    with pytest.raises(PartialFinetuneError, match="encoder layers"):
        PartialFinetuneModel(
            backbone=nn.Linear(2, 2), embedding_dimension=2, class_count=2
        )

    backbone = _TinyBackbone()
    backbone.layernorm = None  # type: ignore[assignment]
    with pytest.raises(PartialFinetuneError, match="layer norm"):
        PartialFinetuneModel(backbone=backbone, embedding_dimension=4, class_count=2)


def test_enabled_augmentation_preserves_an_image_shape() -> None:
    """The reviewed transform returns a PIL image ready for the processor."""
    transform = partial_finetune_module.build_training_augmentation(
        AugmentationConfig(enabled=True)
    )
    assert transform is not None
    result = transform(Image.new("RGB", (32, 32), "gold"))
    assert isinstance(result, Image.Image)
    assert result.size == (224, 224)


def test_pixel_preparation_rejects_invalid_processor_output(tmp_path: Path) -> None:
    """A processor cannot silently return an incorrectly shaped value."""
    examples = _examples(tmp_path)

    class _BadProcessor:
        def __call__(
            self,
            *,
            images: Sequence[Image.Image],
            return_tensors: str,
            **kwargs: object,
        ) -> dict[str, torch.Tensor]:
            del images, return_tensors, kwargs
            return {"pixel_values": torch.ones(2)}

    with pytest.raises(PartialFinetuneError, match="batched pixels"):
        partial_finetune_module._pixel_values(  # pyright: ignore[reportPrivateUsage]
            processor=_BadProcessor(),
            examples=examples[:1],
            augmentation=None,
        )


def test_image_loading_applies_and_validates_augmentation(tmp_path: Path) -> None:
    """Augmentation must preserve the PIL boundary used by the processor."""
    examples = _examples(tmp_path)
    loaded = partial_finetune_module._load_images(  # pyright: ignore[reportPrivateUsage]
        examples=examples[:1], augmentation=nn.Identity()
    )
    try:
        assert loaded[0].mode == "RGB"
    finally:
        loaded[0].close()

    class _InvalidAugmentation(nn.Module):
        def forward(self, image: Image.Image) -> torch.Tensor:
            del image
            return torch.ones(1)

    with pytest.raises(PartialFinetuneError, match="PIL image"):
        partial_finetune_module._load_images(  # pyright: ignore[reportPrivateUsage]
            examples=examples[:1], augmentation=_InvalidAugmentation()
        )


def test_training_requires_paired_components_and_warm_start(tmp_path: Path) -> None:
    """Injected components and requested warm starts must be complete pairs."""
    examples = _examples(tmp_path)
    config = PartialFinetuneConfig(
        device="cpu",
        warm_start_linear_probe=False,
        augmentation=AugmentationConfig(enabled=False),
    )
    with pytest.raises(PartialFinetuneError, match="supplied together"):
        train_partial_finetune(
            examples=examples,
            dataset_fingerprint="fixture",
            config=config,
            artifact_root=tmp_path,
            processor=_TinyProcessor(),
        )

    warm_config = dataclass_config_replace(config, warm_start_linear_probe=True)
    with pytest.raises(PartialFinetuneError, match="warm start artifacts"):
        train_partial_finetune(
            examples=examples,
            dataset_fingerprint="fixture",
            config=warm_config,
            artifact_root=tmp_path,
            processor=_TinyProcessor(),
            backbone=_TinyBackbone(),
        )


def test_amp_overflow_skips_update_and_recovers_on_next_step() -> None:
    """A temporary scaled gradient overflow is not a fatal training error."""
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0, growth_interval=100)
    original_weight = model.weight.detach().clone()

    overflowing_loss = model(torch.ones((1, 1))).sum() * torch.tensor(float("inf"))
    scaler.scale(overflowing_loss).backward()  # pyright: ignore[reportUnknownMemberType]
    skipped_norm = partial_finetune_module._complete_optimizer_step(  # pyright: ignore[reportPrivateUsage]
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        gradient_clip_norm=1.0,
    )

    assert skipped_norm is None
    assert torch.equal(model.weight, original_weight)
    assert scaler.get_scale() == 1.0

    optimizer.zero_grad(set_to_none=True)
    finite_loss = model(torch.ones((1, 1))).sum()
    scaler.scale(finite_loss).backward()  # pyright: ignore[reportUnknownMemberType]
    finite_norm = partial_finetune_module._complete_optimizer_step(  # pyright: ignore[reportPrivateUsage]
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        gradient_clip_norm=1.0,
    )

    assert finite_norm == pytest.approx(1.0)
    assert not torch.equal(model.weight, original_weight)


def test_nonfinite_full_precision_gradient_remains_fatal() -> None:
    """The recovery path applies only when an active scaler skips the update."""
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    loss = model(torch.ones((1, 1))).sum() * torch.tensor(float("inf"))
    scaler.scale(loss).backward()  # pyright: ignore[reportUnknownMemberType]

    with pytest.raises(PartialFinetuneError, match="nonfinite gradient norm"):
        partial_finetune_module._complete_optimizer_step(  # pyright: ignore[reportPrivateUsage]
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip_norm=1.0,
        )


def test_load_rejects_corrupt_metadata_and_checkpoint(tmp_path: Path) -> None:
    """Published inference artifacts are verified before model construction."""
    checkpoint = tmp_path / "delta.safetensors"
    checkpoint.write_bytes(b"not a checkpoint")
    metadata = tmp_path / "metadata.json"
    metadata.write_text("not JSON", encoding="utf-8")
    with pytest.raises(PartialFinetuneError, match="could not load"):
        partial_finetune_module.load_partial_finetune(
            checkpoint_path=checkpoint,
            metadata_path=metadata,
            requested_device="cpu",
        )

    metadata.write_text("{}", encoding="utf-8")
    with pytest.raises(PartialFinetuneError, match="hash"):
        partial_finetune_module.load_partial_finetune(
            checkpoint_path=checkpoint,
            metadata_path=metadata,
            requested_device="cpu",
        )


def dataclass_replace(example: EmbeddingExample, **changes: object) -> EmbeddingExample:
    """Return one modified immutable example for a failure case."""
    from dataclasses import replace

    return replace(example, **changes)


def dataclass_config_replace(
    config: PartialFinetuneConfig, **changes: object
) -> PartialFinetuneConfig:
    """Return one modified immutable training configuration."""
    from dataclasses import replace

    return replace(config, **changes)
