"""Tests for partial DINOv2 fine tuning boundaries and artifacts."""

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


def test_tiny_training_publishes_a_loadable_delta(tmp_path: Path) -> None:
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


def test_config_rejects_an_aggressive_crop() -> None:
    """Identity details cannot be removed by an unsafe configured crop."""
    with pytest.raises(PartialFinetuneError, match="crop scale"):
        AugmentationConfig(crop_scale_minimum=0.5)
