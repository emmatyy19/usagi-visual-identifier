"""Tests for the verified partial fine tuning embedding adapter."""

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from usagi_visual_identifier.embeddings.artifacts import EmbeddingError
from usagi_visual_identifier.embeddings.partial_finetune import PartialFinetuneEncoder
from usagi_visual_identifier.training.partial_finetune import PartialFinetuneModel


class _Processor:
    """Create deterministic pixel rows for adapter tests."""

    def __init__(self, *, include_pixels: bool = True) -> None:
        self.include_pixels = include_pixels

    def __call__(
        self, *, images: Sequence[Image.Image], return_tensors: str
    ) -> Mapping[str, torch.Tensor]:
        assert return_tensors == "pt"
        if not self.include_pixels:
            return {}
        return {"pixel_values": torch.ones((len(images), 2), dtype=torch.float32)}


class _EmbeddingModel(nn.Module):
    """Expose the small embed boundary used by the adapter."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return pixel_values + self.anchor


class _ParameterlessModel(nn.Module):
    """Represent an invalid model without a discoverable device."""


def test_partial_finetune_encoder_exposes_identity_and_encodes() -> None:
    """The adapter reports checkpoint identity and returns float32 rows."""
    encoder = PartialFinetuneEncoder(
        model=cast(PartialFinetuneModel, _EmbeddingModel()),
        processor=_Processor(),
        run_id="run-1",
        checkpoint_sha256="checkpoint-sha",
    )
    images = [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]

    embeddings = encoder.encode(images)

    assert encoder.model_id == "partial-finetune/run-1"
    assert encoder.model_revision == "checkpoint-sha"
    assert encoder.device_name == "cpu"
    np.testing.assert_array_equal(embeddings, np.ones((2, 2), dtype=np.float32))


def test_partial_finetune_encoder_rejects_invalid_inputs() -> None:
    """Empty batches and processor output without pixels fail clearly."""
    encoder = PartialFinetuneEncoder(
        model=cast(PartialFinetuneModel, _EmbeddingModel()),
        processor=_Processor(include_pixels=False),
        run_id="run-1",
        checkpoint_sha256="checkpoint-sha",
    )
    with pytest.raises(EmbeddingError, match="empty batch"):
        encoder.encode([])
    with pytest.raises(EmbeddingError, match="did not return pixel values"):
        encoder.encode([Image.new("RGB", (2, 2))])


def test_partial_finetune_encoder_rejects_parameterless_model() -> None:
    """A model without parameters cannot establish an inference device."""
    with pytest.raises(EmbeddingError, match="has no parameters"):
        PartialFinetuneEncoder(
            model=cast(PartialFinetuneModel, _ParameterlessModel()),
            processor=_Processor(),
            run_id="run-1",
            checkpoint_sha256="checkpoint-sha",
        )
