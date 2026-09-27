"""Tests for DINOv2 loading and device behavior without network access."""

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from usagi_visual_identifier.embeddings.artifacts import EmbeddingError
from usagi_visual_identifier.embeddings.dinov2 import (
    Dinov2Encoder,
    select_torch_device,
)


def test_select_torch_device_prefers_cuda_then_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic selection follows the documented accelerator order."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert select_torch_device().type == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_torch_device().type == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert select_torch_device().type == "cpu"


def test_select_torch_device_rejects_unavailable_or_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit invalid accelerator requests fail instead of falling back."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(EmbeddingError, match="CUDA"):
        select_torch_device(requested="cuda")
    with pytest.raises(EmbeddingError, match="unsupported"):
        select_torch_device(requested="quantum")


def test_dinov2_encoder_uses_pinned_loaders_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter loads once and returns the CLS token as float32."""
    calls: list[tuple[str, str, str]] = []

    class FakeProcessor:
        def __call__(
            self, *, images: Sequence[Image.Image], return_tensors: str
        ) -> dict[str, torch.Tensor]:
            assert return_tensors == "pt"
            return {"pixel_values": torch.ones((len(images), 3, 2, 2))}

    class FakeProcessorLoader:
        @staticmethod
        def from_pretrained(model_id: str, *, revision: str) -> FakeProcessor:
            calls.append(("processor", model_id, revision))
            return FakeProcessor()

    class FakeModel:
        def to(self, device: torch.device) -> "FakeModel":
            assert device.type == "cpu"
            return self

        def eval(self) -> "FakeModel":
            return self

        def __call__(self, **inputs: torch.Tensor) -> SimpleNamespace:
            batch_size = inputs["pixel_values"].shape[0]
            hidden = torch.arange(batch_size * 6, dtype=torch.float32).reshape(
                batch_size, 2, 3
            )
            return SimpleNamespace(last_hidden_state=hidden)

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(
            model_id: str, *, revision: str, use_safetensors: bool
        ) -> FakeModel:
            assert use_safetensors is True
            calls.append(("model", model_id, revision))
            return FakeModel()

    module = SimpleNamespace(
        AutoImageProcessor=FakeProcessorLoader,
        AutoModel=FakeModelLoader,
    )

    def fake_import(name: str) -> SimpleNamespace:
        assert name == "transformers"
        return module

    monkeypatch.setattr(
        "usagi_visual_identifier.embeddings.dinov2.importlib.import_module",
        fake_import,
    )
    encoder = Dinov2Encoder(
        model_id="example/model",
        model_revision="commit",
        requested_device="cpu",
    )
    images = [Image.new("RGB", (2, 2), "red"), Image.new("RGB", (2, 2), "blue")]
    values = encoder.encode(images)

    assert calls == [
        ("processor", "example/model", "commit"),
        ("model", "example/model", "commit"),
    ]
    assert encoder.device_name == "cpu"
    assert encoder.model_id == "example/model"
    assert encoder.model_revision == "commit"
    assert values.dtype == np.float32
    assert values.shape == (2, 3)
    assert np.array_equal(values[0], np.asarray([0.0, 1.0, 2.0], dtype=np.float32))
    for image in images:
        image.close()
