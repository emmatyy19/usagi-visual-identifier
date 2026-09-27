"""Pinned DINOv2 encoder with explicit local device selection."""

import importlib
from collections.abc import Mapping, Sequence
from typing import Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from usagi_visual_identifier.embeddings.artifacts import EmbeddingError

DEFAULT_DINOV2_MODEL_ID = "facebook/dinov2-small"
DEFAULT_DINOV2_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"


class _ImageProcessor(Protocol):
    """Typed boundary for the dynamically loaded image processor."""

    def __call__(
        self, *, images: Sequence[Image.Image], return_tensors: str
    ) -> Mapping[str, torch.Tensor]: ...


class _ModelOutput(Protocol):
    """Output fields used from a DINOv2 forward pass."""

    last_hidden_state: torch.Tensor


class _VisionModel(Protocol):
    """Typed boundary for the dynamically loaded transformer model."""

    def to(self, device: torch.device) -> "_VisionModel": ...

    def eval(self) -> "_VisionModel": ...

    def __call__(self, **inputs: torch.Tensor) -> _ModelOutput: ...


def select_torch_device(*, requested: str = "auto") -> torch.device:
    """Select CUDA, then Apple Metal, then CPU unless explicitly requested."""
    allowed = {"auto", "cuda", "mps", "cpu"}
    if requested not in allowed:
        raise EmbeddingError(f"unsupported device selection: {requested}")

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise EmbeddingError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise EmbeddingError("Apple Metal was requested but is not available")
    return torch.device(requested)


class Dinov2Encoder:
    """Frozen DINOv2 CLS token embeddings from a pinned model revision."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_DINOV2_MODEL_ID,
        model_revision: str = DEFAULT_DINOV2_REVISION,
        requested_device: str = "auto",
    ) -> None:
        """Load the processor and frozen model onto the selected device."""
        self._model_id = model_id
        self._model_revision = model_revision
        self._device = select_torch_device(requested=requested_device)

        transformers = importlib.import_module("transformers")
        processor_loader = transformers.AutoImageProcessor
        model_loader = transformers.AutoModel
        self._processor = cast(
            _ImageProcessor,
            processor_loader.from_pretrained(
                model_id,
                revision=model_revision,
            ),
        )
        model = cast(
            _VisionModel,
            model_loader.from_pretrained(
                model_id,
                revision=model_revision,
                use_safetensors=True,
            ),
        )
        self._model = model.to(self._device).eval()

    @property
    def model_id(self) -> str:
        """Return the Hugging Face model identifier."""
        return self._model_id

    @property
    def model_revision(self) -> str:
        """Return the pinned Hugging Face revision."""
        return self._model_revision

    @property
    def device_name(self) -> str:
        """Return the selected PyTorch device."""
        return str(self._device)

    def encode(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        """Return one frozen CLS embedding for each input image."""
        if not images:
            raise EmbeddingError("DINOv2 cannot encode an empty image batch")
        processed = self._processor(images=images, return_tensors="pt")
        inputs = {name: tensor.to(self._device) for name, tensor in processed.items()}
        with torch.inference_mode():
            output = self._model(**inputs)
            cls_embeddings = output.last_hidden_state[:, 0, :]
        return np.asarray(
            cls_embeddings.detach().cpu().float().numpy(), dtype=np.float32
        )
