"""Image encoder backed by one verified partial fine tuning checkpoint."""

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from usagi_visual_identifier.embeddings.artifacts import EmbeddingError
from usagi_visual_identifier.training.partial_finetune import PartialFinetuneModel


class ImageProcessor(Protocol):
    """Processor boundary needed for embedding inference."""

    def __call__(
        self, *, images: Sequence[Image.Image], return_tensors: str
    ) -> Mapping[str, torch.Tensor]:
        """Return tensor inputs for a batch of images."""
        ...


class PartialFinetuneEncoder:
    """Normalized CLS embeddings from a selected tuned DINOv2 backbone."""

    def __init__(
        self,
        *,
        model: PartialFinetuneModel,
        processor: ImageProcessor,
        run_id: str,
        checkpoint_sha256: str,
    ) -> None:
        """Keep the already verified model and its exact checkpoint identity."""
        try:
            self._device = next(model.parameters()).device
        except StopIteration as error:
            raise EmbeddingError("fine tuned model has no parameters") from error
        self._model = model
        self._processor = processor
        self._run_id = run_id
        self._checkpoint_sha256 = checkpoint_sha256

    @property
    def model_id(self) -> str:
        """Return a stable artifact directory identifier."""
        return f"partial-finetune/{self._run_id}"

    @property
    def model_revision(self) -> str:
        """Return the verified checkpoint hash."""
        return self._checkpoint_sha256

    @property
    def device_name(self) -> str:
        """Return the device holding the tuned model."""
        return str(self._device)

    def encode(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        """Return one normalized tuned CLS embedding per image."""
        if not images:
            raise EmbeddingError("fine tuned encoder cannot encode an empty batch")
        processed = self._processor(images=images, return_tensors="pt")
        pixels = processed.get("pixel_values")
        if not isinstance(pixels, torch.Tensor):
            raise EmbeddingError("image processor did not return pixel values")
        with torch.inference_mode():
            embeddings = self._model.embed(pixels.to(self._device))
        return np.asarray(embeddings.detach().cpu().float().numpy(), dtype=np.float32)
