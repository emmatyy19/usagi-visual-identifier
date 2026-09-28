"""Partial DINOv2 fine tuning on prepared owner photographs."""

import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as functional
from torchvision.transforms import (  # pyright: ignore[reportMissingTypeStubs]
    InterpolationMode,
    v2,
)

from usagi_visual_identifier.data.models import DatasetSplit
from usagi_visual_identifier.embeddings import (
    DEFAULT_DINOV2_MODEL_ID,
    DEFAULT_DINOV2_REVISION,
    EmbeddingExample,
    select_torch_device,
)
from usagi_visual_identifier.training.linear_probe import load_linear_probe

_safetensors = importlib.import_module("safetensors.torch")
_load_safetensors = cast(
    Callable[[str | Path, str], dict[str, torch.Tensor]],
    _safetensors.load_file,
)
_save_safetensors = cast(
    Callable[[dict[str, torch.Tensor], str | Path], None],
    _safetensors.save_file,
)


class PartialFinetuneError(ValueError):
    """Raised when a partial fine tuning run would be unsafe or invalid."""


class ImageProcessor(Protocol):
    """Typed image processor boundary used by real and test components."""

    def __call__(
        self,
        *,
        images: Sequence[Image.Image],
        return_tensors: str,
        **kwargs: object,
    ) -> Mapping[str, torch.Tensor]:
        """Return model inputs for one image batch."""
        ...


class BackboneOutput(Protocol):
    """Output field required from the DINOv2 backbone."""

    last_hidden_state: torch.Tensor


@dataclass(frozen=True)
class AugmentationConfig:
    """Identity preserving training augmentation limits."""

    enabled: bool = True
    crop_scale_minimum: float = 0.88
    crop_ratio_minimum: float = 0.92
    crop_ratio_maximum: float = 1.08
    rotation_degrees: float = 6.0
    brightness: float = 0.12
    contrast: float = 0.12
    saturation: float = 0.08
    hue: float = 0.02

    def __post_init__(self) -> None:
        """Reject destructive or undefined augmentation settings."""
        if not 0.75 <= self.crop_scale_minimum <= 1.0:
            raise PartialFinetuneError("crop scale minimum must be from 0.75 to 1.0")
        if not 0.8 <= self.crop_ratio_minimum <= 1.0:
            raise PartialFinetuneError("crop ratio minimum must be from 0.8 to 1.0")
        if not 1.0 <= self.crop_ratio_maximum <= 1.25:
            raise PartialFinetuneError("crop ratio maximum must be from 1.0 to 1.25")
        if not 0 <= self.rotation_degrees <= 10:
            raise PartialFinetuneError("rotation must be from 0 to 10 degrees")
        for name, value, maximum in (
            ("brightness", self.brightness, 0.2),
            ("contrast", self.contrast, 0.2),
            ("saturation", self.saturation, 0.2),
            ("hue", self.hue, 0.05),
        ):
            if not 0 <= value <= maximum:
                raise PartialFinetuneError(f"{name} augmentation is too strong")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON ready representation."""
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True)
class PartialFinetuneConfig:
    """Complete configuration for one partial fine tuning run."""

    seed: int = 42
    epochs: int = 40
    batch_size: int = 8
    backbone_learning_rate: float = 0.00001
    head_learning_rate: float = 0.0005
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0
    patience: int = 8
    minimum_delta: float = 0.00001
    device: str = "cuda"
    mixed_precision: bool = True
    warm_start_linear_probe: bool = True
    augmentation: AugmentationConfig = AugmentationConfig()
    model_id: str = DEFAULT_DINOV2_MODEL_ID
    model_revision: str = DEFAULT_DINOV2_REVISION

    def __post_init__(self) -> None:
        """Reject values that make training invalid or misleading."""
        if self.seed < 0:
            raise PartialFinetuneError("seed cannot be negative")
        if self.epochs < 1:
            raise PartialFinetuneError("epochs must be positive")
        if self.batch_size < 1:
            raise PartialFinetuneError("batch size must be positive")
        for name, value in (
            ("backbone learning rate", self.backbone_learning_rate),
            ("head learning rate", self.head_learning_rate),
        ):
            if not math.isfinite(value) or value <= 0:
                raise PartialFinetuneError(f"{name} must be finite and positive")
        if self.backbone_learning_rate >= self.head_learning_rate:
            raise PartialFinetuneError(
                "backbone learning rate must be lower than head learning rate"
            )
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise PartialFinetuneError("weight decay must be finite and nonnegative")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
            raise PartialFinetuneError("gradient clip norm must be finite and positive")
        if self.patience < 1:
            raise PartialFinetuneError("patience must be positive")
        if not math.isfinite(self.minimum_delta) or self.minimum_delta < 0:
            raise PartialFinetuneError("minimum delta must be finite and nonnegative")
        if self.device not in {"cpu", "mps", "cuda", "auto"}:
            raise PartialFinetuneError("device must be auto, cpu, mps, or cuda")
        if not self.model_id or not self.model_revision:
            raise PartialFinetuneError("a pinned model ID and revision are required")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON ready representation."""
        value = cast(dict[str, object], asdict(self))
        value["augmentation"] = self.augmentation.as_dict()
        return value


@dataclass(frozen=True)
class PartialFinetuneData:
    """Selected image examples and stable class mapping."""

    train_examples: tuple[EmbeddingExample, ...]
    validation_examples: tuple[EmbeddingExample, ...]
    class_ids: tuple[str, ...]


@dataclass(frozen=True)
class PartialFinetuneRun:
    """Published paths and selected validation results."""

    run_id: str
    output_directory: Path
    checkpoint_path: Path
    metadata_path: Path
    history_path: Path
    validation_report_path: Path
    selected_epoch: int
    validation_top1_accuracy: float
    validation_top3_accuracy: float
    validation_macro_recall: float


def _object_dict(value: object, *, name: str) -> dict[str, object]:
    """Validate an untyped JSON object and return string keyed values."""
    if not isinstance(value, dict):
        raise PartialFinetuneError(f"{name} must be a JSON object")
    result: dict[str, object] = {}
    for key, item in cast(dict[object, object], value).items():
        if not isinstance(key, str):
            raise PartialFinetuneError(f"{name} keys must be strings")
        result[key] = item
    return result


def _required_bool(values: Mapping[str, object], key: str) -> bool:
    """Read one required Boolean configuration value."""
    value = values.get(key)
    if not isinstance(value, bool):
        raise PartialFinetuneError(f"{key} must be a Boolean")
    return value


def _required_int(values: Mapping[str, object], key: str) -> int:
    """Read one required integer configuration value."""
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise PartialFinetuneError(f"{key} must be an integer")
    return value


def _required_float(values: Mapping[str, object], key: str) -> float:
    """Read one required numeric configuration value."""
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PartialFinetuneError(f"{key} must be numeric")
    return float(value)


def _optional_string(values: Mapping[str, object], key: str, default: str) -> str:
    """Read one optional string configuration value."""
    value = values.get(key, default)
    if not isinstance(value, str):
        raise PartialFinetuneError(f"{key} must be a string")
    return value


def partial_finetune_config_from_mapping(
    values: Mapping[str, object], *, seed_override: int | None = None
) -> PartialFinetuneConfig:
    """Build a typed configuration from validated JSON values."""
    augmentation_values = _object_dict(values.get("augmentation"), name="augmentation")
    augmentation = AugmentationConfig(
        enabled=_required_bool(augmentation_values, "enabled"),
        crop_scale_minimum=_required_float(augmentation_values, "crop_scale_minimum"),
        crop_ratio_minimum=_required_float(augmentation_values, "crop_ratio_minimum"),
        crop_ratio_maximum=_required_float(augmentation_values, "crop_ratio_maximum"),
        rotation_degrees=_required_float(augmentation_values, "rotation_degrees"),
        brightness=_required_float(augmentation_values, "brightness"),
        contrast=_required_float(augmentation_values, "contrast"),
        saturation=_required_float(augmentation_values, "saturation"),
        hue=_required_float(augmentation_values, "hue"),
    )
    return PartialFinetuneConfig(
        seed=(
            seed_override
            if seed_override is not None
            else _required_int(values, "seed")
        ),
        epochs=_required_int(values, "epochs"),
        batch_size=_required_int(values, "batch_size"),
        backbone_learning_rate=_required_float(values, "backbone_learning_rate"),
        head_learning_rate=_required_float(values, "head_learning_rate"),
        weight_decay=_required_float(values, "weight_decay"),
        gradient_clip_norm=_required_float(values, "gradient_clip_norm"),
        patience=_required_int(values, "patience"),
        minimum_delta=_required_float(values, "minimum_delta"),
        device=_optional_string(values, "device", "cuda"),
        mixed_precision=_required_bool(values, "mixed_precision"),
        warm_start_linear_probe=_required_bool(values, "warm_start_linear_probe"),
        augmentation=augmentation,
        model_id=_optional_string(values, "model_id", DEFAULT_DINOV2_MODEL_ID),
        model_revision=_optional_string(
            values, "model_revision", DEFAULT_DINOV2_REVISION
        ),
    )


def load_partial_finetune_config(
    *, path: Path, seed_override: int | None = None
) -> PartialFinetuneConfig:
    """Load and validate one JSON experiment configuration."""
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PartialFinetuneError(f"could not load configuration: {error}") from error
    return partial_finetune_config_from_mapping(
        _object_dict(raw, name="fine tuning configuration"),
        seed_override=seed_override,
    )


class PartialFinetuneModel(nn.Module):
    """DINOv2 with only its final block, final norm, and head trainable."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        embedding_dimension: int,
        class_count: int,
    ) -> None:
        """Attach a normalized affine classifier to a vision backbone."""
        super().__init__()
        if embedding_dimension < 1 or class_count < 2:
            raise PartialFinetuneError(
                "partial fine tuning requires a positive dimension and two classes"
            )
        self.backbone = backbone
        self.classifier = nn.Linear(embedding_dimension, class_count)
        self._final_block_name, self._final_norm_name = _freeze_except_final_block(
            backbone
        )

    @property
    def final_block_name(self) -> str:
        """Return the selected backbone block name."""
        return self._final_block_name

    @property
    def final_norm_name(self) -> str:
        """Return the selected final normalization layer name."""
        return self._final_norm_name

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return logits from normalized CLS embeddings."""
        raw_output: object = self.backbone(pixel_values=pixel_values)
        output = cast(BackboneOutput, raw_output)
        cls_embedding = output.last_hidden_state[:, 0, :]
        normalized = functional.normalize(cls_embedding, p=2, dim=1)
        return self.classifier(normalized)


def prepare_partial_finetune_data(
    *, examples: Sequence[EmbeddingExample]
) -> PartialFinetuneData:
    """Select only known training and validation images without reading test files."""
    train = tuple(
        example for example in examples if example.split is DatasetSplit.TRAIN
    )
    validation = tuple(
        example for example in examples if example.split is DatasetSplit.VALIDATION
    )
    if not train or not validation:
        raise PartialFinetuneError("both training and validation images are required")
    if any(
        not example.identity_id.startswith("usagi-")
        for example in (*train, *validation)
    ):
        raise PartialFinetuneError(
            "unknown identities cannot enter closed set partial fine tuning"
        )
    train_classes = {example.identity_id for example in train}
    validation_classes = {example.identity_id for example in validation}
    if train_classes != validation_classes:
        raise PartialFinetuneError(
            "training and validation identities must match exactly"
        )
    train_sessions = {example.capture_session_id for example in train}
    validation_sessions = {example.capture_session_id for example in validation}
    if train_sessions & validation_sessions:
        raise PartialFinetuneError("a capture session crosses training and validation")
    return PartialFinetuneData(
        train_examples=train,
        validation_examples=validation,
        class_ids=tuple(sorted(train_classes)),
    )


def _freeze_except_final_block(backbone: nn.Module) -> tuple[str, str]:
    """Freeze a DINOv2 backbone and expose only its final block and norm."""
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    encoder = getattr(backbone, "encoder", None)
    layers = getattr(encoder, "layer", None)
    final_norm = getattr(backbone, "layernorm", None)
    if not isinstance(layers, nn.ModuleList) or not layers:
        raise PartialFinetuneError("backbone does not expose DINOv2 encoder layers")
    if not isinstance(final_norm, nn.Module):
        raise PartialFinetuneError("backbone does not expose a final layer norm")
    final_block = layers[-1]
    for parameter in final_block.parameters():
        parameter.requires_grad = True
    for parameter in final_norm.parameters():
        parameter.requires_grad = True
    return f"encoder.layer.{len(layers) - 1}", "layernorm"


def _embedding_dimension(backbone: nn.Module) -> int:
    """Read and validate the pinned backbone hidden size."""
    configuration = getattr(backbone, "config", None)
    value = getattr(configuration, "hidden_size", None)
    if not isinstance(value, int) or value < 1:
        raise PartialFinetuneError("backbone does not expose a valid hidden size")
    return value


def _load_components(
    *, config: PartialFinetuneConfig
) -> tuple[ImageProcessor, nn.Module]:
    """Load the pinned processor and backbone through a typed boundary."""
    transformers = importlib.import_module("transformers")
    processor_raw: object = transformers.AutoImageProcessor.from_pretrained(
        config.model_id,
        revision=config.model_revision,
    )
    backbone_raw: object = transformers.AutoModel.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        use_safetensors=True,
    )
    if not isinstance(backbone_raw, nn.Module):
        raise PartialFinetuneError("loaded backbone is not a Torch module")
    return cast(ImageProcessor, processor_raw), backbone_raw


def _set_determinism(seed: int) -> None:
    """Seed supported generators and request deterministic operations."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def build_training_augmentation(config: AugmentationConfig) -> nn.Module | None:
    """Build conservative PIL augmentations without horizontal reflection."""
    if not config.enabled:
        return None
    return v2.Compose(
        [
            v2.RandomResizedCrop(
                size=(224, 224),
                scale=(config.crop_scale_minimum, 1.0),
                ratio=(config.crop_ratio_minimum, config.crop_ratio_maximum),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.RandomRotation(
                degrees=config.rotation_degrees,  # pyright: ignore[reportArgumentType]
                interpolation=InterpolationMode.BICUBIC,
                fill=128,
            ),
            v2.ColorJitter(
                brightness=config.brightness,
                contrast=config.contrast,
                saturation=config.saturation,
                hue=config.hue,
            ),
        ]
    )


def _load_images(
    *, examples: Sequence[EmbeddingExample], augmentation: nn.Module | None
) -> list[Image.Image]:
    """Load detached RGB images and optionally apply training augmentation."""
    images: list[Image.Image] = []
    try:
        for example in examples:
            with Image.open(example.image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            if augmentation is not None:
                augmented: object = augmentation(image)
                if not isinstance(augmented, Image.Image):
                    image.close()
                    raise PartialFinetuneError(
                        "augmentation did not return a PIL image"
                    )
                image.close()
                image = augmented
            images.append(image)
    except Exception:
        for image in images:
            image.close()
        raise
    return images


def _pixel_values(
    *,
    processor: ImageProcessor,
    examples: Sequence[EmbeddingExample],
    augmentation: nn.Module | None,
) -> torch.Tensor:
    """Load, transform, normalize, and batch prepared images."""
    images = _load_images(examples=examples, augmentation=augmentation)
    try:
        options: dict[str, object] = {}
        if augmentation is not None:
            options = {"do_resize": False, "do_center_crop": False}
        processed = processor(images=images, return_tensors="pt", **options)
    finally:
        for image in images:
            image.close()
    pixels = processed.get("pixel_values")
    if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4:
        raise PartialFinetuneError("image processor did not return batched pixels")
    return pixels


def _targets(
    *, examples: Sequence[EmbeddingExample], class_to_index: Mapping[str, int]
) -> torch.Tensor:
    """Convert stable identities to classifier indexes."""
    return torch.tensor(
        [class_to_index[example.identity_id] for example in examples],
        dtype=torch.long,
    )


def _metrics(
    *, logits: torch.Tensor, targets: torch.Tensor, class_count: int
) -> dict[str, float]:
    """Calculate balanced closed set metrics."""
    predicted = torch.argmax(logits, dim=1)
    correct = predicted == targets
    recalls: list[float] = []
    for class_index in range(class_count):
        mask = targets == class_index
        recalls.append(float(correct[mask].float().mean().item()))
    top_count = min(3, class_count)
    top_indices = torch.topk(logits, k=top_count, dim=1).indices
    top3 = torch.any(top_indices == targets.unsqueeze(1), dim=1)
    return {
        "top1_accuracy": float(correct.float().mean().item()),
        "top3_accuracy": float(top3.float().mean().item()),
        "macro_recall": float(np.mean(recalls)),
    }


def _batches(
    *, examples: Sequence[EmbeddingExample], batch_size: int, order: Sequence[int]
) -> Sequence[tuple[EmbeddingExample, ...]]:
    """Build ordered batches without multiprocessing or hidden worker state."""
    ordered = [examples[index] for index in order]
    return tuple(
        tuple(ordered[start : start + batch_size])
        for start in range(0, len(ordered), batch_size)
    )


def _evaluate(
    *,
    model: PartialFinetuneModel,
    processor: ImageProcessor,
    examples: Sequence[EmbeddingExample],
    class_to_index: Mapping[str, int],
    batch_size: int,
    device: torch.device,
    criterion: nn.CrossEntropyLoss,
) -> tuple[float, dict[str, float], torch.Tensor, torch.Tensor]:
    """Evaluate deterministic image preprocessing without gradients."""
    model.eval()
    logits_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in _batches(
            examples=examples,
            batch_size=batch_size,
            order=tuple(range(len(examples))),
        ):
            pixels = _pixel_values(
                processor=processor,
                examples=batch,
                augmentation=None,
            ).to(device)
            targets = _targets(examples=batch, class_to_index=class_to_index).to(device)
            logits_batches.append(model(pixels).cpu())
            target_batches.append(targets.cpu())
    logits = torch.cat(logits_batches)
    targets = torch.cat(target_batches)
    loss = float(criterion(logits, targets).item())
    return (
        loss,
        _metrics(logits=logits, targets=targets, class_count=len(class_to_index)),
        logits,
        targets,
    )


def _trainable_state(model: PartialFinetuneModel) -> dict[str, torch.Tensor]:
    """Copy only trained tensors to CPU for selection and publication."""
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }


def _load_trainable_state(
    *, model: PartialFinetuneModel, state: Mapping[str, torch.Tensor]
) -> None:
    """Load a complete trainable delta while preserving frozen base weights."""
    expected = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(state) != expected:
        raise PartialFinetuneError(
            "fine tuning checkpoint does not match the trainable parameter set"
        )
    current = model.state_dict()
    current.update(state)
    model.load_state_dict(current, strict=True)


def _sha256(path: Path) -> str:
    """Hash one artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    """Publish formatted JSON with an atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _run_id(
    *,
    config: PartialFinetuneConfig,
    dataset_fingerprint: str,
    initial_checkpoint_sha256: str | None,
) -> str:
    """Create a stable run identifier from all controlled inputs."""
    payload = {
        "configuration": config.as_dict(),
        "dataset_fingerprint": dataset_fingerprint,
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def _validation_report(
    *,
    data: PartialFinetuneData,
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss: float,
) -> dict[str, object]:
    """Build aggregate, per identity, and per image validation evidence."""
    probabilities = torch.softmax(logits, dim=1).numpy(force=True)
    target_values = targets.numpy(force=True)
    predictions = np.argmax(probabilities, axis=1)
    metrics = _metrics(
        logits=logits,
        targets=targets,
        class_count=len(data.class_ids),
    )
    per_identity: dict[str, dict[str, int | float]] = {}
    for class_index, class_id in enumerate(data.class_ids):
        mask = target_values == class_index
        count = int(np.count_nonzero(mask))
        correct = int(np.count_nonzero(predictions[mask] == target_values[mask]))
        per_identity[class_id] = {
            "correct": correct,
            "count": count,
            "recall": correct / count,
        }
    queries: list[dict[str, object]] = []
    for row, example in enumerate(data.validation_examples):
        order = np.argsort(-probabilities[row], kind="stable")
        top_index = int(order[0])
        runner_up_index = int(order[1])
        true_index = int(target_values[row])
        queries.append(
            {
                "image_id": example.image_id,
                "true_identity": data.class_ids[true_index],
                "predicted_identity": data.class_ids[top_index],
                "correct": top_index == true_index,
                "top_probability": float(probabilities[row, top_index]),
                "probability_margin": float(
                    probabilities[row, top_index] - probabilities[row, runner_up_index]
                ),
                "cross_entropy": float(
                    -math.log(
                        max(
                            float(probabilities[row, true_index]),
                            float(np.finfo(np.float32).tiny),
                        )
                    )
                ),
                "ranked_identities": [
                    {
                        "identity_id": data.class_ids[int(index)],
                        "probability": float(probabilities[row, index]),
                        "logit": float(logits[row, index].item()),
                    }
                    for index in order
                ],
            }
        )
    return {
        "schema_version": 1,
        "split": "validation",
        "query_count": len(data.validation_examples),
        "class_count": len(data.class_ids),
        "loss": loss,
        **metrics,
        "per_identity": per_identity,
        "queries": queries,
    }


def train_partial_finetune(
    *,
    examples: Sequence[EmbeddingExample],
    dataset_fingerprint: str,
    config: PartialFinetuneConfig,
    artifact_root: Path,
    linear_probe_checkpoint_path: Path | None = None,
    linear_probe_metadata_path: Path | None = None,
    processor: ImageProcessor | None = None,
    backbone: nn.Module | None = None,
) -> PartialFinetuneRun:
    """Train the final DINOv2 block and publish a verified model delta."""
    data = prepare_partial_finetune_data(examples=examples)
    _set_determinism(config.seed)
    device = select_torch_device(requested=config.device)
    if (processor is None) != (backbone is None):
        raise PartialFinetuneError("processor and backbone must be supplied together")
    if processor is None or backbone is None:
        processor, backbone = _load_components(config=config)
    model = PartialFinetuneModel(
        backbone=backbone,
        embedding_dimension=_embedding_dimension(backbone),
        class_count=len(data.class_ids),
    )

    initial_checkpoint_sha256: str | None = None
    if config.warm_start_linear_probe:
        if linear_probe_checkpoint_path is None or linear_probe_metadata_path is None:
            raise PartialFinetuneError("linear probe warm start artifacts are required")
        linear_probe, linear_metadata = load_linear_probe(
            checkpoint_path=linear_probe_checkpoint_path,
            metadata_path=linear_probe_metadata_path,
        )
        if linear_metadata.get("class_ids") != list(data.class_ids):
            raise PartialFinetuneError("linear probe class mapping does not match")
        if linear_probe.classifier.weight.shape != model.classifier.weight.shape:
            raise PartialFinetuneError("linear probe dimensions do not match DINOv2")
        model.classifier.load_state_dict(linear_probe.classifier.state_dict())
        initial_checkpoint_sha256 = _sha256(linear_probe_checkpoint_path)

    model = model.to(device)
    class_to_index = {class_id: index for index, class_id in enumerate(data.class_ids)}
    head_parameters = list(model.classifier.parameters())
    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("classifier.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": config.backbone_learning_rate},
            {"params": head_parameters, "lr": config.head_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
    )
    criterion = nn.CrossEntropyLoss()
    augmentation = build_training_augmentation(config.augmentation)
    amp_enabled = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    generator = torch.Generator(device="cpu")
    _ = generator.manual_seed(config.seed)

    history: list[dict[str, object]] = []
    best_state = _trainable_state(model)
    best_validation_loss = math.inf
    selected_epoch = 0
    epochs_without_improvement = 0
    training_started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        order_tensor = torch.randperm(len(data.train_examples), generator=generator)
        order = [
            int(order_tensor[position].item())
            for position in range(order_tensor.numel())
        ]
        gradient_norms: list[float] = []
        for batch in _batches(
            examples=data.train_examples,
            batch_size=config.batch_size,
            order=order,
        ):
            pixels = _pixel_values(
                processor=processor,
                examples=batch,
                augmentation=augmentation,
            ).to(device)
            targets = _targets(examples=batch, class_to_index=class_to_index).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss = criterion(model(pixels), targets)
            scaled_loss = cast(torch.Tensor, scaler.scale(loss))
            scaled_loss.backward()  # pyright: ignore[reportUnknownMemberType]
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            gradient_norms.append(float(norm.detach().cpu().item()))
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        train_loss, train_metrics, _, _ = _evaluate(
            model=model,
            processor=processor,
            examples=data.train_examples,
            class_to_index=class_to_index,
            batch_size=config.batch_size,
            device=device,
            criterion=criterion,
        )
        validation_loss, validation_metrics, _, _ = _evaluate(
            model=model,
            processor=processor,
            examples=data.validation_examples,
            class_to_index=class_to_index,
            batch_size=config.batch_size,
            device=device,
            criterion=criterion,
        )
        if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
            raise PartialFinetuneError("training produced a nonfinite loss")
        if not all(math.isfinite(value) for value in gradient_norms):
            raise PartialFinetuneError("training produced a nonfinite gradient norm")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "backbone_learning_rate": optimizer.param_groups[0]["lr"],
                "head_learning_rate": optimizer.param_groups[1]["lr"],
                "maximum_gradient_norm": max(gradient_norms),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
            }
        )
        print(
            f"Epoch {epoch:03d} | train loss {train_loss:.4f} | "
            f"validation loss {validation_loss:.4f} | "
            f"validation Top 1 {validation_metrics['top1_accuracy']:.3f}",
            flush=True,
        )
        if validation_loss < best_validation_loss - config.minimum_delta:
            best_validation_loss = validation_loss
            selected_epoch = epoch
            best_state = _trainable_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break
    training_seconds = time.perf_counter() - training_started

    _load_trainable_state(model=model, state=best_state)
    validation_loss, validation_metrics, logits, targets = _evaluate(
        model=model,
        processor=processor,
        examples=data.validation_examples,
        class_to_index=class_to_index,
        batch_size=config.batch_size,
        device=device,
        criterion=criterion,
    )
    report = _validation_report(
        data=data,
        logits=logits,
        targets=targets,
        loss=validation_loss,
    )

    run_id = _run_id(
        config=config,
        dataset_fingerprint=dataset_fingerprint,
        initial_checkpoint_sha256=initial_checkpoint_sha256,
    )
    output_directory = artifact_root / "models/partial_finetune" / run_id
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "trained_delta.safetensors"
    metadata_path = output_directory / "metadata.json"
    history_path = output_directory / "training_history.json"
    validation_report_path = output_directory / "validation_report.json"
    temporary_checkpoint = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    _save_safetensors(best_state, temporary_checkpoint)
    temporary_checkpoint.replace(checkpoint_path)
    _atomic_json(history_path, {"schema_version": 1, "epochs": history})
    _atomic_json(validation_report_path, report)
    trainable_names = sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "configuration": config.as_dict(),
        "device": str(device),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "dataset_fingerprint": dataset_fingerprint,
        "class_ids": list(data.class_ids),
        "class_to_index": class_to_index,
        "train_split": "train",
        "validation_split": "validation",
        "test_split_accessed": False,
        "train_image_count": len(data.train_examples),
        "validation_image_count": len(data.validation_examples),
        "embedding_dimension": _embedding_dimension(backbone),
        "final_block_name": model.final_block_name,
        "final_norm_name": model.final_norm_name,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "total_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "initial_linear_probe_sha256": initial_checkpoint_sha256,
        "loss": "cross_entropy",
        "optimizer": "adamw",
        "scheduler": "cosine_annealing",
        "mixed_precision_used": amp_enabled,
        "deterministic_algorithms": True,
        "training_seconds": training_seconds,
        "completed_epochs": len(history),
        "selected_epoch": selected_epoch,
        "selection_metric": "validation_loss",
        "checkpoint_format": "safetensors_trainable_delta",
        "checkpoint_sha256": _sha256(checkpoint_path),
        "validation_metrics": validation_metrics,
        "validation_loss": validation_loss,
    }
    _atomic_json(metadata_path, metadata)
    _atomic_json(
        artifact_root / "models/partial_finetune/latest.json",
        {"schema_version": 1, "run_id": run_id},
    )
    return PartialFinetuneRun(
        run_id=run_id,
        output_directory=output_directory,
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        history_path=history_path,
        validation_report_path=validation_report_path,
        selected_epoch=selected_epoch,
        validation_top1_accuracy=validation_metrics["top1_accuracy"],
        validation_top3_accuracy=validation_metrics["top3_accuracy"],
        validation_macro_recall=validation_metrics["macro_recall"],
    )


def load_partial_finetune(
    *, checkpoint_path: Path, metadata_path: Path, requested_device: str = "auto"
) -> tuple[PartialFinetuneModel, ImageProcessor, dict[str, object]]:
    """Verify a trained delta, rebuild its pinned base, and load it for inference."""
    try:
        raw: object = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PartialFinetuneError(
            f"could not load fine tuning metadata: {error}"
        ) from error
    metadata = _object_dict(raw, name="fine tuning metadata")
    if metadata.get("checkpoint_sha256") != _sha256(checkpoint_path):
        raise PartialFinetuneError(
            "fine tuning checkpoint hash does not match metadata"
        )
    configuration = _object_dict(
        metadata.get("configuration"), name="fine tuning configuration"
    )
    config = partial_finetune_config_from_mapping(configuration)
    class_ids_raw = metadata.get("class_ids")
    if not isinstance(class_ids_raw, list):
        raise PartialFinetuneError("fine tuning class mapping is invalid")
    class_id_values = cast(list[object], class_ids_raw)
    class_ids = [value for value in class_id_values if isinstance(value, str)]
    if len(class_ids) != len(class_id_values):
        raise PartialFinetuneError("fine tuning class mapping is invalid")
    load_config = replace(config, device=requested_device)
    processor, backbone = _load_components(config=load_config)
    model = PartialFinetuneModel(
        backbone=backbone,
        embedding_dimension=_embedding_dimension(backbone),
        class_count=len(class_ids),
    )
    try:
        state = _load_safetensors(checkpoint_path, "cpu")
    except Exception as error:
        raise PartialFinetuneError(
            f"could not load fine tuning checkpoint: {error}"
        ) from error
    _load_trainable_state(model=model, state=state)
    device = select_torch_device(requested=requested_device)
    return model.to(device).eval(), processor, metadata
