"""Deterministic linear probe training on frozen image embeddings."""

import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from usagi_visual_identifier.embeddings.artifacts import EmbeddingArtifact


class LinearProbeError(ValueError):
    """Raised when a linear probe run would be invalid or irreproducible."""


@dataclass(frozen=True)
class LinearProbeConfig:
    """Complete hyperparameter configuration for one deterministic run."""

    seed: int = 42
    epochs: int = 300
    batch_size: int = 16
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    patience: int = 40
    minimum_delta: float = 0.00001
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Reject values that make training undefined."""
        if self.seed < 0:
            raise LinearProbeError("seed cannot be negative")
        if self.epochs < 1:
            raise LinearProbeError("epochs must be positive")
        if self.batch_size < 1:
            raise LinearProbeError("batch size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise LinearProbeError("learning rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise LinearProbeError("weight decay must be finite and nonnegative")
        if self.patience < 1:
            raise LinearProbeError("patience must be positive")
        if not math.isfinite(self.minimum_delta) or self.minimum_delta < 0:
            raise LinearProbeError("minimum delta must be finite and nonnegative")
        if self.device not in {"cpu", "mps", "cuda"}:
            raise LinearProbeError("device must be cpu, mps, or cuda")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON ready configuration."""
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True)
class LinearProbeData:
    """Validated train and validation arrays with a stable label mapping."""

    train_embeddings: NDArray[np.float32]
    train_targets: NDArray[np.int64]
    train_image_ids: tuple[str, ...]
    validation_embeddings: NDArray[np.float32]
    validation_targets: NDArray[np.int64]
    validation_image_ids: tuple[str, ...]
    class_ids: tuple[str, ...]


@dataclass(frozen=True)
class LinearProbeRun:
    """Published artifact paths and selected metrics for one training run."""

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


class LinearProbe(nn.Module):
    """One affine classifier over frozen embedding coordinates."""

    def __init__(self, *, embedding_dimension: int, class_count: int) -> None:
        """Create the only trainable layer."""
        super().__init__()
        if embedding_dimension < 1 or class_count < 2:
            raise LinearProbeError(
                "linear probe requires a positive dimension and at least two classes"
            )
        self.classifier = nn.Linear(embedding_dimension, class_count)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return one unnormalized logit per identity."""
        return self.classifier(embeddings)


def prepare_linear_probe_data(*, artifact: EmbeddingArtifact) -> LinearProbeData:
    """Create strict Session 1 train and Session 2 validation matrices."""
    if "test" in artifact.splits:
        raise LinearProbeError(
            "linear probe training refuses artifacts containing sealed test embeddings"
        )
    unexpected = sorted(set(artifact.splits) - {"train", "validation"})
    if unexpected:
        raise LinearProbeError(f"unsupported embedding splits: {unexpected}")

    train_indices = [
        index for index, split in enumerate(artifact.splits) if split == "train"
    ]
    validation_indices = [
        index for index, split in enumerate(artifact.splits) if split == "validation"
    ]
    if not train_indices or not validation_indices:
        raise LinearProbeError("both train and validation embeddings are required")

    train_classes = {artifact.identity_ids[index] for index in train_indices}
    validation_classes = {artifact.identity_ids[index] for index in validation_indices}
    if train_classes != validation_classes:
        raise LinearProbeError(
            "train and validation identities must match exactly; "
            f"train only: {sorted(train_classes - validation_classes)}, "
            f"validation only: {sorted(validation_classes - train_classes)}"
        )
    class_ids = tuple(sorted(train_classes))
    class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}

    def targets(indices: list[int]) -> NDArray[np.int64]:
        return np.asarray(
            [class_to_index[artifact.identity_ids[index]] for index in indices],
            dtype=np.int64,
        )

    return LinearProbeData(
        train_embeddings=np.asarray(
            artifact.embeddings[train_indices], dtype=np.float32
        ),
        train_targets=targets(train_indices),
        train_image_ids=tuple(artifact.image_ids[index] for index in train_indices),
        validation_embeddings=np.asarray(
            artifact.embeddings[validation_indices], dtype=np.float32
        ),
        validation_targets=targets(validation_indices),
        validation_image_ids=tuple(
            artifact.image_ids[index] for index in validation_indices
        ),
        class_ids=class_ids,
    )


def _select_device(requested: str) -> torch.device:
    """Select only an explicitly configured and available training device."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise LinearProbeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise LinearProbeError("Apple Metal was requested but is unavailable")
    return torch.device(requested)


def _set_determinism(seed: int) -> None:
    """Seed supported generators and request deterministic Torch operations."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _metrics(
    *, logits: torch.Tensor, targets: torch.Tensor, class_count: int
) -> dict[str, float]:
    """Calculate balanced closed set classification metrics."""
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


def _float_tensor(values: NDArray[np.float32]) -> torch.Tensor:
    """Cross the NumPy to Torch boundary with an explicit float type."""
    return torch.from_numpy(values)  # pyright: ignore[reportUnknownMemberType]


def _target_tensor(values: NDArray[np.int64]) -> torch.Tensor:
    """Cross the NumPy to Torch boundary with an explicit target type."""
    return torch.from_numpy(values)  # pyright: ignore[reportUnknownMemberType]


def _evaluate(
    *,
    model: LinearProbe,
    features: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.CrossEntropyLoss,
    class_count: int,
) -> tuple[float, dict[str, float], torch.Tensor]:
    """Evaluate without gradients and return CPU logits."""
    model.eval()
    with torch.inference_mode():
        logits = model(features)
        loss = float(criterion(logits, targets).item())
    return (
        loss,
        _metrics(logits=logits, targets=targets, class_count=class_count),
        logits.cpu(),
    )


def _clone_state(model: LinearProbe) -> dict[str, torch.Tensor]:
    """Copy a model state onto CPU for stable checkpoint selection."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _sha256(path: Path) -> str:
    """Hash one published file."""
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


def _run_id(*, config: LinearProbeConfig, artifact: EmbeddingArtifact) -> str:
    """Create a stable identifier from configuration and embedding provenance."""
    payload = {
        "configuration": config.as_dict(),
        "embedding_artifact_sha256": artifact.metadata.get("artifact_sha256"),
        "dataset_fingerprint": artifact.metadata.get("dataset_fingerprint"),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def _validation_report(
    *,
    data: LinearProbeData,
    logits: torch.Tensor,
    loss: float,
) -> dict[str, object]:
    """Build aggregate and per query validation evidence."""
    probabilities = np.asarray(
        torch.softmax(logits, dim=1).detach().cpu().numpy(), dtype=np.float32
    )
    targets = data.validation_targets
    predictions = np.argmax(probabilities, axis=1)
    metrics = _metrics(
        logits=logits,
        targets=_target_tensor(targets),
        class_count=len(data.class_ids),
    )
    per_identity: dict[str, dict[str, int | float]] = {}
    for class_index, class_id in enumerate(data.class_ids):
        mask = targets == class_index
        count = int(np.count_nonzero(mask))
        correct = int(np.count_nonzero(predictions[mask] == targets[mask]))
        per_identity[class_id] = {
            "correct": correct,
            "count": count,
            "recall": correct / count,
        }

    queries: list[dict[str, object]] = []
    for row, image_id in enumerate(data.validation_image_ids):
        order = np.argsort(-probabilities[row], kind="stable")
        top_index = int(order[0])
        runner_up_index = int(order[1])
        true_index = int(targets[row])
        queries.append(
            {
                "image_id": image_id,
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
        "query_count": len(data.validation_image_ids),
        "class_count": len(data.class_ids),
        "loss": loss,
        **metrics,
        "per_identity": per_identity,
        "queries": queries,
    }


def train_linear_probe(
    *,
    artifact: EmbeddingArtifact,
    config: LinearProbeConfig,
    artifact_root: Path,
) -> LinearProbeRun:
    """Train, select, evaluate, and publish one deterministic linear probe."""
    data = prepare_linear_probe_data(artifact=artifact)
    device = _select_device(config.device)
    _set_determinism(config.seed)

    train_features = _float_tensor(data.train_embeddings).to(device)
    train_targets = _target_tensor(data.train_targets).to(device)
    validation_features = _float_tensor(data.validation_embeddings).to(device)
    validation_targets = _target_tensor(data.validation_targets).to(device)
    generator = torch.Generator(device="cpu")
    _ = generator.manual_seed(config.seed)
    dataset = TensorDataset(
        _float_tensor(data.train_embeddings),
        _target_tensor(data.train_targets),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )

    model = LinearProbe(
        embedding_dimension=data.train_embeddings.shape[1],
        class_count=len(data.class_ids),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, object]] = []
    best_state = _clone_state(model)
    best_validation_loss = math.inf
    selected_epoch = 0
    epochs_without_improvement = 0
    training_started = time.perf_counter()
    epoch_progress = tqdm(
        range(1, config.epochs + 1),
        desc="Linear probe",
        unit="epoch",
        dynamic_ncols=True,
        disable=None,
    )
    for epoch in epoch_progress:
        model.train()
        for batch_features, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features.to(device))
            loss = criterion(logits, batch_targets.to(device))
            loss.backward()
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]

        train_loss, train_metrics, _ = _evaluate(
            model=model,
            features=train_features,
            targets=train_targets,
            criterion=criterion,
            class_count=len(data.class_ids),
        )
        validation_loss, validation_metrics, _ = _evaluate(
            model=model,
            features=validation_features,
            targets=validation_targets,
            criterion=criterion,
            class_count=len(data.class_ids),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
            }
        )
        epoch_progress.set_postfix(  # pyright: ignore[reportUnknownMemberType]
            {
                "train_loss": f"{train_loss:.4f}",
                "validation_loss": f"{validation_loss:.4f}",
                "validation_top1": f"{validation_metrics['top1_accuracy']:.3f}",
            }
        )
        if validation_loss < best_validation_loss - config.minimum_delta:
            best_validation_loss = validation_loss
            selected_epoch = epoch
            best_state = _clone_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break
    training_seconds = time.perf_counter() - training_started

    model.load_state_dict(best_state)
    final_validation_loss, final_metrics, final_logits = _evaluate(
        model=model,
        features=validation_features,
        targets=validation_targets,
        criterion=criterion,
        class_count=len(data.class_ids),
    )
    report = _validation_report(
        data=data,
        logits=final_logits,
        loss=final_validation_loss,
    )

    run_id = _run_id(config=config, artifact=artifact)
    output_directory = artifact_root / "models/linear_probe" / run_id
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "checkpoint.pt"
    history_path = output_directory / "training_history.json"
    validation_report_path = output_directory / "validation_report.json"
    metadata_path = output_directory / "metadata.json"

    temporary_checkpoint = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    torch.save(best_state, temporary_checkpoint)
    temporary_checkpoint.replace(checkpoint_path)
    _atomic_json(history_path, {"schema_version": 1, "epochs": history})
    _atomic_json(validation_report_path, report)
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "configuration": config.as_dict(),
        "device": str(device),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "encoder_model_id": artifact.metadata.get("model_id"),
        "encoder_model_revision": artifact.metadata.get("model_revision"),
        "embedding_artifact_sha256": artifact.metadata.get("artifact_sha256"),
        "dataset_fingerprint": artifact.metadata.get("dataset_fingerprint"),
        "embedding_dimension": data.train_embeddings.shape[1],
        "class_ids": list(data.class_ids),
        "class_to_index": {
            class_id: index for index, class_id in enumerate(data.class_ids)
        },
        "train_split": "train",
        "validation_split": "validation",
        "test_split_accessed": False,
        "train_image_count": len(data.train_image_ids),
        "validation_image_count": len(data.validation_image_ids),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "architecture": "linear",
        "loss": "cross_entropy",
        "optimizer": "adamw",
        "deterministic_algorithms": True,
        "training_seconds": training_seconds,
        "completed_epochs": len(history),
        "selected_epoch": selected_epoch,
        "selection_metric": "validation_loss",
        "checkpoint_sha256": _sha256(checkpoint_path),
        "validation_metrics": final_metrics,
        "validation_loss": final_validation_loss,
    }
    _atomic_json(metadata_path, metadata)
    _atomic_json(
        artifact_root / "models/linear_probe/latest.json",
        {"schema_version": 1, "run_id": run_id},
    )
    return LinearProbeRun(
        run_id=run_id,
        output_directory=output_directory,
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        history_path=history_path,
        validation_report_path=validation_report_path,
        selected_epoch=selected_epoch,
        validation_top1_accuracy=final_metrics["top1_accuracy"],
        validation_top3_accuracy=final_metrics["top3_accuracy"],
        validation_macro_recall=final_metrics["macro_recall"],
    )


def load_linear_probe(
    *, checkpoint_path: Path, metadata_path: Path, device: str = "cpu"
) -> tuple[LinearProbe, dict[str, object]]:
    """Verify provenance and load a published linear probe for inference."""
    try:
        raw: object = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LinearProbeError(
            f"could not load linear probe metadata: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise LinearProbeError("linear probe metadata must be a JSON object")
    metadata = cast(dict[str, object], raw)
    if metadata.get("checkpoint_sha256") != _sha256(checkpoint_path):
        raise LinearProbeError("linear probe checkpoint hash does not match metadata")
    dimension = metadata.get("embedding_dimension")
    class_ids_raw = metadata.get("class_ids")
    if not isinstance(dimension, int) or not isinstance(class_ids_raw, list):
        raise LinearProbeError("linear probe metadata has invalid dimensions or labels")
    class_ids = cast(list[object], class_ids_raw)
    if not all(isinstance(value, str) for value in class_ids):
        raise LinearProbeError("linear probe metadata has invalid dimensions or labels")
    selected_device = _select_device(device)
    model = LinearProbe(
        embedding_dimension=dimension,
        class_count=len(class_ids),
    ).to(selected_device)
    try:
        state = cast(
            dict[str, torch.Tensor],
            torch.load(
                checkpoint_path,
                map_location=selected_device,
                weights_only=True,
            ),
        )
        model.load_state_dict(state)
    except (OSError, RuntimeError, ValueError) as error:
        raise LinearProbeError(
            f"could not load linear probe checkpoint: {error}"
        ) from error
    model.eval()
    return model, metadata
