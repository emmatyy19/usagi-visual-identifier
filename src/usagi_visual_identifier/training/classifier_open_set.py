"""Validation only unknown rejection using a fine tuned classifier head."""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import torch
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from usagi_visual_identifier.embeddings.artifacts import (
    EmbeddingError,
    EmbeddingExample,
)
from usagi_visual_identifier.training.partial_finetune import PartialFinetuneModel


class ImageProcessor(Protocol):
    """Processor boundary needed for classifier inference."""

    def __call__(
        self, *, images: Sequence[Image.Image], return_tensors: str
    ) -> Mapping[str, torch.Tensor]:
        """Return tensor inputs for a batch of images."""
        ...


class ClassifierConfidence(StrEnum):
    """Confidence scores derived from classifier probabilities."""

    MAX_PROBABILITY = "max_probability"
    PROBABILITY_MARGIN = "probability_margin"


@dataclass(frozen=True)
class ClassifierQuery:
    """One validation prediction from the classifier head."""

    image_id: str
    true_identity: str
    predicted_identity: str
    confidence: float
    is_known: bool
    is_correct: bool
    accepted: bool = False

    def as_dict(self) -> dict[str, object]:
        """Return a JSON ready query."""
        return {
            "image_id": self.image_id,
            "true_identity": self.true_identity,
            "predicted_identity": self.predicted_identity,
            "confidence": self.confidence,
            "is_known": self.is_known,
            "is_correct": self.is_correct,
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class ClassifierThreshold:
    """Classifier behavior at one threshold."""

    threshold: float
    known_correct_accepted: int
    known_count: int
    unknown_accepted: int
    unknown_count: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON ready operating point."""
        return {
            "threshold": self.threshold,
            "known_correct_accepted": self.known_correct_accepted,
            "known_count": self.known_count,
            "known_correct_acceptance_rate": (
                self.known_correct_accepted / self.known_count
            ),
            "unknown_accepted": self.unknown_accepted,
            "unknown_count": self.unknown_count,
            "unknown_false_acceptance_rate": (
                self.unknown_accepted / self.unknown_count
            ),
        }


@dataclass(frozen=True)
class ClassifierOpenSetReport:
    """Selected classifier confidence threshold and validation predictions."""

    method: ClassifierConfidence
    selected: ClassifierThreshold
    queries: tuple[ClassifierQuery, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON ready report."""
        return {
            "schema_version": 1,
            "selection_data": "validation_only",
            "confidence_method": self.method.value,
            "selected": self.selected.as_dict(),
            "queries": [query.as_dict() for query in self.queries],
        }


def _threshold(
    *, queries: tuple[ClassifierQuery, ...], minimum_known_correct_acceptance: float
) -> tuple[ClassifierThreshold, tuple[ClassifierQuery, ...]]:
    """Select the safest threshold that retains enough correct known photos."""
    known = tuple(query for query in queries if query.is_known)
    unknown = tuple(query for query in queries if not query.is_known)
    if not known or not unknown:
        raise EmbeddingError("classifier evaluation requires known and unknown queries")
    points: list[ClassifierThreshold] = []
    for threshold in sorted({query.confidence for query in queries}, reverse=True):
        points.append(
            ClassifierThreshold(
                threshold=threshold,
                known_correct_accepted=sum(
                    query.is_correct and query.confidence >= threshold
                    for query in known
                ),
                known_count=len(known),
                unknown_accepted=sum(
                    query.confidence >= threshold for query in unknown
                ),
                unknown_count=len(unknown),
            )
        )
    required = math.ceil(minimum_known_correct_acceptance * len(known))
    eligible = tuple(
        point for point in points if point.known_correct_accepted >= required
    )
    if not eligible:
        raise EmbeddingError(
            "classifier cannot retain the required correct known photos"
        )
    selected = min(
        eligible,
        key=lambda point: (
            point.unknown_accepted,
            -point.known_correct_accepted,
            -point.threshold,
        ),
    )
    accepted = tuple(
        replace(query, accepted=query.confidence >= selected.threshold)
        for query in queries
    )
    return selected, accepted


def evaluate_classifier_open_set(
    *,
    model: PartialFinetuneModel,
    processor: ImageProcessor,
    examples: Sequence[EmbeddingExample],
    class_ids: tuple[str, ...],
    method: ClassifierConfidence,
    minimum_known_correct_acceptance: float = 0.95,
    batch_size: int = 8,
) -> ClassifierOpenSetReport:
    """Evaluate classifier confidence on known and unknown validation photos."""
    if any(example.split.value != "validation" for example in examples):
        raise EmbeddingError("classifier threshold selection is validation only")
    if batch_size < 1:
        raise EmbeddingError("batch size must be positive")
    device = next(model.parameters()).device
    known_ids = frozenset(class_ids)
    queries: list[ClassifierQuery] = []
    for start in tqdm(
        range(0, len(examples), batch_size),
        desc="Classifier validation",
        unit="batch",
    ):
        batch = examples[start : start + batch_size]
        images: list[Image.Image] = []
        try:
            for example in batch:
                with Image.open(example.image_path) as opened:
                    images.append(ImageOps.exif_transpose(opened).convert("RGB"))
            processed = processor(images=images, return_tensors="pt")
        finally:
            for image in images:
                image.close()
        pixels = processed.get("pixel_values")
        if not isinstance(pixels, torch.Tensor):
            raise EmbeddingError("image processor did not return pixel values")
        with torch.inference_mode():
            probabilities = torch.softmax(model(pixels.to(device)), dim=1).cpu()
        top = torch.topk(probabilities, k=2, dim=1)
        for offset, example in enumerate(batch):
            predicted_index = int(top.indices[offset, 0].item())
            confidence = float(top.values[offset, 0].item())
            if method is ClassifierConfidence.PROBABILITY_MARGIN:
                confidence -= float(top.values[offset, 1].item())
            predicted = class_ids[predicted_index]
            is_known = example.identity_id in known_ids
            queries.append(
                ClassifierQuery(
                    image_id=example.image_id,
                    true_identity=example.identity_id,
                    predicted_identity=predicted,
                    confidence=confidence,
                    is_known=is_known,
                    is_correct=is_known and predicted == example.identity_id,
                )
            )
    selected, accepted = _threshold(
        queries=tuple(queries),
        minimum_known_correct_acceptance=minimum_known_correct_acceptance,
    )
    return ClassifierOpenSetReport(
        method=method,
        selected=selected,
        queries=accepted,
    )


def write_classifier_open_set_reports(
    *, path: Path, reports: tuple[ClassifierOpenSetReport, ...]
) -> None:
    """Atomically write classifier validation reports."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    value = {
        "schema_version": 1,
        "reports": [report.as_dict() for report in reports],
    }
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)
