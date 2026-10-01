"""One time closed set classifier evaluation on the sealed test session."""

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from usagi_visual_identifier.data.models import DatasetSplit
from usagi_visual_identifier.embeddings.artifacts import (
    EmbeddingError,
    EmbeddingExample,
)
from usagi_visual_identifier.training.partial_finetune import PartialFinetuneModel


class ImageProcessor(Protocol):
    """Processor boundary needed for final classifier inference."""

    def __call__(
        self, *, images: Sequence[Image.Image], return_tensors: str
    ) -> Mapping[str, torch.Tensor]:
        """Return tensor inputs for a batch of images."""
        ...


@dataclass(frozen=True)
class ClassifierTestReport:
    """Frozen classifier metrics on the known final test session."""

    query_count: int
    top1_accuracy: float
    top3_accuracy: float
    macro_recall: float
    probability_margin_threshold: float
    accepted_count: int
    correct_accepted_count: int
    queries: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON ready final test report."""
        return {
            "schema_version": 1,
            "evaluation_role": "final_known_test",
            "selection_allowed": False,
            "query_count": self.query_count,
            "top1_accuracy": self.top1_accuracy,
            "top3_accuracy": self.top3_accuracy,
            "macro_recall": self.macro_recall,
            "probability_margin_threshold": self.probability_margin_threshold,
            "accepted_count": self.accepted_count,
            "known_acceptance_rate": self.accepted_count / self.query_count,
            "correct_accepted_count": self.correct_accepted_count,
            "known_correct_acceptance_rate": (
                self.correct_accepted_count / self.query_count
            ),
            "queries": list(self.queries),
        }


def evaluate_classifier_test(
    *,
    model: PartialFinetuneModel,
    processor: ImageProcessor,
    examples: Sequence[EmbeddingExample],
    class_ids: tuple[str, ...],
    probability_margin_threshold: float,
    batch_size: int = 8,
) -> ClassifierTestReport:
    """Evaluate a frozen classifier and threshold on Session 3 exactly once."""
    if not examples or any(
        example.split is not DatasetSplit.TEST for example in examples
    ):
        raise EmbeddingError("final classifier evaluation requires only test examples")
    if batch_size < 1:
        raise EmbeddingError("batch size must be positive")
    if not 0 <= probability_margin_threshold <= 1:
        raise EmbeddingError("probability margin threshold must be from zero to one")

    device = next(model.parameters()).device
    class_to_index = {identity_id: index for index, identity_id in enumerate(class_ids)}
    correct_by_identity: dict[str, list[bool]] = defaultdict(list)
    query_rows: list[dict[str, object]] = []
    top1_correct = 0
    top3_correct = 0
    accepted_count = 0
    correct_accepted_count = 0

    for start in tqdm(
        range(0, len(examples), batch_size),
        desc="Final classifier test",
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
        top_count = min(3, len(class_ids))
        top = torch.topk(probabilities, k=top_count, dim=1)
        for offset, example in enumerate(batch):
            true_index = class_to_index.get(example.identity_id)
            if true_index is None:
                raise EmbeddingError(
                    f"test identity is absent from classifier: {example.identity_id}"
                )
            ranked_indices = [
                cast(int, top.indices[offset, rank].item()) for rank in range(top_count)
            ]
            predicted_index = ranked_indices[0]
            is_correct = predicted_index == true_index
            in_top3 = true_index in ranked_indices
            margin = float(top.values[offset, 0].item() - top.values[offset, 1].item())
            accepted = margin >= probability_margin_threshold
            top1_correct += int(is_correct)
            top3_correct += int(in_top3)
            accepted_count += int(accepted)
            correct_accepted_count += int(accepted and is_correct)
            correct_by_identity[example.identity_id].append(is_correct)
            query_rows.append(
                {
                    "image_id": example.image_id,
                    "true_identity": example.identity_id,
                    "predicted_identity": class_ids[predicted_index],
                    "top3_identities": [class_ids[index] for index in ranked_indices],
                    "probability_margin": margin,
                    "accepted": accepted,
                    "is_correct": is_correct,
                }
            )
    query_count = len(examples)
    recalls = [sum(values) / len(values) for values in correct_by_identity.values()]
    return ClassifierTestReport(
        query_count=query_count,
        top1_accuracy=top1_correct / query_count,
        top3_accuracy=top3_correct / query_count,
        macro_recall=float(np.mean(recalls)),
        probability_margin_threshold=probability_margin_threshold,
        accepted_count=accepted_count,
        correct_accepted_count=correct_accepted_count,
        queries=tuple(query_rows),
    )


def write_classifier_test_report(*, path: Path, report: ClassifierTestReport) -> None:
    """Atomically write the private final classifier report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
