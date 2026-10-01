"""Tests for fine tuned classifier unknown rejection."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import torch
from PIL import Image
from torch import nn

from usagi_visual_identifier.data.models import DatasetSplit
from usagi_visual_identifier.embeddings.artifacts import EmbeddingExample
from usagi_visual_identifier.training.classifier_open_set import (
    ClassifierConfidence,
    evaluate_classifier_open_set,
)
from usagi_visual_identifier.training.classifier_test import evaluate_classifier_test
from usagi_visual_identifier.training.partial_finetune import PartialFinetuneModel


class _Processor:
    """Turn each image's red channel into deterministic test logits."""

    def __call__(
        self, *, images: Sequence[Image.Image], return_tensors: str
    ) -> Mapping[str, torch.Tensor]:
        assert return_tensors == "pt"
        rows: list[list[float]] = []
        for image in images:
            pixel = cast(tuple[int, int, int], image.getpixel((0, 0)))
            red = pixel[0]
            rows.append([red / 255, 1 - red / 255])
        return {"pixel_values": torch.tensor(rows, dtype=torch.float32)}


class _Model(nn.Module):
    """Pass the processor's invented logits through unchanged."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return pixel_values + self.anchor * 0


def _example(
    tmp_path: Path,
    number: int,
    identity: str,
    red: int,
    split: DatasetSplit = DatasetSplit.VALIDATION,
) -> EmbeddingExample:
    """Create one validation image and aligned example."""
    path = tmp_path / f"{number}.jpg"
    Image.new("RGB", (2, 2), (red, 0, 0)).save(path)
    return EmbeddingExample(
        image_id=f"img-{number:06d}",
        identity_id=identity,
        capture_session_id="session-validation",
        split=split,
        processed_relative_path=path.name,
        image_path=path,
    )


def test_classifier_threshold_retains_known_and_rejects_unknown(tmp_path: Path) -> None:
    """Classifier probability thresholds use known and unknown validation rows."""
    examples = (
        _example(tmp_path, 1, "usagi-0001", 250),
        _example(tmp_path, 2, "usagi-0002", 5),
        _example(tmp_path, 3, "unknown-0001", 128),
        _example(tmp_path, 4, "unknown-0002", 130),
    )
    report = evaluate_classifier_open_set(
        model=cast(PartialFinetuneModel, _Model()),
        processor=_Processor(),
        examples=examples,
        class_ids=("usagi-0001", "usagi-0002"),
        method=ClassifierConfidence.MAX_PROBABILITY,
        minimum_known_correct_acceptance=1.0,
        batch_size=2,
    )
    assert report.selected.known_correct_accepted == 2
    assert report.selected.unknown_accepted == 0
    assert all(query.accepted for query in report.queries if query.is_known)


def test_classifier_final_test_applies_frozen_margin(tmp_path: Path) -> None:
    """The final evaluator reports ranking and applies an unchanged threshold."""
    examples = (
        _example(tmp_path, 1, "usagi-0001", 250, DatasetSplit.TEST),
        _example(tmp_path, 2, "usagi-0002", 5, DatasetSplit.TEST),
    )
    report = evaluate_classifier_test(
        model=cast(PartialFinetuneModel, _Model()),
        processor=_Processor(),
        examples=examples,
        class_ids=("usagi-0001", "usagi-0002"),
        probability_margin_threshold=0.4,
        batch_size=2,
    )
    assert report.top1_accuracy == 1.0
    assert report.top3_accuracy == 1.0
    assert report.accepted_count == 2
    assert report.correct_accepted_count == 2
