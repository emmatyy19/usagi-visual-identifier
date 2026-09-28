"""Render conservative training augmentations for human review."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps

from usagi_visual_identifier.embeddings import load_prepared_examples
from usagi_visual_identifier.training import (
    build_training_augmentation,
    load_partial_finetune_config,
    prepare_partial_finetune_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Save one original and four variants for every known identity."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/partial_finetune/conservative.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/augmentation_review/conservative.png"),
    )
    arguments = parser.parse_args()
    config = load_partial_finetune_config(path=PROJECT_ROOT / arguments.config)
    transform = build_training_augmentation(config.augmentation)
    if transform is None:
        raise ValueError("the selected configuration has no augmentation")
    examples, _ = load_prepared_examples(
        private_root=PROJECT_ROOT / "data/private",
        processed_root=PROJECT_ROOT / "data/processed",
    )
    data = prepare_partial_finetune_data(examples=examples)
    representatives = [
        next(
            example
            for example in data.train_examples
            if example.identity_id == class_id
        )
        for class_id in data.class_ids
    ]
    _ = torch.manual_seed(config.seed)  # pyright: ignore[reportUnknownMemberType]
    figure, axes = plt.subplots(
        len(representatives),
        5,
        figsize=(12, 2.5 * len(representatives)),
    )
    for row, example in enumerate(representatives):
        with Image.open(example.image_path) as opened:
            original = ImageOps.exif_transpose(opened).convert("RGB")
        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"{example.identity_id}\noriginal")
        for column in range(1, 5):
            augmented = transform(original)
            if not isinstance(augmented, Image.Image):
                original.close()
                raise TypeError("augmentation did not return a PIL image")
            axes[row, column].imshow(augmented)
            axes[row, column].set_title(f"variant {column}")
            augmented.close()
        original.close()
        for axis in axes[row]:
            axis.axis("off")
    figure.suptitle(  # pyright: ignore[reportUnknownMemberType]
        "Conservative identity preserving augmentation review"
    )
    figure.tight_layout()
    output = (PROJECT_ROOT / arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    figure.savefig(  # pyright: ignore[reportUnknownMemberType]
        temporary, format="png", dpi=140, bbox_inches="tight"
    )
    temporary.replace(output)
    plt.close(figure)
    print(f"Augmentation review: {output}")


if __name__ == "__main__":
    main()
