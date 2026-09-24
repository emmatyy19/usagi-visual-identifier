# Usagi Visual Catalog

A learning-focused computer-vision project for recognizing specific Chiikawa
Usagi merchandise variants from photographs and connecting confident matches
to useful catalog information.

## Motivation

I originally wanted a small computer-vision project because my master's
schedule will not leave room for a formal computer-vision course. I still want
the hands-on part: collecting imperfect real data, choosing an honest
evaluation, adapting a pretrained model, and learning from the mistakes it
makes.

The idea became much more personal when I looked at my Chiikawa Usagi dolls. I
am a fan, although very far from having a complete collection—some collectors
have an entire wall of them. I have a handful that I especially like and was
lucky enough to get, while many more variants exist than I could name or own.
That made me curious whether a photograph could identify exactly which doll it
is and eventually connect it to its release, availability, and market history.

Many Usagi dolls share the same face and body shape. The distinguishing evidence
is often a small costume, hat, prop, embroidery detail, size difference, or
release-specific accessory. That makes the collection a compact but meaningful
fine-grained visual-recognition problem—and gives the technical work a reason I
will still care about after the first model runs.

This project is an alternative path toward hands-on computer-vision experience:
build an honest baseline, collect a small real dataset, fine-tune a pretrained
vision model on consumer hardware, analyze where it fails, and turn the result
into a useful personal catalog.

## Initial problem

Given a photograph containing one cropped doll, return:

1. the most visually similar catalog entries;
2. calibrated evidence indicating whether the best result is trustworthy; and
3. an explicit `unknown` result when the pictured variant is absent from the
   catalog.

The first milestone is deliberately limited to individually photographed dolls
from the owner's collection. Detecting multiple dolls in a shelf photograph,
identifying unowned products, and tracking market listings are later phases.

## Why retrieval rather than only classification?

The catalog will grow. A closed classifier can perform well on known classes,
but usually needs retraining when a new variant is introduced. The initial
system instead represents each photograph with a pretrained visual embedding
and retrieves nearby reference images. This provides a useful no-training
baseline and supports adding reference entries incrementally.

Fine-tuning remains a central learning objective. The planned experiments move
from frozen embeddings, to a linear classifier, to partial backbone fine-tuning,
and finally to supervised metric learning. Each stage must earn its complexity
through held-out evaluation.

## Planned system boundary

```text
single-doll photo
       |
       v
preprocessing and crop validation
       |
       v
vision encoder -> normalized embedding -> nearest catalog references
                                             |
                               confidence / unknown decision
                                             |
                                             v
                                  local catalog metadata
```

Commerce information is intentionally outside the visual model. A successful
identity becomes a stable catalog key that can later join to release details,
official availability, and time-stamped marketplace observations.

## Learning questions

- How well do general-purpose visual embeddings separate nearly identical
  merchandise variants without training?
- Which augmentations improve robustness without erasing identity-defining
  details?
- Does a linear probe outperform nearest-neighbor retrieval?
- How much does partial fine-tuning improve held-out capture sessions?
- Can metric learning improve retrieval while still allowing new catalog
  entries to be added without retraining a classifier?
- How reliably can the system reject a variant it has never seen?
- Which visual details cause the most common confusions?

## Documentation

- [Roadmap](docs/roadmap.md) tracks milestone-level progress.
- [Visual identification design](docs/design/001-visual-identification.md)
  defines the initial technical and experimental contract.
- [Dataset protocol](docs/data/dataset-protocol.md) defines capture, labeling,
  splitting, privacy, and provenance rules.
- [Photo capture guide](docs/data/photo-capture-guide.md) gives the exact first
  photography workflow, shot list, and folder layout.
- [Evaluation plan](docs/evaluation.md) defines metrics, slices, comparisons,
  and promotion gates.
- [Experiment records](docs/experiments/README.md) will document results and
  design changes as the project develops.

## Project status

Foundation. The problem, data protocol, evaluation rules, and roadmap are
defined. The repository now has a reproducible Python environment and quality
tooling; dataset inventory and capture are next.

## Development setup

The project uses Python 3.12 and [`uv`](https://docs.astral.sh/uv/) for Python,
dependency, environment, and package management.

Create the locked environment, including development tools:

```bash
uv sync --all-groups
```

Install the Git hooks:

```bash
uv run pre-commit install
```

Confirm that PyTorch can see the intended accelerator:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

`torch.cuda.is_available()` should be `True` on the RTX 3060 machine. A false
result does not mean the project is broken: CPU and Apple Metal environments
remain useful for development, but the NVIDIA machine may need a PyTorch build
matched to its driver. Record that machine's operating system, driver, GPU
memory, and PyTorch CUDA version before changing the locked source configuration.

The runtime stack includes PyTorch and torchvision, Hugging Face Transformers,
Pillow and ImageHash, NumPy and pandas, scikit-learn, PyTorch Metric Learning,
Matplotlib, Pydantic, Typer, and Rich. The development group adds JupyterLab,
Ruff, strict Pyright, pytest with coverage, and pre-commit.

## Quality checks

Run the same checks used by continuous integration:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
uv build
```

## Privacy and rights

Personal photographs, collection inventory, purchase details, credentials,
downloaded product images, generated embeddings, and model artifacts remain
local and are excluded from Git. Public examples will be added only when their
redistribution rights and identifying content have been reviewed.
