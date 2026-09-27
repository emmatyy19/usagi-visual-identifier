# Usagi Visual Identifier

I am building a computer vision system that can look at a photo of a Chiikawa
Usagi doll and identify the specific merchandise variant. Once the visual part
works, I want to connect each result to useful information such as the release
name, original price, current availability, and recent market prices.

## Why I am building this

I wanted a small computer vision project because my master's schedule will not
leave room for a formal computer vision course. I still want the practical
experience of collecting real data, defining a fair evaluation, adapting a
pretrained model, and understanding why it makes mistakes.

The idea became more personal when I looked at my Usagi dolls. I am a fan, but
I am very far from having a complete collection. Some collectors have an entire
wall of them. I have a handful that I especially like and was lucky enough to
get. There are also many more variants that I do not own and cannot identify by
name yet.

That made me wonder whether I could take a photo of a doll and find out exactly
which one it is. Many Usagi dolls share the same face and body shape, so the
answer often depends on a small costume, hat, bow, prop, embroidery detail, or
size difference. This makes the collection a fun and meaningful vision problem
that I genuinely care about.

## What I want the first version to do

The first version will receive one photo containing one doll. It will return:

1. The three most visually similar dolls in the known collection.
2. A score that helps explain how confident the match is.
3. An `unknown` result when the doll does not match anything in the collection.

I will start with dolls that I own and can photograph myself. Group photos,
unowned variants, product information, and market tracking will come later.

## Why I am starting with image retrieval

The collection can keep growing. A classifier works well when every possible
class is known ahead of time, but it normally needs more training when a new
doll is added.

The first system will use a pretrained vision model to turn each photo into a
numeric representation called an embedding. It will compare a new photo with
stored reference photos and return the closest matches. This gives me a useful
baseline before any training and makes it easier to add another doll later.

I still want hands on experience with training. I plan to compare frozen
embeddings, a linear classifier, partial model tuning, and metric learning. I
will only keep the extra complexity when the evaluation shows that it helps.

## Planned flow

```text
photo
  ↓
image validation and preprocessing
  ↓
vision model
  ↓
normalized embedding
  ↓
similarity search against reference photos
  ↓
confidence check
  ↓
known identity or unknown result
  ↓
local product information
```

The vision model will only decide which doll is pictured. Product information
and changing market data will remain separate so they can be updated without
changing the model.

## What I want to learn

1. How well can a general vision model distinguish nearly identical dolls
   without training?
2. Which image changes make the system more reliable without hiding the small
   details that identify a doll?
3. Does a simple linear classifier improve on image retrieval?
4. How much does partial model tuning help on photos from a different session?
5. Can metric learning improve retrieval while still letting me add new dolls?
6. How reliably can the system reject a doll it has never seen?
7. Which visual details cause the most common mistakes?

## Documentation

1. The [roadmap](docs/roadmap.md) tracks the planned milestones.
2. The [visual identification design](docs/design/001-visual-identification.md)
   explains the technical plan and experiment rules.
3. The [dataset protocol](docs/data/dataset-protocol.md) explains labeling,
   data splits, privacy, and provenance.
4. The [photo capture guide](docs/data/photo-capture-guide.md) gives the exact
   first photo workflow, shot list, and folder structure.
5. The [evaluation plan](docs/evaluation.md) defines the metrics and comparison
   rules.
6. The [experiment records](docs/experiments/README.md) will document what I
   tried, what happened, and what changed afterward.
7. The [data preparation guide](docs/data/data-preparation.md) explains how I
   turn private phone photos into validated model inputs.
8. The [frozen DINOv2 baseline](docs/baselines/frozen-dinov2.md) explains the
   first embedding and retrieval experiment.
9. The [embedding walkthrough](notebooks/embedding_baseline_walkthrough.ipynb)
   provides an executed visual showcase of the baseline results and all 98
   Session 1 photographs.

## Current status

The planning and repository setup are complete. The first capture session has
been collected and prepared, and the frozen DINOv2 embedding baseline can run
locally on Apple Metal, NVIDIA CUDA, or CPU. The Session 1 result is only a
pipeline sanity check. The next meaningful measurement requires an independent
Session 2, while Session 3 will remain sealed for final testing.

After those sessions are ready, I will compare nearest image retrieval,
identity prototypes, top k identity means, and similarity weighted neighbor
voting without changing the encoder. Then I will train a linear classification
head while DINOv2 stays frozen. Partial tuning of the final transformer block
comes after that. Metric learning and patch level matching will only be added
when the earlier results give a clear reason to try them.

## Development setup

This project uses Python 3.12 and [`uv`](https://docs.astral.sh/uv/) to manage
Python, dependencies, the virtual environment, and the package build.

Create the locked environment:

```bash
uv sync --all-groups
```

Install the Git hooks:

```bash
uv run pre-commit install
```

Check whether PyTorch can use the available accelerator:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'No CUDA')"
```

The GPU environment has been tested on an NVIDIA GeForce RTX 3060 with 12 GB of
memory. PyTorch 2.14.0 with CUDA 13.0 detects the GPU correctly. The project can
also run on a CPU or Apple Metal for development and smaller tests.

The main libraries are PyTorch, torchvision, Transformers, PyTorch Metric
Learning, Pillow, ImageHash, NumPy, pandas, `scikit-learn`, Matplotlib, Pydantic,
Typer, and Rich. Development tools include JupyterLab, Ruff, Pyright, pytest,
coverage, and `pre-commit`.

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

The showcase notebook includes reviewed, resized renderings of the doll photos
I took for Session 1. I am comfortable sharing those images as part of this
project, and the prepared copies do not include the original photo metadata.

The original HEIC files, standalone prepared dataset, collection inventory,
purchase details, credentials, downloaded third party images, generated
embeddings, and model files stay local and are excluded from Git.

This is an independent educational fan project and is not affiliated with
Nagano, Chiikawa, or their official rights holders. The photographs are © 2026
Emma Tong. All rights reserved.

## License

The source code and written documentation are available under the
[MIT License](LICENSE). The license does not apply to the photographs embedded
in the showcase notebook. Those photographs remain © 2026 Emma Tong. All
rights reserved. Chiikawa characters and merchandise belong to their respective
rights holders.
