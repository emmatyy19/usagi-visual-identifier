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
confident match, possible match, or no confident local match
  ↓
local product information
  ↓ when I choose to continue
optional web assisted discovery with source links
  ↓
manual confirmation before saving anything
```

The vision model will only decide which doll is pictured. Product information
and changing market data will remain separate so they can be updated without
changing the model.

The catalog is intentionally personal rather than complete. It can contain
dolls I own, dolls I know exist, dolls I want, and dolls I do not want. A web
lookup can help me investigate an unfamiliar doll, but it will return suggestions
and sources rather than silently creating a trusted catalog record.

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
   provides an executed visual showcase of the baseline results, all 98
   Session 1 photographs, and all 42 Session 2 validation photographs.
10. The [aggregation comparison](docs/baselines/aggregation-comparison.md)
    explains the four frozen retrieval rules and how to compare them fairly.
11. The [single image inference guide](docs/baselines/single-image-inference.md)
    explains how to predict one photo and measure local latency in the debugger.
12. The [linear probe guide](docs/training/linear-probe.md) explains the first
    supervised training workflow and its reproducible artifacts.
13. The [linear probe walkthrough](notebooks/linear_probe_walkthrough.ipynb)
    explains the reusable training process, logits, softmax, cross entropy,
    learning curves, validation results, confidence, and learned weights.
14. The [partial fine tuning guide](docs/training/partial-finetuning.md) explains
    the protected data transfer, GPU readiness check, training matrix, saved
    model delta, and result transfer workflow.
15. The [partial fine tuning walkthrough](notebooks/partial_finetune_walkthrough.ipynb)
    compares all six RTX 3060 runs, learning curves, paired seed results,
    confidence margins, difficult views, training cost, and the augmentation
    decision.
16. The [unknown rejection walkthrough](notebooks/unknown_rejection_walkthrough.ipynb)
    shows the known and unknown score distributions, selected threshold,
    identity level results, false accepts, and known rejections.
17. The [open set model selection record](docs/experiments/005-unknown-rejection-validation.md)
    compares four embedding confidence signals and the selected partial fine
    tuning checkpoint without opening Session 3.
18. The [open set model selection walkthrough](notebooks/open_set_model_selection_walkthrough.ipynb)
    visualizes the frozen rule, checkpoint choice, tuned embedding results, and
    classifier confidence results.
19. The [final known test](docs/experiments/005b-final-known-test.md) records the
    one time Session 3 results and the decision to keep unknown rejection as a
    validation result.
20. The [personal catalog and discovery design](docs/design/002-personal-catalog-and-discovery.md)
    defines the local catalog boundary, uncertainty states, optional web lookup,
    manual verification, and secrets policy.
21. The [complete project journey](notebooks/project_journey_walkthrough.ipynb)
    summarizes the dataset, frozen retrieval, linear probe, partial fine tuning,
    unknown validation, final test, lessons, and personal catalog direction.
22. The [personal catalog application guide](docs/application/personal-catalog.md)
    explains the private database, migration strategy, current API, debugger
    launch, and next browser application slice.

## Current status

The first two capture sessions have been collected and prepared. Session 1 has
98 training photographs and Session 2 has 42 independent validation
photographs. All four frozen DINOv2 aggregation rules correctly identified all
42 Session 2 queries. Since the added aggregation complexity did not improve
the validation metrics, nearest image remains the preferred frozen baseline.

Session 3 was opened once after the checkpoint and rejection rules were frozen.
Frozen and tuned nearest image retrieval each identified 41 of 42 known photos
correctly. Tuned prototype retrieval and the classifier each identified 40 of
42, and every system placed the correct identity in its top three for all 42
photos. Session 3 is now closed to further model or threshold selection.

The unknown item workflow is ready for the next photo session. It keeps each
unknown physical product under a stable local ID, includes unknowns in the same
prepared image audit, and selects a nearest image cosine threshold from
validation data only. The selection rule retains at least 95 percent of
correctly identified known validation queries while minimizing false acceptance
of unknown items.

Session 4 now contains 51 unknown validation photographs from four physical
objects. Maximum similarity was the strongest confidence rule for frozen
DINOv2. A threshold of `0.820814` retained 40 of 42 known queries and falsely
accepted 7 of 51 unknown queries. The selected partial fine tuning checkpoint
improved the same rule to 41 correct known accepts and 4 unknown false accepts.
Tuned prototype similarity and classifier probability margin both reached zero
unknown false accepts at the required known retention. I chose not to collect
or redistribute third party social media photos, so there is no independent
unknown final test. These unknown rejection results remain validation evidence
and are not presented as proof of reliable rejection on new Usagi variants.

The frozen baseline now also has a debugger workflow for predicting one photo
and measuring warm inference latency. The first Apple Metal benchmark reached a
median warm prediction time of about 91 milliseconds against the 98 image
training gallery.

The first supervised linear probe is also complete. It trains only a 2,695
parameter classification head over frozen embeddings and keeps Session 3
sealed. It matched nearest image retrieval with perfect Session 2 validation
accuracy, so nearest image remains the preferred product baseline.

The partial fine tuning experiment is complete on the RTX 3060. All six runs
reached perfect Session 2 accuracy. Conservative augmentation lowered mean
validation loss by about 44 percent and improved the most difficult Lobster
view for every seed. Run `c62a6ff28d2e` was frozen because it had the lowest
known validation loss among the conservative runs. It did not improve final
Top 1 accuracy over frozen nearest image retrieval.

The private catalog application is now underway. Its first foundation uses a
versioned SQLite database, typed catalog records, and a local HTTP API. It keeps
ownership, interest, verification, and the optional link to a photographed
identity separate. The next slice will add the browser interface and connect a
photo upload to the selected retrieval model and uncertainty rule.

I also plan to experiment with my Meta Ray Ban Display glasses through a native
iPhone companion. The browser and glasses flows will share the same Python
identifier and personal catalog. The glasses app will be another client, not a
separate model or database. Metric learning and multi doll detection remain
optional future work rather than requirements for the first useful application.

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
I took for Session 1 and Session 2. I am comfortable sharing those images as
part of this project, and the prepared copies do not include the original photo
metadata.

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
