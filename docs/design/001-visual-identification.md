# Design 001: Visual Identification and Retrieval

## Status

Active

## Context

Chiikawa Usagi merchandise contains many visually similar variants. The base
character appearance is shared, while identity often depends on localized
details such as clothing, headwear, props, scale, fabric, or embroidery. The
catalog is also open-ended: new and previously unseen variants can appear after
the first model is trained.

The project is intended to teach practical computer vision on a single consumer
GPU. It should expose representation learning, transfer learning, fine-tuning,
retrieval, open-set recognition, evaluation, and failure analysis without
requiring foundation-model pretraining.

## Problem statement

Given a photograph containing one primary, cropped doll, rank catalog identities
by visual similarity and decide whether the query is represented in the known
catalog.

The initial output contains:

- up to three candidate catalog IDs;
- similarity or calibrated confidence evidence for each candidate;
- reference-image provenance for inspection;
- the model and index versions; and
- either an accepted identity or an explicit `unknown` decision.

## Terminology

- **Catalog identity:** one specific merchandise variant, not merely the Usagi
  character.
- **Reference or gallery image:** a labeled image stored in the retrieval index.
- **Query image:** an image submitted for identification.
- **Known query:** a query whose catalog identity exists in the gallery.
- **Unknown query:** a query whose identity is deliberately absent from the
  gallery.
- **Closed-set evaluation:** every query belongs to a known identity.
- **Open-set evaluation:** queries may represent unknown identities.
- **Capture session:** images acquired together under substantially shared
  camera, background, lighting, and staging conditions.

## Goals

- Produce a useful no-training retrieval baseline with a pretrained vision
  encoder.
- Measure generalization across independently captured sessions.
- Gain hands-on experience with linear probing and partial fine-tuning.
- Explore metric learning that preserves incremental catalog growth.
- Reject unsupported identities instead of always returning a confident label.
- Keep the visual system independent of volatile product and market metadata.
- Fit development and inference comfortably on an RTX 3060-class GPU.

## Non-goals for the initial system

- Detecting multiple dolls or segmenting cluttered shelf images.
- Identifying variants absent from both training data and the reference gallery.
- Inferring an exact commercial product name from visual appearance alone.
- Estimating authenticity, physical condition, or monetary value.
- Training a vision foundation model from scratch.
- Publishing original phone files, unreviewed personal photos, or downloaded
  product photography.
- Providing a public service or mobile application.

## Proposed architecture

```text
query image
    |
    v
decode and validate -> deterministic evaluation transform
    |
    v
pretrained vision encoder
    |
    v
L2-normalized embedding
    |
    +---------------- cosine similarity ----------------+
                                                     gallery
                                                       |
                                                       v
                                            ranked reference images
                                                       |
                                                       v
                                        aggregate by catalog identity
                                                       |
                                                       v
                                      accept identity or return unknown
                                                       |
                                                       v
                                             join catalog metadata
```

The first index can be an in-memory matrix with exact cosine similarity. A
vector database or approximate-nearest-neighbor service is unnecessary at the
expected catalog size.

## Model strategy

### Stage A: Frozen visual encoder

DINOv2 Small is the planned first encoder. Each gallery and query image is
converted to an L2-normalized embedding. Exact cosine similarity ranks gallery
images. This establishes the minimum-complexity baseline.

Four item level aggregation rules will be compared:

1. **Nearest image:** an item's score is its most similar reference image.
2. **Prototype:** average the normalized reference embeddings for an item, then
   normalize the mean and compare the query to that prototype.
3. **Top k mean:** average the strongest few reference scores for each item.
4. **Weighted neighbor vote:** allow several nearby references to vote, with
   stronger similarities receiving more weight.

The rule is chosen with validation sessions only.

Patch level similarity is not part of the initial comparison. It will be added
only if independent validation errors show that the global embedding repeatedly
misses small local identity details.

### Stage B: Linear probe

The encoder remains frozen while a linear classification head learns the known
catalog identities. This measures how linearly separable the pretrained features
are. It also provides a simple supervised baseline, but it is not expected to
be the long-term catalog-growth mechanism.

### Stage C: Partial fine-tuning

The classification head and final encoder block are trained while earlier
blocks stay frozen. Mixed precision, small batches, gradient accumulation when
needed, early stopping, and separate backbone/head learning rates keep the run
within consumer-GPU limits and reduce catastrophic overfitting.

Full-backbone fine-tuning is considered only after partial fine-tuning has a
stable baseline and a measured limitation.

### Stage D: Metric learning

A trainable projection head—and later, if justified, the final encoder
block—is optimized with supervised contrastive or triplet-style loss. The
objective is to pull same-identity images from different sessions together and
separate confusing variants.

Loss and batch-mining choices must be evaluated through retrieval metrics rather
than training loss alone.

## Data boundary

Original images and private inventory metadata remain outside version control.
A local manifest maps image paths to stable catalog IDs, capture sessions,
sources, and split assignments. See the
[dataset protocol](../data/dataset-protocol.md).

Derived embeddings, checkpoints, indexes, and run outputs are also local
artifacts. Public experiment records may contain reviewed, resized photographs
when I have explicitly approved them. They must not contain original photo
metadata, private paths, or unreviewed source files.

## Identity boundary

A catalog ID represents the finest variant that can be labeled consistently
from reliable product information. Two items should not be assigned distinct
visual classes when their documented difference is invisible in ordinary
photos—for example, packaging-only differences while both are photographed
unboxed.

When identity is uncertain, the inventory records a provisional entry and
excludes it from supervised training until resolved. Labels must not be inferred
solely from the model's prediction.

## Split and leakage boundary

Images from a capture session are indivisible for splitting. Near-duplicate
frames, augmented versions, crops from the same original, and derived images
must remain in the same split as their source.

The initial split policy is:

- training sessions: model fitting and gallery construction;
- validation sessions: model, augmentation, aggregation, and threshold choices;
- test sessions: one-time final evaluation after those choices are frozen.

No test image may become a gallery reference, training example, hard negative,
or threshold-tuning example.

## Unknown-item decision

The system must not equate "nearest" with "correct." A query is accepted only
when validation evidence supports the operating rule. Candidate signals include:

- top-1 cosine similarity;
- margin between the first and second identities;
- consistency among the nearest reference images; and
- classifier calibration when a classification head is evaluated.

The initial rule should be simple and interpretable, likely a global top-1
similarity threshold with an optional minimum top-1/top-2 margin. More complex
confidence models require clear validation improvement.

## Catalog boundary

Visual identity resolves to a stable local key. Metadata such as name, Japanese
name, character, collection, release date, original price, size, official URL,
and notes attach to that key. Personal ownership and purchase history belong in
private tables or files.

Availability and market observations are time-dependent records rather than
properties of the visual model. Every observation should carry a source,
retrieval timestamp, currency, condition, and listing state.

## Interfaces

The first interface uses local debugger scripts and VS Code launch choices for:

- validating and preparing an image manifest;
- extracting or refreshing gallery embeddings;
- evaluating a saved model and index configuration; and
- identifying one cropped query image.

A local web UI is deferred until the underlying output contract is stable.

## Reproducibility

Every experiment should record:

- code revision;
- model checkpoint identifier and license;
- preprocessing and augmentation configuration;
- random seeds;
- dataset-manifest fingerprint;
- split and gallery-definition versions;
- trainable parameter groups and optimizer settings;
- hardware and peak GPU memory when training; and
- aggregate and per-slice evaluation results.

## Risks and mitigations

### Background shortcut learning

If each doll is photographed against a unique background, the model can learn
the setting instead of the merchandise. Capture every identity across multiple
shared backgrounds and hold out complete sessions.

### Test leakage through repeated frames

Burst photographs can make random image splits misleadingly easy. Keep capture
sessions and source-image families intact.

### Augmentation destroys identity

Aggressive crops, color jitter, or erasing may remove the very accessory that
distinguishes a variant. Begin conservatively and document augmentation
ablations.

### Catalog labels are wrong or too granular

Model quality cannot exceed label quality. Record provenance for product names,
keep unresolved IDs provisional, and merge classes that are not visually
distinguishable under the input contract.

### Overconfident handling of unknowns

Nearest-neighbor retrieval always has a nearest result. Include meaningful
unknown validation examples and report false-accept rates at the selected
threshold.

### Marketplace scope overwhelms the vision project

Defer automated listing acquisition until identification works. Model catalog
metadata and commerce observations as separate modules with separate tests.

## Initial decisions

- Retrieval precedes classification as the user-facing formulation.
- DINOv2 Small is the first baseline encoder.
- Exact cosine search is sufficient for the initial catalog.
- Cheap aggregation comparisons come before model training.
- Fine-tuning begins with a linear head, then the final backbone block.
- Patch level matching requires evidence that global features miss local cues.
- Capture-session splitting is mandatory.
- Unknown rejection is a first-class requirement, not a post-launch addition.
- Multi-object detection and market tracking are later phases.

## Open decisions

- Is DINOv2 or DINOv3 better under the same local protocol?
- Which planned aggregation rule generalizes best to Session 2?
- Which augmentations preserve tiny identity cues?
- Which metric-learning objective is most stable at the available class count?
- What known-acceptance/unknown-false-acceptance tradeoff feels useful in the
  personal workflow?
