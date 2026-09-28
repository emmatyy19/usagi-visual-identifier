# Project Roadmap

This roadmap tracks public milestone-level progress. Detailed personal notes,
collection inventory, product-research leads, and near-term working checklists
belong in the Git-ignored `private/` directory.

## Current position

The repository contains seven catalog identities, 98 prepared Session 1
training photographs, 42 prepared Session 2 validation photographs, and a
reproducible frozen DINOv2 retrieval baseline. All four aggregation rules
correctly classified every Session 2 query, so nearest image remains the
preferred rule because it is the simplest tied method and preserves direct
reference provenance. The linear probe is complete. The partial fine tuning
pipeline and private GPU transfer workflow are ready, but the RTX 3060 results
are pending. Session 3 contains 42 sealed test photographs and remains closed
while validation driven model and threshold choices are completed.

The modeling order is frozen retrieval, simple aggregation comparisons, a
linear probe, partial tuning of the final transformer block, and then retrieval
oriented metric learning. Patch level matching is optional and will only be
added if the failure analysis shows that global embeddings consistently miss
small identity details.

## Phase 0: Define the experiment

- [x] Define the project motivation and initial user workflow
- [x] Separate visual identification from catalog and commerce data
- [x] Choose retrieval as the initial formulation
- [x] Define a staged transfer-learning and fine-tuning plan
- [x] Define capture-session-aware dataset splitting
- [x] Define closed-set and open-set evaluation
- [x] Inventory the first 5-10 owned variants using stable local identifiers
- [x] Decide which collection fields are public, private, or deferred
- [x] Confirm the available RTX 3060 VRAM and local CUDA environment

### Exit criteria

The first catalog identities are unambiguous, the capture protocol can be
followed repeatedly, and the evaluation rules are frozen before looking at test
results.

## Phase 1: Repository and data foundation

- [x] Create a Python 3.12 package using `uv` and a `src/` layout
- [x] Configure Ruff, strict Pyright, pytest, pre-commit, and CI
- [x] Define and apply deterministic seeds for model experiments
- [x] Define typed catalog-item and image-manifest records
- [x] Implement manifest validation with actionable error messages
- [x] Create safe synthetic image fixtures for automated tests
- [x] Implement deterministic image preprocessing
- [x] Add a command that audits class counts, sessions, and split leakage
- [x] Document the exact local directory and metadata contracts

### Exit criteria

A contributor can reproduce the environment, validate a synthetic dataset, and
prove that no capture session or source image crosses dataset splits.

## Phase 2: Collect the first dataset

- [x] Assign stable IDs to 5-10 owned variants
- [x] Conduct at least three distinct capture sessions per variant
- [x] Vary pose, distance, background, and lighting according to the protocol
- [x] Record image provenance and capture-session metadata
- [x] Reserve one complete session per variant as the untouched test set
- [ ] Collect non-Usagi plushes and excluded Usagi variants as unknown examples
- [x] Audit Session 1 duplicates, corrupted images, labels, and personal data
- [x] Publish a reviewed Session 1 summary without the original HEIC files
- [x] Audit and publish the reviewed Session 2 validation summary

### Exit criteria

Every known identity has independent train, validation, and test sessions; the
unknown set includes both easy and visually similar negatives; and the dataset
audit passes.

## Phase 3: Frozen-embedding retrieval baseline

- [x] Extract normalized embeddings with DINOv2 Small
- [x] Build a brute-force cosine-similarity index
- [x] Support one or many reference images per catalog item
- [x] Implement nearest image, prototype, top k mean, and weighted neighbor rules
- [x] Compare the four rules on Session 2 and choose one with validation results
- [x] Return top-k matches with distances and reference provenance
- [x] Evaluate top-1, top-3, per-class recall, and common confusion pairs
- [x] Measure Apple Metal inference latency
- [ ] Measure CPU and NVIDIA GPU inference latency
- [x] Record the Session 1 sanity check as Experiment 001
- [x] Record the Session 2 validation comparison as Experiment 002

### Exit criteria

The complete query-to-result pipeline runs locally and produces a reproducible
baseline on untouched capture sessions.

## Phase 4: Linear-probe baseline

- [x] Freeze the visual backbone and train a linear classification head
- [x] Tune only on the training and validation sessions
- [x] Compare classification accuracy with nearest-neighbor retrieval
- [x] Inspect calibration and confidence failure cases
- [x] Document how adding a catalog identity differs between both approaches
- [x] Record the comparison as Experiment 003

### Exit criteria

The project has a supervised baseline and a clear, measured explanation of the
tradeoff between fixed-class prediction and incremental retrieval.

## Phase 5: Partial fine-tuning

- [x] Define conservative augmentations that preserve merchandise identity
- [x] Implement a model that unfreezes only the final encoder block
- [x] Configure mixed precision and VRAM-aware batch sizing
- [x] Track train/validation curves and save only local checkpoints
- [x] Build a verified private transfer workflow for GPU training
- [ ] Run the predetermined configurations and seeds on the RTX 3060
- [ ] Compare against the frozen retrieval and linear-probe baselines
- [ ] Run ablations for augmentation and trainable depth
- [ ] Inspect attention or nearest-neighbor examples for qualitative evidence
- [ ] Record the results as Experiment 004

### Exit criteria

Fine-tuning produces a repeatable held-out improvement or is explicitly rejected
as unnecessary based on the evidence.

## Phase 6: Retrieval-oriented metric learning

- [ ] Select a supervised contrastive or triplet-loss formulation
- [ ] Build balanced batches containing multiple images per identity
- [ ] Define positive, confusing-negative, and unknown sampling rules
- [ ] Fine-tune a projection head and then, if justified, the final backbone block
- [ ] Evaluate recall at k and embedding-space separation
- [ ] Test adding a held-out identity using references without retraining
- [ ] Record the results as Experiment 005

### Exit criteria

The chosen embedding model outperforms the frozen baseline and demonstrates a
practical path for adding new catalog identities.

## Phase 7: Unknown-item rejection

- [ ] Define validation-only threshold-selection rules
- [ ] Compare global and per-identity thresholds
- [ ] Measure known acceptance versus unknown false acceptance
- [ ] Test easy negatives, same-character unknowns, and near-duplicate variants
- [ ] Expose uncertainty instead of forcing every query into a known identity
- [ ] Freeze the first operating point before final test evaluation

### Exit criteria

The system has a documented operating threshold and refuses uncertain matches
at a measured rate suitable for personal catalog use.

## Phase 8: Local catalog application

- [ ] Define a small SQLite catalog schema and migration strategy
- [ ] Link model identities to canonical names and release metadata
- [ ] Build a local command-line or lightweight web interface
- [ ] Display top matches, reference evidence, and unknown warnings
- [ ] Allow owner corrections without silently treating them as training data
- [ ] Preserve model, index, and catalog version provenance

### Exit criteria

A local user can submit one cropped doll photograph, review likely matches, and
open the selected catalog record.

## Phase 9: Multi-doll photographs

- [ ] Assemble and annotate a small detection dataset
- [ ] Establish a manual-crop oracle before training a detector
- [ ] Detect or segment dolls in group and shelf photographs
- [ ] Pass each crop through the existing identity pipeline
- [ ] Evaluate detection and identification errors separately and end to end

### Exit criteria

The system can enumerate dolls in a group photograph while making the source of
each failure—localization, cropping, or identity—observable.

## Phase 10: Availability and market information

- [ ] Research permitted official-store and marketplace data sources
- [ ] Define canonical listing matching independently of visual identification
- [ ] Store source, currency, condition, timestamp, and listing status
- [ ] Distinguish asking prices from completed-sale evidence
- [ ] Add staleness and provenance indicators to every displayed value
- [ ] Avoid automated collection where terms or access rules do not permit it

### Exit criteria

Catalog records can display traceable, time-stamped availability information
without presenting stale or incomparable prices as authoritative market value.

## Explicitly deferred

- Training a foundation model from scratch
- Identifying every Chiikawa character or product category
- Scraping marketplaces before source permissions and data semantics are clear
- Cloud deployment, vector databases, and distributed infrastructure
- Generative answers before the underlying identity and metadata are reliable
