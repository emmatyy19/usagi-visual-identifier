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
reference provenance. The linear probe and six partial fine tuning runs are
complete. Conservative augmentation lowered mean Session 2 validation loss by
about 44 percent and improved the hardest Lobster view for all three seeds.
Session 3 has now been evaluated once and is closed to further selection.
Frozen and tuned nearest image retrieval each identified 41 of 42 final known
photos correctly. Prototype retrieval and the classifier each identified 40 of
42. All four systems achieved 42 of 42 Top 3 accuracy.

The unknown validation workflow is implemented and Session 4 contains 51
photographs from four unknown physical products. Maximum similarity was the best
frozen DINOv2 confidence rule. Partial fine tuning improved its result from 40
correct known accepts and 7 unknown false accepts to 41 correct known accepts
and 4 unknown false accepts. Tuned prototype similarity and classifier
probability margin each reached zero unknown false accepts while retaining 40
correct known photos. I chose not to collect third party social media images,
so there is no independent unknown final test. These remain validation results,
and independent unknown testing is deferred.

The core modeling comparison is complete. Frozen nearest image retrieval and
tuned nearest image retrieval both identified 41 of 42 final known photos. The
extra training did not improve final Top 1 accuracy, so deeper tuning and metric
learning are no longer required milestones. They remain optional learning
experiments.

The next product goal is a private personal catalog. It will track dolls I own,
dolls I know exist, dolls I want, dolls I do not want, source links, and notes
about availability. Local retrieval will search my photographed collection.
When there is no confident local match, an optional web assisted lookup can
suggest names and source pages for me to review. It will not silently add web
results to the catalog or treat them as verified facts.

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
- [x] Collect non-Usagi plushes and excluded Usagi variants as unknown examples
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
- [ ] Measure CPU and NVIDIA GPU inference latency if deployment profiling becomes useful
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
- [x] Run the predetermined configurations and seeds on the RTX 3060
- [x] Compare against the frozen retrieval and linear-probe baselines
- [x] Compare augmentation and trainable scope using frozen, head only, final block, augmented, and unaugmented runs
- [x] Inspect the least decisive images for qualitative evidence
- [x] Record the results as Experiment 004

### Exit criteria

Fine-tuning produces a repeatable held-out improvement or is explicitly rejected
as unnecessary based on the evidence.

The existing comparisons already provide the useful ablations. Deeper
unfreezing is not justified because partial fine tuning did not improve final
Top 1 accuracy.

## Phase 6: Optional retrieval oriented metric learning

This phase is no longer required for the personal catalog. With seven known
identities and no practical source of unseen identity photographs, it would be
hard to show that metric learning generalizes beyond the current collection.
It remains available as a future learning exercise.

- [ ] Select a supervised contrastive or triplet-loss formulation
- [ ] Build balanced batches containing multiple images per identity
- [ ] Define positive, confusing-negative, and unknown sampling rules
- [ ] Fine-tune a projection head and then, if justified, the final backbone block
- [ ] Evaluate recall at k and embedding-space separation
- [ ] Test adding a held-out identity using references without retraining
- [ ] Record the results as Experiment 006

### Exit criteria

If revisited, the chosen embedding model must outperform the frozen baseline on
independent data and demonstrate a practical path for adding identities without
retraining.

## Phase 7: Unknown-item rejection

- [x] Define validation-only threshold-selection rules
- [x] Implement unknown photo preparation and threshold curve reporting
- [x] Compare global confidence methods on validation data
- [x] Decide against per identity thresholds because each identity has only six validation photographs
- [x] Measure known acceptance versus unknown false acceptance
- [x] Test easy negatives and same-character unknowns
- [ ] Expose uncertainty instead of forcing every query into a known identity
- [x] Freeze the first operating point before final test evaluation
- [x] Run the one time final known Session 3 evaluation
- [ ] Run an independent unknown final test if suitable owner captured images become available

### Exit criteria

The system has a documented validation operating threshold. The application
must describe uncertain results as possible matches or no confident local match.
Reliable rejection of new Usagi variants remains unconfirmed without an
independent unknown final test.

## Phase 8: Local catalog application

- [ ] Define a small SQLite schema and migration strategy for the personal catalog
- [ ] Track owned, known to exist, wanted, and not wanted status independently
- [ ] Store a preferred name, distinguishing details, notes, and verification status
- [ ] Store reviewed source links and optional availability notes with timestamps
- [ ] Link photographed identities to personal catalog records without requiring a complete official catalog
- [ ] Build a private local web interface for uploading one cropped doll photograph
- [ ] Display top matches, reference evidence, and calibrated uncertainty wording
- [ ] Allow manual confirmation and correction without silently creating training data
- [ ] Preserve model, gallery, threshold, and catalog version provenance
- [ ] Keep credentials, query history, and personal collection details out of Git

### Exit criteria

I can submit one cropped doll photograph, review likely local matches, confirm
or reject the suggestion, and open or create a personal catalog record.

## Phase 9: Optional web assisted discovery

- [ ] Add an explicit opt in lookup when there is no confident local match
- [ ] Use an official image or web search API rather than scraping search pages
- [ ] Return candidate names and source page links instead of copying web images into the repository
- [ ] Optionally use an LLM to compare visible details and summarize candidate evidence
- [ ] Require manual confirmation before saving any discovered catalog information
- [ ] Record lookup provider, source URL, and review date for saved evidence
- [ ] Add local request limits, timeouts, error handling, and cost controls
- [ ] Load all provider credentials from ignored environment configuration

### Exit criteria

An uncertain local result can start a private, traceable lookup. The application
shows suggestions and source links while keeping me responsible for the final
catalog decision.

## Phase 10: Personal availability and acquisition notes

- [ ] Research permitted official store and marketplace sources only when I need them
- [ ] Save reviewed listing links without treating one listing as a canonical product record
- [ ] Store source, currency, condition, timestamp, and listing status when useful
- [ ] Distinguish asking prices from completed sale evidence
- [ ] Display staleness and provenance for every saved availability observation
- [ ] Keep collection automated only where an official API and its terms permit it

### Exit criteria

Personal catalog records can hold traceable availability notes without claiming
to be a complete or authoritative market database.

## Explicitly deferred

- Training a foundation model from scratch
- Identifying every Chiikawa character or product category
- Scraping marketplaces before source permissions and data semantics are clear
- Cloud deployment, vector databases, and distributed infrastructure
- Generative answers before the underlying identity and metadata are reliable
- Metric learning unless a larger, independently testable identity dataset becomes available
- Multi doll detection until the single doll personal catalog workflow is useful
