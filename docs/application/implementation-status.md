# Implementation status

Status reviewed October 23, 2026. This page describes working behavior, not
future intent.

## Implemented

| Area | Current behavior | Boundary |
| --- | --- | --- |
| Dataset workflow | Manifest validation, image preparation, and session based splits | Original data stays local |
| Modeling | Frozen retrieval, aggregation comparisons, linear probe, partial fine tuning | Seven known identities |
| Evaluation | Final known Session 3 evaluation and unknown validation | No independent unknown final test |
| Catalog | SQLite migrations and typed create, read, and update API | No delete operation |
| Sources | Add and edit links, notes, availability, and manual check dates | No automatic shop checks or price tracking |
| Shelf | Responsive browse, filter, create, and edit pages | No hosted service or authentication |
| Public viewer | Seven published photos, search, sorting, and doll details | Static collection only; Pages workflow prepared |
| Covers | Repeatable import from first Session 1 photos | Import requires local collection files |
| Identification | Browser upload and shared JSON API | One cropped doll per photo |
| Loading feedback | Immediate submit feedback, progress indicator, and slower request messages across local pages | Indeterminate progress; no estimated completion time |
| Evidence | Top three identities, nearest training references, catalog links | Suggestions need human review |
| Uncertainty | Existing frozen similarity threshold | Not calibrated probabilities |
| Query handling | Format and size validation, orientation correction, metadata removal, temporary file cleanup | Query photos are never retained |
| Decisions | Confirm a suggestion, select another catalog doll, or mark unknown | Explicit decisions only; no automatic training examples |
| Provenance | Saved model revision, gallery hash, dataset fingerprint, threshold, scores, and catalog snapshots | Only identification relevant catalog fields are copied |

## Still planned

1. Add optional web assisted discovery with reviewed source links.
2. Build a native iPhone companion and test glasses capture and display.

These are not available through the current interface. Metric learning and
multi doll detection remain optional experiments, not prerequisites.

## What someone can reproduce from a clone

The locked environment, application code, deterministic synthetic tests, and
empty catalog workflow are available. Executed notebooks and screenshots show
the completed experiments and app behavior without requiring local artifacts.

The collection database, raw images, prepared dataset, embedding galleries, and
training checkpoints are excluded. Full experiment reruns and identification
against my seven dolls require those artifacts. Public availability of the code
does not make those artifacts available or grant rights to the embedded photos.

See the [application guide](personal-catalog.md), [showcase](../showcase/README.md),
and [roadmap](../roadmap.md) for usage, examples, and the remaining milestones.
