# Implementation status

Status reviewed October 17, 2026, after the photo identification implementation
was merged in PR 17. This page describes working behavior, not future intent.

## Implemented

| Area | Current behavior | Boundary |
| --- | --- | --- |
| Dataset workflow | Manifest validation, image preparation, and session based splits | Original data stays local |
| Modeling | Frozen retrieval, aggregation comparisons, linear probe, partial fine tuning | Seven known identities |
| Evaluation | Final known Session 3 evaluation and unknown validation | No independent unknown final test |
| Catalog | SQLite migrations and typed create, read, and update API | No delete operation |
| Shelf | Responsive browse, filter, create, and edit pages | No hosted service or authentication |
| Public viewer | Seven published photos, search, sorting, and doll details | Static collection only; Pages workflow prepared |
| Covers | Repeatable import from first Session 1 photos | Import requires local collection files |
| Identification | Browser upload and shared JSON API | One cropped doll per photo |
| Evidence | Top three identities, nearest training references, catalog links | Suggestions need human review |
| Uncertainty | Existing frozen similarity threshold | Not calibrated probabilities |
| Query handling | Format and size validation, orientation correction, metadata removal, temporary file cleanup | No durable query history |
| Provenance | Model revision, gallery hash, dataset fingerprint, threshold, and timing in responses | No saved catalog version snapshot |

## Still planned

1. Save explicit identification confirmations and corrections with provenance.
2. Add source link and availability operations. The initial source link table
   exists, but there is no corresponding user workflow yet.
3. Add optional web assisted discovery with reviewed source links.
4. Build a native iPhone companion and test glasses capture and display.

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
