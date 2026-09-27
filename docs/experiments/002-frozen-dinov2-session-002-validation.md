# Experiment 002: Frozen DINOv2 Session 2 Validation

## Status

Complete validation comparison

## Question

Which frozen embedding aggregation rule generalizes best from the Session 1
gallery to independent Session 2 photographs?

## Dataset

1. Manifest fingerprint:
   `27c73c3bc69640bb10b36f4846270847dea4fe871cedeee516b7f58badd9a634`
2. Known identities: 7
3. Gallery: 98 Session 1 training images, 14 per identity
4. Queries: 42 Session 2 validation images, 6 per identity
5. Unknown identities: not measured

## Configuration

1. Encoder: `facebook/dinov2-small`
2. Revision: `ed25f3a31f01632728cabb09d1542f84ab7b0056`
3. Embedding: 384 dimensional CLS token, normalized to unit length
4. Aggregation rules: nearest image, identity prototype, top k mean with three
   references, and weighted neighbors with five references and temperature
   `0.07`
5. Trainable parameters: 0
6. Hardware: Apple Silicon with PyTorch Metal

## Protocol

Every Session 2 image was used once as a query. Every query was compared with
the same 98 image Session 1 gallery. No Session 2 image entered the gallery.
All aggregation rules received identical embeddings and queries.

Session 2 is validation data, so it may be used to choose the aggregation rule.
Session 3 remains sealed and may not be used for this decision.

## Results

| Method | Top 1 | Top 3 | Macro recall | MRR |
| --- | ---: | ---: | ---: | ---: |
| Nearest image | 1.000 | 1.000 | 1.000 | 1.000 |
| Identity prototype | 1.000 | 1.000 | 1.000 | 1.000 |
| Top k mean | 1.000 | 1.000 | 1.000 | 1.000 |
| Weighted neighbors | 1.000 | 1.000 | 1.000 | 1.000 |

All four methods classified all 42 queries correctly.

For nearest image, the winning cosine score had a minimum of `0.7783` and a
mean of `0.8757`. The first versus second place margin had a minimum of `0.0108`
and a mean of `0.0651`.

## Failure and margin analysis

There are no incorrect predictions to review. The narrowest margins still
identify useful risk areas. Lobster side views were close to Matcha, Airport,
or Chiikawa Park fluffy. Airport side views were also close to Matcha. These
predictions were correct, but they were less decisive than most front and
detail views.

Weighted neighbor scores are vote shares rather than cosine similarities, so
their score scale was not compared directly with the other three methods.

## Decision

Keep nearest image as the preferred frozen baseline. No more complex rule
improved a primary validation metric, and nearest image provides the clearest
reference provenance for inspecting a result.

The perfect score is limited evidence rather than proof that the problem is
solved. Session 2 contains only 42 balanced known identity queries from one
shared setup. Unknown rejection remains untested.

## Follow up

1. Capture Session 3 under a clearly different setup and seal it immediately.
2. Collect unknown validation examples before choosing a rejection threshold.
3. Measure latency and expose a single image prediction path.
4. Train a linear probe and compare it with this frozen baseline using Session
   2 for validation only.
5. Try partial fine tuning for learning value, but require a measured benefit.
6. Freeze every choice before opening Session 3 for final evaluation.
