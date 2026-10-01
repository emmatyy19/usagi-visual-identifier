# Experiment 005: Unknown Rejection Validation

## Status

Complete on Apple Metal. Session 3 remains sealed.

## Question

Which confidence signal best distinguishes known catalog dolls from objects and
doll variants that are not in the catalog, and does partial fine tuning improve
that separation?

## Frozen protocol

1. Session 1 supplies 98 known gallery photographs.
2. Session 2 supplies 42 known validation queries.
3. Session 4 supplies 51 unknown validation queries from four physical objects.
4. Session 3 remains absent from the embedding artifact.
5. Frozen DINOv2 Small is compared using maximum similarity, identity margin,
   five neighbor agreement, and prototype similarity.
6. The threshold must correctly identify and accept at least 95 percent of known
   validation queries.
7. Among eligible thresholds, select the point with the lowest unknown false
   acceptance rate. Break ties by retaining more correct known queries, then by
   choosing the higher threshold.
8. The partial fine tuning checkpoint is selected first using known Session 2
   validation loss. Unknown photos do not influence checkpoint selection.

## Unknown identities

| Identity | Informal name | Photographs | Role |
| --- | --- | ---: | --- |
| `unknown-0001` | Crocheted banana | 14 | Easy unrelated object |
| `unknown-0002` | Crocheted rabbit | 14 | Medium visually related rabbit |
| `unknown-0003` | Usagi posture pal | 11 | Hard unseen Usagi variant |
| `unknown-0004` | Sloth plush | 12 | Easy unrelated plush |

## Results

### Frozen DINOv2 confidence comparison

| Confidence signal | Correct known accepted | Unknown accepted |
| --- | ---: | ---: |
| Maximum similarity | 40 of 42 | 7 of 51 |
| Identity margin | 40 of 42 | 13 of 51 |
| Five neighbor agreement | 40 of 42 | 22 of 51 |
| Prototype similarity | 40 of 42 | 11 of 51 |

Maximum similarity remains the frozen confidence rule. It had the lowest
unknown false acceptance count while satisfying the known retention
requirement. More elaborate confidence scores did not improve the frozen model.

### Original maximum similarity baseline

The selected cosine threshold was `0.820814`. It correctly identified and
accepted 40 of 42 known queries, or 95.2 percent. Every accepted known query was
correct. Airport image `img-000105` and Magical image `img-000117` were ranked
correctly but rejected because their scores were below the threshold.

Seven of 51 unknown photographs were falsely accepted, giving an aggregate
unknown false acceptance rate of 13.7 percent. All seven came from Usagi posture
pal. The other three unknown identities had zero false accepts.

| Unknown identity | False accepts | Rate | Mean top score | Maximum top score |
| --- | ---: | ---: | ---: | ---: |
| Crocheted banana | 0 of 14 | 0.0% | 0.505 | 0.651 |
| Crocheted rabbit | 0 of 14 | 0.0% | 0.603 | 0.653 |
| Usagi posture pal | 7 of 11 | 63.6% | 0.842 | 0.874 |
| Sloth plush | 0 of 12 | 0.0% | 0.762 | 0.801 |

Six accepted posture pal photographs selected Lobster claws as their nearest
known identity. One selected Chiikawa Park fluffy. The shared Usagi face, ears,
color, and body shape produced high similarity even though the defining product
accessories did not match.

## Decision

The exact partial fine tuning checkpoint was frozen as run `c62a6ff28d2e`,
epoch 22, before evaluating unknown photos. It was the conservative
augmentation run with the lowest known Session 2 validation loss. Its checkpoint
SHA256 is
`79a41f985a890c40c08575e16d60e80d7070a709594de07a6a5d3b711b85f6ef`.

Using the same maximum similarity rule, tuned embeddings correctly accepted 41
of 42 known photos and falsely accepted 4 of 51 unknown photos. This improves on
the frozen model's 40 known and 7 unknown result without changing the confidence
definition.

Two additional tuned results are promising but remain validation findings.
Prototype similarity accepted 40 correct known photos and zero unknown photos.
The classifier head's probability margin also accepted 40 correct known photos
and zero unknown photos. Maximum classifier probability accepted one unknown.
Softmax confidence is not proof that an image belongs to the catalog, so the
probability margin result still needs truly novel final test objects.

Keep maximum similarity as the frozen baseline rule and carry both tuned
prototype similarity and classifier probability margin into the final
comparison. At this decision point Session 3 remained sealed. It was later
opened once for the final known test recorded in
[Experiment 005b](005b-final-known-test.md).

Per identity thresholds remain deferred because each known identity has only six
validation photographs. Forty two separate validation photos are already small;
splitting them into seven threshold calibration groups would make the result too
sensitive to individual images.

I later chose not to collect or redistribute third party social media photos.
There is therefore no independent unknown final test, and every unknown result
in this record remains a validation result.

The
[executed walkthrough](../../notebooks/unknown_rejection_walkthrough.ipynb)
contains the complete score distributions, threshold curve, false accepts, and
known rejections. The
[model selection walkthrough](../../notebooks/open_set_model_selection_walkthrough.ipynb)
adds the confidence comparison, frozen checkpoint, tuned embedding results, and
classifier confidence results.
