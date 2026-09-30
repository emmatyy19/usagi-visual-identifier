# Experiment 005: Unknown Rejection Validation

## Status

Complete on Apple Metal. Session 3 remains sealed.

## Question

Can maximum nearest image cosine similarity distinguish known catalog dolls from
objects and doll variants that are not in the catalog?

## Frozen protocol

1. Session 1 supplies 98 known gallery photographs.
2. Session 2 supplies 42 known validation queries.
3. Session 4 supplies 51 unknown validation queries from four physical objects.
4. Session 3 remains absent from the embedding artifact.
5. Frozen DINOv2 Small and nearest image retrieval produce one top cosine score
   per query.
6. The threshold must correctly identify and accept at least 95 percent of known
   validation queries.
7. Among eligible thresholds, select the point with the lowest unknown false
   acceptance rate. Break ties by retaining more correct known queries, then by
   choosing the higher threshold.

## Unknown identities

| Identity | Informal name | Photographs | Role |
| --- | --- | ---: | --- |
| `unknown-0001` | Crocheted banana | 14 | Easy unrelated object |
| `unknown-0002` | Crocheted rabbit | 14 | Medium visually related rabbit |
| `unknown-0003` | Usagi posture pal | 11 | Hard unseen Usagi variant |
| `unknown-0004` | Sloth plush | 12 | Easy unrelated plush |

## Results

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

Keep the global cosine threshold as the first open set baseline, but do not
treat it as sufficient for the application. The aggregate rate hides a major
failure on the most realistic difficult unknown category.

Compare score margin, neighbor agreement, prototype distance, and per identity
thresholds on the same validation set before freezing an operating rule.
Session 3 remains sealed during this comparison.

The
[executed walkthrough](../../notebooks/unknown_rejection_walkthrough.ipynb)
contains the complete score distributions, threshold curve, false accepts, and
known rejections.
