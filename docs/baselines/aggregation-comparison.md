# Frozen Retrieval Aggregation Comparison

This comparison asks how several reference photographs should become one score
for a catalog identity. Every method uses the same frozen DINOv2 embeddings and
the same cosine similarities. No model weight changes.

## Methods

### Nearest image

The identity score is its strongest reference image similarity. This method can
match a specific viewpoint well, but one misleading reference can decide the
result.

### Identity prototype

All reference embeddings for one identity are averaged and normalized. The
query is compared with that single prototype. This method is stable and cheap,
but averaging may blur useful differences between front, side, and rear views.

### Top k mean

The strongest few reference similarities for each identity are averaged. This
requires support from more than one photograph without forcing every viewpoint
into one prototype. The default reference count is three.

### Weighted neighbors

The strongest references across the complete gallery vote for their identities.
A softmax gives more weight to references with higher similarity. The default
uses five neighbors and a temperature of `0.07`.

This score is vote mass between zero and one. It is not a cosine similarity and
must not share an unknown rejection threshold with the other methods.

## Run it in the debugger

First build the DINOv2 embedding artifact. Then open Run and Debug in VS Code
and select `Compare Retrieval Aggregations`.

The debugger launch compares the Session 1 training gallery with Session 2
validation queries. It writes:

```text
artifacts/evaluations/dinov2_small/aggregation_comparison_train_to_validation.json
```

The report is a local generated artifact and is excluded from Git. The Session
1 implementation check remains available by setting both splits to `train`.

## Explore it visually

The aggregation section in the
[`embedding_baseline_walkthrough.ipynb`](../../notebooks/embedding_baseline_walkthrough.ipynb)
notebook compares the aggregate metrics, per identity accuracy, prediction
disagreements, changed errors, reference evidence, weighted votes, and prototype
construction. It also includes all 42 Session 2 photographs, the validation
metrics, score and margin distributions, and the closest correct decisions.

## Session 1 implementation check

The first run produced these same session results:

| Method | Top 1 | Top 3 | Macro recall | MRR |
| --- | ---: | ---: | ---: | ---: |
| Nearest image | 0.918 | 0.980 | 0.918 | 0.953 |
| Identity prototype | 0.980 | 1.000 | 0.980 | 0.990 |
| Top k mean | 0.969 | 1.000 | 0.969 | 0.983 |
| Weighted neighbors | 0.929 | 0.980 | 0.929 | 0.958 |

The prototype result is encouraging, but it is not evidence that prototypes
will work best on new photographs. Every query and reference still came from
Session 1. This table verifies the comparison pipeline and records its behavior
before independent validation data exists.

## How to interpret the current run

Session 1 can confirm that every method is implemented correctly and expose
obvious behavior differences. It cannot select the preferred method because
the gallery and queries share one capture session. The report records this as
an exploratory same split comparison and sets `selection_allowed` to `false`.

## Session 2 validation result

The Session 1 gallery contains 98 images. Session 2 contributes 42 independent
validation queries, six for each identity.

| Method | Top 1 | Top 3 | Macro recall | MRR |
| --- | ---: | ---: | ---: | ---: |
| Nearest image | 1.000 | 1.000 | 1.000 | 1.000 |
| Identity prototype | 1.000 | 1.000 | 1.000 | 1.000 |
| Top k mean | 1.000 | 1.000 | 1.000 | 1.000 |
| Weighted neighbors | 1.000 | 1.000 | 1.000 | 1.000 |

All methods tie on every primary metric. Nearest image remains the preferred
frozen baseline because it is the simplest tied rule and preserves a direct
reference image for every prediction. Several Lobster and Airport side views
have narrow first versus second place margins, so future capture sessions
should continue to include these difficult views.

This is not a universal 100 percent accuracy claim. The validation set is
small, contains only known identities, and shares one Session 2 setup. Unknown
rejection still needs its own validation examples and operating threshold.
Session 3 remains sealed until all model and threshold choices are frozen.

## Evidence in each result

Each ranked identity includes:

1. Its aggregate score.
2. The reference image IDs that contributed to the score.
3. The original cosine similarity for each contributing reference.

This makes the result inspectable even when the aggregate score is an average
or weighted vote rather than one image similarity.
