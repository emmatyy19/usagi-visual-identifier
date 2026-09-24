# Evaluation Plan

## Principles

- Measure generalization across capture sessions, not memorization of adjacent
  frames.
- Compare every trained system with the frozen-encoder retrieval baseline.
- Choose models, aggregation rules, and thresholds using validation data only.
- Keep the final test set untouched until the experiment configuration is
  frozen.
- Separate known-item ranking from unknown-item rejection.
- Accompany aggregate metrics with per-identity and qualitative failure review.

## Evaluation units

The primary unit is a query image. Because some identities or sessions may have
more images than others, report both micro-averaged results over queries and
macro-averaged results giving equal weight to each identity.

When multiple related queries come from one source photo or short burst, also
report session-level summaries so repeated frames do not dominate conclusions.

## Closed-set retrieval metrics

Report at least:

- **Top-1 accuracy:** fraction of known queries whose first-ranked identity is
  correct.
- **Top-3 accuracy:** fraction whose correct identity appears in the first three.
- **Macro recall:** mean per-identity top-1 recall.
- **Mean reciprocal rank:** rewards placing the correct identity near the top.
- **Confusion pairs:** identity pairs responsible for repeated errors.

If retrieval ranks individual reference images, convert them to an identity
ranking before calculating identity metrics. Document the aggregation rule.

## Classification metrics

For linear-probe and fine-tuned classifiers, report:

- top-1 and top-3 accuracy;
- macro precision, recall, and F1;
- confusion matrix;
- negative log likelihood; and
- expected calibration error with the binning rule documented.

Classification results do not replace retrieval evaluation when the intended
user workflow remains retrieval-based.

## Open-set metrics

At the selected operating point, report:

- known-query acceptance rate;
- unknown-query false-acceptance rate;
- accepted-known identification accuracy;
- false rejections of known queries; and
- false acceptances broken down by unknown difficulty.

Also plot or tabulate the tradeoff across thresholds. AUROC can summarize score
separation but must not replace an explicit operating point.

The threshold-selection objective will be frozen after inspecting validation
behavior. A reasonable first objective is to minimize unknown false accepts
subject to retaining an agreed fraction of correctly identified known queries.

## Catalog-growth evaluation

To test whether retrieval supports new identities:

1. hold one or more catalog identities out of model training;
2. add only a small reference gallery for those identities after training;
3. query with independent sessions from the held-out identities; and
4. measure recall without updating model weights.

This directly evaluates the claimed advantage over a closed classifier.

## Robustness slices

Where metadata permits, break results down by:

- capture session;
- lighting type;
- viewing angle;
- background;
- crop tightness or doll scale;
- partial occlusion;
- accessory size; and
- identities with near-duplicate appearance.

Slices are diagnostic when small; avoid presenting noisy subgroup percentages
as precise population estimates.

## Efficiency

Record:

- encoder parameter count;
- trainable parameter count;
- peak GPU memory during training;
- training wall time;
- single-image inference latency on GPU and CPU;
- gallery-index build time; and
- serialized model and index sizes.

## Comparison matrix

Every major experiment should preserve a table like:

| System | Trainable parts | Top-1 | Top-3 | Macro recall | Unknown false accept | Latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Frozen DINOv2 + nearest image | None | TBD | TBD | TBD | TBD | TBD |
| Frozen DINOv2 + prototypes | None | TBD | TBD | TBD | TBD | TBD |
| Linear probe | Head | TBD | TBD | TBD | TBD | TBD |
| Partial fine-tune | Head + final block | TBD | TBD | TBD | TBD | TBD |
| Metric learning | Projection + selected blocks | TBD | TBD | TBD | TBD | TBD |

## Promotion rules

A more complex model replaces the current preferred system only if it:

1. improves the predeclared primary validation metric across independent
   sessions;
2. does not materially worsen unknown false acceptance at the chosen known
   acceptance level;
3. improves more than one identity rather than exploiting class imbalance;
4. remains practical on the available hardware; and
5. survives qualitative review for background or shortcut behavior.

Final test metrics confirm a frozen decision; they do not trigger another round
of tuning on the same test set.

## Failure review

For every major experiment, review representative examples of:

- correct high-confidence matches;
- correct low-confidence matches;
- wrong high-confidence matches;
- known queries rejected as unknown;
- unknown queries falsely accepted; and
- errors resolved or introduced relative to the previous baseline.

The public experiment record should describe the visible pattern without
publishing private source images unless explicitly approved.

