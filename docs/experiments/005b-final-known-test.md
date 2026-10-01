# Experiment 005b: Final Known Test

## Status

Complete on Apple Metal. Session 3 has been opened once and is no longer
available for model selection.

## Question

How well do the frozen systems identify the seven catalog dolls in a third
independent capture session, and how many correct predictions survive the
validation selected rejection thresholds?

## Frozen protocol

1. Session 1 supplies the 98 reference photographs.
2. Session 3 supplies 42 final known test photographs, six per identity.
3. The checkpoint, aggregation methods, and thresholds were written to
   `configs/open_set/final_known_test.json` before Session 3 was accessed.
4. No model or threshold selection is allowed from this result.
5. There is no independent unknown final test. All unknown rejection numbers
   remain validation results from Session 4.

## Results

| System | Top 1 | Top 3 | Correct accepted | Known acceptance |
| --- | ---: | ---: | ---: | ---: |
| Frozen DINOv2 nearest image | 41 of 42 | 42 of 42 | 33 of 42 | 78.6% |
| Tuned DINOv2 nearest image | 41 of 42 | 42 of 42 | 35 of 42 | 83.3% |
| Tuned DINOv2 prototype | 40 of 42 | 42 of 42 | 34 of 42 | 81.0% |
| Tuned classifier probability margin | 40 of 42 | 42 of 42 | 38 of 42 | 90.5% |

Every accepted prediction was correct. The thresholds therefore behaved
conservatively, but none retained the 95 percent known acceptance target on the
new capture session. This is the main final test finding. The validation
thresholds did not transfer as well as raw identity ranking.

Frozen and tuned nearest image retrieval made the same single error. Pompompurin
image `img-000170` was predicted as Chiikawa Park fluffy. Prototype retrieval
and the classifier also missed that photograph, and both additionally predicted
Airport image `img-000152` as Matcha ice cream. All correct identities still
appeared in the top three.

## Decision

Nearest image retrieval remains the preferred identification baseline. Partial
fine tuning did not improve final Top 1 accuracy, although tuned nearest image
accepted two more correct photos under its own frozen threshold.

The classifier probability margin retained the most known photos, but its raw
Top 1 result was lower than nearest image retrieval. The zero unknown false
accepts seen on Session 4 remain validation evidence only. Without an
independent unknown test collection, this project does not claim reliable
unknown rejection on new Usagi variants.

The experiment is complete without collecting third party social media images.
Independent unknown testing remains future work if suitable owner captured or
permission granted images become available later.
