# Experiment 003: Frozen DINOv2 Linear Probe

## Status

Complete validation experiment

## Question

Can a supervised linear head over frozen DINOv2 embeddings improve or match the
nearest image retrieval baseline on independent Session 2 photographs?

## Hypothesis

The linear probe will reach high validation accuracy because the frozen
embeddings already separate the identities. It may produce smoother identity
boundaries, but it is unlikely to beat the perfect nearest image validation
accuracy.

## Baseline

Nearest image aggregation over the 98 image Session 1 gallery. It achieved
perfect Top 1, Top 3, and macro recall on Session 2.

## Dataset

1. Manifest fingerprint:
   `27c73c3bc69640bb10b36f4846270847dea4fe871cedeee516b7f58badd9a634`
2. Known identities: 7
3. Training: 98 Session 1 embeddings, 14 per identity
4. Validation: 42 Session 2 embeddings, 6 per identity
5. Test: not accessed
6. Unknown identities: not measured

## Configuration

1. Encoder: frozen `facebook/dinov2-small`
2. Encoder revision: `ed25f3a31f01632728cabb09d1542f84ab7b0056`
3. Input: normalized 384 dimensional CLS embedding
4. Head: one affine layer with 2,695 trainable parameters
5. Loss: cross entropy
6. Optimizer: AdamW
7. Learning rate: `0.01`
8. Weight decay: `0.0001`
9. Batch size: 16
10. Maximum epochs: 300
11. Selection: minimum validation loss
12. Early stopping patience: 40 epochs
13. Minimum improvement: `0.00001`
14. Seed: 42
15. Hardware: Mac CPU

## Protocol

The head trained only on Session 1. Session 2 was evaluated after every epoch
and selected the checkpoint with the lowest validation loss. The stable catalog
IDs were sorted to create the class mapping. Python, NumPy, PyTorch, and the
data loader were seeded.

The training code rejects any embedding artifact containing the test split.
Session 3 was not embedded, loaded, or evaluated.

## Success criteria

The candidate should match or improve nearest image Top 1 accuracy without a
large per identity regression. The run must reproduce from one debugger entry,
record its full configuration and provenance, and load only a hash verified
checkpoint.

## Results

| Metric | Nearest image | Linear probe | Difference |
| --- | ---: | ---: | ---: |
| Top 1 accuracy | 1.000 | 1.000 | 0.000 |
| Top 3 accuracy | 1.000 | 1.000 | 0.000 |
| Macro recall | 1.000 | 1.000 | 0.000 |
| Incorrect queries | 0 | 0 | 0 |

The selected checkpoint was epoch 278 of 300. Training loss was `0.0083`,
validation loss was `0.0356`, and every identity had validation recall of
`1.000`. The complete CPU training loop took `0.365` seconds.

## Confidence analysis

The lowest probability margin belonged to `img-000138`, a Lobster claws side
view. The probe assigned Lobster claws probability `0.5972` and Chiikawa Park
fluffy probability `0.2082`. The prediction was correct but meaningfully less
decisive than the rest.

The next lowest margin was another Lobster claws view, `img-000137`, which was
closest to Matcha ice cream. This agrees with the frozen retrieval analysis:
Lobster side views remain the clearest known identity weakness.

Mean top probability was `0.9681`. A five bin reliability check produced an
expected calibration error of `0.0319`. With only 42 queries and no mistakes,
this is a descriptive diagnostic rather than evidence that the probabilities
are calibrated for new data or unknown dolls.

The validation loss began increasing slightly after epoch 278 while training
loss continued falling. Selecting epoch 278 preserved the best observed
validation loss instead of automatically keeping the final weights.

## Conclusion

The linear probe matches but does not improve the frozen retrieval baseline on
the primary validation metrics. It remains valuable as a reproducible
supervised baseline and as hands on experience with logits, softmax, cross
entropy, optimization, model selection, and checkpoint provenance.

Nearest image remains the preferred product baseline because it supports new
identities without retraining and provides direct reference evidence. The
linear probe remains the comparison model for partial fine tuning.

## Follow up

1. Collect and evaluate unknown validation examples.
2. Run the linear probe walkthrough notebook.
3. Define conservative identity preserving augmentations.
4. Partially tune only the final DINOv2 transformer block.
5. Keep Session 3 sealed until all choices are frozen.
