# Experiment 004: Partial DINOv2 Fine Tuning

## Status

Complete on the RTX 3060. Session 3 remains sealed.

## Question

Does training only the final DINOv2 block improve validation loss, difficult
view margins, or stability compared with frozen retrieval and the linear
probe?

## Hypothesis

Partial tuning may adapt the representation to small merchandise details, but
the dataset is small and both existing methods already have perfect validation
accuracy. Conservative augmentation may reduce overfitting. A stable reduction
in validation loss or improvement in the Lobster side views would be stronger
evidence than higher confidence alone.

## Frozen protocol

1. Session 1 contains 98 training images.
2. Session 2 contains 42 validation images.
3. Session 3 remains sealed and is absent from the GPU transfer bundle.
4. DINOv2 Small stays pinned to revision
   `ed25f3a31f01632728cabb09d1542f84ab7b0056`.
5. Only the final transformer block, final layer normalization, and classifier
   are trainable.
6. The classifier starts from the selected linear probe.
7. Validation loss selects the checkpoint.
8. No augmentation and conservative augmentation each use seeds `42`, `123`,
   and `2026`.

## Artifacts

The training code records the configuration, software and CUDA versions, GPU
name, dataset fingerprint, class mapping, trainable parameter names, selected
epoch, metrics, and checkpoint hash. The checkpoint uses SafeTensors and stores
only the trained delta.

## Results

All six runs reached perfect Session 2 top 1 accuracy, top 3 accuracy, and macro
recall. Accuracy therefore remains at the same ceiling as frozen nearest image
retrieval and the linear probe.

| Augmentation | Seed | Selected epoch | Validation loss | AMP skips |
| --- | ---: | ---: | ---: | ---: |
| None | 42 | 26 | 0.006019 | 0 |
| None | 123 | 3 | 0.006266 | 0 |
| None | 2026 | 15 | 0.005924 | 0 |
| Conservative | 42 | 22 | 0.002307 | 1 |
| Conservative | 123 | 9 | 0.004439 | 0 |
| Conservative | 2026 | 19 | 0.003471 | 1 |

Mean selected validation loss was `0.006070` without augmentation and
`0.003406` with conservative augmentation. This is a reduction of about 44
percent. Conservative augmentation produced the lower loss for every matched
seed.

The most difficult validation photograph remained `img-000138`, a side view of
Lobster claws. Its mean first versus second probability margin increased from
`0.709` without augmentation to `0.862` with augmentation. The margin improved
for all three seeds. The
[executed walkthrough](../../notebooks/partial_finetune_walkthrough.ipynb)
contains the complete curves, tables, and qualitative review.

The first CUDA smoke attempt exposed an overly strict safety check. PyTorch AMP
correctly skipped a batch with overflowing FP16 gradients and reduced its
dynamic scale, but the training loop treated that recovered overflow as fatal.
The corrected loop records skipped steps and continues after a valid update. It
still fails if an entire epoch produces no valid optimizer step.

## Decision

Use conservative augmentation for the partial fine tuning candidate. The result
is stable across the three planned seeds and improves the known difficult view.
This does not yet prove that partial fine tuning is the preferred production
model because the validation set contains only 42 photographs. Freeze one
checkpoint selection rule before comparing frozen retrieval, the linear probe,
and partial fine tuning on Session 3.
