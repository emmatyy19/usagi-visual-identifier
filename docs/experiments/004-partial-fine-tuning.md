# Experiment 004: Partial DINOv2 Fine Tuning

## Status

Training pipeline ready. GPU results pending.

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

Pending the RTX 3060 runs.

## Decision

Pending comparison across all six runs. Session 3 will not be opened as part
of this experiment.
