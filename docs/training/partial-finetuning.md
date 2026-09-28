# Partial DINOv2 Fine Tuning

This experiment trains the final DINOv2 transformer block, the final layer
normalization, and the seven class head. Every earlier DINOv2 parameter stays
frozen.

The training computer receives Session 1 and Session 2 only. The transfer
bundle does not contain Session 3 photographs, so the final test set cannot be
opened accidentally during training.

## What changes from the linear probe

The linear probe reads cached 384 dimensional embeddings. Partial fine tuning
must read the prepared images because gradients need to flow through the final
transformer block.

The diagram below follows the DINOv2 Small configuration used by this project.
Gray components are fixed. Yellow components contain weights updated by this
experiment.

```mermaid
flowchart LR
    A[Prepared RGB image] --> B[DINOv2 image processor]
    B --> C[14 by 14 pixel patches]
    C --> D[Patch embeddings plus CLS token<br/>and positional embeddings]
    D --> E[Transformer blocks 1 through 11]
    E --> F[Transformer block 12<br/>encoder.layer.11]
    F --> G[Final layer normalization]
    G --> H[CLS representation<br/>384 values]
    H --> I[L2 normalization]
    I --> J[Affine classification head<br/>384 inputs to 7 outputs]
    J --> K[Seven logits<br/>one per doll identity]
    K --> L[Softmax probabilities<br/>for reporting]

    classDef frozen fill:#e5e7eb,stroke:#6b7280,color:#111827
    classDef trained fill:#fcd34d,stroke:#92400e,color:#451a03,stroke-width:3px
    classDef operation fill:#dbeafe,stroke:#2563eb,color:#172554
    classDef output fill:#dcfce7,stroke:#15803d,color:#14532d

    class B,C,D,E frozen
    class F,G,J trained
    class A,H,I operation
    class K,L output
```

The image processor and patch creation are fixed operations with no weights to
train. The CLS token, positional embeddings, patch embeddings, and first eleven
transformer blocks keep their pretrained weights. They still participate in
every forward pass, but the optimizer does not change their parameters. The
final transformer block, final layer normalization, and affine classification
head are trained. L2 normalization and softmax have no learned weights.

The model calls its blocks `encoder.layer.0` through `encoder.layer.11`, so
`encoder.layer.11` is the twelfth and final block. DINOv2 Small uses 14 pixel
patches, 12 transformer blocks, and a hidden size of 384. These details come
from the pinned model configuration rather than assumptions in the training
code.

The existing linear probe initializes the classification head. This starts the
experiment from a validated decision layer instead of replacing it with random
weights.

## Controlled configurations

The repository contains three reviewed configurations.

1. `smoke.json` runs one epoch and writes to a disposable artifact root.
2. `no_augmentation.json` measures partial tuning without random image changes.
3. `conservative.json` adds small crops, rotations, and color changes.

The conservative profile intentionally excludes horizontal reflection, random
erasing, and aggressive crops. Those transforms could reverse tags or remove
small props that define the merchandise identity.

Both full configurations use a backbone learning rate of `0.00001` and a head
learning rate of `0.0005`. The lower backbone rate reduces the risk of damaging
the useful pretrained representation. AdamW, cosine learning rate decay,
gradient clipping, validation loss selection, and early stopping are applied.
CUDA mixed precision is enabled on the GPU computer.

I review the actual transformations before GPU training:

```bash
uv run python scripts/review_finetune_augmentations.py
```

The contact sheet is written to
`artifacts/augmentation_review/conservative.png`. I check that clothing,
headwear, props, embroidery, and tags remain recognizable in every variant.

## Private input transfer

The photo files and model artifacts remain ignored by Git. On the Mac, run:

```bash
cd /Users/emmatong/Projects/usagi-visual-identifier
git switch main
git pull --ff-only
uv sync --all-groups
uv run python scripts/export_training_bundle.py
```

The command prints a file under `artifacts/transfers`. The current dataset
produces:

```text
artifacts/transfers/training-inputs-27c73c3bc696.zip
```

This private archive is about 322 MB. It contains the 140 prepared Session 1
and Session 2 JPEGs, their filtered manifests, and the verified linear probe.
It contains no Session 3 image. I can move it with a private Google Drive
folder, a USB drive, or another private file transfer method. I should not
commit it to Git.

## Prepare the Windows GPU computer

These commands assume the repository is already cloned in WSL at
`~/usagi-visual-identifier`. I first update the code and environment:

```bash
cd ~/usagi-visual-identifier
git switch main
git pull --ff-only
uv sync --all-groups
```

I place the training ZIP somewhere Windows can access. For example, a file in
the Windows Downloads folder is usually available in WSL under
`/mnt/c/Users/<WindowsUser>/Downloads/`.

I import it with the real Windows user name and exact ZIP name:

```bash
uv run python scripts/import_transfer_bundle.py \
  /mnt/c/Users/<WindowsUser>/Downloads/training-inputs-27c73c3bc696.zip
```

The importer verifies every recorded file size and SHA256 hash, rejects unsafe
archive paths, confirms that only training and validation splits are present,
and refuses to replace a different local file.

## Check the GPU and inputs

```bash
uv run python scripts/check_finetune_readiness.py \
  --config configs/partial_finetune/conservative.json
```

The expected report includes:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
Training images: 98
Validation images: 42
Classes: 7
Test images selected: 0
Readiness check passed
```

The first model run downloads the pinned DINOv2 Small weights from Hugging
Face. Later runs reuse the local cache.

## Run a one epoch smoke test

```bash
uv run python scripts/train_partial_finetune.py \
  --config configs/partial_finetune/smoke.json \
  --artifact-root artifacts/smoke
```

The smoke result is deliberately stored outside the real experiment directory.
It verifies image loading, model download, forward and backward passes, CUDA
mixed precision, validation, and checkpoint publication.

## Run the predetermined experiment matrix

I use the same three seeds for both configurations. This makes the augmentation
comparison less dependent on one random batch order.

```bash
for seed in 42 123 2026; do
  uv run python scripts/train_partial_finetune.py \
    --config configs/partial_finetune/no_augmentation.json \
    --seed "$seed"
done

for seed in 42 123 2026; do
  uv run python scripts/train_partial_finetune.py \
    --config configs/partial_finetune/conservative.json \
    --seed "$seed"
done
```

Each run has an independent deterministic run ID. A completed run writes:

```text
artifacts/models/partial_finetune/<run-id>/trained_delta.safetensors
artifacts/models/partial_finetune/<run-id>/metadata.json
artifacts/models/partial_finetune/<run-id>/training_history.json
artifacts/models/partial_finetune/<run-id>/validation_report.json
```

The SafeTensors checkpoint contains only trained parameters. The frozen base is
reconstructed later from the pinned DINOv2 model revision.

If one run fails, completed seed directories remain valid. I rerun only the
failed command with its configuration and seed.

## Export every result

After all six full runs finish:

```bash
uv run python scripts/export_finetune_bundle.py --all
```

This creates:

```text
artifacts/transfers/partial-finetune-experiment.zip
```

I copy that ZIP back to the Mac with the same private transfer method.

## Import results on the Mac

From the Mac repository:

```bash
uv run python scripts/import_transfer_bundle.py \
  ~/Downloads/partial-finetune-experiment.zip
```

The imported runs appear under `artifacts/models/partial_finetune`. They are
ready for the Experiment 004 comparison notebook. The notebook will compare
all seeds with frozen retrieval and the linear probe before any Session 3
evaluation.

## What success means

Session 2 accuracy is already perfect for both frozen retrieval and the linear
probe. I will therefore compare more than accuracy:

1. Validation cross entropy
2. Probability margins, especially Lobster side views
3. Stability across the three seeds
4. The gap between training and validation loss
5. The effect of conservative augmentation
6. Training time and GPU memory context

A more complicated model is not automatically better. If partial fine tuning
only increases confidence or variance without producing a stable validation
benefit, frozen retrieval remains the preferred system.

## References

1. [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
2. [Official DINOv2 implementation from Meta Research](https://github.com/facebookresearch/dinov2)
3. [DINOv2 Small model configuration](https://huggingface.co/facebook/dinov2-small/blob/main/config.json)
4. [DINOv2 documentation in Transformers](https://huggingface.co/docs/transformers/model_doc/dinov2)
