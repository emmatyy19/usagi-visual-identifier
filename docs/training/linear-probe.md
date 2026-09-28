# Linear Probe Training

The linear probe is the first supervised model in this project. DINOv2 remains
completely frozen. Its 384 dimensional embeddings become inputs to one affine
layer with seven outputs, one for each known doll identity.

The layer contains 2,695 trainable parameters:

```text
384 input values × 7 classes + 7 bias values
```

This experiment separates reusable training code from analysis. The source
package owns the model, data contract, optimization, evaluation, checkpointing,
and provenance. The notebook only reads completed artifacts and explains them.

## Data boundary

Session 1 supplies 98 training embeddings. Session 2 supplies 42 validation
embeddings. The training pipeline refuses to run if the input artifact contains
any Session 3 test embedding.

The validation loss selects the checkpoint. Validation accuracy is reported
but does not select between epochs. Session 3 remains sealed until every model
and threshold choice is frozen.

## Run it in the debugger

First build the frozen embedding artifact with only training and validation
splits. Then open Run and Debug in VS Code and select `Train Linear Probe`.

The committed launch uses:

| Setting | Value |
| --- | ---: |
| Seed | 42 |
| Maximum epochs | 300 |
| Batch size | 16 |
| Learning rate | 0.01 |
| Weight decay | 0.0001 |
| Early stopping patience | 40 |
| Minimum validation loss improvement | 0.00001 |
| Device | CPU |

CPU is intentional. The head is tiny, finishes in seconds, and does not benefit
meaningfully from accelerator overhead. The later partial DINOv2 tuning stage
is where the RTX 3060 becomes useful.

## Reproducibility

Each run seeds Python, NumPy, and PyTorch and requests deterministic Torch
operations. The data loader uses its own seeded generator. Class indices come
from sorted stable catalog IDs rather than CSV order.

The run ID is a digest of the complete training configuration, embedding
artifact hash, and dataset fingerprint. Repeating the same run resolves to the
same ID and reproduces the selected weights in the automated tests.

## Local artifacts

One run writes:

```text
artifacts/models/linear_probe/<run-id>/
├── checkpoint.pt
├── metadata.json
├── training_history.json
└── validation_report.json
```

The checkpoint contains only the selected model state. Its SHA256 digest is
stored in metadata and verified before loading. The metadata also records the
model revision, embedding provenance, class mapping, configuration, library
versions, selected epoch, selection metric, and whether test data was accessed.

Every JSON file and checkpoint is written through a temporary file and then
atomically replaced. The `latest.json` pointer is published last so readers do
not discover an incomplete run.

Generated training artifacts remain outside Git. The committed experiment
record contains the configuration, aggregate results, decision, and important
limitations.

## Adding another identity

Retrieval can add a new identity by embedding and indexing new references. The
linear probe cannot produce a new class until its output layer is expanded and
trained again. This fixed class requirement is the main operational tradeoff
for its direct probability output.

## Current result

Run `582a6d5094cc` selected epoch 278 using minimum validation loss. It reached
perfect Top 1, Top 3, and macro recall on all 42 Session 2 validation images.
Its validation cross entropy was `0.0356`.
The complete 300 epoch CPU training loop took about `0.365` seconds on the Mac.

Nearest image retrieval also reached perfect validation accuracy. The linear
probe therefore does not improve the primary accuracy metric. It does provide
a supervised baseline, direct class logits, and a useful training workflow.
