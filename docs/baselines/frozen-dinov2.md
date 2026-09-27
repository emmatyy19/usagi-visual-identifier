# Frozen DINOv2 Baseline

This baseline asks how useful a general pretrained vision model is before any
training on the doll collection. It uses DINOv2 Small as a frozen feature
extractor. No model parameter changes during this experiment.

## Run it in the debugger

First run the `Prepare Session 1` launch so the processed manifest and JPEG
files are current. Then open Run and Debug in VS Code and select
`Build DINOv2 Embedding Baseline`.

The first run downloads the pinned public model from Hugging Face. Later runs
reuse the local Hugging Face cache. Model weights are not stored in this
repository.

The launch uses automatic device selection:

1. CUDA is preferred when an NVIDIA GPU is available.
2. Apple Metal is used on a compatible Mac.
3. CPU is the final fallback.

The default batch size is eight. It is intentionally conservative for both the
Mac and the RTX 3060.

The default artifact includes the training and validation splits only. The
debugger explicitly sets `USAGI_ALLOW_FINAL_TEST` to `false`, so rerunning the
builder cannot encode Session 3 by accident. The final evaluation will require
changing both the included splits and this deliberate override after every
model and threshold choice is frozen.

## What the script saves

The script writes the following ignored local artifacts:

```text
artifacts/embeddings/facebook_dinov2-small/embeddings.npz
artifacts/embeddings/facebook_dinov2-small/metadata.json
artifacts/evaluations/dinov2_small/session_001_leave_one_out.json
```

The embedding artifact aligns each normalized vector with its image ID,
identity ID, capture session, split, and processed path. The metadata records
the pinned model revision, selected device, dataset fingerprint, dimensions,
and artifact hash.

## Session 1 sanity check

Every Session 1 image becomes a query once. Its exact row is removed from the
gallery, and the remaining images are ranked by cosine similarity. Each
identity receives the score of its nearest reference image.

This check confirms that the extraction and retrieval pipeline works. It is not
evidence of generalization because the queries and gallery share the same
capture session. Session 2 will provide the first meaningful validation set.

## Explore the result visually

Open
[`notebooks/embedding_baseline_walkthrough.ipynb`](../../notebooks/embedding_baseline_walkthrough.ipynb)
after building the local artifacts. Run all cells to inspect sample embeddings,
cosine similarity, pair distributions, image and identity heatmaps, aggregate
metrics, the confusion matrix, and every incorrect query beside its nearest
wrong reference.

The committed notebook includes the executed outputs and all 98 reviewed
Session 1 photographs. This makes the complete baseline visible without the
local model artifacts. The original HEIC files and the standalone prepared
dataset remain outside Git.
