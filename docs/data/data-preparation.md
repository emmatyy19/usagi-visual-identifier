# Data Preparation

This step turns private phone originals into images that common Python vision
tools can read consistently. The originals remain unchanged. Every generated
file stays local because both `data/private/` and `data/processed/` are ignored
by Git.

## Prepare a session in the debugger

Open the Run and Debug view in VS Code. Select `Prepare Session 1`, then start
debugging. The repository includes three launch choices:

1. `Prepare Session 1` assigns the session to training.
2. `Prepare Session 2` assigns the session to validation.
3. `Prepare Session 3` assigns the session to testing.

Each launch choice runs `scripts/prepare_data.py` with the matching session and
split. Put a breakpoint in that script or in
`src/usagi_visual_identifier/data/preparation.py` to follow the work step by
step. `justMyCode` is disabled so the debugger can enter the reusable pipeline.

The debug script performs the following work:

1. It reads and validates `data/private/catalog.csv`.
2. It discovers the requested session beneath each catalog identity.
3. It creates or updates `data/private/images.csv` without changing old IDs.
4. It decodes HEIC, JPEG, and PNG originals and corrects stored orientation.
5. It writes RGB JPEG copies beneath `data/processed/images/`.
6. It hashes every original and prepared file.
7. It creates one contact sheet for each identity and session.
8. It writes an audit summary with counts and the dataset fingerprint.

Running the same launch again is safe. Existing image and source IDs remain
stable. When a later session is added, it receives the next available IDs.

## Review the result

Open `data/processed/audit/summary.json` and confirm the expected count for
every identity. Then review the contact sheets beneath
`data/processed/audit/<session>/`.

Check that each sheet contains only the intended doll, that orientation is
correct, and that no private information is visible. A successful command
means the files are readable and structurally valid. It does not replace this
visual review.

## Generated files

```text
data/private/images.csv
data/processed/images.csv
data/processed/images/<catalog ID>/<session>/<photo>.jpg
data/processed/audit/<session>/<catalog ID>.jpg
data/processed/audit/summary.json
```

The private manifest describes the original files. The processed manifest
connects each original to its JPEG derivative, hashes, and dimensions. The
dataset fingerprint changes whenever the catalog or original manifest changes.
