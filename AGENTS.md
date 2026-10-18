# Agent Guidelines

These instructions apply to the entire repository.

## Project purpose

This is a learning-focused computer-vision project for identifying specific
Chiikawa Usagi merchandise variants from photographs and linking confident
matches to structured catalog metadata.

## Working conventions

- Keep the first implementation local, small, and measurable.
- Establish a frozen pretrained-embedding baseline before fine-tuning.
- Treat visual identification and changing commerce data as separate systems.
- Put reusable code under `src/usagi_visual_identifier/`; notebooks are for
  exploration and must not become application dependencies.
- Record meaningful experiments under `docs/experiments/` using the template in
  that directory. Include failures and negative results.
- Update public documentation when evidence changes a design decision.
- Never commit private collection notes, purchase history, credentials, raw
  photographs, downloaded marketplace images, model weights, or generated
  experiment artifacts.
- Only commit images whose redistribution rights are understood. Prefer tiny
  invented fixtures for automated tests.
- `site/` contains the owner approved public collection snapshot and resized
  cover photos. Publish only explicitly reviewed fields and images; never export
  private database records or notes automatically into the site.

## Writing style

Write public documentation in a direct, natural, first person voice that sounds
like the repository owner. Prefer short sentences and straightforward wording.
Avoid em dashes, en dashes, and hyphens used as sentence punctuation. Avoid
hyphenated wording when a normal alternative is clear. Hyphens required by
commands, package names, identifiers, and file paths are allowed.

## Evaluation discipline

- Split photographs by capture session, not by randomly splitting adjacent
  images or frames.
- Preserve a final test set until model and threshold choices are frozen.
- Evaluate closed-set retrieval and unknown-item rejection separately.
- Report top-1 and top-3 retrieval accuracy, confusion pairs, latency, and
  open-set threshold behavior; do not present accuracy alone.
- Compare every trained model against the frozen-backbone retrieval baseline.

## Git workflow

- Keep `main` stable and use focused branches such as `docs/<topic>`,
  `feat/<topic>`, and `experiment/<topic>`.
- Do not stage, commit, push, or create a pull request without explicit user
  authorization.
- Preserve unrelated user changes and keep commits focused.

## Python toolchain

- Use Python 3.12 and manage dependencies exclusively with `uv`.
- Centralize package and tool configuration in `pyproject.toml`.
- Use Ruff for formatting and linting, Pyright in strict mode for type checking,
  and pytest for deterministic tests.
- Fully type reusable functions and keep data transforms shared between
  training and inference.

## Required verification

Run these commands before considering a change complete:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
uv build
```
