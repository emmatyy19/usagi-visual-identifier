# Experiment Records

This directory documents meaningful modeling experiments, including negative
results. Records make decisions auditable without committing original images,
embeddings, checkpoints, or machine specific output. A record may include
reviewed photographs that I explicitly approved for publication.

Name records sequentially:

```text
001-frozen-dinov2-session-001.md
002-frozen-dinov2-session-002-validation.md
003-linear-probe.md
004-partial-fine-tuning.md
005-unknown-rejection-validation.md
005b-final-known-test.md
```

Copy [`template.md`](template.md) when beginning an experiment. Complete the
protocol and success criteria before examining final results. Update the record
afterward with aggregate metrics, failure analysis, conclusions, and follow-up
questions.

An experiment record should be reproducible from committed code and local data
matching the recorded manifest fingerprint. Never paste private paths, personal
metadata, third-party copyrighted images, credentials, or raw listings into a
public record.
