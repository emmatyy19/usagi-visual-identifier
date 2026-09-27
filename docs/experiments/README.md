# Experiment Records

This directory documents meaningful modeling experiments, including negative
results. Records make decisions auditable without committing original images,
embeddings, checkpoints, or machine specific output. A record may include
reviewed photographs that I explicitly approved for publication.

Name records sequentially:

```text
001-frozen-dinov2-retrieval.md
002-linear-probe.md
003-partial-fine-tuning.md
```

Copy [`template.md`](template.md) when beginning an experiment. Complete the
protocol and success criteria before examining final results. Update the record
afterward with aggregate metrics, failure analysis, conclusions, and follow-up
questions.

An experiment record should be reproducible from committed code and local data
matching the recorded manifest fingerprint. Never paste private paths, personal
metadata, third-party copyrighted images, credentials, or raw listings into a
public record.
