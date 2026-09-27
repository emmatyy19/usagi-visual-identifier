"""Identify one doll photograph with the frozen DINOv2 gallery."""

import json
import os
import time
from pathlib import Path

from usagi_visual_identifier.data.loaders import load_catalog_csv
from usagi_visual_identifier.embeddings import (
    DEFAULT_DINOV2_MODEL_ID,
    DEFAULT_DINOV2_REVISION,
    Dinov2Encoder,
    FrozenRetrievalPredictor,
    load_embedding_artifact,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
EMBEDDING_ROOT = ARTIFACT_ROOT / "embeddings" / "facebook_dinov2-small"


def _nonnegative_integer(name: str, default: int) -> int:
    """Read one nonnegative integer from the environment."""
    value = int(os.environ.get(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def main() -> None:
    """Load the frozen predictor once, identify one image, and optionally benchmark."""
    query_value = os.environ.get("USAGI_QUERY_IMAGE", "").strip()
    if not query_value:
        raise ValueError("USAGI_QUERY_IMAGE must point to one photo")
    query_path = Path(query_value).expanduser().resolve()
    device = os.environ.get("USAGI_DEVICE", "auto")
    warmup_runs = _nonnegative_integer("USAGI_LATENCY_WARMUP", 0)
    measured_runs = _nonnegative_integer("USAGI_LATENCY_RUNS", 0)

    artifact = load_embedding_artifact(
        artifact_path=EMBEDDING_ROOT / "embeddings.npz",
        metadata_path=EMBEDDING_ROOT / "metadata.json",
    )
    catalog = load_catalog_csv(path=PROJECT_ROOT / "data/private/catalog.csv")
    catalog_names = {item.catalog_id: item.display_name for item in catalog.items}

    model_started = time.perf_counter()
    encoder = Dinov2Encoder(
        model_id=DEFAULT_DINOV2_MODEL_ID,
        model_revision=DEFAULT_DINOV2_REVISION,
        requested_device=device,
    )
    predictor = FrozenRetrievalPredictor(artifact=artifact, encoder=encoder)
    model_load_seconds = time.perf_counter() - model_started
    prediction = predictor.predict(path=query_path, top_k=3)

    print(f"Device: {predictor.device_name}")
    print(f"Gallery images: {predictor.gallery_size}")
    print(f"Model startup: {model_load_seconds:.3f} seconds")
    print(f"First prediction: {prediction.total_seconds:.4f} seconds")
    print("Rank  Identity                 Score    Best reference")
    for rank, match in enumerate(prediction.matches, start=1):
        name = catalog_names.get(match.identity_id, match.identity_id)
        reference = match.reference_image_ids[0]
        print(f"{rank:<5} {name:<24} {match.score:>6.4f}   {reference}")

    if measured_runs == 0:
        return
    summary = predictor.benchmark(
        path=query_path,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )
    report = {
        "device": predictor.device_name,
        "gallery_image_count": predictor.gallery_size,
        "model_load_seconds": model_load_seconds,
        "warmup_runs": summary.warmup_runs,
        "measured_runs": summary.measured_runs,
        "mean_seconds": summary.mean_seconds,
        "median_seconds": summary.median_seconds,
        "p95_seconds": summary.p95_seconds,
        "minimum_seconds": summary.minimum_seconds,
        "maximum_seconds": summary.maximum_seconds,
    }
    report_path = ARTIFACT_ROOT / "benchmarks/dinov2_small/single_image_latency.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Warm latency over {summary.measured_runs} runs: "
        f"median {summary.median_seconds:.4f}s, "
        f"mean {summary.mean_seconds:.4f}s, "
        f"p95 {summary.p95_seconds:.4f}s"
    )
    print(f"Benchmark report: {report_path}")


if __name__ == "__main__":
    main()
