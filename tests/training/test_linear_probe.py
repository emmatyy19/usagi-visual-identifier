"""Tests for deterministic linear probe training and artifacts."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from usagi_visual_identifier.embeddings import EmbeddingArtifact
from usagi_visual_identifier.training import (
    LinearProbeConfig,
    LinearProbeError,
    load_linear_probe,
    prepare_linear_probe_data,
    train_linear_probe,
)


def _artifact() -> EmbeddingArtifact:
    """Create three perfectly separable identities across two sessions."""
    base = np.eye(3, dtype=np.float32)
    embeddings = np.asarray(
        [
            base[0],
            base[0] * 0.95 + 0.05,
            base[1],
            base[1] * 0.95 + 0.05,
            base[2],
            base[2] * 0.95 + 0.05,
            base[0],
            base[1],
            base[2],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return EmbeddingArtifact(
        image_ids=tuple(f"img-{index:06d}" for index in range(1, 10)),
        identity_ids=(
            "usagi-0001",
            "usagi-0001",
            "usagi-0002",
            "usagi-0002",
            "usagi-0003",
            "usagi-0003",
            "usagi-0001",
            "usagi-0002",
            "usagi-0003",
        ),
        capture_session_ids=(
            "session-001",
            "session-001",
            "session-001",
            "session-001",
            "session-001",
            "session-001",
            "session-002",
            "session-002",
            "session-002",
        ),
        splits=(
            "train",
            "train",
            "train",
            "train",
            "train",
            "train",
            "validation",
            "validation",
            "validation",
        ),
        processed_relative_paths=tuple(f"images/{index}.jpg" for index in range(9)),
        embeddings=embeddings,
        metadata={
            "model_id": "example/model",
            "model_revision": "commit",
            "artifact_sha256": "embedding-hash",
            "dataset_fingerprint": "dataset-hash",
        },
    )


def _config() -> LinearProbeConfig:
    """Return a fast deterministic test configuration."""
    return LinearProbeConfig(
        seed=7,
        epochs=100,
        batch_size=3,
        learning_rate=0.05,
        patience=15,
        device="cpu",
    )


def test_training_is_reproducible_and_loadable(tmp_path: Path) -> None:
    """The same inputs produce the same selected weights and complete reports."""
    first = train_linear_probe(
        artifact=_artifact(), config=_config(), artifact_root=tmp_path
    )
    first_model, first_metadata = load_linear_probe(
        checkpoint_path=first.checkpoint_path,
        metadata_path=first.metadata_path,
    )
    first_state = {
        name: value.detach().clone() for name, value in first_model.state_dict().items()
    }
    second = train_linear_probe(
        artifact=_artifact(), config=_config(), artifact_root=tmp_path
    )
    second_model, second_metadata = load_linear_probe(
        checkpoint_path=second.checkpoint_path,
        metadata_path=second.metadata_path,
    )

    assert first.run_id == second.run_id
    assert first.selected_epoch == second.selected_epoch
    assert first.validation_top1_accuracy == 1.0
    assert first.validation_top3_accuracy == 1.0
    assert first_metadata["test_split_accessed"] is False
    assert first_metadata["trainable_parameter_count"] == 12
    assert first_metadata["selected_epoch"] == second_metadata["selected_epoch"]
    for name, value in second_model.state_dict().items():
        assert torch.equal(first_state[name], value)

    history = json.loads(first.history_path.read_text(encoding="utf-8"))
    report = json.loads(first.validation_report_path.read_text(encoding="utf-8"))
    assert history["epochs"][0]["validation_loss"] > report["loss"]
    assert report["query_count"] == 3
    assert len(report["queries"][0]["ranked_identities"]) == 3


def test_training_rejects_test_embeddings_and_mismatched_classes() -> None:
    """Sealed rows and incompatible split labels cannot enter training."""
    artifact = _artifact()
    with_test = replace(
        artifact,
        splits=(*artifact.splits[:-1], "test"),
    )
    with pytest.raises(LinearProbeError, match="sealed test"):
        prepare_linear_probe_data(artifact=with_test)

    mismatched = replace(
        artifact,
        identity_ids=(*artifact.identity_ids[:-1], "usagi-9999"),
    )
    with pytest.raises(LinearProbeError, match="must match exactly"):
        prepare_linear_probe_data(artifact=mismatched)


def test_checkpoint_hash_detects_tampering(tmp_path: Path) -> None:
    """A modified checkpoint cannot be loaded under unchanged metadata."""
    result = train_linear_probe(
        artifact=_artifact(), config=_config(), artifact_root=tmp_path
    )
    result.checkpoint_path.write_bytes(b"not a checkpoint")

    with pytest.raises(LinearProbeError, match="hash does not match"):
        load_linear_probe(
            checkpoint_path=result.checkpoint_path,
            metadata_path=result.metadata_path,
        )
