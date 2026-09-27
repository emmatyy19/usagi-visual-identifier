"""Tests for embedding artifact construction and loading."""

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from usagi_visual_identifier.data.models import DatasetSplit
from usagi_visual_identifier.embeddings.artifacts import (
    EmbeddingError,
    build_embedding_artifact,
    load_embedding_artifact,
    load_prepared_examples,
    select_embedding_examples,
)


class PreparedFixture(Protocol):
    """Paths supplied by the shared prepared dataset fixture."""

    private_root: Path
    processed_root: Path


class MeanColorEncoder:
    """Small deterministic encoder used instead of a downloaded model."""

    @property
    def model_id(self) -> str:
        """Return an invented model ID."""
        return "test/mean-color"

    @property
    def model_revision(self) -> str:
        """Return an invented revision."""
        return "revision-1"

    @property
    def device_name(self) -> str:
        """Return the test device."""
        return "cpu"

    def encode(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        """Represent each image by its mean RGB value."""
        rows = [
            np.asarray(image, dtype=np.float32).mean(axis=(0, 1)) for image in images
        ]
        return np.asarray(rows, dtype=np.float32)


def test_build_and_load_embedding_artifact(
    prepared_fixture: PreparedFixture, tmp_path: Path
) -> None:
    """Built vectors remain aligned, normalized, hashed, and pickle free."""
    examples, fingerprint = load_prepared_examples(
        private_root=prepared_fixture.private_root,
        processed_root=prepared_fixture.processed_root,
    )

    result = build_embedding_artifact(
        examples=examples,
        dataset_fingerprint_value=fingerprint,
        encoder=MeanColorEncoder(),
        artifact_root=tmp_path / "artifacts",
        batch_size=3,
    )
    artifact = load_embedding_artifact(
        artifact_path=result.artifact_path,
        metadata_path=result.metadata_path,
    )

    assert result.image_count == 4
    assert result.embedding_dimension == 3
    assert result.device_name == "cpu"
    assert result.elapsed_seconds > 0
    assert result.images_per_second > 0
    assert artifact.image_ids == (
        "img-000001",
        "img-000002",
        "img-000003",
        "img-000004",
    )
    assert artifact.embeddings.shape == (4, 3)
    assert np.allclose(np.linalg.norm(artifact.embeddings, axis=1), 1.0)
    assert artifact.metadata["model_revision"] == "revision-1"
    assert artifact.metadata["embedding_pooling"] == "cls_token"


def test_embedding_artifact_detects_tampering(
    prepared_fixture: PreparedFixture, tmp_path: Path
) -> None:
    """Metadata protects a saved matrix from unnoticed replacement."""
    examples, fingerprint = load_prepared_examples(
        private_root=prepared_fixture.private_root,
        processed_root=prepared_fixture.processed_root,
    )
    result = build_embedding_artifact(
        examples=examples,
        dataset_fingerprint_value=fingerprint,
        encoder=MeanColorEncoder(),
        artifact_root=tmp_path / "artifacts",
    )
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    metadata["artifact_sha256"] = "wrong"
    result.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(EmbeddingError, match="hash does not match"):
        load_embedding_artifact(
            artifact_path=result.artifact_path,
            metadata_path=result.metadata_path,
        )


def test_prepared_examples_require_complete_matching_manifests(
    prepared_fixture: PreparedFixture,
) -> None:
    """Missing and conflicting prepared rows cannot shift embedding labels."""
    manifest_path = prepared_fixture.processed_root / "images.csv"
    rows = manifest_path.read_text(encoding="utf-8").splitlines()
    manifest_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(EmbeddingError, match="missing image ID"):
        load_prepared_examples(
            private_root=prepared_fixture.private_root,
            processed_root=prepared_fixture.processed_root,
        )


def test_embedding_split_selection_keeps_test_sealed(
    prepared_fixture: PreparedFixture,
) -> None:
    """Test examples require an explicit final evaluation override."""
    examples, _ = load_prepared_examples(
        private_root=prepared_fixture.private_root,
        processed_root=prepared_fixture.processed_root,
    )
    test_example = replace(
        examples[0],
        image_id="img-test",
        capture_session_id="session-003",
        split=DatasetSplit.TEST,
    )
    all_examples = (*examples, test_example)

    selected = select_embedding_examples(
        examples=all_examples,
        included_splits=frozenset({DatasetSplit.TRAIN}),
    )
    assert len(selected) == 4

    with pytest.raises(EmbeddingError, match="test embeddings are sealed"):
        select_embedding_examples(
            examples=all_examples,
            included_splits=frozenset({DatasetSplit.TEST}),
        )
    assert select_embedding_examples(
        examples=all_examples,
        included_splits=frozenset({DatasetSplit.TEST}),
        allow_test=True,
    ) == (test_example,)
    with pytest.raises(EmbeddingError, match="at least one embedding split"):
        select_embedding_examples(examples=all_examples, included_splits=frozenset())


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.asarray([[0.0, 0.0]], dtype=np.float32), "zero length"),
        (np.asarray([[np.nan, 1.0]], dtype=np.float32), "nonfinite"),
    ],
)
def test_embedding_build_rejects_invalid_encoder_output(
    prepared_fixture: PreparedFixture,
    tmp_path: Path,
    values: NDArray[np.float32],
    message: str,
) -> None:
    """Invalid encoder values fail before an artifact is published."""
    examples, fingerprint = load_prepared_examples(
        private_root=prepared_fixture.private_root,
        processed_root=prepared_fixture.processed_root,
    )

    class InvalidEncoder(MeanColorEncoder):
        def encode(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
            return np.repeat(values, len(images), axis=0)

    with pytest.raises(EmbeddingError, match=message):
        build_embedding_artifact(
            examples=examples,
            dataset_fingerprint_value=fingerprint,
            encoder=InvalidEncoder(),
            artifact_root=tmp_path / "artifacts",
        )


def test_embedding_build_rejects_empty_input(tmp_path: Path) -> None:
    """An empty dataset and an invalid batch size fail clearly."""
    with pytest.raises(EmbeddingError, match="at least one"):
        build_embedding_artifact(
            examples=(),
            dataset_fingerprint_value="fingerprint",
            encoder=MeanColorEncoder(),
            artifact_root=tmp_path,
        )
