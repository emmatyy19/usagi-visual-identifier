"""Identification boundary tests with a deterministic encoder and invented photos."""

from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import cast

import httpx2
import numpy as np
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from PIL import Image

from usagi_visual_identifier.api import app as app_module
from usagi_visual_identifier.api.identification import (
    IdentificationResult,
    IdentificationService,
    prepare_upload,
)
from usagi_visual_identifier.embeddings.artifacts import EmbeddingArtifact
from usagi_visual_identifier.embeddings.inference import FrozenRetrievalPredictor


class Encoder:
    """Return an exact known vector without downloading weights."""

    model_id = "test"
    model_revision = "test-revision"
    device_name = "cpu"

    def encode(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        """Produce one predictable image embedding."""
        return np.asarray([[1.0, 0.0] for _ in images], dtype=np.float32)


def _photo(image_format: str = "PNG", size: tuple[int, int] = (12, 12)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, "pink").save(buffer, format=image_format)
    return buffer.getvalue()


def _service(root: Path, *, score: float = 1.0) -> IdentificationService:
    artifact = EmbeddingArtifact(
        image_ids=("train-a", "sealed-test"),
        identity_ids=("usagi-0001", "usagi-0002"),
        capture_session_ids=("session-001", "session-003"),
        splits=("train", "test"),
        processed_relative_paths=("a.jpg", "test.jpg"),
        embeddings=np.asarray(
            [[score, (1 - score**2) ** 0.5], [1, 0]], dtype=np.float32
        ),
        metadata={
            "model_id": "test",
            "model_revision": "test-revision",
            "artifact_sha256": "gallery-hash",
            "dataset_fingerprint": "dataset-hash",
        },
    )
    (root / "a.jpg").write_bytes(_photo("JPEG"))
    return IdentificationService(
        predictor=FrozenRetrievalPredictor(artifact=artifact, encoder=Encoder()),
        artifact=artifact,
        threshold=0.82,
        processed_root=root,
    )


def test_upload_shares_service_and_returns_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTML and API return real reference evidence without writing catalog records."""
    service = _service(tmp_path)
    calls: list[Path] = []

    def load(root: Path) -> IdentificationService:
        calls.append(root)
        return service

    monkeypatch.setattr(app_module, "load_identification_service", load)
    with TestClient(
        app_module.create_app(database_path=tmp_path / "db.sqlite3")
    ) as raw:
        client = cast(httpx2.Client, raw)
        assert client.get("/identify").status_code == 200
        assert not calls
        client.post(
            "/api/catalog",
            json={"preferred_name": "Test doll", "linked_identity_id": "usagi-0001"},
        )
        before = client.get("/api/catalog").json()
        response = client.post(
            "/api/identify", files={"photo": ("../fake.png", _photo(), "image/png")}
        )
        assert response.status_code == 200
        result = response.json()
        assert result["state"] == "possible_match"
        assert result["matches"][0]["name"] == "Test doll"
        assert result["matches"][0]["record_id"]
        assert result["gallery_sha256"] == "gallery-hash"
        html = client.post("/identify", files={"photo": ("x.png", _photo())})
        assert "Possible match: Test doll" in html.text
        assert "/identification/references/train-a" in html.text
        assert "Similarity is not a probability" in html.text
        assert client.get("/identification/references/train-a").status_code == 200
        assert client.get("/identification/references/sealed-test").status_code == 404
        assert client.get("/identification/references/missing").status_code == 404
        assert client.get("/api/catalog").json() == before
        assert len(calls) == 1


def test_unknown_result_remains_a_suggestion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A low score is not presented as a known identity or a probability."""
    service = _service(tmp_path, score=0.5)

    def load(_: Path) -> IdentificationService:
        return service

    monkeypatch.setattr(app_module, "load_identification_service", load)
    with TestClient(
        app_module.create_app(database_path=tmp_path / "db.sqlite3")
    ) as raw:
        client = cast(httpx2.Client, raw)
        response = client.post("/api/identify", files={"photo": ("x.png", _photo())})
        assert response.json()["state"] == "no_confident_match"
        html = client.post("/identify", files={"photo": ("x.png", _photo())})
        assert "No confident local match" in html.text
        assert "No linked personal catalog record yet" in html.text


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        (b"", 422),
        (b"not an image", 422),
        (_photo("GIF"), 422),
        (b"a" * (10 * 1024 * 1024 + 1), 413),
        (_photo(size=(5700, 5700)), 413),
    ],
)
def test_invalid_upload_does_not_load_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: bytes, status: int
) -> None:
    """Reject invalid content and resource limits before costly inference."""

    def forbidden(_: Path) -> IdentificationService:
        raise AssertionError("invalid upload must not load the model")

    monkeypatch.setattr(app_module, "load_identification_service", forbidden)
    with TestClient(
        app_module.create_app(database_path=tmp_path / "db.sqlite3")
    ) as raw:
        client = cast(httpx2.Client, raw)
        for url in ("/api/identify", "/identify"):
            assert (
                client.post(url, files={"photo": ("x.png", payload)}).status_code
                == status
            )


def test_missing_artifacts_are_readable_and_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable model does not prevent browsing or expose filesystem paths."""

    def unavailable(_: Path) -> IdentificationService:
        raise OSError("private path must not be returned")

    monkeypatch.setattr(app_module, "load_identification_service", unavailable)
    with TestClient(
        app_module.create_app(database_path=tmp_path / "db.sqlite3")
    ) as raw:
        client = cast(httpx2.Client, raw)
        response = client.post("/identify", files={"photo": ("x.png", _photo())})
        assert response.status_code == 503
        assert "Local model artifacts are unavailable" in response.text
        assert "private path" not in response.text
        assert client.get("/catalog").status_code == 200


def test_upload_removes_metadata(tmp_path: Path) -> None:
    """The temporary lossless image keeps pixels but not source EXIF tags."""
    buffer = BytesIO()
    image = Image.new("RGB", (12, 10), "pink")
    exif = Image.Exif()
    exif[270] = "private photo description"
    image.save(buffer, format="JPEG", exif=exif)
    buffer.seek(0)
    destination = tmp_path / "query.png"
    prepare_upload(UploadFile(file=buffer), destination)
    with Image.open(destination) as prepared:
        assert prepared.size == (12, 10)
        assert not prepared.getexif()


def test_inference_failure_cleans_up_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Temporary files are removed even when the model fails after decoding."""
    service = _service(tmp_path)
    observed: list[Path] = []

    def load(_: Path) -> IdentificationService:
        return service

    def fail(path: Path) -> IdentificationResult:
        assert path.is_file()
        observed.append(path)
        raise RuntimeError("private backend detail")

    monkeypatch.setattr(app_module, "load_identification_service", load)
    monkeypatch.setattr(service, "identify", fail)
    with TestClient(
        app_module.create_app(database_path=tmp_path / "db.sqlite3")
    ) as raw:
        client = cast(httpx2.Client, raw)
        response = client.post("/api/identify", files={"photo": ("x.png", _photo())})
        assert response.status_code == 503
        assert "private backend detail" not in response.text
        assert observed and not observed[0].exists()
        assert not observed[0].parent.exists()
