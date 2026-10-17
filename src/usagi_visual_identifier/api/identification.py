"""Local photo identification without retaining query photos or changing records."""

import importlib
import json
import math
import os
import threading
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, cast

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from usagi_visual_identifier.api.dependencies import CatalogRepositoryDependency
from usagi_visual_identifier.embeddings.artifacts import (
    EmbeddingArtifact,
    EmbeddingError,
    load_embedding_artifact,
)
from usagi_visual_identifier.embeddings.dinov2 import Dinov2Encoder
from usagi_visual_identifier.embeddings.inference import FrozenRetrievalPredictor

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 32_000_000


class PhotoMatch(BaseModel):
    """One candidate and its actual nearest training reference."""

    identity_id: str
    score: float
    reference_image_id: str
    name: str
    record_id: str | None = None


class IdentificationResult(BaseModel):
    """Versioned response shared by browser and future companion clients."""

    schema_version: int = 1
    state: Literal["possible_match", "no_confident_match"]
    matches: list[PhotoMatch]
    threshold: float
    model_id: str
    model_revision: str
    gallery_sha256: str
    dataset_fingerprint: str
    aggregation: str = "nearest_image"
    elapsed_seconds: float


class IdentificationService:
    """Reuse a predictor with serialized inference and a training only gallery."""

    def __init__(
        self,
        *,
        predictor: FrozenRetrievalPredictor,
        artifact: EmbeddingArtifact,
        threshold: float,
        processed_root: Path,
    ) -> None:
        """Keep the exact artifact and frozen operating point together."""
        self.predictor = predictor
        if not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("similarity threshold must be between -1 and 1")
        self.artifact = artifact
        self.threshold = threshold
        self.lock = threading.Lock()
        root = processed_root.resolve()
        self.references: dict[str, Path] = {}
        for image_id, relative, split in zip(
            artifact.image_ids,
            artifact.processed_relative_paths,
            artifact.splits,
            strict=True,
        ):
            path = (root / relative).resolve()
            if split == "train" and path.is_relative_to(root):
                self.references[image_id] = path

    def identify(self, path: Path) -> IdentificationResult:
        """Return suggestions, never an automatically confirmed identity."""
        with self.lock:
            prediction = self.predictor.predict(path=path, top_k=3)
        metadata = self.artifact.metadata
        return IdentificationResult(
            state="possible_match"
            if prediction.matches[0].score >= self.threshold
            else "no_confident_match",
            matches=[
                PhotoMatch(
                    identity_id=match.identity_id,
                    name=match.identity_id,
                    score=match.score,
                    reference_image_id=match.reference_image_ids[0],
                )
                for match in prediction.matches
            ],
            threshold=self.threshold,
            model_id=str(metadata["model_id"]),
            model_revision=str(metadata["model_revision"]),
            gallery_sha256=str(metadata["artifact_sha256"]),
            dataset_fingerprint=str(metadata["dataset_fingerprint"]),
            elapsed_seconds=prediction.total_seconds,
        )


def load_identification_service(root: Path) -> IdentificationService:
    """Load the existing frozen baseline and its unchanged validation threshold."""
    directory = root / "artifacts/embeddings/facebook_dinov2-small"
    artifact = load_embedding_artifact(
        artifact_path=directory / "embeddings.npz",
        metadata_path=directory / "metadata.json",
    )
    config = json.loads((root / "configs/open_set/final_known_test.json").read_text())
    threshold = float(config["operating_points"]["frozen_max_similarity"])
    encoder = Dinov2Encoder(requested_device=os.environ.get("USAGI_DEVICE", "auto"))
    return IdentificationService(
        predictor=FrozenRetrievalPredictor(artifact=artifact, encoder=encoder),
        artifact=artifact,
        threshold=threshold,
        processed_root=root / "data/processed",
    )


def prepare_upload(upload: UploadFile, destination: Path) -> None:
    """Bound file size and decoded dimensions, then discard original metadata."""
    content = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Choose a photo smaller than 10 MB.")
    if not content:
        raise HTTPException(422, "Choose a photo first.")
    module = importlib.import_module("pillow_heif")
    cast(Callable[[], None], module.register_heif_opener)()
    try:
        with Image.open(BytesIO(content)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP", "HEIF"}:
                raise HTTPException(422, "Use a JPEG, PNG, WebP, or HEIC photo.")
            if source.width * source.height > MAX_PIXELS:
                raise HTTPException(413, "Resize the photo to 32 megapixels or less.")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.info.clear()
            image.save(destination, "PNG")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(422, "This photo could not be decoded.") from error


def register_identification_routes(
    app: FastAPI, *, factory: Callable[[], IdentificationService]
) -> None:
    """Attach browser and JSON entry points backed by one lazy service."""
    templates = Jinja2Templates(directory=Path(__file__).parent / "web/templates")
    service: IdentificationService | None = None
    initialization_lock = threading.Lock()

    def get_service() -> IdentificationService:
        nonlocal service
        with initialization_lock:
            if service is None:
                try:
                    service = factory()
                except (OSError, ValueError, EmbeddingError) as error:
                    raise HTTPException(
                        503,
                        "Local model artifacts are unavailable. Check the server configuration.",
                    ) from error
        return service

    def identify(
        photo: UploadFile, catalog: CatalogRepositoryDependency
    ) -> IdentificationResult:
        with TemporaryDirectory(prefix="usagi-query-") as directory:
            path = Path(directory) / "query.png"
            try:
                prepare_upload(photo, path)
                try:
                    result = get_service().identify(path)
                except (RuntimeError, EmbeddingError) as error:
                    raise HTTPException(
                        503,
                        "Local identification failed. Try again or check the server configuration.",
                    ) from error
            finally:
                photo.file.close()
        records = {
            record.linked_identity_id: record for record in catalog.list_records()
        }
        for match in result.matches:
            record = records.get(match.identity_id)
            if record is not None:
                match.name = record.preferred_name
                match.record_id = record.record_id
        return result

    @app.post("/api/identify", response_model=IdentificationResult)
    def identify_api(
        photo: UploadFile, catalog: CatalogRepositoryDependency
    ) -> IdentificationResult:
        """Identify one uploaded photo without persisting it."""
        return identify(photo, catalog)

    @app.get("/identify", response_class=HTMLResponse)
    def identify_page(request: Request) -> HTMLResponse:
        """Show the single doll upload form."""
        return templates.TemplateResponse(
            request, "identify.html", {"result": None, "error": None}
        )

    @app.post("/identify", response_class=HTMLResponse)
    def identify_browser(
        request: Request, photo: UploadFile, catalog: CatalogRepositoryDependency
    ) -> HTMLResponse:
        """Render reference evidence or a readable upload error."""
        try:
            result = identify(photo, catalog)
        except HTTPException as error:
            return templates.TemplateResponse(
                request,
                "identify.html",
                {"result": None, "error": error.detail},
                status_code=error.status_code,
            )
        return templates.TemplateResponse(
            request, "identify.html", {"result": result, "error": None}
        )

    @app.get("/identification/references/{image_id}")
    def reference_image(image_id: str) -> FileResponse:
        """Serve only registered training references, not arbitrary local files."""
        path = get_service().references.get(image_id)
        if path is None or not path.is_file():
            raise HTTPException(404, "Reference image unavailable.")
        return FileResponse(
            path,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, no-cache"},
        )
