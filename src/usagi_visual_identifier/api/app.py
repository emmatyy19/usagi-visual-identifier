"""FastAPI application for the private personal catalog."""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status

from usagi_visual_identifier.api.dependencies import CatalogRepositoryDependency
from usagi_visual_identifier.api.web import register_web_routes
from usagi_visual_identifier.catalog.database import CatalogDatabase
from usagi_visual_identifier.catalog.models import (
    CatalogRecord,
    CatalogRecordCreate,
    CatalogRecordUpdate,
)
from usagi_visual_identifier.catalog.repository import (
    CatalogConflictError,
    CatalogRecordNotFoundError,
)

DEFAULT_DATABASE_PATH = Path("private/catalog.sqlite3")


def _configured_database_path() -> Path:
    """Read the local catalog path without requiring environment tooling."""
    configured = os.environ.get("USAGI_CATALOG_DB")
    return Path(configured) if configured else DEFAULT_DATABASE_PATH


def create_app(*, database_path: Path | None = None) -> FastAPI:
    """Create an application with an explicit, testable database dependency."""
    database = CatalogDatabase(database_path or _configured_database_path())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        database.initialize()
        yield

    app = FastAPI(
        title="Usagi Personal Catalog",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.catalog_database = database

    register_web_routes(app=app)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        """Report that the process and startup migration completed."""
        return {"status": "ok"}

    @app.get("/api/catalog", response_model=list[CatalogRecord])
    def list_catalog(
        catalog: CatalogRepositoryDependency,
    ) -> tuple[CatalogRecord, ...]:
        """List the complete private personal catalog."""
        return catalog.list_records()

    @app.post(
        "/api/catalog",
        response_model=CatalogRecord,
        status_code=status.HTTP_201_CREATED,
    )
    def create_catalog_record(
        value: CatalogRecordCreate,
        catalog: CatalogRepositoryDependency,
    ) -> CatalogRecord:
        """Create a manually entered personal catalog record."""
        try:
            return catalog.create_record(value)
        except CatalogConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/catalog/{record_id}", response_model=CatalogRecord)
    def get_catalog_record(
        record_id: str,
        catalog: CatalogRepositoryDependency,
    ) -> CatalogRecord:
        """Return one personal catalog record."""
        try:
            return catalog.get_record(record_id)
        except CatalogRecordNotFoundError as error:
            raise HTTPException(
                status_code=404, detail="catalog record not found"
            ) from error

    @app.patch("/api/catalog/{record_id}", response_model=CatalogRecord)
    def update_catalog_record(
        record_id: str,
        changes: CatalogRecordUpdate,
        catalog: CatalogRepositoryDependency,
    ) -> CatalogRecord:
        """Apply a validated partial change without destructive deletion."""
        try:
            return catalog.update_record(record_id=record_id, changes=changes)
        except CatalogRecordNotFoundError as error:
            raise HTTPException(
                status_code=404, detail="catalog record not found"
            ) from error
        except CatalogConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    return app


app = create_app()
