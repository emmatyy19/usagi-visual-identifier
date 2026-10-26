"""Browser and JSON operations for manually reviewed catalog sources."""

from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from usagi_visual_identifier.api.dependencies import CatalogRepositoryDependency
from usagi_visual_identifier.catalog.database import CatalogDatabase
from usagi_visual_identifier.catalog.repository import CatalogRecordNotFoundError
from usagi_visual_identifier.catalog.sources import (
    Availability,
    SourceInput,
    SourceLink,
    SourceRepository,
)


def source_repository(request: Request) -> SourceRepository:
    """Use the application catalog database."""
    database = request.app.state.catalog_database
    assert isinstance(database, CatalogDatabase)
    return SourceRepository(database)


SourceDependency = Annotated[SourceRepository, Depends(source_repository)]


def register_source_routes(app: FastAPI) -> None:
    """Register source management without external network calls."""
    templates = Jinja2Templates(directory=Path(__file__).parent / "web/templates")

    @app.get("/api/catalog/{record_id}/sources")
    def list_sources(
        record_id: str, sources: SourceDependency, catalog: CatalogRepositoryDependency
    ) -> list[SourceLink]:
        """List sources for an existing doll."""
        try:
            catalog.get_record(record_id)
        except CatalogRecordNotFoundError as error:
            raise HTTPException(404, "Doll not found.") from error
        return sources.list(record_id)

    def save(
        record_id: str,
        value: SourceInput,
        sources: SourceRepository,
        source_id: str | None = None,
    ) -> SourceLink:
        try:
            return sources.save(record_id, value, source_id)
        except LookupError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/catalog/{record_id}/sources", status_code=201)
    def create_source(
        record_id: str, value: SourceInput, sources: SourceDependency
    ) -> SourceLink:
        """Create a manually entered source."""
        return save(record_id, value, sources)

    @app.put("/api/catalog/{record_id}/sources/{source_id}")
    def update_source(
        record_id: str, source_id: str, value: SourceInput, sources: SourceDependency
    ) -> SourceLink:
        """Replace editable fields on a source belonging to this doll."""
        return save(record_id, value, sources, source_id)

    def render(
        request: Request,
        record_id: str,
        sources: SourceRepository,
        catalog: CatalogRepositoryDependency,
        *,
        error: str | None = None,
        draft: dict[str, str] | None = None,
        source_id: str = "",
        status_code: int = 200,
    ) -> HTMLResponse:
        try:
            record = catalog.get_record(record_id)
        except CatalogRecordNotFoundError as missing:
            raise HTTPException(404, "Doll not found.") from missing
        return templates.TemplateResponse(
            request,
            "sources.html",
            {
                "record": record,
                "sources": sources.list(record_id),
                "options": [
                    (item.value, item.value.replace("_", " ").capitalize())
                    for item in Availability
                ],
                "error": error,
                "draft": draft,
                "editing_id": source_id,
            },
            status_code=status_code,
        )

    @app.get("/catalog/{record_id}/sources", response_class=HTMLResponse)
    def sources_page(
        request: Request,
        record_id: str,
        sources: SourceDependency,
        catalog: CatalogRepositoryDependency,
    ) -> HTMLResponse:
        """Show saved sources and an empty add form."""
        return render(request, record_id, sources, catalog)

    @app.post("/catalog/{record_id}/sources", response_model=None)
    async def save_source_page(
        request: Request,
        record_id: str,
        sources: SourceDependency,
        catalog: CatalogRepositoryDependency,
    ) -> Response:
        """Preserve form input on invalid links or duplicate sources."""
        form = await request.form()
        fields = ("url", "label", "notes", "availability", "checked_now")
        draft = {key: str(form.get(key, "")) for key in fields}
        source_id = str(form.get("source_id", ""))
        try:
            value = SourceInput.model_validate(
                {**draft, "checked_now": draft["checked_now"] == "on"}
            )
            save(record_id, value, sources, source_id or None)
        except (ValidationError, HTTPException) as error:
            message = (
                "Use a valid http or https link without login credentials and check the field lengths."
                if isinstance(error, ValidationError)
                else str(error.detail)
            )
            return render(
                request,
                record_id,
                sources,
                catalog,
                error=message,
                draft=draft,
                source_id=source_id,
                status_code=422
                if isinstance(error, ValidationError)
                else error.status_code,
            )
        return RedirectResponse(
            f"/catalog/{record_id}/sources?saved=1", status_code=303
        )
