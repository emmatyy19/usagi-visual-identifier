"""Server rendered browser routes for the private personal catalog."""

from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, Form, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from usagi_visual_identifier.api.dependencies import CatalogRepositoryDependency
from usagi_visual_identifier.catalog.database import CatalogDatabase
from usagi_visual_identifier.catalog.models import (
    CatalogRecord,
    CatalogRecordCreate,
    CatalogRecordUpdate,
    InterestLevel,
    VerificationStatus,
)
from usagi_visual_identifier.catalog.repository import (
    CatalogConflictError,
    CatalogRecordNotFoundError,
)

OwnershipFilter = Literal["all", "owned", "not_owned"]
INTEREST_OPTIONS = (
    (InterestLevel.UNKNOWN, "Not sure yet"),
    (InterestLevel.NOT_INTERESTED, "Not interested"),
    (InterestLevel.NICE_TO_HAVE, "Nice to have"),
    (InterestLevel.WOULD_BUY, "Would buy"),
    (InterestLevel.ACTIVELY_LOOKING, "Actively looking"),
)
INTEREST_LABELS = dict(INTEREST_OPTIONS)
VERIFICATION_OPTIONS = (
    (VerificationStatus.UNVERIFIED, "Not verified"),
    (VerificationStatus.MANUALLY_VERIFIED, "Manually verified"),
)


def _template_context(request: Request, **values: object) -> dict[str, object]:
    """Return shared labels and request state for every template."""
    return {
        "request": request,
        "interest_options": INTEREST_OPTIONS,
        "interest_labels": INTEREST_LABELS,
        "verification_options": VERIFICATION_OPTIONS,
        **values,
    }


def _optional_identity(value: str) -> str | None:
    """Convert a blank form field into the catalog model's null value."""
    cleaned = value.strip()
    return cleaned or None


def _validation_message(error: ValidationError) -> str:
    """Return the first concise model validation message for a form."""
    errors = error.errors()
    if not errors:
        return "The catalog record could not be validated."
    return str(errors[0]["msg"]).removeprefix("Value error, ")


def register_web_routes(*, app: FastAPI) -> None:
    """Attach HTML pages and local static assets to the application."""
    web_root = Path(__file__).parent / "web"
    templates = Jinja2Templates(directory=web_root / "templates")
    app.mount("/static", StaticFiles(directory=web_root / "static"), name="static")

    def cover_path(request: Request, record: CatalogRecord) -> Path | None:
        database = request.app.state.catalog_database
        assert isinstance(database, CatalogDatabase)
        if record.linked_identity_id is None:
            return None
        path = (
            database.path.parent / "catalog-covers" / f"{record.linked_identity_id}.jpg"
        )
        return path if path.is_file() else None

    @app.get("/catalog/{record_id}/cover", include_in_schema=False)
    def record_cover(
        request: Request, record_id: str, catalog: CatalogRepositoryDependency
    ) -> FileResponse:
        """Serve only the linked record's private cover, never arbitrary paths."""
        try:
            record = catalog.get_record(record_id)
        except CatalogRecordNotFoundError as error:
            raise HTTPException(status_code=404) from error
        path = cover_path(request, record)
        if path is None:
            raise HTTPException(status_code=404)
        return FileResponse(
            path,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, no-cache"},
        )

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """Open the catalog as the application home page."""
        return RedirectResponse(url="/catalog", status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/catalog", response_class=HTMLResponse, include_in_schema=False)
    def catalog_page(
        request: Request,
        catalog: CatalogRepositoryDependency,
        ownership: Annotated[OwnershipFilter, Query()] = "all",
        interest: Annotated[str, Query()] = "",
    ) -> HTMLResponse:
        """Browse and filter the private personal catalog."""
        records = catalog.list_records()
        if interest and interest not in InterestLevel:
            raise HTTPException(status_code=422, detail="Invalid interest level")
        if ownership == "owned":
            interest = ""
        visible = tuple(
            record
            for record in records
            if (ownership == "all" or record.owned is (ownership == "owned"))
            and (
                not interest or (not record.owned and record.interest_level == interest)
            )
        )
        return templates.TemplateResponse(
            request,
            "catalog.html",
            _template_context(
                request,
                records=visible,
                covers={
                    record.record_id
                    for record in visible
                    if cover_path(request, record)
                },
                total_count=len(records),
                owned_count=sum(record.owned for record in records),
                looking_count=sum(
                    not record.owned
                    and record.interest_level is InterestLevel.ACTIVELY_LOOKING
                    for record in records
                ),
                ownership=ownership,
                selected_interest=interest,
            ),
        )

    @app.get("/catalog/new", response_class=HTMLResponse, include_in_schema=False)
    def new_catalog_record_page(request: Request) -> HTMLResponse:
        """Show a blank personal catalog form."""
        return templates.TemplateResponse(
            request,
            "record_form.html",
            _template_context(
                request,
                page_title="Add a doll",
                heading="Add a doll",
                submit_label="Save doll",
                action="/catalog/new",
                record=None,
                error=None,
            ),
        )

    @app.post("/catalog/new", response_class=HTMLResponse, include_in_schema=False)
    def create_catalog_record_page(
        request: Request,
        catalog: CatalogRepositoryDependency,
        preferred_name: Annotated[str, Form()],
        product_type: Annotated[str, Form()] = "plush",
        distinguishing_details: Annotated[str, Form()] = "",
        notes: Annotated[str, Form()] = "",
        verification_status: Annotated[
            VerificationStatus, Form()
        ] = VerificationStatus.UNVERIFIED,
        known_to_exist: Annotated[bool, Form()] = False,
        owned: Annotated[bool, Form()] = False,
        interest_level: Annotated[InterestLevel, Form()] = InterestLevel.UNKNOWN,
        linked_identity_id: Annotated[str, Form()] = "",
    ) -> Response:
        """Validate and create one record from the browser form."""
        try:
            value = CatalogRecordCreate(
                preferred_name=preferred_name,
                product_type=product_type,
                distinguishing_details=distinguishing_details,
                notes=notes,
                verification_status=verification_status,
                known_to_exist=known_to_exist,
                owned=owned,
                interest_level=interest_level,
                linked_identity_id=_optional_identity(linked_identity_id),
            )
            created = catalog.create_record(value)
        except (ValidationError, CatalogConflictError) as error:
            message = (
                _validation_message(error)
                if isinstance(error, ValidationError)
                else str(error)
            )
            draft = CatalogRecordCreate.model_construct(
                preferred_name=preferred_name,
                product_type=product_type,
                distinguishing_details=distinguishing_details,
                notes=notes,
                verification_status=verification_status,
                known_to_exist=known_to_exist,
                owned=owned,
                interest_level=interest_level,
                linked_identity_id=_optional_identity(linked_identity_id),
            )
            return templates.TemplateResponse(
                request,
                "record_form.html",
                _template_context(
                    request,
                    page_title="Add a doll",
                    heading="Add a doll",
                    submit_label="Save doll",
                    action="/catalog/new",
                    record=draft,
                    error=message,
                ),
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            )
        return RedirectResponse(
            url=f"/catalog/{created.record_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.get(
        "/catalog/{record_id}", response_class=HTMLResponse, include_in_schema=False
    )
    def catalog_record_page(
        request: Request, record_id: str, catalog: CatalogRepositoryDependency
    ) -> HTMLResponse:
        """Show one editable personal catalog record."""
        try:
            record = catalog.get_record(record_id)
        except CatalogRecordNotFoundError:
            return templates.TemplateResponse(
                request,
                "not_found.html",
                _template_context(request, page_title="Doll not found"),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return templates.TemplateResponse(
            request,
            "record_form.html",
            _template_context(
                request,
                page_title=record.preferred_name,
                heading=record.preferred_name,
                submit_label="Save changes",
                action=f"/catalog/{record.record_id}",
                record=record,
                error=None,
            ),
        )

    @app.post(
        "/catalog/{record_id}", response_class=HTMLResponse, include_in_schema=False
    )
    def update_catalog_record_page(
        request: Request,
        record_id: str,
        catalog: CatalogRepositoryDependency,
        preferred_name: Annotated[str, Form()],
        product_type: Annotated[str, Form()] = "plush",
        distinguishing_details: Annotated[str, Form()] = "",
        notes: Annotated[str, Form()] = "",
        verification_status: Annotated[
            VerificationStatus, Form()
        ] = VerificationStatus.UNVERIFIED,
        known_to_exist: Annotated[bool, Form()] = False,
        owned: Annotated[bool, Form()] = False,
        interest_level: Annotated[InterestLevel, Form()] = InterestLevel.UNKNOWN,
        linked_identity_id: Annotated[str, Form()] = "",
    ) -> Response:
        """Validate and persist all editable browser fields."""
        try:
            changes = CatalogRecordUpdate(
                preferred_name=preferred_name,
                product_type=product_type,
                distinguishing_details=distinguishing_details,
                notes=notes,
                verification_status=verification_status,
                known_to_exist=known_to_exist,
                owned=owned,
                interest_level=interest_level,
                linked_identity_id=_optional_identity(linked_identity_id),
            )
            catalog.update_record(record_id=record_id, changes=changes)
        except CatalogRecordNotFoundError:
            return templates.TemplateResponse(
                request,
                "not_found.html",
                _template_context(request, page_title="Doll not found"),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        except (ValidationError, CatalogConflictError) as error:
            message = (
                _validation_message(error)
                if isinstance(error, ValidationError)
                else str(error)
            )
            draft = CatalogRecord.model_construct(
                record_id=record_id,
                preferred_name=preferred_name,
                product_type=product_type,
                distinguishing_details=distinguishing_details,
                notes=notes,
                verification_status=verification_status,
                known_to_exist=known_to_exist,
                owned=owned,
                interest_level=interest_level,
                linked_identity_id=_optional_identity(linked_identity_id),
            )
            return templates.TemplateResponse(
                request,
                "record_form.html",
                _template_context(
                    request,
                    page_title=preferred_name or "Edit doll",
                    heading=preferred_name or "Edit doll",
                    submit_label="Save changes",
                    action=f"/catalog/{record_id}",
                    record=draft,
                    error=message,
                ),
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            )
        return RedirectResponse(
            url=f"/catalog/{record_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )
