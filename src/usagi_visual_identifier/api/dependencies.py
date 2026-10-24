"""Shared FastAPI dependencies for JSON and browser routes."""

from typing import Annotated

from fastapi import Depends, Request

from usagi_visual_identifier.catalog.database import CatalogDatabase
from usagi_visual_identifier.catalog.repository import CatalogRepository
from usagi_visual_identifier.catalog.reviews import ReviewRepository


def catalog_repository(request: Request) -> CatalogRepository:
    """Build a repository from the initialized application database."""
    stored = request.app.state.catalog_database
    if not isinstance(stored, CatalogDatabase):
        raise RuntimeError("catalog database is not configured")
    return CatalogRepository(stored)


CatalogRepositoryDependency = Annotated[
    CatalogRepository,
    Depends(catalog_repository),
]


def review_repository(request: Request) -> ReviewRepository:
    """Use the same private database for explicitly saved decisions."""
    stored = request.app.state.catalog_database
    if not isinstance(stored, CatalogDatabase):
        raise RuntimeError("catalog database is not configured")
    return ReviewRepository(stored)


ReviewRepositoryDependency = Annotated[ReviewRepository, Depends(review_repository)]
