"""Explicit identification decisions, stored separately from training data."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, JsonValue

from usagi_visual_identifier.catalog.database import CatalogDatabase


class CatalogSnapshot(BaseModel):
    """Identification relevant catalog fields as seen during the query."""

    record_id: str
    preferred_name: str
    linked_identity_id: str | None
    distinguishing_details: str
    updated_at: datetime


class IdentificationReview(BaseModel):
    """A human decision and the exact evidence presented before that decision."""

    query_id: str
    decision: Literal["confirmed", "corrected", "unknown"]
    selected_record: CatalogSnapshot | None
    candidates: list[CatalogSnapshot]
    result: dict[str, JsonValue]
    created_at: datetime


class ReviewRepository:
    """Persist only explicit reviews in the private catalog database."""

    def __init__(self, database: CatalogDatabase) -> None:
        """Use the initialized application database."""
        self.database = database

    def get(self, query_id: str) -> IdentificationReview | None:
        """Read a saved review without requiring a live model or pending query."""
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT review_json FROM identification_reviews WHERE query_id = ?",
                (query_id,),
            ).fetchone()
        return IdentificationReview.model_validate_json(row[0]) if row else None

    def save(self, review: IdentificationReview) -> IdentificationReview:
        """Keep the first decision on repeated submissions, including concurrent ones."""
        with self.database.connect() as connection, connection:
            connection.execute(
                """INSERT OR IGNORE INTO identification_reviews
                (query_id, review_json, created_at) VALUES (?, ?, ?)""",
                (
                    review.query_id,
                    review.model_dump_json(),
                    review.created_at.isoformat(),
                ),
            )
        saved = self.get(review.query_id)
        assert saved is not None
        return saved
