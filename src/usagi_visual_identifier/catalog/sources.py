"""Manually maintained links and availability observations."""

import sqlite3
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from usagi_visual_identifier.catalog.database import CatalogDatabase


class Availability(StrEnum):
    """Availability observed at one source, not a claim about every seller."""

    UNKNOWN = "unknown"
    AVAILABLE = "available"
    SOLD_OUT = "sold_out"
    PREORDER = "preorder"
    DISCONTINUED = "discontinued"


class SourceInput(BaseModel):
    """Editable fields shared by browser and JSON clients."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    url: HttpUrl
    label: str = Field(default="", max_length=200)
    notes: str = Field(default="", max_length=2000)
    availability: Availability = Availability.UNKNOWN
    checked_now: bool = False

    @model_validator(mode="after")
    def reject_credentials(self) -> Self:
        """Keep embedded login credentials out of saved links."""
        if self.url.username is not None or self.url.password is not None:
            raise ValueError("Use a link without embedded login credentials.")
        return self


class SourceLink(BaseModel):
    """A private source and its most recent manual availability check."""

    source_id: str
    record_id: str
    url: HttpUrl
    label: str
    notes: str
    availability: Availability
    reviewed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class SourceRepository:
    """Read and save source links without fetching their contents."""

    def __init__(self, database: CatalogDatabase) -> None:
        """Keep the private database explicit."""
        self.database = database

    def list(self, record_id: str) -> list[SourceLink]:
        """Return sources belonging to one catalog record."""
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM source_links WHERE record_id = ? ORDER BY created_at, source_id",
                (record_id,),
            ).fetchall()
        return [SourceLink.model_validate(dict(row)) for row in rows]

    def save(
        self, record_id: str, value: SourceInput, source_id: str | None = None
    ) -> SourceLink:
        """Create or replace a source, preserving check dates only for unchanged claims."""
        now = datetime.now(UTC).isoformat()
        with self.database.connect() as connection, connection:
            if (
                connection.execute(
                    "SELECT 1 FROM catalog_records WHERE record_id = ?", (record_id,)
                ).fetchone()
                is None
            ):
                raise LookupError("Doll not found.")
            previous = None
            if source_id is not None:
                previous = connection.execute(
                    "SELECT * FROM source_links WHERE record_id = ? AND source_id = ?",
                    (record_id, source_id),
                ).fetchone()
                if previous is None:
                    raise LookupError("Source not found for this doll.")
            reviewed_at = now if value.checked_now else None
            if (
                previous
                and not value.checked_now
                and previous["url"] == str(value.url)
                and previous["availability"] == value.availability
            ):
                reviewed_at = previous["reviewed_at"]
            selected_id = source_id or f"source-{uuid4().hex}"
            try:
                connection.execute(
                    """INSERT INTO source_links
                    (source_id, record_id, url, label, notes, availability, reviewed_at, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_id) DO UPDATE SET url=excluded.url,
                    label=excluded.label, notes=excluded.notes,
                    availability=excluded.availability, reviewed_at=excluded.reviewed_at,
                    updated_at=excluded.updated_at""",
                    (
                        selected_id,
                        record_id,
                        str(value.url),
                        value.label,
                        value.notes,
                        value.availability.value,
                        reviewed_at,
                        previous["created_at"] if previous else now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("This link is already saved for this doll.") from error
            row = connection.execute(
                "SELECT * FROM source_links WHERE source_id = ?", (selected_id,)
            ).fetchone()
            assert row is not None
            return SourceLink.model_validate(dict(row))
