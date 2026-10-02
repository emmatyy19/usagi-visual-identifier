"""Persistence operations for personal catalog records."""

import sqlite3
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from usagi_visual_identifier.catalog.database import CatalogDatabase
from usagi_visual_identifier.catalog.models import (
    CatalogRecord,
    CatalogRecordCreate,
    CatalogRecordUpdate,
)


class CatalogRecordNotFoundError(LookupError):
    """Raised when a requested personal catalog record does not exist."""


class CatalogConflictError(ValueError):
    """Raised when a unique catalog relationship already exists."""


class CatalogRepository:
    """Read and write catalog records through one initialized database."""

    def __init__(self, database: CatalogDatabase) -> None:
        """Keep the database dependency explicit for application and tests."""
        self._database = database

    def list_records(self) -> tuple[CatalogRecord, ...]:
        """Return all records in stable name and ID order."""
        with self._database.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM catalog_records
                ORDER BY preferred_name COLLATE NOCASE, record_id
                """
            ).fetchall()
        return tuple(self._record_from_row(row) for row in rows)

    def get_record(self, record_id: str) -> CatalogRecord:
        """Return one record or raise a domain specific missing error."""
        with self._database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM catalog_records WHERE record_id = ?", (record_id,)
            ).fetchone()
        if row is None:
            raise CatalogRecordNotFoundError(record_id)
        return self._record_from_row(row)

    def create_record(self, value: CatalogRecordCreate) -> CatalogRecord:
        """Create one record using a stable random identifier."""
        now = datetime.now(UTC)
        record = CatalogRecord(
            record_id=f"item-{uuid4().hex}",
            created_at=now,
            updated_at=now,
            **value.model_dump(),
        )
        try:
            with self._database.connect() as connection, connection:
                connection.execute(
                    """
                    INSERT INTO catalog_records (
                        record_id, preferred_name, product_type,
                        distinguishing_details, notes, verification_status,
                        known_to_exist, owned, interest_level,
                        linked_identity_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._record_values(record),
                )
        except sqlite3.IntegrityError as error:
            raise CatalogConflictError(
                "a catalog record already uses that visual identity"
            ) from error
        return record

    def update_record(
        self, *, record_id: str, changes: CatalogRecordUpdate
    ) -> CatalogRecord:
        """Validate and persist a partial change to one record."""
        current = self.get_record(record_id)
        merged = current.model_dump(exclude={"record_id", "created_at", "updated_at"})
        merged.update(changes.model_dump(exclude_unset=True))
        updated = CatalogRecord(
            record_id=current.record_id,
            created_at=current.created_at,
            updated_at=datetime.now(UTC),
            **merged,
        )
        try:
            with self._database.connect() as connection, connection:
                cursor = connection.execute(
                    """
                    UPDATE catalog_records SET
                        preferred_name = ?, product_type = ?,
                        distinguishing_details = ?, notes = ?,
                        verification_status = ?, known_to_exist = ?, owned = ?,
                        interest_level = ?, linked_identity_id = ?,
                        updated_at = ?
                    WHERE record_id = ?
                    """,
                    (
                        updated.preferred_name,
                        updated.product_type,
                        updated.distinguishing_details,
                        updated.notes,
                        updated.verification_status.value,
                        int(updated.known_to_exist),
                        int(updated.owned),
                        updated.interest_level.value,
                        updated.linked_identity_id,
                        updated.updated_at.isoformat(),
                        updated.record_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise CatalogRecordNotFoundError(record_id)
        except sqlite3.IntegrityError as error:
            raise CatalogConflictError(
                "a catalog record already uses that visual identity"
            ) from error
        return updated

    @staticmethod
    def _record_values(record: CatalogRecord) -> tuple[object, ...]:
        """Return database values in the insert column order."""
        return (
            record.record_id,
            record.preferred_name,
            record.product_type,
            record.distinguishing_details,
            record.notes,
            record.verification_status.value,
            int(record.known_to_exist),
            int(record.owned),
            record.interest_level.value,
            record.linked_identity_id,
            record.created_at.isoformat(),
            record.updated_at.isoformat(),
        )

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> CatalogRecord:
        """Validate one SQLite row at the persistence boundary."""
        keys = frozenset(row.keys())
        required = {
            "record_id",
            "preferred_name",
            "product_type",
            "distinguishing_details",
            "notes",
            "verification_status",
            "known_to_exist",
            "owned",
            "interest_level",
            "linked_identity_id",
            "created_at",
            "updated_at",
        }
        if not required.issubset(keys):
            raise RuntimeError("catalog database row is missing required columns")
        return CatalogRecord.model_validate(
            {
                "record_id": cast(str, row["record_id"]),
                "preferred_name": cast(str, row["preferred_name"]),
                "product_type": cast(str, row["product_type"]),
                "distinguishing_details": cast(str, row["distinguishing_details"]),
                "notes": cast(str, row["notes"]),
                "verification_status": cast(str, row["verification_status"]),
                "known_to_exist": bool(row["known_to_exist"]),
                "owned": bool(row["owned"]),
                "interest_level": cast(str, row["interest_level"]),
                "linked_identity_id": cast(str | None, row["linked_identity_id"]),
                "created_at": cast(str, row["created_at"]),
                "updated_at": cast(str, row["updated_at"]),
            }
        )
