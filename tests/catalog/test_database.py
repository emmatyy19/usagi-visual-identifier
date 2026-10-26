"""Tests for versioned personal catalog migrations."""

import sqlite3
from pathlib import Path

import pytest

from usagi_visual_identifier.catalog.database import (
    CatalogDatabase,
    CatalogMigrationError,
)


def test_database_initialization_is_repeatable(tmp_path: Path) -> None:
    """A fresh database migrates once and can safely initialize again."""
    database = CatalogDatabase(tmp_path / "nested" / "catalog.sqlite3")
    database.initialize()
    database.initialize()

    with database.connect() as connection:
        migrations = connection.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version"
        ).fetchall()
        tables = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()

    assert [(row["version"], row["name"]) for row in migrations] == [
        (1, "create_personal_catalog"),
        (2, "create_identification_reviews"),
        (3, "add_source_availability"),
    ]
    assert {"catalog_records", "source_links", "schema_migrations"} <= tables
    assert foreign_keys is not None
    assert foreign_keys[0] == 1


def test_database_rejects_changed_migration_history(tmp_path: Path) -> None:
    """An applied version cannot silently change its migration identity."""
    database = CatalogDatabase(tmp_path / "catalog.sqlite3")
    database.initialize()
    with database.connect() as connection, connection:
        connection.execute(
            "UPDATE schema_migrations SET name = 'unexpected' WHERE version = 1"
        )

    with pytest.raises(CatalogMigrationError, match="applied as unexpected"):
        database.initialize()


def test_database_schema_enforces_interest_values(tmp_path: Path) -> None:
    """SQLite keeps invented interest levels out beyond the Python boundary."""
    database = CatalogDatabase(tmp_path / "catalog.sqlite3")
    database.initialize()
    with database.connect() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO catalog_records (
                record_id, preferred_name, product_type, verification_status,
                known_to_exist, owned, interest_level, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "item-0123456789abcdef0123456789abcdef",
                "Impossible",
                "plush",
                "unverified",
                1,
                0,
                "must_have_at_any_cost",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
