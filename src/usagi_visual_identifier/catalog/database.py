"""SQLite connection and migration management for the personal catalog."""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from usagi_visual_identifier.catalog.migrations import MIGRATIONS


class CatalogMigrationError(RuntimeError):
    """Raised when the catalog schema cannot be migrated safely."""


class CatalogDatabase:
    """Own connections to one private SQLite catalog file."""

    def __init__(self, path: Path) -> None:
        """Keep the explicit database path without opening it yet."""
        self.path = path

    def initialize(self) -> None:
        """Create the database directory and apply pending migrations."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    applied_at TEXT NOT NULL
                )
                """
            )
            applied_rows = connection.execute(
                "SELECT version, name FROM schema_migrations ORDER BY version"
            ).fetchall()
            applied = {int(row["version"]): str(row["name"]) for row in applied_rows}
            for migration in MIGRATIONS:
                previous_name = applied.get(migration.version)
                if previous_name is not None and previous_name != migration.name:
                    raise CatalogMigrationError(
                        f"migration {migration.version} was applied as {previous_name}, "
                        f"not {migration.name}"
                    )
                if previous_name is not None:
                    continue
                with connection:
                    for statement in migration.statements:
                        connection.execute(statement)
                    connection.execute(
                        """
                        INSERT INTO schema_migrations(version, name, applied_at)
                        VALUES (?, ?, ?)
                        """,
                        (
                            migration.version,
                            migration.name,
                            datetime.now(UTC).isoformat(),
                        ),
                    )

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection]:
        """Yield a configured connection that always enforces foreign keys."""
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
        finally:
            connection.close()
