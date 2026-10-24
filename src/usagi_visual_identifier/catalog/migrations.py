"""Ordered SQLite migrations for the personal catalog."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Migration:
    """One atomic catalog schema change."""

    version: int
    name: str
    statements: tuple[str, ...]


MIGRATIONS = (
    Migration(
        version=1,
        name="create_personal_catalog",
        statements=(
            """
            CREATE TABLE catalog_records (
                record_id TEXT PRIMARY KEY,
                preferred_name TEXT NOT NULL CHECK (length(trim(preferred_name)) > 0),
                product_type TEXT NOT NULL CHECK (length(trim(product_type)) > 0),
                distinguishing_details TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                verification_status TEXT NOT NULL
                    CHECK (verification_status IN ('unverified', 'manually_verified')),
                known_to_exist INTEGER NOT NULL DEFAULT 1
                    CHECK (known_to_exist IN (0, 1)),
                owned INTEGER NOT NULL DEFAULT 0 CHECK (owned IN (0, 1)),
                interest_level TEXT NOT NULL DEFAULT 'unknown'
                    CHECK (interest_level IN (
                        'unknown', 'not_interested', 'nice_to_have',
                        'would_buy', 'actively_looking'
                    )),
                linked_identity_id TEXT UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK (NOT (owned = 1 AND known_to_exist = 0))
            )
            """,
            """
            CREATE INDEX catalog_records_preferred_name_index
            ON catalog_records(preferred_name COLLATE NOCASE)
            """,
            """
            CREATE TABLE source_links (
                source_id TEXT PRIMARY KEY,
                record_id TEXT NOT NULL REFERENCES catalog_records(record_id)
                    ON DELETE CASCADE,
                url TEXT NOT NULL CHECK (length(trim(url)) > 0),
                label TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                reviewed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (record_id, url)
            )
            """,
            """
            CREATE INDEX source_links_record_id_index
            ON source_links(record_id)
            """,
        ),
    ),
    Migration(
        version=2,
        name="create_identification_reviews",
        statements=(
            """
            CREATE TABLE identification_reviews (
                query_id TEXT PRIMARY KEY,
                review_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
        ),
    ),
)
