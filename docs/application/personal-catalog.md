# Personal Catalog Application

The personal catalog is one private application with a shared Python service
and SQLite database. The browser interface will be the first client. A later
native iPhone companion for my Meta Ray Ban Display glasses will call the same
service.

## Current application

The first implementation provides:

1. versioned SQLite migrations;
2. typed catalog create, update, and response records;
3. independent known and owned fields plus a graded personal interest level;
4. an optional link from a personal record to one photographed `usagi-####`
   identity;
5. local API endpoints for health, listing, creation, lookup, and updates;
6. database constraints that repeat important application validation;
7. a responsive browser interface for browsing, filtering, creating, and
   editing catalog records.

The migration also creates a source link table. Source link and availability
operations will be added in a later application slice.

The interest level is one of `unknown`, `not_interested`, `nice_to_have`,
`would_buy`, or `actively_looking`. Named levels keep the meaning clearer than
a numeric rating while still supporting useful wish list sorting. Ownership is
separate because I may already own a doll and still want another copy.

## Local data

The default database is `private/catalog.sqlite3`. The entire `private/`
directory is ignored by Git. A different path can be selected through
`USAGI_CATALOG_DB`.

The application creates the parent directory, database file, and pending schema
migrations when it starts. Each migration has an integer version and stable
name. An applied version cannot silently change its name.

## Run through the debugger

Choose `Run Personal Catalog API` in the VS Code Run and Debug panel. The
service starts at `http://127.0.0.1:8000` and stores the database under the
ignored `private/` directory.

Open the catalog at `http://127.0.0.1:8000/catalog`. FastAPI also exposes
interactive API documentation at `http://127.0.0.1:8000/docs`.

## Run from the terminal

```bash
uv run python scripts/run_catalog_app.py
```

For automatic restart while editing Python files:

```bash
uv run uvicorn usagi_visual_identifier.api.app:app --reload
```

## Current endpoints

```text
GET    /api/health
GET    /api/catalog
POST   /api/catalog
GET    /api/catalog/{record_id}
PATCH  /api/catalog/{record_id}
```

The browser interface provides these pages:

```text
GET    /catalog
GET    /catalog/new
POST   /catalog/new
GET    /catalog/{record_id}
POST   /catalog/{record_id}
```

The catalog page can filter by ownership and any of the five personal interest
levels. The forms preserve handwritten names and notes, keep verification
explicit, and allow an optional link to one photographed model identity.

Interest applies only to dolls I do not own. Owned dolls hide the interest field
and badge, and do not count as actively looking. A previous interest choice is
kept so it is available if ownership changes later.

There is no delete endpoint yet. Manual entry and correction are the important
first operations, and avoiding destructive behavior keeps the first version
safer.

## Owned collection photos

The local shelf can start with the photographed collection. In the debugger,
choose `Import Owned Catalog and Covers`, or run:

```bash
uv run python scripts/import_owned_catalog.py
```

The import adds missing owned records from the private catalog and uses the first
Session 1 photograph for each cover. Running it again preserves existing catalog
edits. The photos are resized with their orientation corrected and metadata removed.
The database and cover images stay inside the ignored private directory, outside Git.
This does not change the training data or model labels.

## Next application slice

The next slice will add a one image upload endpoint and an application service
that converts the selected retrieval model output into the three documented
result states. The browser will show top matches and reference evidence before
any catalog change is saved.
