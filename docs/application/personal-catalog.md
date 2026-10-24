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
stored separately, but the interface only shows interest for dolls I do not own.

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

## Identify a photo

Open `/identify` or choose `Identify a doll` in the navigation. Upload one cropped
doll photo as JPEG, PNG, WebP, or HEIC, up to 10 MB and 32 megapixels. The server
corrects orientation, removes metadata, and uses a temporary lossless image.
Query files are deleted when the request finishes, including failed requests.

The first application version uses frozen DINOv2 Small with nearest image
retrieval. It tied tuned retrieval on final Top 1 accuracy and avoids loading a
separate training checkpoint. It searches only the training gallery and uses the
unchanged `frozen_max_similarity` operating point in
`configs/open_set/final_known_test.json`.

The page returns `Possible match` above the threshold or `No confident local
match` below it. Both show the closest alternatives with their actual training
reference images. Similarity is not a probability. There is no new threshold or
automatic confirmed state. I compare the details and open the relevant catalog
record myself. Each linked suggestion has a `That's the one` button. I can also
choose any other catalog doll or mark the photo unknown. A saved receipt shows
my choice and links to the selected record, or to manual catalog entry for an
unknown doll. These actions do not change ownership or other catalog fields.

The same service is available at `POST /api/identify` with a multipart `photo`
field. Responses include model revision, gallery hash, dataset fingerprint,
threshold, identity scores, reference IDs, current linked catalog record IDs,
and inference timing. Each response also includes a random query ID. Evidence
waits in memory for up to one hour, with a limit of 100 pending queries. Restarting
the server clears unsaved results. This local workflow uses one server process.

Only an explicit decision writes to the private SQLite `identification_reviews`
table. The saved evidence includes the original response, candidate catalog
snapshots, the selected record snapshot, and a timestamp. Snapshots contain
names, identity links, distinguishing details, and record update timestamps;
they exclude personal notes. Confirmed means I chose the first suggestion,
corrected means I chose another record, and unknown means I did not identify it.
These labels describe my decision, not new evidence of model accuracy.

Repeated submissions of the same choice return the existing receipt. A different
choice for an already saved query is rejected; a new identification starts a new
decision. Photos are deleted after inference, and no training examples are
created. Saved receipts survive restarts and can be revisited through their URL.
There is no history browsing screen yet.

Run from the repository root, or set `USAGI_PROJECT_ROOT`. Identification needs
the existing frozen embedding artifact in
`artifacts/embeddings/facebook_dinov2-small`, its metadata, the prepared training
photos under `data/processed`, and the pinned DINOv2 model. The model loads only
on first identification and is reused. If not cached, the model loader may
download model files from Hugging Face; query images are never sent there.
Set `HF_HUB_OFFLINE=1` to require cached model files. `USAGI_DEVICE` selects
`auto`, `mps`, `cuda`, or `cpu`. Browsing still works without the model artifacts.

Keep the server bound to localhost. Authentication, remote hosting, and glasses
integration are not part of this slice.
