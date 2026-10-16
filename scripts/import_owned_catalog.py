"""Import photographed dolls and Session 1 covers into the private catalog."""

import importlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

from PIL import Image, ImageOps

from usagi_visual_identifier.catalog import (
    CatalogDatabase,
    CatalogRecordCreate,
    CatalogRepository,
)
from usagi_visual_identifier.data.loaders import (
    load_catalog_csv,
    load_image_manifest_csv,
)


def main() -> None:
    """Add missing owned records without overwriting personal edits."""
    root = Path(__file__).resolve().parents[1]
    database = CatalogDatabase(
        Path(os.environ.get("USAGI_CATALOG_DB", root / "private/catalog.sqlite3"))
    )
    database.initialize()
    repository = CatalogRepository(database)
    linked = {record.linked_identity_id for record in repository.list_records()}
    catalog = load_catalog_csv(path=root / "data/private/catalog.csv")
    manifest = load_image_manifest_csv(path=root / "data/private/images.csv")
    covers = database.path.parent / "catalog-covers"
    covers.mkdir(parents=True, exist_ok=True)
    heif = importlib.import_module("pillow_heif")
    cast(Callable[[], None], heif.register_heif_opener)()
    for item in catalog.items:
        candidates = sorted(
            (
                row
                for row in manifest.records
                if row.catalog_id == item.catalog_id
                and row.capture_session_id == "session-001"
            ),
            key=lambda row: row.relative_path,
        )
        if not candidates:
            continue
        target = covers / f"{item.catalog_id}.jpg"
        if not target.exists():
            with Image.open(
                root / "data/private" / candidates[0].relative_path
            ) as source:
                cover = ImageOps.exif_transpose(source).convert("RGB")
                cover.thumbnail((900, 900))
                cover.save(target, "JPEG", quality=88, exif=b"")
        if item.catalog_id not in linked:
            repository.create_record(
                CatalogRecordCreate(
                    preferred_name=item.display_name,
                    product_type=item.product_type,
                    distinguishing_details=", ".join(item.distinguishing_cues),
                    owned=True,
                    linked_identity_id=item.catalog_id,
                )
            )
        print(f"Ready: {item.display_name}")


if __name__ == "__main__":
    main()
