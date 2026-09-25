"""Deterministic fingerprints for validated private dataset manifests."""

import hashlib
import json

from usagi_visual_identifier.data.models import Catalog, ImageManifest

FINGERPRINT_SCHEMA_VERSION = 1


def dataset_fingerprint(*, catalog: Catalog, manifest: ImageManifest) -> str:
    """Return an order independent SHA256 dataset fingerprint.

    Parameters
    ----------
    catalog
        Validated catalog records.
    manifest
        Validated image records.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """
    payload = {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "catalog": [
            item.model_dump(mode="json")
            for item in sorted(catalog.items, key=lambda item: item.catalog_id)
        ],
        "images": [
            record.model_dump(mode="json")
            for record in sorted(manifest.records, key=lambda record: record.image_id)
        ],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
