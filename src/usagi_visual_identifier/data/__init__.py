"""Data contracts and loading tools for the image collection."""

from usagi_visual_identifier.data.fingerprint import dataset_fingerprint
from usagi_visual_identifier.data.loaders import (
    DataLoadError,
    load_catalog_csv,
    load_image_manifest_csv,
)
from usagi_visual_identifier.data.models import (
    Catalog,
    CatalogItem,
    DatasetSplit,
    ImageManifest,
    ImageRecord,
    ImageSource,
    LabelStatus,
)
from usagi_visual_identifier.data.preparation import (
    PreparationError,
    PreparationResult,
    prepare_session,
)

__all__ = [
    "Catalog",
    "CatalogItem",
    "DataLoadError",
    "DatasetSplit",
    "ImageManifest",
    "ImageRecord",
    "ImageSource",
    "LabelStatus",
    "PreparationError",
    "PreparationResult",
    "dataset_fingerprint",
    "load_catalog_csv",
    "load_image_manifest_csv",
    "prepare_session",
]
