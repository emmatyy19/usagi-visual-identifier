"""Strict CSV loaders for private catalog and image manifests."""

import csv
from collections.abc import Iterator, Mapping
from pathlib import Path

from pydantic import ValidationError

from usagi_visual_identifier.data.csv_rows import (
    CatalogCsvRow,
    ImageManifestCsvRow,
    catalog_csv_columns,
    image_manifest_csv_columns,
)
from usagi_visual_identifier.data.models import (
    Catalog,
    CatalogItem,
    ImageManifest,
    ImageRecord,
)

CATALOG_COLUMNS = catalog_csv_columns()
IMAGE_MANIFEST_COLUMNS = image_manifest_csv_columns()


class DataLoadError(ValueError):
    """Raised when a private data file violates its declared contract."""


def _read_csv_rows(
    *, path: Path, expected_columns: tuple[str, ...]
) -> Iterator[tuple[int, Mapping[str, str]]]:
    """Yield rows only when the CSV header matches exactly.

    Parameters
    ----------
    path
        CSV file to read.
    expected_columns
        Required columns in required order.

    Yields
    ------
    tuple[int, Mapping[str, str]]
        One based file line number and row values.

    Raises
    ------
    DataLoadError
        If the file cannot be read or its header does not match.
    """
    try:
        handle = path.open(encoding="utf-8-sig", newline="")
    except OSError as error:
        raise DataLoadError(f"could not open {path}: {error}") from error

    with handle:
        reader = csv.DictReader(handle)
        actual_columns = tuple(reader.fieldnames or ())
        if actual_columns != expected_columns:
            raise DataLoadError(
                f"unexpected columns in {path}; expected {expected_columns}, "
                f"received {actual_columns}"
            )
        for line_number, row in enumerate(reader, start=2):
            clean_row: dict[str, str] = {}
            for column in expected_columns:
                value = row.get(column)
                if not isinstance(value, str):
                    raise DataLoadError(f"malformed CSV row at {path}:{line_number}")
                clean_row[column] = value
            if None in row:
                raise DataLoadError(f"malformed CSV row at {path}:{line_number}")
            yield line_number, clean_row


def _row_error(
    *, path: Path, line_number: int, error: ValidationError
) -> DataLoadError:
    """Create a concise validation error with row context.

    Parameters
    ----------
    path
        Source CSV path.
    line_number
        Source line number.
    error
        Pydantic validation error.

    Returns
    -------
    DataLoadError
        Contextualized data loading error.
    """
    return DataLoadError(f"invalid row at {path}:{line_number}: {error}")


def load_catalog_csv(*, path: Path) -> Catalog:
    """Load and validate the private catalog CSV.

    Parameters
    ----------
    path
        Catalog CSV path.

    Returns
    -------
    Catalog
        Validated catalog.

    Raises
    ------
    DataLoadError
        If a header, row, or catalog invariant is invalid.
    """
    items: list[CatalogItem] = []
    for line_number, row in _read_csv_rows(path=path, expected_columns=CATALOG_COLUMNS):
        try:
            items.append(CatalogCsvRow.model_validate(row).to_catalog_item())
        except ValidationError as error:
            raise _row_error(path=path, line_number=line_number, error=error) from error

    try:
        return Catalog(items=tuple(items))
    except ValidationError as error:
        raise DataLoadError(f"invalid catalog in {path}: {error}") from error


def load_image_manifest_csv(*, path: Path) -> ImageManifest:
    """Load and validate the private image manifest CSV.

    Parameters
    ----------
    path
        Image manifest CSV path.

    Returns
    -------
    ImageManifest
        Validated image manifest.

    Raises
    ------
    DataLoadError
        If a header, row, or manifest invariant is invalid.
    """
    records: list[ImageRecord] = []
    for line_number, row in _read_csv_rows(
        path=path, expected_columns=IMAGE_MANIFEST_COLUMNS
    ):
        try:
            records.append(ImageManifestCsvRow.model_validate(row).to_image_record())
        except ValidationError as error:
            raise _row_error(path=path, line_number=line_number, error=error) from error

    try:
        return ImageManifest(records=tuple(records))
    except ValidationError as error:
        raise DataLoadError(f"invalid image manifest in {path}: {error}") from error
