"""Strict CSV loaders for private catalog and image manifests."""

import csv
from collections.abc import Iterator, Mapping
from pathlib import Path

from pydantic import ValidationError

from usagi_visual_identifier.data.models import (
    Catalog,
    CatalogItem,
    ImageManifest,
    ImageRecord,
)

CATALOG_COLUMNS = (
    "catalog_id",
    "display_name",
    "label_status",
    "distinguishing_cues",
    "product_type",
    "included_in_model",
)

IMAGE_MANIFEST_COLUMNS = (
    "image_id",
    "relative_path",
    "catalog_id",
    "unknown_identity_id",
    "capture_session_id",
    "source_image_id",
    "split",
    "source",
)


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


def _parse_boolean(*, value: str, path: Path, line_number: int) -> bool:
    """Parse one explicit lowercase boolean.

    Parameters
    ----------
    value
        Raw CSV field.
    path
        Source path for error context.
    line_number
        Source line for error context.

    Returns
    -------
    bool
        Parsed value.

    Raises
    ------
    DataLoadError
        If the value is not `true` or `false`.
    """
    if value == "true":
        return True
    if value == "false":
        return False
    raise DataLoadError(
        f"expected true or false at {path}:{line_number}; received {value!r}"
    )


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
        cues = tuple(
            cue.strip() for cue in row["distinguishing_cues"].split("|") if cue.strip()
        )
        try:
            items.append(
                CatalogItem.model_validate(
                    {
                        "catalog_id": row["catalog_id"],
                        "display_name": row["display_name"],
                        "label_status": row["label_status"],
                        "distinguishing_cues": cues,
                        "product_type": row["product_type"],
                        "included_in_model": _parse_boolean(
                            value=row["included_in_model"],
                            path=path,
                            line_number=line_number,
                        ),
                    }
                )
            )
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
            records.append(
                ImageRecord.model_validate(
                    {
                        "image_id": row["image_id"],
                        "relative_path": row["relative_path"],
                        "catalog_id": row["catalog_id"] or None,
                        "unknown_identity_id": row["unknown_identity_id"] or None,
                        "capture_session_id": row["capture_session_id"],
                        "source_image_id": row["source_image_id"],
                        "split": row["split"],
                        "source": row["source"],
                    }
                )
            )
        except ValidationError as error:
            raise _row_error(path=path, line_number=line_number, error=error) from error

    try:
        return ImageManifest(records=tuple(records))
    except ValidationError as error:
        raise DataLoadError(f"invalid image manifest in {path}: {error}") from error
