"""CSV boundary models for catalog and image manifest rows."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from usagi_visual_identifier.data.models import (
    CatalogId,
    CatalogItem,
    DatasetSplit,
    ImageId,
    ImageRecord,
    ImageSource,
    LabelStatus,
    SessionId,
    SourceImageId,
    UnknownId,
)


class CatalogCsvRow(BaseModel):
    """One catalog row in its CSV representation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    catalog_id: CatalogId
    display_name: str = Field(min_length=1)
    label_status: LabelStatus
    distinguishing_cues: tuple[str, ...]
    product_type: str = Field(min_length=1)
    included_in_model: bool

    @field_validator("distinguishing_cues", mode="before")
    @classmethod
    def parse_distinguishing_cues(cls, value: object) -> tuple[str, ...]:
        """Convert pipe separated text into individual cues."""
        if not isinstance(value, str):
            raise ValueError("distinguishing cues must be pipe separated text")
        return tuple(cue.strip() for cue in value.split("|") if cue.strip())

    @field_validator("included_in_model", mode="before")
    @classmethod
    def parse_included_in_model(cls, value: object) -> bool:
        """Accept only explicit lowercase CSV booleans."""
        if value == "true":
            return True
        if value == "false":
            return False
        raise ValueError("included_in_model must be true or false")

    def to_catalog_item(self) -> CatalogItem:
        """Convert this validated CSV row into a domain record."""
        return CatalogItem.model_validate(self.model_dump())


class ImageManifestCsvRow(BaseModel):
    """One image manifest row in its CSV representation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    image_id: ImageId
    relative_path: str = Field(min_length=1)
    catalog_id: CatalogId | None
    unknown_identity_id: UnknownId | None
    capture_session_id: SessionId
    source_image_id: SourceImageId
    split: DatasetSplit
    source: ImageSource

    @field_validator("catalog_id", "unknown_identity_id", mode="before")
    @classmethod
    def parse_optional_identity(cls, value: object) -> object:
        """Convert an empty CSV identity field into `None`."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def to_image_record(self) -> ImageRecord:
        """Convert this validated CSV row into a domain record."""
        return ImageRecord.model_validate(self.model_dump())


def catalog_csv_columns() -> tuple[str, ...]:
    """Return the catalog CSV columns in declared order."""
    return tuple(CatalogCsvRow.model_fields)


def image_manifest_csv_columns() -> tuple[str, ...]:
    """Return the image manifest CSV columns in declared order."""
    return tuple(ImageManifestCsvRow.model_fields)
