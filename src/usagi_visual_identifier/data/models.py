"""Validated records for catalog items and image manifests."""

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CATALOG_ID_PATTERN = r"^usagi-\d{4}$"
IMAGE_ID_PATTERN = r"^img-\d{6}$"
SESSION_ID_PATTERN = r"^session-\d{3}$"
SOURCE_IMAGE_ID_PATTERN = r"^src-\d{6}$"
UNKNOWN_ID_PATTERN = r"^unknown-\d{4}$"

CatalogId = Annotated[str, Field(pattern=CATALOG_ID_PATTERN)]
ImageId = Annotated[str, Field(pattern=IMAGE_ID_PATTERN)]
SessionId = Annotated[str, Field(pattern=SESSION_ID_PATTERN)]
SourceImageId = Annotated[str, Field(pattern=SOURCE_IMAGE_ID_PATTERN)]
UnknownId = Annotated[str, Field(pattern=UNKNOWN_ID_PATTERN)]


class LabelStatus(StrEnum):
    """Verification state for a catalog identity."""

    PROVISIONAL = "provisional"
    VERIFIED = "verified"
    EXCLUDED = "excluded"


class DatasetSplit(StrEnum):
    """Allowed dataset partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ImageSource(StrEnum):
    """Provenance category for an image."""

    OWNER_CAPTURE = "owner_capture"
    SYNTHETIC = "synthetic"
    THIRD_PARTY = "third_party"


class CatalogItem(BaseModel):
    """One visually distinguishable merchandise identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    catalog_id: CatalogId
    display_name: str = Field(min_length=1)
    label_status: LabelStatus
    distinguishing_cues: tuple[str, ...] = ()
    product_type: str = Field(min_length=1)
    included_in_model: bool

    @field_validator("distinguishing_cues")
    @classmethod
    def validate_distinguishing_cues(cls, cues: tuple[str, ...]) -> tuple[str, ...]:
        """Reject blank or repeated visual cues."""
        cleaned = tuple(cue.strip() for cue in cues)
        if any(not cue for cue in cleaned):
            raise ValueError("distinguishing cues cannot be blank")
        normalized = tuple(cue.casefold() for cue in cleaned)
        if len(normalized) != len(set(normalized)):
            raise ValueError("distinguishing cues must be unique")
        return cleaned

    @model_validator(mode="after")
    def validate_model_inclusion(self) -> Self:
        """Allow only verified identities into supervised modeling."""
        if self.included_in_model and self.label_status is not LabelStatus.VERIFIED:
            raise ValueError("only verified catalog items can be included in the model")
        return self


class Catalog(BaseModel):
    """A validated collection of unique catalog items."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[CatalogItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Self:
        """Reject duplicate catalog identifiers."""
        identifiers = tuple(item.catalog_id for item in self.items)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("catalog IDs must be unique")
        return self

    def by_id(self) -> dict[str, CatalogItem]:
        """Return catalog items indexed by stable identifier."""
        return {item.catalog_id: item for item in self.items}


class ImageRecord(BaseModel):
    """One original image or one declared derivative."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    image_id: ImageId
    relative_path: str = Field(min_length=1)
    catalog_id: CatalogId | None = None
    unknown_identity_id: UnknownId | None = None
    capture_session_id: SessionId
    source_image_id: SourceImageId
    split: DatasetSplit
    source: ImageSource

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        """Require a normalized relative path with forward slashes."""
        if "\\" in value:
            raise ValueError("relative paths must use forward slashes")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("image paths must remain beneath the dataset root")
        if value != path.as_posix() or value in {"", "."}:
            raise ValueError("image paths must be normalized relative paths")
        return value

    @model_validator(mode="after")
    def validate_identity_reference(self) -> Self:
        """Require exactly one known or unknown identity reference."""
        references = (self.catalog_id is not None, self.unknown_identity_id is not None)
        if sum(references) != 1:
            raise ValueError("each image must reference one known or unknown identity")
        return self

    @property
    def identity_key(self) -> str:
        """Return the known or unknown identity for this image."""
        if self.catalog_id is not None:
            return self.catalog_id
        if self.unknown_identity_id is None:
            raise RuntimeError("validated image record has no identity")
        return self.unknown_identity_id


class ImageManifest(BaseModel):
    """A manifest whose sessions and source families cannot cross splits."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    records: tuple[ImageRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest_invariants(self) -> Self:
        """Reject duplicate IDs and split leakage."""
        image_ids: set[str] = set()
        session_splits: dict[str, DatasetSplit] = {}
        source_families: dict[str, tuple[DatasetSplit, str]] = {}

        for record in self.records:
            if record.image_id in image_ids:
                raise ValueError(f"duplicate image ID: {record.image_id}")
            image_ids.add(record.image_id)

            session_split = session_splits.setdefault(
                record.capture_session_id, record.split
            )
            if session_split is not record.split:
                raise ValueError(
                    f"capture session crosses splits: {record.capture_session_id}"
                )

            source_family = source_families.setdefault(
                record.source_image_id, (record.split, record.identity_key)
            )
            if source_family != (record.split, record.identity_key):
                raise ValueError(
                    f"source image family crosses a split or identity: "
                    f"{record.source_image_id}"
                )

        return self
