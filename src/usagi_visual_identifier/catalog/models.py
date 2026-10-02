"""Typed records for the private personal catalog."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from usagi_visual_identifier.data.models import CATALOG_ID_PATTERN

LinkedIdentityId = Annotated[str, Field(pattern=CATALOG_ID_PATTERN)]


class VerificationStatus(StrEnum):
    """How strongly the personal catalog details have been checked."""

    UNVERIFIED = "unverified"
    MANUALLY_VERIFIED = "manually_verified"


class InterestLevel(StrEnum):
    """Personal acquisition interest from undecided to active searching."""

    UNKNOWN = "unknown"
    NOT_INTERESTED = "not_interested"
    NICE_TO_HAVE = "nice_to_have"
    WOULD_BUY = "would_buy"
    ACTIVELY_LOOKING = "actively_looking"


class CatalogRecordBase(BaseModel):
    """Fields shared by create requests and stored catalog records."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    preferred_name: str = Field(min_length=1, max_length=200)
    product_type: str = Field(default="plush", min_length=1, max_length=100)
    distinguishing_details: str = Field(default="", max_length=2_000)
    notes: str = Field(default="", max_length=10_000)
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    known_to_exist: bool = True
    owned: bool = False
    interest_level: InterestLevel = InterestLevel.UNKNOWN
    linked_identity_id: LinkedIdentityId | None = None

    @model_validator(mode="after")
    def validate_ownership(self) -> Self:
        """Require an owned item to be acknowledged as existing."""
        if self.owned and not self.known_to_exist:
            raise ValueError("an owned catalog record must be known to exist")
        return self


class CatalogRecordCreate(CatalogRecordBase):
    """User supplied fields for one new personal catalog record."""


class CatalogRecordUpdate(BaseModel):
    """Partial user supplied changes for an existing catalog record."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    preferred_name: str | None = Field(default=None, min_length=1, max_length=200)
    product_type: str | None = Field(default=None, min_length=1, max_length=100)
    distinguishing_details: str | None = Field(default=None, max_length=2_000)
    notes: str | None = Field(default=None, max_length=10_000)
    verification_status: VerificationStatus | None = None
    known_to_exist: bool | None = None
    owned: bool | None = None
    interest_level: InterestLevel | None = None
    linked_identity_id: LinkedIdentityId | None = None

    @model_validator(mode="after")
    def validate_nonempty_update(self) -> Self:
        """Require at least one explicit field in every patch."""
        if not self.model_fields_set:
            raise ValueError("at least one catalog field must be updated")
        return self


class CatalogRecord(CatalogRecordBase):
    """One stored item in the personal catalog."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    record_id: str = Field(pattern=r"^item-[0-9a-f]{32}$")
    created_at: datetime
    updated_at: datetime
