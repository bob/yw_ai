"""Schemas for the public sequences API."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, HttpUrl, NonNegativeInt, PositiveInt

Language = Literal["en", "uk", "ru"]


class ApiSchema(BaseModel):
    """Base model for responses whose shape is controlled by the API contract."""

    model_config = ConfigDict(extra="forbid")


class CatalogItemSchema(ApiSchema):
    """Localized catalog item returned by the public catalog APIs."""

    id: PositiveInt
    uuid: str | None
    name: str
    bilateral: bool
    image_url: HttpUrl | None
    catalog_tag_names: list[str]


class SequenceElementSchema(ApiSchema):
    """An active breathing-mode element returned for a public sequence."""

    uuid: UUID | None
    order: PositiveInt
    start_mirror: bool
    end_mirror: bool
    use_in_breathing_mode: bool
    catalog_item: CatalogItemSchema | None


class SequenceSchema(ApiSchema):
    """A localized public sequence returned by the API."""

    id: PositiveInt
    name: str
    slug: str
    description: str | None
    lang: Language
    elements_count: NonNegativeInt
    elements: list[SequenceElementSchema]
