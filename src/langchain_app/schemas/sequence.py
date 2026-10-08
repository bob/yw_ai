"""Schemas for the public sequences API."""

from typing import Literal
from uuid import UUID

from pydantic import NonNegativeInt, PositiveInt

from langchain_app.schemas.base import ApiSchema
from langchain_app.schemas.catalog import CatalogItemSchema

Language = Literal["en", "uk", "ru"]


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
