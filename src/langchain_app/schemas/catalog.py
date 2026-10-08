"""Catalog data and search arguments, independent of sequences and tools."""

from typing import Any

from pydantic import Field, HttpUrl, PositiveInt, field_validator

from langchain_app.schemas.base import ApiSchema


class CatalogItemSchema(ApiSchema):
    """A catalog item with its API-provided name and tags."""

    id: PositiveInt
    uuid: str | None
    name: str
    bilateral: bool
    image_url: HttpUrl | None
    catalog_tag_names: list[str]


class CatalogItemSearchResult(ApiSchema):
    """HTTP result whose optional fields depend on the requested options."""

    id: PositiveInt
    uuid: str | None
    name: str
    bilateral: bool
    image_url: HttpUrl | None
    catalog_tag_names: list[str] | None = None
    rigman_data: dict[str, Any] | None = None
    second_side_rigman_data: dict[str, Any] | None = None


class CatalogItemsSearchArguments(ApiSchema):
    """Local search options; the catalog language is fixed by the client."""

    name: str | None = None
    uuid: str | None = None
    bilateral: bool | None = Field(default=None, strict=True)
    catalog_tag_names: list[str] = Field(default_factory=list)
    with_rigman: bool = Field(default=False, strict=True)
    with_tags: bool = Field(default=True, strict=True)
    limit: int = Field(default=10, ge=0, strict=True)
    rand: bool = Field(default=False, strict=True)

    @field_validator("name", "uuid")
    @classmethod
    def normalize_filter(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return cls._nonblank(value)

    @field_validator("catalog_tag_names")
    @classmethod
    def normalize_tags(cls, values: list[str]) -> list[str]:
        return [cls._nonblank(value) for value in values]

    @staticmethod
    def _nonblank(value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("search filters must not be blank")
        return value
