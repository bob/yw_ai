"""Materialized sequence variants, independent of their UI projections."""

from collections.abc import Iterable
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, PositiveInt, computed_field

from langchain_app.schemas.base import ApiSchema
from langchain_app.schemas.catalog import CatalogItemSchema


class GeneratedSequenceElement(ApiSchema):
    element_id: UUID = Field(default_factory=uuid4)
    order: PositiveInt
    catalog_item: CatalogItemSchema
    start_mirror: bool
    end_mirror: bool


class GeneratedSequence(ApiSchema):
    uuid: UUID = Field(default_factory=uuid4)
    family_uuid: UUID = Field(default_factory=uuid4)
    parent_uuid: UUID | None = None
    operation: Literal["generate", "regenerate", "replace"] = "generate"
    generation_request: str
    operation_request: str
    name: str = "New Sequence"
    description: str | None = None
    language: Literal["en"] = "en"
    elements: list[GeneratedSequenceElement]

    @computed_field
    @property
    def elements_count(self) -> int:
        return len(self.elements)

    @classmethod
    def from_catalog_items(
        cls, items: Iterable[CatalogItemSchema], *, request: str,
        parent: GeneratedSequence | None = None,
    ) -> GeneratedSequence:
        return cls(
            generation_request=parent.generation_request if parent else request,
            operation_request=request,
            **({
                "family_uuid": parent.family_uuid,
                "parent_uuid": parent.uuid,
                "operation": "regenerate",
            } if parent else {}),
            elements=[
                GeneratedSequenceElement(
                    order=order, catalog_item=item.model_copy(deep=True),
                    start_mirror=item.bilateral, end_mirror=item.bilateral,
                )
                for order, item in enumerate(items, start=1)
            ],
        )

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> GeneratedSequence:
        # elements_count is derived, never accepted as a second source of truth.
        return cls.model_validate({k: v for k, v in snapshot.items() if k != "elements_count"})

    def model_context(self) -> dict:
        """Keep every occurrence and catalog field except image URLs in prompts."""
        return self.model_dump(mode="json", exclude={
            "elements": {"__all__": {"catalog_item": {"image_url"}}},
        })
