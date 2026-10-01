"""Generate yoga sequences from catalog items."""

from collections.abc import Iterator
from random import choice

from langchain_app.schemas import CatalogItemSchema, Language
from langchain_app.services.catalog_item import CatalogItemService
from langchain_app.services.type_safe import TypeSafeService


class SequenceGenerator:
    """Select catalog items for a generated yoga sequence."""

    def __init__(
        self,
        catalog_item_service: CatalogItemService,
        type_safe_service: TypeSafeService,
    ) -> None:
        self._catalog_item_service = catalog_item_service
        self._type_safe_service = type_safe_service

    def generate(
        self,
        request: str,
        *,
        language: Language,
        count: int,
    ) -> Iterator[CatalogItemSchema]:
        """Yield the catalog items selected for a requested yoga sequence."""

        if count < 0:
            raise ValueError("count must not be negative")
        if count == 0:
            return

        catalog_items = [
            item
            for item in self._catalog_item_service.search(language=language)
            if item.image_url is not None
        ]
        starting_items = [
            item
            for item in catalog_items
            if "Other/Starting" in item.catalog_tag_names
        ]
        if not starting_items:
            raise ValueError(
                "No catalog items with tag Other/Starting and thumbnails are available"
            )

        starting_item = choice(starting_items)
        selected_items = [starting_item]
        yield starting_item

        for _ in range(count - 1):
            catalog_item = self._type_safe_service.choose_next_catalog_item(
                catalog_items,
                selected_items=selected_items,
                request=request,
                elements_count=count,
            )
            selected_items.append(catalog_item)
            yield catalog_item
