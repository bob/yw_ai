"""Generate yoga sequences from catalog items."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from random import choice, sample
from typing import Literal

from langchain_app.schemas import CatalogItemSchema, Language
from langchain_app.services.catalog_item import CatalogItemService
from langchain_app.services.type_safe import TypeSafeService

STARTING_TAG = "Other/Starting"
FINISHING_TAG = "Other/Finishing"
BATCH_SIZE = 10


@dataclass(frozen=True)
class GeneratorSearchProgress:
    """Optional observation of candidate loading, independent of agent/UI code."""

    phase: Literal["started", "finished"]
    outcome: Literal["ok", "error"] = "ok"
    candidate_count: int | None = None


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
        progress_observer: Callable[[GeneratorSearchProgress], None] | None = None,
    ) -> Iterator[CatalogItemSchema]:
        """Yield the catalog items selected for a requested yoga sequence."""

        if count < 0:
            raise ValueError("count must not be negative")
        if count == 0:
            return

        if progress_observer is not None:
            progress_observer(GeneratorSearchProgress("started"))
        try:
            catalog_items = [
                CatalogItemSchema.model_validate(
                    item.model_dump(exclude={"rigman_data", "second_side_rigman_data"})
                )
                for item in self._catalog_item_service.search(
                    with_tags=True, with_rigman=False, limit=0, rand=True
                )
                if item.image_url is not None
            ]
        except Exception:
            if progress_observer is not None:
                progress_observer(GeneratorSearchProgress("finished", outcome="error"))
            raise
        if progress_observer is not None:
            progress_observer(GeneratorSearchProgress("finished", candidate_count=len(catalog_items)))
        starting_items = [
            item
            for item in catalog_items
            if STARTING_TAG in item.catalog_tag_names
        ]
        if not starting_items:
            raise ValueError(
                f"No catalog items with tag {STARTING_TAG} and thumbnails are available"
            )

        starting_item = choice(starting_items)
        selected_items = [starting_item]
        yield starting_item

        for _ in range(count - 1):
            selected_ids = {item.id for item in selected_items}
            available_items = [
                item for item in catalog_items if item.id not in selected_ids
            ]
            if len(selected_items) < count - 1:
                candidates = sample(
                    available_items, min(BATCH_SIZE, len(available_items))
                )
            else:
                candidates = [
                    item
                    for item in available_items
                    if FINISHING_TAG in item.catalog_tag_names
                ]
                if not candidates:
                    candidates = available_items
            catalog_item = self._type_safe_service.choose_next_catalog_item(
                candidates,
                selected_items=selected_items,
                request=request,
                elements_count=count,
            )
            selected_items.append(catalog_item)
            yield catalog_item
