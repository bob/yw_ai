"""Generate yoga sequences from catalog items."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from math import floor, isfinite
from random import choice, choices, sample
from typing import Literal

from langchain_app.schemas import CatalogItemSchema, Language
from langchain_app.services.catalog_item import CatalogItemService
from langchain_app.services.type_safe import TypeSafeSelectionError, TypeSafeService

STARTING_TAG = "Other/Starting"
FINISHING_TAG = "Other/Finishing"
BATCH_SIZE = 10
MAX_SEQUENCE_ELEMENTS = 45
POSE_DURATION_SECONDS = 40
LYING_TAG_PREFIX = "Position/Lying"
SITTING_TAG_PREFIX = "Position/Sitting"
STANDING_TAG = "Position/Standing"
SUPINE_TAG = "Position/Lying on the back"


def _is_lying(item: CatalogItemSchema) -> bool:
    return any(tag.startswith(LYING_TAG_PREFIX) for tag in item.catalog_tag_names)


def _duration_units(item: CatalogItemSchema) -> int:
    """A mirrored catalog pose is performed on both sides."""
    return 2 if item.bilateral else 1


def _random_starting_item(items: list[CatalogItemSchema]) -> CatalogItemSchema:
    """Weight positions independently of how many starting poses each contains."""
    groups = [
        [item for item in items if _is_lying(item)],
        [item for item in items if any(
            tag.startswith(SITTING_TAG_PREFIX) for tag in item.catalog_tag_names
        )],
        [item for item in items if STANDING_TAG in item.catalog_tag_names],
    ]
    available = [(group, weight) for group, weight in zip(groups, (1, 6, 3)) if group]
    if not available:
        return choice(items)
    group = choices(
        [group for group, _ in available],
        weights=[weight for _, weight in available],
        k=1,
    )[0]
    return choice(group)


def _second_pose_candidates(
    items: list[CatalogItemSchema], starting_item: CatalogItemSchema,
) -> list[CatalogItemSchema]:
    if _is_lying(starting_item):
        candidates = [item for item in items if SUPINE_TAG in item.catalog_tag_names]
        requirement = "lying on the back"
    elif STANDING_TAG in starting_item.catalog_tag_names:
        candidates = [item for item in items if not _is_lying(item)]
        requirement = "not lying down"
    else:
        return items
    if not candidates:
        raise TypeSafeSelectionError(
            f"No unused catalog items with thumbnails are available for a second pose {requirement}"
        )
    return candidates


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
        starting_catalog_item_id: int | None = None,
        duration_seconds: float | None = None,
        progress_observer: Callable[[GeneratorSearchProgress], None] | None = None,
    ) -> Iterator[CatalogItemSchema]:
        """Yield at most 45 poses, using 40/80 seconds when duration is given.

        Without duration, count is exact. With duration, derive the upper count
        and stop once the selected poses use the rounded time budget.
        """

        if count < 0:
            raise ValueError("count must not be negative")
        if count > MAX_SEQUENCE_ELEMENTS:
            raise ValueError(f"count must not exceed {MAX_SEQUENCE_ELEMENTS}")
        duration_units = None
        if duration_seconds is not None:
            if not isfinite(duration_seconds) or duration_seconds <= 0:
                raise ValueError("duration_seconds must be positive and finite")
            duration_units = max(1, floor(duration_seconds / POSE_DURATION_SECONDS + 0.5))
            count = min(duration_units, MAX_SEQUENCE_ELEMENTS)
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
        if starting_catalog_item_id is not None:
            starting_item = next(
                (item for item in catalog_items if item.id == starting_catalog_item_id), None,
            )
            if starting_item is None:
                raise ValueError(
                    f"Starting catalog item {starting_catalog_item_id} is unavailable or has no thumbnail"
                )
        else:
            starting_items = [
                item for item in catalog_items if STARTING_TAG in item.catalog_tag_names
            ]
            if not starting_items:
                raise ValueError(
                    f"No catalog items with tag {STARTING_TAG} and thumbnails are available"
                )
            starting_item = _random_starting_item(starting_items)
        selected_items = [starting_item]
        used_duration_units = _duration_units(starting_item)
        yield starting_item

        while len(selected_items) < count:
            remaining_units = (
                duration_units - used_duration_units if duration_units is not None else None
            )
            if remaining_units is not None and remaining_units <= 0:
                break
            elements_count = (
                min(count, len(selected_items) + remaining_units)
                if remaining_units is not None else count
            )
            selected_ids = {item.id for item in selected_items}
            available_items = [
                item for item in catalog_items if item.id not in selected_ids
            ]
            if len(selected_items) == 1:
                available_items = _second_pose_candidates(available_items, starting_item)
            is_last_element = len(selected_items) == elements_count - 1
            finishing_items = [
                item for item in available_items
                if FINISHING_TAG in item.catalog_tag_names
                and (is_last_element or (
                    remaining_units is not None and _duration_units(item) >= remaining_units
                ))
            ]
            if not is_last_element and not finishing_items:
                candidates = sample(
                    available_items, min(BATCH_SIZE, len(available_items))
                )
            else:
                candidates = finishing_items or available_items
                elements_count = len(selected_items) + 1
            catalog_item = self._type_safe_service.choose_next_catalog_item(
                candidates,
                selected_items=selected_items,
                request=request,
                elements_count=elements_count,
            )
            selected_items.append(catalog_item)
            used_duration_units += _duration_units(catalog_item)
            yield catalog_item
