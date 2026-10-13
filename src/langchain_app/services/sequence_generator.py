"""Generate yoga sequences from catalog items."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from math import floor, isfinite
from random import choice, choices, sample
from typing import Literal
from uuid import UUID

from langchain_app.schemas import CatalogItemSchema, GeneratedSequence, Language
from langchain_app.services.catalog_item import CatalogItemService
from langchain_app.services.sequence_editor import same_catalog_item
from langchain_app.services.sequence_rules import (
    MAX_SEQUENCE_ELEMENTS,
    STARTING_TAG,
    STANDING_TAG,
    has_catalog_uuid as _has_catalog_uuid,
    is_lying as _is_lying,
    second_pose_allowed as _second_pose_allowed,
    starting_pose_allowed,
)
from langchain_app.services.type_safe import TypeSafeSelectionError, TypeSafeService

FINISHING_TAG = "Other/Finishing"
BATCH_SIZE = 10
POSE_DURATION_SECONDS = 40
SITTING_TAG_PREFIX = "Position/Sitting"


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
        requirement = "lying on the back"
    elif STANDING_TAG in starting_item.catalog_tag_names:
        requirement = "not lying down"
    else:
        return items
    candidates = [item for item in items if _second_pose_allowed(starting_item, item)]
    if not candidates:
        raise TypeSafeSelectionError(
            f"No unused catalog items with thumbnails are available for a second pose {requirement}"
        )
    return candidates


def _pose_candidates(
    items: list[CatalogItemSchema], *, used_items: Sequence[CatalogItemSchema],
    position: int, first_item: CatalogItemSchema,
    fixed_second_item: CatalogItemSchema | None = None,
) -> list[CatalogItemSchema]:
    """Apply identity and position constraints before sampling in either mode."""
    candidates = [
        item for item in items
        if not any(same_catalog_item(item, used) for used in used_items)
        and starting_pose_allowed(item, position)
    ]
    if position == 1 and fixed_second_item is not None:
        candidates = [
            item for item in candidates if _second_pose_allowed(item, fixed_second_item)
        ]
    if position == 2:
        candidates = _second_pose_candidates(candidates, first_item)
    return candidates


def _candidate_batch(
    items: list[CatalogItemSchema], *, is_last_element: bool,
    remaining_units: int | None = None,
) -> tuple[list[CatalogItemSchema], bool]:
    """Prefer a finishing pose only when it can end the entire sequence."""
    finishing_items = [
        item for item in items
        if FINISHING_TAG in item.catalog_tag_names
        and (is_last_element or (
            remaining_units is not None and _duration_units(item) >= remaining_units
        ))
    ]
    if is_last_element or finishing_items:
        return finishing_items or items, True
    return sample(items, min(BATCH_SIZE, len(items))), False


@dataclass(frozen=True)
class GeneratorSearchProgress:
    """Optional observation of candidate loading, independent of agent/UI code."""

    phase: Literal["started", "finished"]
    outcome: Literal["ok", "error"] = "ok"
    candidate_count: int | None = None


@dataclass(frozen=True)
class GeneratorAdditionProgress:
    """Selection counts for an atomic block, without provisional catalog items."""

    completed: int
    total: int


class SequenceGenerator:
    """Select catalog items for a generated yoga sequence."""

    def __init__(
        self,
        catalog_item_service: CatalogItemService,
        type_safe_service: TypeSafeService,
    ) -> None:
        self._catalog_item_service = catalog_item_service
        self._type_safe_service = type_safe_service

    def _load_catalog_items(
        self, progress_observer: Callable[[GeneratorSearchProgress], None] | None,
        *, require_catalog_uuid: bool = False,
    ) -> list[CatalogItemSchema]:
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
                and (not require_catalog_uuid or _has_catalog_uuid(item))
            ]
        except Exception:
            if progress_observer is not None:
                progress_observer(GeneratorSearchProgress("finished", outcome="error"))
            raise
        if progress_observer is not None:
            progress_observer(GeneratorSearchProgress("finished", candidate_count=len(catalog_items)))
        return catalog_items

    def add_elements(
        self, sequence: GeneratedSequence, request: str, *, language: Language,
        count: int, insert_before_element_id: UUID | None = None,
        progress_observer: Callable[[GeneratorSearchProgress | GeneratorAdditionProgress], None] | None = None,
    ) -> list[CatalogItemSchema]:
        """Select an entire insertion block without mutating or regenerating its source.

        Count is exact and must fit the available space. Only a complete block is
        returned; errors discard partial selections. The observer reports catalog
        loading and selection counts, and never publishes provisional elements.
        """
        if not sequence.elements:
            raise ValueError("The source sequence must not be empty")
        available = MAX_SEQUENCE_ELEMENTS - len(sequence.elements)
        if type(count) is not int or not 1 <= count <= available:
            raise ValueError(f"count must be a positive integer within the available space ({available})")
        if not request.strip():
            raise ValueError("request must not be blank")
        insertion_index = len(sequence.elements)
        if insert_before_element_id is not None:
            insertion_index = next((
                index for index, element in enumerate(sequence.elements)
                if element.element_id == insert_before_element_id
            ), -1)
            if insertion_index == -1:
                raise ValueError("The insertion anchor is not in the source sequence")

        if progress_observer is not None:
            progress_observer(GeneratorAdditionProgress(0, count))
        catalog_items = self._load_catalog_items(progress_observer, require_catalog_uuid=True)
        fixed_prefix = sequence.elements[:insertion_index]
        fixed_suffix = sequence.elements[insertion_index:]
        source_items = [element.catalog_item for element in sequence.elements]
        additions: list[CatalogItemSchema] = []
        for offset in range(count):
            position = insertion_index + offset + 1
            first_item = additions[0] if insertion_index == 0 and additions else source_items[0]
            available_items = _pose_candidates(
                catalog_items, used_items=[*source_items, *additions], position=position,
                first_item=first_item,
                fixed_second_item=source_items[0] if insertion_index == 0 and count == 1 else None,
            )
            candidates, _ = _candidate_batch(
                available_items, is_last_element=position == len(source_items) + count,
            )
            selected = self._type_safe_service.choose_addition_catalog_item(
                candidates, fixed_prefix=fixed_prefix, fixed_suffix=fixed_suffix,
                selected_additions=additions, generation_request=sequence.generation_request,
                request=request, count=count,
            )
            additions.append(selected.model_copy(deep=True))
            if progress_observer is not None:
                progress_observer(GeneratorAdditionProgress(len(additions), count))
        return additions

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

        catalog_items = self._load_catalog_items(progress_observer)
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
            available_items = _pose_candidates(
                catalog_items, used_items=selected_items,
                position=len(selected_items) + 1, first_item=starting_item,
            )
            candidates, is_last_element = _candidate_batch(
                available_items, is_last_element=len(selected_items) == elements_count - 1,
                remaining_units=remaining_units,
            )
            if is_last_element:
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
