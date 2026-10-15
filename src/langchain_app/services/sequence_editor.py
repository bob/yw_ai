"""Validated edits that create new sequence variants without changing history."""

from collections.abc import Iterable
from uuid import UUID, uuid4

from pydantic import ValidationError

from langchain_app.schemas import (
    CatalogItemSchema,
    GeneratedSequence,
    GeneratedSequenceElement,
)
from langchain_app.services.sequence_rules import (
    MAX_SEQUENCE_ELEMENTS,
    has_catalog_uuid,
    second_pose_allowed,
    starting_pose_allowed,
)


class SequenceEditorError(ValueError):
    """An expected edit failure that can be returned to the assistant model."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def same_catalog_item(first: CatalogItemSchema, second: CatalogItemSchema) -> bool:
    return first.id == second.id or bool(
        first.uuid and second.uuid and first.uuid.casefold() == second.uuid.casefold()
    )


def insert_elements(
    sequence: GeneratedSequence, *, items: Iterable[CatalogItemSchema],
    count: int, request: str, insert_before_element_id: UUID | None = None,
) -> GeneratedSequence:
    """Validate an exact block and create a detached variant before any commit.

    Count is the already normalized number of additions, not a requested total.
    Existing catalog data and flags are preserved, including legacy repetitions.
    Only a changed first/second pair is checked against the shared pose rules.
    """
    if not sequence.elements:
        raise SequenceEditorError("empty_sequence", "The source sequence must not be empty")
    if type(count) is not int or count < 1:
        raise SequenceEditorError("invalid_arguments", "count must be a positive integer")
    if not isinstance(request, str) or not request.strip():
        raise SequenceEditorError("invalid_arguments", "request must not be blank")
    insertion_index = len(sequence.elements)
    if insert_before_element_id is not None:
        insertion_index = next((
            index for index, element in enumerate(sequence.elements)
            if element.element_id == insert_before_element_id
        ), -1)
        if insertion_index == -1:
            raise SequenceEditorError("sequence_element_not_found", "The insertion anchor is not in the selected sequence")
    if len(sequence.elements) + count > MAX_SEQUENCE_ELEMENTS:
        raise SequenceEditorError("invalid_addition_result", "The addition would exceed the sequence element limit")
    try:
        additions = list(items)
    except TypeError as error:
        raise SequenceEditorError("invalid_addition_result", "The addition must be a block of catalog items") from error
    if len(additions) != count:
        raise SequenceEditorError("invalid_addition_result", "The addition must contain exactly count catalog items")

    validated: list[CatalogItemSchema] = []
    source_items = [element.catalog_item for element in sequence.elements]
    used_items = list(source_items)
    for offset, item in enumerate(additions):
        if not isinstance(item, CatalogItemSchema):
            raise SequenceEditorError("invalid_addition_result", "The addition must contain catalog item schemas")
        try:
            # Revalidate fields even for model_copy/model_construct results.
            candidate = CatalogItemSchema.model_validate(item.model_dump(warnings=False), strict=True)
        except ValidationError as error:
            raise SequenceEditorError("invalid_addition_result", "The addition contains an invalid catalog item") from error
        if candidate.image_url is None or not has_catalog_uuid(candidate):
            raise SequenceEditorError("invalid_addition_result", "Every new item needs a thumbnail and a valid catalog UUID")
        if any(same_catalog_item(candidate, used) for used in used_items):
            raise SequenceEditorError("invalid_addition_result", "The addition contains a repeated catalog item")
        if not starting_pose_allowed(candidate, insertion_index + offset + 1):
            raise SequenceEditorError("invalid_addition_result", "New starting poses must occupy only the first position")
        validated.append(candidate.model_copy(deep=True))
        used_items.append(candidate)

    if insertion_index <= 1:
        result_items = [*source_items[:insertion_index], *validated, *source_items[insertion_index:]]
        if not second_pose_allowed(result_items[0], result_items[1]):
            raise SequenceEditorError("invalid_addition_result", "The addition violates the first/second pose constraints")

    variant = sequence.model_copy(deep=True, update={
        "uuid": uuid4(), "parent_uuid": sequence.uuid, "operation": "add",
        "operation_request": request,
    })
    variant.elements[insertion_index:insertion_index] = [
        GeneratedSequenceElement(
            order=insertion_index + offset + 1, catalog_item=item,
            start_mirror=item.bilateral, end_mirror=item.bilateral,
        )
        for offset, item in enumerate(validated)
    ]
    for order, element in enumerate(variant.elements, start=1):
        element.order = order
    return variant


def replace_element(
    sequence: GeneratedSequence, *, element_id: UUID,
    replacement_catalog_item_id: int, candidates: Iterable[CatalogItemSchema],
    request: str,
) -> GeneratedSequence | None:
    """Replace one occurrence; return None for a verified replacement with itself."""
    element = next((item for item in sequence.elements if item.element_id == element_id), None)
    if element is None:
        raise SequenceEditorError("sequence_element_not_found", "The occurrence is not in the selected sequence")
    replacement = next((item for item in candidates if item.id == replacement_catalog_item_id), None)
    if replacement is None:
        raise SequenceEditorError("catalog_item_not_verified", "Search the catalog again to verify this candidate")
    if replacement.image_url is None or not replacement.uuid or not replacement.uuid.strip():
        raise SequenceEditorError("catalog_item_not_usable", "The candidate needs an image URL and catalog UUID")
    if same_catalog_item(element.catalog_item, replacement):
        return None
    if any(
        other.element_id != element_id and same_catalog_item(other.catalog_item, replacement)
        for other in sequence.elements
    ):
        raise SequenceEditorError("duplicate_catalog_item", "The candidate already occurs in another position")

    variant = sequence.model_copy(deep=True, update={
        "uuid": uuid4(), "parent_uuid": sequence.uuid, "operation": "replace",
        "operation_request": request,
    })
    updated = next(item for item in variant.elements if item.element_id == element_id)
    updated.catalog_item = replacement.model_copy(deep=True)
    updated.start_mirror = updated.end_mirror = replacement.bilateral
    return variant


def swap_elements(
    sequence: GeneratedSequence, *, first_element_id: UUID,
    second_element_id: UUID, request: str,
) -> GeneratedSequence | None:
    """Swap two occurrences; return None when both IDs select the same occurrence."""
    first_index = next((
        index for index, item in enumerate(sequence.elements)
        if item.element_id == first_element_id
    ), None)
    second_index = next((
        index for index, item in enumerate(sequence.elements)
        if item.element_id == second_element_id
    ), None)
    if first_index is None or second_index is None:
        raise SequenceEditorError("sequence_element_not_found", "The occurrence is not in the selected sequence")
    if first_element_id == second_element_id:
        return None

    variant = sequence.model_copy(deep=True, update={
        "uuid": uuid4(), "parent_uuid": sequence.uuid, "operation": "swap",
        "operation_request": request,
    })
    variant.elements[first_index], variant.elements[second_index] = (
        variant.elements[second_index], variant.elements[first_index],
    )
    for order, element in enumerate(variant.elements, start=1):
        element.order = order
    return variant
