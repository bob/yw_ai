"""Validated edits that create new sequence variants without changing history."""

from collections.abc import Iterable
from uuid import UUID, uuid4

from langchain_app.schemas import CatalogItemSchema, GeneratedSequence


class SequenceEditorError(ValueError):
    """An expected edit failure that can be returned to the assistant model."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def same_catalog_item(first: CatalogItemSchema, second: CatalogItemSchema) -> bool:
    return first.id == second.id or bool(
        first.uuid and second.uuid and first.uuid.casefold() == second.uuid.casefold()
    )


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
