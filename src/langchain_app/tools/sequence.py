"""Sequence tools; selection remains in the SequenceGenerator service."""

import json
from typing import Annotated
from uuid import UUID

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import Field

from langchain_app.schemas import CatalogItemSchema, GeneratedSequence
from langchain_app.schemas.assistant import AssistantState
from langchain_app.services.catalog_errors import CatalogServiceError
from langchain_app.services.sequence_editor import (
    SequenceEditorError,
    insert_elements,
    remove_elements,
    replace_element,
    swap_elements,
)
from langchain_app.services.sequence_generator import (
    MAX_SEQUENCE_ELEMENTS,
    generation_count,
)
from langchain_app.services.type_safe import TypeSafeServiceError
from langchain_app.tools.context import AssistantToolContext, tool_error
from langchain_app.tools.results import commit_sequence


@tool
def generate_sequence(
    request: Annotated[str, Field(min_length=1)],
    runtime: ToolRuntime[AssistantToolContext, AssistantState],
    count: Annotated[int | None, Field(
        gt=0, strict=True,
        description="Explicit requested pose count; null for duration-only requests or unspecified size.",
    )] = None,
    starting_catalog_item_id: Annotated[int | None, Field(
        gt=0, strict=True,
        description="Verified catalog item ID requested as the first pose; null uses a random start.",
    )] = None,
    duration_seconds: Annotated[float | None, Field(
        gt=0, allow_inf_nan=False,
        description="Requested duration in seconds; used only when count is null.",
    )] = None,
) -> Command | ToolMessage:
    """Create a new sequence whenever generation is requested, including when
    a sequence is already selected. "Создай секвенцию" requires this tool.

    Pass an explicit count unchanged; otherwise pass duration_seconds, or leave
    both null for 9 poses. Code gives count priority, calculates duration using
    40/80 seconds per ordinary/mirrored pose, and caps the result at 45 elements.
    A request for 50 poses means count=50: run without asking to accept the cap.
    For a requested first pose, resolve its ID by search or selected_sequence
    and pass starting_catalog_item_id; Other/Starting is not required. Clarify
    an ambiguous first pose; report an unavailable pose or missing thumbnail
    without substituting a random start. Null uses the code's random start.
    """
    if not request.strip():
        return tool_error(runtime, "invalid_arguments", "request must not be blank")
    context = runtime.context
    if context.generation_started or runtime.state.get("turn_operation_committed", False):
        return tool_error(runtime, "operation_already_started", "Generate at most once per request")
    requested_count = generation_count(count=count, duration_seconds=duration_seconds)
    effective_count = min(requested_count, MAX_SEQUENCE_ELEMENTS)
    if count is not None:
        duration_seconds = None
    context.generation_started = True
    generator = context.generator_factory()
    items = []
    for item in generator.generate(
        request, language="en", count=effective_count, starting_catalog_item_id=starting_catalog_item_id,
        **({"duration_seconds": duration_seconds} if duration_seconds is not None else {}),
        progress_observer=runtime.stream_writer,
    ):
        if len(items) >= MAX_SEQUENCE_ELEMENTS:
            return tool_error(
                runtime, "sequence_limit_exceeded",
                f"Generation must not return more than {MAX_SEQUENCE_ELEMENTS} elements",
            )
        items.append(item.model_copy(deep=True))
        runtime.stream_writer(item)
    if not items:
        return tool_error(runtime, "empty_sequence", "Generation returned no elements")
    sequence = GeneratedSequence.from_catalog_items(
        items, request=context.user_request, parent=context.regeneration_parent,
    )
    generation = None
    limit_applied = requested_count > MAX_SEQUENCE_ELEMENTS and (
        duration_seconds is None
        or sum(2 if item.bilateral else 1 for item in items) < requested_count
    )
    if limit_applied:
        generation = {"limit_applied": True, "limit": MAX_SEQUENCE_ELEMENTS}
        if count is not None:
            generation["requested_count"] = count
    return commit_sequence(runtime, sequence, generation=generation)


@tool
def add_sequence_elements(
    sequence_uuid: UUID,
    count: Annotated[int, Field(
        gt=0, strict=True,
        description="Requested number of NEW elements; the tool caps additions to available space.",
    )],
    request: Annotated[str, Field(min_length=1)],
    runtime: ToolRuntime[AssistantToolContext, AssistantState],
    insert_before_element_id: UUID | None = None,
) -> Command | ToolMessage:
    """Add one TypeSafe-selected block to the selected sequence.

    The sequence may grow beyond its original requested count; only the total
    limit of 45 elements is binding.
    Count NEW elements: default 1; "fill to T" means T-N, "fill to the limit"
    means 45-N. If the target size is reached or lower, explain without calling.
    Pass the full requested delta; code caps it to free space without confirmation,
    including when only part of the requested block fits. Mirrored poses
    count as one. For duration-only additions, clarify the number of elements:
    "Добавь асаны ещё на пять минут" -> ask how many; do not call this tool.
    Null insert_before_element_id appends; otherwise insert before that UUID.
    To prepend use the first occurrence; to insert after P use the next UUID
    or null after the last. TypeSafe chooses poses using fixed surroundings;
    do not supply catalog IDs, prepared elements, or mirror flags.
    """
    context = runtime.context
    if context.require_generation:
        return tool_error(runtime, "generation_required", "This application command requires generate_sequence")
    if context.generation_started or runtime.state.get("turn_operation_committed", False):
        return tool_error(runtime, "operation_already_started", "Change at most one sequence per request")
    active_uuid = runtime.state.get("active_sequence_uuid")
    sequences = runtime.state.get("sequences", {})
    snapshot = sequences.get(active_uuid) if active_uuid is not None else None
    if snapshot is None:
        return tool_error(runtime, "sequence_not_found", "Addition requires a selected sequence.")
    source = GeneratedSequence.from_snapshot(snapshot)
    if sequence_uuid != source.uuid:
        return tool_error(runtime, "stale_sequence", "Use the UUID of the currently selected sequence")
    if not request.strip():
        return tool_error(runtime, "invalid_arguments", "request must not be blank")
    if not source.elements:
        return tool_error(runtime, "empty_sequence", "The source sequence must not be empty")
    insertion_index = len(source.elements)
    if insert_before_element_id is not None:
        insertion_index = next((
            index for index, element in enumerate(source.elements)
            if element.element_id == insert_before_element_id
        ), -1)
        if insertion_index == -1:
            return tool_error(runtime, "sequence_element_not_found", "The insertion anchor is not in the selected sequence")
    available = max(0, MAX_SEQUENCE_ELEMENTS - len(source.elements))
    if available == 0:
        return ToolMessage(
            content=json.dumps({
                "status": "no_change", "code": "sequence_limit_reached",
                "message": f"The selected sequence has no space within the {MAX_SEQUENCE_ELEMENTS}-element limit",
                "uuid": str(source.uuid), "requested_count": count,
                "added_count": 0, "limit_applied": True,
            }),
            tool_call_id=runtime.tool_call_id or "",
        )
    added_count = min(count, available)
    generator = context.generator_factory()
    try:
        items = generator.add_elements(
            source, request, language=source.language, count=added_count,
            insert_before_element_id=insert_before_element_id,
            progress_observer=runtime.stream_writer,
        )
        variant = insert_elements(
            source, items=items, count=added_count, request=context.user_request,
            insert_before_element_id=insert_before_element_id,
        )
    except CatalogServiceError as exc:
        return tool_error(runtime, "catalog_search_failed", str(exc))
    except TypeSafeServiceError as exc:
        return tool_error(runtime, "addition_selection_failed", str(exc))
    except SequenceEditorError as exc:
        return tool_error(runtime, exc.code, str(exc))
    return commit_sequence(runtime, variant, addition={
        "requested_count": count, "added_count": added_count,
        "limit_applied": count > added_count, "position": insertion_index + 1,
        "insert_before_element_id": str(insert_before_element_id) if insert_before_element_id else None,
        "elements": [
            {"element_id": str(element.element_id), "order": element.order}
            for element in variant.elements[insertion_index:insertion_index + added_count]
        ],
    })


@tool
def remove_sequence_elements(
    sequence_uuid: UUID,
    element_ids: Annotated[list[UUID], Field(min_length=1, strict=True)],
    runtime: ToolRuntime[AssistantToolContext, AssistantState],
) -> Command | ToolMessage:
    """Remove one explicitly requested set of occurrences from the selected sequence.

    Resolve the COMPLETE requested set before calling: every position must be
    in 1..N and every name must match exactly one occurrence in the snapshot.
    If any reference is invalid, absent or ambiguous, ask without tools; never
    remove a valid subset or clamp an invalid position. A missing target or a
    request to shorten the sequence requires asking which poses to remove.
    Deduplicate references to the same occurrence and execute once, passing
    unique occurrence UUIDs for the whole set, never positions or catalog IDs.
    At least one occurrence must remain. No catalog search or pose selection.

    For a five-pose snapshot, requests to remove second AND sixth, or zeroth
    AND second, require clarification with ZERO calls: the whole set is invalid.
    A missing named pose invalidates a mixed set even if another position exists.
    If the third occurrence is the unique Tree pose, "remove third and Tree pose"
    authorizes removing that single occurrence ONCE, without a confirmation.
    """
    context = runtime.context
    if context.require_generation:
        return tool_error(runtime, "generation_required", "This application command requires generate_sequence")
    if context.generation_started or runtime.state.get("turn_operation_committed", False):
        return tool_error(runtime, "operation_already_started", "Change at most one sequence per request")
    active_uuid = runtime.state.get("active_sequence_uuid")
    sequences = runtime.state.get("sequences", {})
    snapshot = sequences.get(active_uuid) if active_uuid is not None else None
    if snapshot is None:
        return tool_error(runtime, "sequence_not_found", "Removal requires a selected sequence.")
    source = GeneratedSequence.from_snapshot(snapshot)
    if sequence_uuid != source.uuid:
        return tool_error(runtime, "stale_sequence", "Use the UUID of the currently selected sequence")
    try:
        variant = remove_elements(source, element_ids=element_ids, request=context.user_request)
    except SequenceEditorError as exc:
        return tool_error(runtime, exc.code, str(exc))
    targets = set(element_ids)
    removed = [element for element in source.elements if element.element_id in targets]
    return commit_sequence(runtime, variant, removal={
        "removed_count": len(removed),
        "elements": [
            {"element_id": str(element.element_id), "name": element.catalog_item.name, "old_order": element.order}
            for element in removed
        ],
    })


@tool
def replace_sequence_element(
    sequence_uuid: UUID,
    element_id: UUID,
    replacement_catalog_item_id: Annotated[int, Field(gt=0, strict=True)],
    runtime: ToolRuntime[AssistantToolContext, AssistantState],
) -> Command | ToolMessage:
    """Replace element_id in sequence_uuid using replacement_catalog_item_id
    from the latest verified catalog search. Creates a new variant.
    """
    context = runtime.context
    if context.require_generation:
        return tool_error(runtime, "generation_required", "This application command requires generate_sequence")
    if context.generation_started or runtime.state.get("turn_operation_committed", False):
        return tool_error(runtime, "operation_already_started", "Change at most one sequence per request")
    active_uuid = runtime.state.get("active_sequence_uuid")
    sequences = runtime.state.get("sequences", {})
    snapshot = sequences.get(active_uuid) if active_uuid is not None else None
    if snapshot is None:
        return tool_error(
            runtime, "sequence_not_found",
            "Replacement requires a selected sequence.",
        )
    source = GeneratedSequence.from_snapshot(snapshot)
    if sequence_uuid != source.uuid:
        return tool_error(runtime, "stale_sequence", "Use the UUID of the currently selected sequence")
    try:
        variant = replace_element(
            source, element_id=element_id, replacement_catalog_item_id=replacement_catalog_item_id,
            candidates=[CatalogItemSchema.model_validate(item) for item in runtime.state.get("latest_catalog_items", [])],
            request=context.user_request,
        )
    except SequenceEditorError as exc:
        return tool_error(runtime, exc.code, str(exc))
    if variant is None:
        return ToolMessage(
            content=json.dumps({"status": "no_change", "uuid": str(source.uuid)}),
            tool_call_id=runtime.tool_call_id or "",
        )
    old_element = next(item for item in source.elements if item.element_id == element_id)
    new_element = next(item for item in variant.elements if item.element_id == element_id)
    return commit_sequence(runtime, variant, replacement={
        "element_id": str(element_id), "order": old_element.order,
        "old_catalog_item": {"id": old_element.catalog_item.id, "name": old_element.catalog_item.name},
        "new_catalog_item": {"id": new_element.catalog_item.id, "name": new_element.catalog_item.name},
    })


@tool
def swap_sequence_elements(
    sequence_uuid: UUID,
    first_element_id: UUID,
    second_element_id: UUID,
    runtime: ToolRuntime[AssistantToolContext, AssistantState],
) -> Command | ToolMessage:
    """Apply an explicitly requested swap of one fully identified pair.

    Do not call for hypothetical questions, ambiguous references, or requests
    for multiple operations; clarify those first. Exchange first_element_id
    and second_element_id in sequence_uuid,
    creating a new variant. Intermediate positions stay fixed. Distinct
    occurrences of the same catalog item can swap. No catalog search needed.
    A selected snapshot alone never resolves a pronoun: "Поменяй её местами
    с последней позой" with no conversational antecedent requires asking which
    occurrence is meant, with ZERO calls. Never guess the first occurrence.
    """
    context = runtime.context
    if context.require_generation:
        return tool_error(runtime, "generation_required", "This application command requires generate_sequence")
    if context.generation_started or runtime.state.get("turn_operation_committed", False):
        return tool_error(runtime, "operation_already_started", "Change at most one sequence per request")
    active_uuid = runtime.state.get("active_sequence_uuid")
    sequences = runtime.state.get("sequences", {})
    snapshot = sequences.get(active_uuid) if active_uuid is not None else None
    if snapshot is None:
        return tool_error(runtime, "sequence_not_found", "Swapping requires a selected sequence.")
    source = GeneratedSequence.from_snapshot(snapshot)
    if sequence_uuid != source.uuid:
        return tool_error(runtime, "stale_sequence", "Use the UUID of the currently selected sequence")
    try:
        variant = swap_elements(
            source, first_element_id=first_element_id, second_element_id=second_element_id,
            request=context.user_request,
        )
    except SequenceEditorError as exc:
        return tool_error(runtime, exc.code, str(exc))
    if variant is None:
        return ToolMessage(
            content=json.dumps({"status": "no_change", "uuid": str(source.uuid)}),
            tool_call_id=runtime.tool_call_id or "",
        )
    old_elements = {item.element_id: item for item in source.elements}
    new_orders = {item.element_id: item.order for item in variant.elements}
    return commit_sequence(runtime, variant, swap=[
        {
            "element_id": str(element_id), "name": old_elements[element_id].catalog_item.name,
            "old_order": old_elements[element_id].order, "new_order": new_orders[element_id],
        }
        for element_id in (first_element_id, second_element_id)
    ])
