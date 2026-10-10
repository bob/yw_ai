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
from langchain_app.services.sequence_editor import SequenceEditorError, replace_element
from langchain_app.services.sequence_generator import MAX_SEQUENCE_ELEMENTS
from langchain_app.tools.context import AssistantToolContext, tool_error


@tool
def generate_sequence(
    request: Annotated[str, Field(min_length=1)],
    count: Annotated[int, Field(
        gt=0, le=MAX_SEQUENCE_ELEMENTS, strict=True,
        description="Pose count or duration-based upper estimate; mandatory maximum is 45 elements.",
    )],
    runtime: ToolRuntime[AssistantToolContext, AssistantState],
    starting_catalog_item_id: Annotated[int | None, Field(
        gt=0, strict=True,
        description="Verified catalog item ID requested as the first pose; null uses a random start.",
    )] = None,
    duration_seconds: Annotated[float | None, Field(
        gt=0, allow_inf_nan=False,
        description="Requested practice duration in seconds; null when an explicit pose count is given.",
    )] = None,
) -> Command | ToolMessage:
    """Generate a yoga sequence using the user's full request and inferred pose count.

    Preserve every preference and constraint in request. Use an explicit count
    when present and leave duration_seconds null. Otherwise infer an upper count
    from duration using 40 seconds per ordinary pose, round halves up, and pass
    the requested duration_seconds. Mirrored (bilateral) poses take 80 seconds,
    so the generator adjusts the actual element count during selection.
    Mandatory rule: never generate more than 45 elements, including when the
    user requests a higher count or a longer duration. Cap count at 45 and
    explain the limit. A mirrored pose is one element but takes twice the time.
    If the user specifies a first pose, resolve it using catalog search or the
    selected sequence and pass starting_catalog_item_id. It takes priority over
    the random start and does not require the Other/Starting tag. Leave it null
    when no first pose is requested; random positions are lying 10%, sitting 60%,
    standing 30%. After a lying start the second pose must be lying on the back;
    after a standing start prefer standing and never choose a lying second pose.
    """
    if not request.strip():
        return tool_error(runtime, "invalid_arguments", "request must not be blank")
    context = runtime.context
    if context.generation_started or runtime.state.get("turn_operation_committed", False):
        return tool_error(runtime, "operation_already_started", "Generate at most once per request")
    context.generation_started = True
    generator = context.generator_factory()
    items = []
    for item in generator.generate(
        request, language="en", count=count, starting_catalog_item_id=starting_catalog_item_id,
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
    snapshot = sequence.model_dump(mode="json")
    return Command(update={
        "sequences": {str(sequence.uuid): snapshot},
        "active_sequence_uuid": str(sequence.uuid),
        "turn_operation_committed": True,
        "messages": [ToolMessage(
            content=json.dumps({"status": "ok", **snapshot}, ensure_ascii=False),
            tool_call_id=runtime.tool_call_id or "",
        )],
    })


@tool
def replace_sequence_element(
    sequence_uuid: UUID,
    element_id: UUID,
    replacement_catalog_item_id: Annotated[int, Field(gt=0, strict=True)],
    runtime: ToolRuntime[AssistantToolContext, AssistantState],
) -> Command | ToolMessage:
    """Replace one occurrence in the selected sequence with a verified catalog candidate.

    Use UUIDs from the current sequence snapshot and an item ID from catalog
    search. Call only when the user asks to apply a replacement, not to find alternatives.
    Replace at most one element per user request. This does not add, delete, move,
    or independently change the mirroring of an element.
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
    snapshot = variant.model_dump(mode="json")
    old_element = next(item for item in source.elements if item.element_id == element_id)
    new_element = next(item for item in variant.elements if item.element_id == element_id)
    return Command(update={
        "sequences": {str(variant.uuid): snapshot},
        "active_sequence_uuid": str(variant.uuid),
        "turn_operation_committed": True,
        "messages": [ToolMessage(
            content=json.dumps({
                "status": "ok", **snapshot,
                "replacement": {
                    "element_id": str(element_id), "order": old_element.order,
                    "old_catalog_item": old_element.catalog_item.model_dump(mode="json"),
                    "new_catalog_item": new_element.catalog_item.model_dump(mode="json"),
                },
            }, ensure_ascii=False),
            tool_call_id=runtime.tool_call_id or "",
        )],
    })
