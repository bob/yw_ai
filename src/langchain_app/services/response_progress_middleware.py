"""Publish real model/tool work without coupling the agent to a UI."""

import json
from collections.abc import Callable
from time import monotonic
from uuid import UUID, uuid4

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, ToolCallRequest, wrap_model_call, wrap_tool_call
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from langchain_app.schemas import CatalogItemSchema
from langchain_app.schemas.assistant import AssistantState
from langchain_app.schemas.response_progress import (
    AssistantMessageEvent, CatalogItemsDisplayedEvent, SequenceElementEvent,
    StepFinishedEvent, StepProgressEvent, StepStartedEvent,
)
from langchain_app.services.response_progress import PendingSequenceCommit, ProgressSignal
from langchain_app.services.sequence_generator import GeneratorSearchProgress
from langchain_app.tools.context import AssistantToolContext

TOOL_STEPS = {
    "get_catalog_tag_names": ("loading_catalog_tags", "Loading catalog tags..."),
    "search_catalog_items": ("searching_catalog_items", "Searching catalog items..."),
    "show_catalog_items": ("showing_catalog_items", "Showing catalog items..."),
    "generate_sequence": ("generating_sequence", "Generating sequence..."),
    "replace_sequence_element": ("updating_sequence", "Updating sequence..."),
}


@wrap_model_call(state_schema=AssistantState)
def model_response_progress(
    request: ModelRequest[AssistantToolContext],
    handler: Callable[[ModelRequest[AssistantToolContext]], ModelResponse],
) -> ModelResponse:
    writer = request.runtime.stream_writer
    step_id = uuid4()
    started_at = monotonic()
    writer(ProgressSignal(StepStartedEvent, {
        "step_id": step_id, "code": "thinking", "title": "Thinking...",
    }))
    try:
        response = handler(request)
    except Exception:
        writer(ProgressSignal(StepFinishedEvent, {
            "step_id": step_id, "outcome": "error", "duration_seconds": monotonic() - started_at,
            "summary": "Model request failed.",
        }))
        raise
    for message in response.result:
        if isinstance(message, AIMessage) and message.tool_calls and message.text.strip():
            writer(ProgressSignal(AssistantMessageEvent, {
                "step_id": step_id, "message_id": message.id,
                "purpose": "intermediate", "text": message.text,
            }))
    writer(ProgressSignal(StepFinishedEvent, {
        "step_id": step_id, "outcome": "ok", "duration_seconds": monotonic() - started_at,
        "summary": "Model response received.",
    }))
    for message in response.result:
        if not isinstance(message, AIMessage) or message.tool_calls or not message.text.strip():
            continue
        final_step = uuid4()
        final_started_at = monotonic()
        writer(ProgressSignal(StepStartedEvent, {
            "step_id": final_step, "code": "preparing_response", "title": "Preparing response...",
        }))
        writer(ProgressSignal(AssistantMessageEvent, {
            "step_id": final_step, "message_id": message.id, "purpose": "final", "text": message.text,
        }))
        writer(ProgressSignal(StepFinishedEvent, {
            "step_id": final_step, "outcome": "ok", "duration_seconds": monotonic() - final_started_at,
            "summary": "Response prepared.",
        }))
    return response


def _tool_result(result: ToolMessage | Command, tool_name: str) -> tuple[str, str]:
    messages = (result.update or {}).get("messages", []) if isinstance(result, Command) else [result]
    message = next((m for m in messages if isinstance(m, ToolMessage)), None)
    if message is None:
        return "error", "Tool returned no result."
    try:
        payload = json.loads(message.text)
    except (ValueError, TypeError):
        payload = {}
    if message.status == "error" or payload.get("status") == "error":
        return "error", "Could not complete this tool call."
    if payload.get("status") == "no_change":
        return "no_change", "The selected item is already present; no change was needed."
    if tool_name == "get_catalog_tag_names":
        return "ok", f"Loaded {len(payload.get('tag_names', []))} catalog tags."
    if tool_name == "search_catalog_items":
        return "ok", f"Found {payload.get('returned_count', 0)} catalog items."
    if tool_name == "show_catalog_items":
        return "ok", f"Displayed {payload['displayed_count']} catalog items."
    if tool_name == "generate_sequence":
        return "ok", f"Sequence created with {payload['elements_count']} poses."
    return "ok", "Sequence element replaced."


def _tool_response_progress(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    name = request.tool_call["name"]
    # Unregistered tools have no real application operation to report.
    if name not in TOOL_STEPS:
        return handler(request)
    code, title = TOOL_STEPS[name]
    writer = request.runtime.stream_writer
    step_id = uuid4()
    started_at = monotonic()
    identity = {"step_id": step_id, "tool_call_id": request.tool_call["id"]}
    writer(ProgressSignal(StepStartedEvent, {**identity, "code": code, "title": title}))
    count = request.tool_call["args"].get("count") if name == "generate_sequence" else None
    total = count if type(count) is int and count > 0 else None
    order = 0
    search_id = None
    search_started_at = 0.0

    def publish(data: object) -> None:
        nonlocal order, search_id, search_started_at
        if isinstance(data, CatalogItemSchema):
            order += 1
            writer(ProgressSignal(SequenceElementEvent, {
                **identity, "order": order, "catalog_item": data.model_copy(deep=True),
            }))
            writer(ProgressSignal(StepProgressEvent, {
                **identity, "completed": order, "total": total if total is None or order <= total else None,
            }))
        elif isinstance(data, GeneratorSearchProgress):
            if data.phase == "started":
                search_id = uuid4()
                search_started_at = monotonic()
                writer(ProgressSignal(StepStartedEvent, {
                    "step_id": search_id, "tool_call_id": identity["tool_call_id"],
                    "parent_step_id": step_id, "code": "searching_catalog_items",
                    "title": "Searching catalog items...",
                }))
            elif search_id is not None:
                summary = (f"Found {data.candidate_count} usable catalog candidates."
                           if data.outcome == "ok" else "Catalog search failed.")
                writer(ProgressSignal(StepFinishedEvent, {
                    "step_id": search_id, "tool_call_id": identity["tool_call_id"],
                    "outcome": data.outcome, "duration_seconds": monotonic() - search_started_at,
                    "summary": summary,
                }))
                if data.outcome == "ok":
                    # Resume the parent's title before selecting the first pose.
                    writer(ProgressSignal(StepProgressEvent, {
                        **identity, "completed": order, "total": total,
                    }))
                search_id = None
        else:
            writer(data)

    if total is not None:
        writer(ProgressSignal(StepProgressEvent, {**identity, "completed": 0, "total": total}))
    request.runtime.stream_writer = publish
    try:
        result = handler(request)
        outcome, summary = _tool_result(result, name)
        if name == "show_catalog_items" and outcome == "ok" and isinstance(result, ToolMessage):
            items = [
                CatalogItemSchema.model_validate(snapshot).model_copy(deep=True)
                for snapshot in result.artifact["items"]
            ]
            writer(ProgressSignal(CatalogItemsDisplayedEvent, {**identity, "items": items}))
    except Exception:
        writer(ProgressSignal(StepFinishedEvent, {
            **identity, "outcome": "error", "duration_seconds": monotonic() - started_at,
            "summary": "Could not complete this tool call.",
        }))
        raise
    finally:
        request.runtime.stream_writer = writer
    finished = ProgressSignal(StepFinishedEvent, {
        **identity, "outcome": outcome, "duration_seconds": monotonic() - started_at, "summary": summary,
    })
    if (isinstance(result, Command) and result.update is not None
            and outcome == "ok" and result.update.get("turn_operation_committed")):
        writer(PendingSequenceCommit(UUID(result.update["active_sequence_uuid"]), finished))
    else:
        writer(finished)
    return result


tool_response_progress: AgentMiddleware[AssistantState, AssistantToolContext] = wrap_tool_call(
    state_schema=AssistantState, name="tool_response_progress",
)(_tool_response_progress)
