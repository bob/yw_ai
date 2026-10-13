"""LangChain tools for catalog search, tags, and explicit display selection."""

import json
from typing import Annotated

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import Field, ValidationError

from langchain_app.schemas.catalog import CatalogItemSchema
from langchain_app.schemas.assistant import AssistantState
from langchain_app.services.catalog_errors import CatalogServiceError
from langchain_app.tools.context import AssistantToolContext, tool_error
from langchain_app.tools.results import catalog_item_result


@tool
def search_catalog_items(
    runtime: ToolRuntime[AssistantToolContext, AssistantState],
    name: str | None = None,
    uuid: str | None = None,
    bilateral: bool | None = None,
    catalog_tag_names: list[str] | None = None,
    limit: Annotated[int, Field(ge=1, le=50, strict=True)] = 10,
    rand: bool = False,
) -> Command:
    """Search real catalog items using English names and API-provided tags.

    Filters combine with AND; tag filters match any supplied tag (OR).
    Name and tag matching is partial and case-insensitive. UUID matching is exact.
    Results are limited, not ranked by relevance; an empty list is a valid result.
    Searching does not display images. For user browsing, follow with
    show_catalog_items using selected IDs from the latest verified results.
    """
    try:
        items = runtime.context.catalog_item_service.search(
            name=name, uuid=uuid, bilateral=bilateral,
            catalog_tag_names=catalog_tag_names,
            with_tags=True, with_rigman=False, limit=limit, rand=rand,
        )
    except CatalogServiceError as exc:
        return Command(update={
            "latest_catalog_items": [], "messages": [tool_error(runtime, exc.code, str(exc))],
        })
    except ValueError as exc:
        return Command(update={
            "latest_catalog_items": [], "messages": [tool_error(runtime, "invalid_arguments", str(exc))],
        })
    snapshots = [item.model_dump(
        mode="json", exclude={"rigman_data", "second_side_rigman_data"}
    ) for item in items]
    result = {
        "status": "ok", "language": "en", "returned_count": len(items),
        "limit": limit,
        "items": [catalog_item_result(item) for item in items],
    }
    return Command(update={
        "latest_catalog_items": snapshots,
        "messages": [ToolMessage(
            content=json.dumps(result, ensure_ascii=False), tool_call_id=runtime.tool_call_id or "",
        )],
    })


@tool
def get_catalog_tag_names(runtime: ToolRuntime[AssistantToolContext]) -> dict | ToolMessage:
    """Get real English catalog tag names before choosing unknown tag filters."""
    try:
        names = runtime.context.catalog_tag_service.get_names()
    except CatalogServiceError as exc:
        return tool_error(runtime, exc.code, str(exc))
    return {"status": "ok", "language": "en", "tag_names": names}


@tool
def show_catalog_items(
    catalog_item_ids: Annotated[
        list[Annotated[int, Field(gt=0, strict=True)]], Field(min_length=1),
    ],
    runtime: ToolRuntime[AssistantToolContext, AssistantState],
) -> ToolMessage:
    """Show selected real catalog items for the user's requested browsing or selection.

    Use for requests to view available poses, questions about which poses exist,
    follow-up requests for more options, or finding replacement candidates
    without changing a sequence. The user need not say "show"; "do not change
    anything yet" permits display and does not require confirmation. If display
    and a sequence operation are separately requested, display their results
    separately. Choose IDs from the latest successful catalog
    search, in display order. Call once with the complete selection. Do not use
    for internal searches during generation, replacement, technique answers,
    or tag requests alone. Empty or failed searches do not authorize display.
    This does not generate, replace, or select a sequence and is unavailable
    for mandatory Generate or Regenerate commands.
    """
    context = runtime.context
    if context.require_generation or context.regeneration_parent is not None:
        return tool_error(runtime, "generation_required", "This application command requires generate_sequence")
    if context.catalog_items_displayed:
        return tool_error(runtime, "catalog_items_already_displayed", "Show catalog items at most once per request")
    if len(set(catalog_item_ids)) != len(catalog_item_ids):
        return tool_error(runtime, "invalid_arguments", "catalog_item_ids must not contain duplicates")
    try:
        candidates = {
            item.id: item for item in (
                CatalogItemSchema.model_validate(snapshot)
                for snapshot in runtime.state.get("latest_catalog_items", [])
            )
        }
    except ValidationError:
        return tool_error(runtime, "invalid_catalog_items", "Catalog candidates contain invalid data; search again")
    if any(item_id not in candidates for item_id in catalog_item_ids):
        return tool_error(runtime, "catalog_item_not_verified", "Use IDs from the latest verified catalog search")
    snapshots = [candidates[item_id].model_dump(mode="json") for item_id in catalog_item_ids]
    result = ToolMessage(
        content=json.dumps({
            "status": "ok", "displayed_count": len(snapshots), "catalog_item_ids": catalog_item_ids,
        }),
        artifact={"items": snapshots},
        tool_call_id=runtime.tool_call_id or "",
    )
    context.catalog_items_displayed = True
    return result
