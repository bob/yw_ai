"""LangChain tools for the two public catalog endpoints."""

import json
from typing import Annotated

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import Field

from langchain_app.schemas.assistant import AssistantState
from langchain_app.services.catalog_errors import CatalogServiceError
from langchain_app.tools.context import AssistantToolContext, tool_error


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
        "items": snapshots,
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
