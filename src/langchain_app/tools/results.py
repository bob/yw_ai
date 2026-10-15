"""Model-visible projections; full application data belongs in state."""

import json

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from langchain_app.schemas import GeneratedSequence
from langchain_app.schemas.catalog import CatalogItemSchema, CatalogItemSearchResult
from langchain_app.schemas.assistant import AssistantState
from langchain_app.tools.context import AssistantToolContext


def catalog_item_result(item: CatalogItemSchema | CatalogItemSearchResult) -> dict:
    """Allow only the catalog facts needed to choose a verified candidate."""
    return item.model_dump(mode="json", include={
        "id", "uuid", "name", "bilateral", "catalog_tag_names",
    })


def commit_sequence(
    runtime: ToolRuntime[AssistantToolContext, AssistantState],
    sequence: GeneratedSequence,
    *,
    generation: dict | None = None,
    replacement: dict | None = None,
    swap: list[dict] | None = None,
    addition: dict | None = None,
) -> Command:
    """Commit the full snapshot and its compact acknowledgement together."""
    sequence_uuid = str(sequence.uuid)
    result = {
        "status": "ok", "uuid": sequence_uuid,
        "operation": sequence.operation, "elements_count": sequence.elements_count,
    }
    for key, facts in (("generation", generation), ("replacement", replacement), ("swap", swap), ("addition", addition)):
        if facts is not None:
            result[key] = facts
    return Command(update={
        "sequences": {sequence_uuid: sequence.model_dump(mode="json")},
        "active_sequence_uuid": sequence_uuid,
        "turn_operation_committed": True,
        "messages": [ToolMessage(
            content=json.dumps(result, ensure_ascii=False),
            tool_call_id=runtime.tool_call_id or "",
        )],
    })
