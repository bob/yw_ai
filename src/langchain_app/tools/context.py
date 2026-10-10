"""Application dependencies injected into tools, never supplied by the model."""

import json
from collections.abc import Callable
from dataclasses import dataclass

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage

from langchain_app.schemas.generated_sequence import GeneratedSequence
from langchain_app.services.catalog_item import CatalogItemService
from langchain_app.services.catalog_tag import CatalogTagService
from langchain_app.services.sequence_generator import SequenceGenerator


@dataclass
class AssistantToolContext:
    """Dependencies and operation status scoped to a single request."""

    generator_factory: Callable[[], SequenceGenerator]
    catalog_item_service: CatalogItemService
    catalog_tag_service: CatalogTagService
    user_request: str = ""
    require_generation: bool = False
    regeneration_parent: GeneratedSequence | None = None
    generation_started: bool = False
    generation_completed: bool = False
    catalog_items_displayed: bool = False


def tool_error[ContextT, StateT](
    runtime: ToolRuntime[ContextT, StateT], code: str, message: str,
) -> ToolMessage:
    return ToolMessage(
        content=json.dumps({"status": "error", "code": code, "message": message}),
        tool_call_id=runtime.tool_call_id or "",
        status="error",
    )
