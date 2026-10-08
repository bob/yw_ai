"""Conversation state owned by one assistant session."""

from typing import Annotated, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import InputAgentState


def merge_sequences(previous: dict, incoming: dict) -> dict:
    """Add whole snapshots without appending elements or mutating old variants."""
    return {**previous, **incoming}


class AssistantInputState(InputAgentState):
    active_sequence_uuid: NotRequired[str | None]
    turn_operation_committed: NotRequired[bool]


class AssistantState(AgentState):
    sequences: NotRequired[Annotated[dict[str, dict], merge_sequences]]
    active_sequence_uuid: NotRequired[str | None]
    latest_catalog_items: NotRequired[list[dict]]
    turn_operation_committed: NotRequired[bool]
