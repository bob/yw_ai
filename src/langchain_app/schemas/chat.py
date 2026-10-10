"""Schemas for messages stored in the Streamlit chat session."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

from langchain_app.schemas.response_progress import ExecutionSnapshot

if TYPE_CHECKING:
    from langchain_app.ui import AsanaThumbnail, SequenceInfo, SequenceView


class _OptionalChatMessageFields(TypedDict, total=False):
    execution: ExecutionSnapshot | None
    execution_history: list[ExecutionSnapshot]
    sequence_uuid: str | None
    sequence_variant_uuids: list[str]
    sequence: SequenceView | None
    sequence_request: str
    sequence_variants: list[SequenceView]
    selected_sequence_index: int
    sequence_info: SequenceInfo
    thumbnails: list[AsanaThumbnail]


class ChatMessage(_OptionalChatMessageFields):
    role: Literal["user", "assistant"]
    content: str
