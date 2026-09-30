"""Schemas for messages stored in the Streamlit chat session."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from langchain_app.ui import AsanaThumbnail, SequenceInfo, SequenceView


class _OptionalChatMessageFields(TypedDict, total=False):
    sequence: SequenceView | None
    sequence_info: SequenceInfo
    thumbnails: list[AsanaThumbnail]


class ChatMessage(_OptionalChatMessageFields):
    role: Literal["user", "assistant"]
    content: str
