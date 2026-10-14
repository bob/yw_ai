"""Streamlit entrypoint for the yoga assistant."""

import os
from typing import cast

import streamlit as st
from dotenv import load_dotenv

from langchain_app.dependencies import get_assistant_agent
from langchain_app.schemas import ChatMessage, Language
from langchain_app.services.assistant_agent import AssistantAgent
from langchain_app.ui import SequenceRenderer
from langchain_app.ui.chat_input import focus_chat_input
from langchain_app.ui.response_progress import (
    PendingResponse,
    enqueue_response,
    retain_execution,
)

load_dotenv()

SEQUENCE_LANGUAGE: Language = "en"
ASANA_SERVICE_BASE_URL = os.getenv(
    "ASANA_SERVICE_BASE_URL", "http://0.0.0.0:4999"
)
ASANA_CATALOG_FOLDER_UUID = os.getenv(
    "ASANA_CATALOG_FOLDER_UUID", "yoga-public-v1"
)


st.title("Sequence Generator")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending_response", None)
messages = cast(list[ChatMessage], st.session_state.messages)


def session_assistant() -> AssistantAgent:
    """Create dependencies lazily and keep conversation state local to this session."""
    if "assistant_agent" not in st.session_state:
        st.session_state["assistant_agent"] = get_assistant_agent(
            ASANA_SERVICE_BASE_URL, ASANA_CATALOG_FOLDER_UUID,
        )
    return st.session_state["assistant_agent"]


def generate_sequence() -> None:
    if st.session_state.pending_response is not None:
        return
    request = f"Generate a yoga sequence with {st.session_state['sequence-elements-count']} poses."
    st.session_state["sequence-prompt"] = ""
    enqueue_response(PendingResponse(request, require_generation=True))


def submit_chat() -> None:
    if request := st.session_state.get("sequence-prompt"):
        enqueue_response(PendingResponse(request))


pending = cast(PendingResponse | None, st.session_state.pending_response)
busy = pending is not None

# Form for sequence generation on the top
with st.form("sequence-generation-form", border=False):  # noqa: SIM117
    with st.container(horizontal=True, vertical_alignment="center"):
        elements_count = st.selectbox(
            "Number of elements",
            options=range(9, 37),
            index=0,
            key="sequence-elements-count",
            label_visibility="collapsed",
            width=100,
            disabled=busy,
        )
        st.form_submit_button(
            "Generate", type="primary", key="generate-sequence",
            disabled=busy, on_click=generate_sequence,
        )

sequence_renderer = SequenceRenderer(
    generator_factory=session_assistant,
    language=SEQUENCE_LANGUAGE,
    catalog_url=ASANA_SERVICE_BASE_URL,
)
sequence_renderer.render_styles()
sequence_renderer.render_messages(messages, disabled=busy)

reset_prompt = st.session_state.pop("sequence-prompt-reset", False)
if reset_prompt:
    st.session_state["sequence-prompt"] = ""

st.chat_input(
    "Ask about yoga, search the catalog, or generate or edit a sequence…", key="sequence-prompt",
    disabled=busy, submit_mode="disable", on_submit=submit_chat,
)
if reset_prompt:
    focus_chat_input(reset_prompt)

if pending is not None:
    try:
        response = sequence_renderer.render_response(
            pending.request, content="", key="assistant-sequence-live", pending=pending,
        )
        if pending.message_index is None:
            # Every attempt is retained, including errors and empty public replies.
            messages.append(response)
        else:
            message = messages[pending.message_index]
            execution = response.get("execution")
            if execution is not None:
                retain_execution(message, execution)
            if response["content"] or response.get("sequence") is not None:
                message["content"] = response["content"]
            if (sequence := response.get("sequence")) is not None:
                message["sequence"] = sequence
                message["sequence_uuid"] = response.get("sequence_uuid")
                message["sequence_variant_uuids"] = response.get("sequence_variant_uuids", [])
                message["sequence_request"] = response.get("sequence_request", pending.request)
                message["sequence_variants"] = response.get("sequence_variants", [sequence])
                message["selected_sequence_index"] = response.get("selected_sequence_index", 0)
    finally:
        st.session_state.pending_response = None
        st.session_state["sequence-prompt-reset"] = pending.request_id
    st.rerun()
