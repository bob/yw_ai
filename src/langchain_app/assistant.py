"""Streamlit entrypoint for the yoga assistant."""

import os
from typing import cast

import streamlit as st
from dotenv import load_dotenv

from langchain_app.dependencies import get_assistant_agent
from langchain_app.schemas import ChatMessage, Language
from langchain_app.services.assistant_agent import AssistantAgent
from langchain_app.ui import SequenceRenderer

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
messages = cast(list[ChatMessage], st.session_state.messages)


def session_assistant() -> AssistantAgent:
    """Create dependencies lazily and keep conversation state local to this session."""
    if "assistant_agent" not in st.session_state:
        st.session_state["assistant_agent"] = get_assistant_agent(
            ASANA_SERVICE_BASE_URL, ASANA_CATALOG_FOLDER_UUID,
        )
    return st.session_state["assistant_agent"]

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
        )
        generate = st.form_submit_button(
            "Generate", type="primary", key="generate-sequence"
        )

sequence_renderer = SequenceRenderer(
    generator_factory=session_assistant,
    language=SEQUENCE_LANGUAGE,
    catalog_url=ASANA_SERVICE_BASE_URL,
)
sequence_renderer.render_styles()
sequence_renderer.render_messages(messages)

generated_prompt = None
if generate:
    generated_prompt = f"Generate a yoga sequence with {elements_count} poses."
    st.session_state["sequence-prompt"] = generated_prompt
elif prefill := st.session_state.pop("sequence-prompt-prefill", None):
    st.session_state["sequence-prompt"] = prefill

prompt = st.chat_input(
    "Ask about yoga, search the catalog, or generate or edit a sequence…", key="sequence-prompt"
)
request = generated_prompt or prompt
if request:
    messages.append({"role": "user", "content": request})

    with st.chat_message("user"):
        st.markdown(request)

    response = sequence_renderer.render_response(
        request, content="", key="assistant-sequence-live",
        require_generation=generated_prompt is not None,
    )
    if response.get("sequence") is not None or (response["content"] and generated_prompt is None):
        messages.append(response)
        if generated_prompt is not None:
            # Restore the generated text after rendering the stored sequence controls.
            st.session_state["sequence-prompt-prefill"] = generated_prompt
        st.rerun()
