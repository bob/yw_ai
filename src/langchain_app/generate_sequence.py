import os
from typing import cast

import streamlit as st
from dotenv import load_dotenv

from langchain_app.dependencies import get_sequence_agent
from langchain_app.schemas import ChatMessage, Language
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

with st.form("sequence-generation-form", border=False):
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
    generator_factory=lambda: get_sequence_agent(
        ASANA_SERVICE_BASE_URL, ASANA_CATALOG_FOLDER_UUID
    ),
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
    "Describe the yoga sequence you want to generate…", key="sequence-prompt"
)
request = generated_prompt or prompt
if request:
    messages.append({"role": "user", "content": request})

    with st.chat_message("user"):
        st.markdown(request)

    response = sequence_renderer.render_response(
        request, content="", key="assistant-sequence-live"
    )
    if response.get("sequence") is not None:
        messages.append(response)
        if generated_prompt is not None:
            # Restore the generated text after rendering the stored sequence controls.
            st.session_state["sequence-prompt-prefill"] = generated_prompt
        st.rerun()
