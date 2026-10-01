import os
from typing import cast

import streamlit as st
from dotenv import load_dotenv

from langchain_app.dependencies import get_sequence_generator
from langchain_app.schemas import ChatMessage, Language
from langchain_app.ui import SequenceRenderer

load_dotenv()

SEQUENCE_LANGUAGE: Language = "en"
SEQUENCE_ELEMENTS_COUNT = 9
ASANA_SERVICE_BASE_URL = os.getenv(
    "ASANA_SERVICE_BASE_URL", "http://0.0.0.0:4999"
)
ASANA_CATALOG_FOLDER_UUID = os.getenv(
    "ASANA_CATALOG_FOLDER_UUID", "yoga-public-v1"
)


st.title("Sequence Viewer")
sequence_renderer = SequenceRenderer(
    generator_factory=lambda: get_sequence_generator(
        ASANA_SERVICE_BASE_URL, ASANA_CATALOG_FOLDER_UUID
    ),
    language=SEQUENCE_LANGUAGE,
    elements_count=SEQUENCE_ELEMENTS_COUNT,
)
sequence_renderer.render_styles()
st.session_state.setdefault("messages", [])
messages = cast(list[ChatMessage], st.session_state.messages)
sequence_renderer.render_messages(messages)

if prompt := st.chat_input("What is up?"):
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    messages.append(
        sequence_renderer.render_response(
            prompt,
            content="This is a placeholder response from the assistant.",
            key="assistant-sequence-live",
        )
    )
    st.rerun()
