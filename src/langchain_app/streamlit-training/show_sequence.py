import os
import time
from collections.abc import Iterable, Iterator
from typing import Literal, NotRequired, TypedDict, cast

import streamlit as st
from dotenv import load_dotenv

from langchain_app.schemas import Language
from langchain_app.services import SequenceService, SequenceServiceError
from langchain_app.ui import (
    AsanaThumbnail,
    SequenceInfo,
    SequenceRenderer,
    SequenceView,
)

load_dotenv()

SEQUENCE_ID = 199
SEQUENCE_LANGUAGE: Language = "en"
THUMBNAIL_STREAM_DELAY_SECONDS = 0.05
ASANA_SERVICE_BASE_URL = os.getenv(
    "ASANA_SERVICE_BASE_URL", "http://0.0.0.0:4999"
)


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    sequence: NotRequired[SequenceView | None]
    sequence_info: NotRequired[SequenceInfo]
    thumbnails: NotRequired[list[AsanaThumbnail]]


@st.cache_data(ttl=300)
def get_sequence_view(
    base_url: str, sequence_id: int, language: Language
) -> SequenceView:
    with SequenceService(base_url=base_url) as service:
        sequence = service.get_by_id(sequence_id, language=language)

    return {
        "info": {
            "id": sequence.id,
            "name": sequence.name,
            "description": sequence.description,
            "language": sequence.lang,
            "elements_count": sequence.elements_count,
        },
        "thumbnails": [
            {
                "image_url": str(element.catalog_item.image_url),
                "alt": element.catalog_item.name,
                "start_mirror": element.start_mirror,
                "end_mirror": element.end_mirror,
            }
            for element in sequence.elements
            if element.catalog_item is not None
            and element.catalog_item.image_url is not None
        ],
    }


def get_thumbnails_one_by_one(
    thumbnails: Iterable[AsanaThumbnail],
    *,
    delay_seconds: float = THUMBNAIL_STREAM_DELAY_SECONDS,
) -> Iterator[AsanaThumbnail]:
    """Yield thumbnails as they become available from the sequence source."""

    for thumbnail in thumbnails:
        time.sleep(delay_seconds)
        yield thumbnail


st.title("Sequence Viewer")

sequence_renderer = SequenceRenderer()
sequence_renderer.render_styles()

if "messages" not in st.session_state:
    st.session_state.messages = []

messages = cast(list[ChatMessage], st.session_state.messages)

for message_index, message in enumerate(messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        stored_sequence = message.get("sequence")
        if stored_sequence is None and (
            stored_sequence_info := message.get("sequence_info")
        ):
            # Preserve messages created before SequenceRenderer was introduced.
            stored_sequence = SequenceView(
                info=stored_sequence_info,
                thumbnails=message.get("thumbnails", []),
            )

        sequence_renderer.render(
            stored_sequence,
            key=f"assistant-sequence-{message_index}",
        )

if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Here you would typically generate the assistant's response
        assistant_response = "This is a placeholder response from the assistant."
        st.markdown(assistant_response)

        try:
            assistant_sequence = get_sequence_view(
                ASANA_SERVICE_BASE_URL, SEQUENCE_ID, SEQUENCE_LANGUAGE
            )
        except SequenceServiceError as exc:
            st.error(f"Could not load sequence {SEQUENCE_ID}: {exc}")
            assistant_sequence = None
        else:
            live_sequence = SequenceView(
                info=assistant_sequence["info"],
                thumbnails=get_thumbnails_one_by_one(
                    assistant_sequence["thumbnails"]
                ),
            )
            sequence_renderer.render(
                live_sequence,
                key="assistant-sequence-live",
            )

        # Add assistant message to chat history
        messages.append(
            {
                "role": "assistant",
                "content": assistant_response,
                "sequence": assistant_sequence,
            }
        )
