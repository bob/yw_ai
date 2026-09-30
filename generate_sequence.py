import os
from collections.abc import Iterator
from typing import Literal, NotRequired, TypedDict, cast

import streamlit as st
from dotenv import load_dotenv

from langchain_app.schemas import Language
from langchain_app.services import (
    CatalogItemService,
    CatalogItemServiceError,
    TypeSafeService,
    TypeSafeServiceError,
)
from langchain_app.ui import (
    AsanaThumbnail,
    SequenceInfo,
    SequenceRenderer,
    SequenceView,
    create_sequence_view,
    generate_thumbnails_one_by_one,
)

load_dotenv()

SEQUENCE_LANGUAGE: Language = "en"
SEQUENCE_ELEMENTS_COUNT = 9
ASANA_SERVICE_BASE_URL = os.getenv(
    "ASANA_SERVICE_BASE_URL", "http://0.0.0.0:4999"
)
ASANA_CATALOG_FOLDER_UUID = os.getenv(
    "ASANA_CATALOG_FOLDER_UUID", "yoga-public-v1"
)


@st.cache_resource
def get_catalog_item_service(
    base_url: str, catalog_folder_uuid: str
) -> CatalogItemService:
    return CatalogItemService(
        base_url=base_url,
        catalog_folder_uuid=catalog_folder_uuid,
    )


@st.cache_resource
def get_type_safe_service() -> TypeSafeService:
    return TypeSafeService()


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    sequence: NotRequired[SequenceView | None]
    sequence_info: NotRequired[SequenceInfo]
    thumbnails: NotRequired[list[AsanaThumbnail]]


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

        assistant_sequence = create_sequence_view(SEQUENCE_LANGUAGE)
        assistant_sequence["info"]["elements_count"] = SEQUENCE_ELEMENTS_COUNT
        generated_thumbnails: list[AsanaThumbnail] = []

        def remember_generated_thumbnails() -> Iterator[AsanaThumbnail]:
            catalog_item_service = get_catalog_item_service(
                ASANA_SERVICE_BASE_URL,
                ASANA_CATALOG_FOLDER_UUID,
            )
            type_safe_service = get_type_safe_service()
            for thumbnail in generate_thumbnails_one_by_one(
                catalog_item_service,
                type_safe_service,
                language=SEQUENCE_LANGUAGE,
                request=prompt,
                count=SEQUENCE_ELEMENTS_COUNT,
            ):
                generated_thumbnails.append(thumbnail)
                yield thumbnail

        try:
            live_sequence = SequenceView(
                info=assistant_sequence["info"],
                thumbnails=remember_generated_thumbnails(),
            )
            sequence_renderer.render(
                live_sequence,
                key="assistant-sequence-live",
            )
        except (
            CatalogItemServiceError,
            TypeSafeServiceError,
            ValueError,
        ) as exc:
            st.error(f"Could not create sequence: {exc}")
            assistant_sequence = None
        else:
            assistant_sequence["thumbnails"] = generated_thumbnails
            assistant_sequence["info"]["elements_count"] = len(
                generated_thumbnails
            )

        # Add assistant message to chat history
        messages.append(
            {
                "role": "assistant",
                "content": assistant_response,
                "sequence": assistant_sequence,
            }
        )
