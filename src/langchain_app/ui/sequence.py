"""Streamlit rendering for public yoga sequences."""

import time
from collections.abc import Callable, Iterable, Iterator
from time import monotonic
from typing import TypedDict
from uuid import uuid4

import streamlit as st

from langchain_app.schemas import (
    CatalogItemSchema,
    ChatMessage,
    GeneratedSequence,
    Language,
)
from langchain_app.services import (
    AssistantAgent,
    SequenceService,
)
from langchain_app.schemas.response_progress import (
    CatalogItemsDisplayedEvent, SequenceCommittedEvent, SequenceElementEvent,
)
from langchain_app.services.request_errors import public_request_error
from langchain_app.ui.catalog_items import CatalogItemsRenderer
from langchain_app.ui.response_progress import (
    PendingResponse, ResponseProgressRenderer, enqueue_regeneration, render_execution,
)
from langchain_app.ui.sequence_import import render_sequence_import
from langchain_app.ui.thumbnails import (
    DEFAULT_THUMBNAIL_SIZE, AsanaThumbnail, ThumbnailRenderer,
)

THUMBNAIL_STREAM_DELAY_SECONDS = 0.05


class SequenceInfo(TypedDict):
    id: int
    uuid: str | None
    name: str
    description: str | None
    language: Language
    elements_count: int


class SequenceView(TypedDict):
    info: SequenceInfo
    thumbnails: Iterable[AsanaThumbnail]


def create_sequence_view(language: Language) -> SequenceView:
    return {
        "info": {
            "id": 0,
            "uuid": str(uuid4()),
            "name": "New Sequence",
            "description": None,
            "language": language,
            "elements_count": 0,
        },
        "thumbnails": [],
    }


def generated_sequence_view(sequence: GeneratedSequence) -> SequenceView:
    """Project a complete agent snapshot into the display and import format."""
    return {
        "info": {
            "id": 0, "uuid": str(sequence.uuid), "name": sequence.name,
            "description": sequence.description, "language": sequence.language,
            "elements_count": sequence.elements_count,
        },
        "thumbnails": [
            {
                "uuid": element.catalog_item.uuid, "order": element.order,
                "image_url": str(element.catalog_item.image_url) if element.catalog_item.image_url else "",
                "alt": element.catalog_item.name,
                "start_mirror": element.start_mirror, "end_mirror": element.end_mirror,
            }
            for element in sequence.elements
        ],
    }


@st.cache_data(ttl=300)
def get_sequence_view(
    base_url: str, sequence_id: int, language: Language
) -> SequenceView:
    with SequenceService(base_url=base_url) as service:
        sequence = service.get_by_id(sequence_id, language=language)

    return {
        "info": {
            "id": sequence.id,
            "uuid": None,
            "name": sequence.name,
            "description": sequence.description,
            "language": sequence.lang,
            "elements_count": sequence.elements_count,
        },
        "thumbnails": [
            {
                "uuid": element.catalog_item.uuid,
                "order": element.order,
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


def generate_thumbnails_one_by_one(
    catalog_items: Iterable[CatalogItemSchema],
) -> Iterator[AsanaThumbnail]:
    """Convert catalog items to thumbnails one by one."""

    order = 0
    for catalog_item in catalog_items:
        if catalog_item.image_url is None:
            continue

        order += 1
        yield {
            "uuid": catalog_item.uuid,
            "order": order,
            "image_url": str(catalog_item.image_url),
            "alt": catalog_item.name,
            "start_mirror": catalog_item.bilateral,
            "end_mirror": catalog_item.bilateral,
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


class SequenceRenderer:
    """Render sequence details and thumbnails in a Streamlit app."""

    def __init__(
        self,
        *,
        thumbnail_size: int = DEFAULT_THUMBNAIL_SIZE,
        generator_factory: Callable[[], AssistantAgent] | None = None,
        language: Language = "en",
        catalog_url: str | None = None,
    ) -> None:
        self._thumbnails = ThumbnailRenderer(thumbnail_size=thumbnail_size)
        self._catalog_items = CatalogItemsRenderer(thumbnail_size=thumbnail_size)
        self._generator_factory = generator_factory
        self._language: Language = language
        self._catalog_url = catalog_url

    def render_messages(self, messages: list[ChatMessage], *, disabled: bool = False) -> None:
        """Display chat history with controls for each generated sequence."""
        for message_index, message in enumerate(messages):
            with st.chat_message(message["role"]):
                for snapshot in message.get("execution_history", []):
                    render_execution(snapshot)
                if snapshot := message.get("execution"):
                    render_execution(snapshot)
                if items := message.get("catalog_items"):
                    self._catalog_items.render(items, key=f"assistant-catalog-{message_index}")
                self._render_message_sequence(messages, message_index, disabled=disabled)
                if message["content"]:
                    st.markdown(message["content"])

    def _render_message_sequence(
        self, messages: list[ChatMessage], message_index: int, *, disabled: bool = False,
    ) -> None:
        message = messages[message_index]
        assistant = None
        variant_uuids = message.get("sequence_variant_uuids")
        if variant_uuids and self._generator_factory is not None:
            assistant = self._generator_factory()
            # UI views are disposable projections; full catalog data lives in the agent.
            snapshots = [
                snapshot for uuid in variant_uuids
                if (snapshot := assistant.get_sequence(uuid)) is not None
            ]
            if len(snapshots) == len(variant_uuids):
                message["sequence_variants"] = [generated_sequence_view(snapshot) for snapshot in snapshots]
                message["sequence"] = message["sequence_variants"][message.get("selected_sequence_index", 0)]
            else:
                # Legacy cards remain viewable without inventing catalog data.
                assistant = None
        stored_sequence = message.get("sequence")
        if stored_sequence is None and (
            stored_sequence_info := message.get("sequence_info")
        ):
            stored_sequence = SequenceView(
                info=stored_sequence_info,
                thumbnails=message.get("thumbnails", []),
            )
        if stored_sequence is None:
            return

        variants = message.setdefault("sequence_variants", [stored_sequence])
        selected = message.setdefault("selected_sequence_index", 0)
        # Older session messages did not retain their original request.
        request = message.get("sequence_request")
        if request is None:
            request = next(
                (
                    previous["content"]
                    for previous in reversed(messages[:message_index])
                    if previous["role"] == "user"
                ),
                None,
            )
            if request is not None:
                message["sequence_request"] = request

        key = f"assistant-sequence-{message_index}"
        selected_uuid = variants[selected]["info"].get("uuid")
        with st.container(
            horizontal=True,
            horizontal_alignment="left",
            vertical_alignment="center",
        ):
            st.button(
                "Regenerate", key=f"{key}-regenerate", disabled=disabled or request is None,
                on_click=enqueue_regeneration,
                args=(request or "", selected_uuid if assistant is not None else None, message_index),
            )
            if assistant is not None and selected_uuid is not None:
                if selected_uuid == assistant.active_sequence_uuid:
                    st.caption("Active sequence", width="content")
                elif st.button("Use this sequence", key=f"{key}-use", disabled=disabled):
                    assistant.select_sequence(selected_uuid)
                    st.rerun()
            if len(variants) > 1:
                if st.button(
                    "Previous Sequence", icon=":material/chevron_left:",
                    key=f"{key}-previous", disabled=disabled or selected == 0,
                ):
                    selected -= 1
                st.caption(f"{selected + 1}/{len(variants)}", width="content")
                if st.button(
                    "Next Sequence", icon=":material/chevron_right:",
                    key=f"{key}-next", disabled=disabled or selected == len(variants) - 1,
                ):
                    selected += 1

        if selected != message.get("selected_sequence_index", 0):
            message["selected_sequence_index"] = selected
            message["sequence"] = variants[selected]
            selected_uuid = variants[selected]["info"]["uuid"]
            message["sequence_uuid"] = selected_uuid
            if assistant is not None and selected_uuid is not None:
                assistant.select_sequence(selected_uuid)
            st.rerun()
        message["sequence"] = variants[selected]
        self.render(variants[selected], key=key)
        if self._catalog_url is not None:
            if variants[selected]["info"]["id"] == 0 and not variants[selected]["info"].get("uuid"):
                variants[selected]["info"]["uuid"] = str(uuid4())
            render_sequence_import(
                variants[selected], catalog_url=self._catalog_url,
                key=f"{key}-save-{selected}",
            )

    def render_response(
        self, request: str, *, content: str, key: str, require_generation: bool = False,
        pending: PendingResponse | None = None,
    ) -> ChatMessage:
        """Use one event lifecycle for chat, Generate and every Regenerate attempt."""
        pending = pending or PendingResponse(request, require_generation=require_generation)
        assistant = None
        sequence = None
        catalog_items = None
        snapshots = []
        with st.chat_message("assistant"):
            progress = ResponseProgressRenderer(pending.request_id, pending.accepted_at, pending.sequence_uuid)
            catalog_output = st.empty()
            sequence_output = st.empty()
            # Reserve text below both galleries so late replies cannot move thumbnails.
            text_output = st.empty()
            if content:
                text_output.markdown(content)
            provisional = create_sequence_view(self._language)
            thumbnails = []
            provisional["thumbnails"] = thumbnails
            try:
                if self._generator_factory is None:
                    raise ValueError("An assistant factory is required")
                assistant = self._generator_factory()
                for event in assistant.stream_response_events(
                    request, require_generation=pending.require_generation,
                    regenerate_sequence_uuid=pending.sequence_uuid,
                    request_id=pending.request_id,
                ):
                    if not progress.accept(event):
                        continue
                    text = content + "".join(progress.execution.final_texts)
                    if text:
                        text_output.markdown(text)
                    if isinstance(event, CatalogItemsDisplayedEvent) and catalog_items is None:
                        catalog_items = [item.model_copy(deep=True) for item in event.items]
                        with catalog_output.container():
                            self._catalog_items.render(catalog_items, key=f"{key}-catalog")
                    elif isinstance(event, SequenceElementEvent):
                        for thumbnail in generate_thumbnails_one_by_one([event.catalog_item]):
                            thumbnail["order"] = event.order
                            thumbnails.append(thumbnail)
                        with sequence_output.container():
                            self.render(provisional, key=f"{key}-{event.event_index}")
                    elif isinstance(event, SequenceCommittedEvent):
                        committed = assistant.get_sequence(str(event.sequence_uuid))
                        if committed is None:
                            raise ValueError("Committed sequence is missing from agent state")
                        sequence = generated_sequence_view(committed)
                        with sequence_output.container():
                            self.render(sequence, key=f"{key}-{event.event_index}")
            except Exception as exc:
                # The producer normally emitted its terminal error already.
                # Dependency initialization failures have no stream to close.
                progress.execution.fail(
                    max(0.0, monotonic() - pending.accepted_at),
                    error_message=public_request_error(exc),
                )
            execution = progress.finish()
            if sequence is None:
                sequence_output.empty()
            if sequence is not None and assistant is not None:
                snapshots = assistant.get_sequence_variants(sequence["info"]["uuid"])
        sequence_uuid = sequence["info"]["uuid"] if sequence is not None else None
        variants = [generated_sequence_view(snapshot) for snapshot in snapshots]
        selected = next((index for index, snapshot in enumerate(snapshots) if str(snapshot.uuid) == sequence_uuid), 0)
        message: ChatMessage = {
            "role": "assistant",
            "content": content + "".join(progress.execution.final_texts),
            "execution": execution,
            "sequence": sequence,
            "sequence_uuid": sequence_uuid,
            "sequence_variant_uuids": [str(snapshot.uuid) for snapshot in snapshots],
            "sequence_request": snapshots[selected].generation_request if snapshots else request,
            "sequence_variants": variants,
            "selected_sequence_index": selected,
        }
        if catalog_items is not None:
            message["catalog_items"] = catalog_items
        return message

    def render_styles(self) -> None:
        """Add shared thumbnail styles once at the page entry point."""
        self._thumbnails.render_styles()

    def render(
        self,
        sequence: SequenceView | None,
        *,
        key: str,
    ) -> None:
        """Render a sequence as its thumbnails become available."""

        if sequence is None:
            return

        self._render_info(sequence["info"])
        self._render_images(
            sequence["thumbnails"],
            key=key,
        )

    @staticmethod
    def _render_info(sequence_info: SequenceInfo) -> None:
        st.markdown(f"**{sequence_info['name']}**")

        if sequence_info["description"]:
            st.markdown(sequence_info["description"])

        count = sequence_info["elements_count"]
        count_label = f"{count} poses · " if count else ""
        st.caption(
            f"Sequence #{sequence_info['id']} · {count_label}"
            f"{sequence_info['language'].upper()}"
        )

    @staticmethod
    def _thumbnail_html(thumbnail: AsanaThumbnail) -> str:
        return ThumbnailRenderer._thumbnail_html(thumbnail)

    def _render_images(
        self,
        thumbnails: Iterable[AsanaThumbnail],
        *,
        key: str,
    ) -> None:
        self._thumbnails.render(thumbnails, key=key)
