"""Streamlit rendering for public yoga sequences."""

import time
from collections.abc import Callable, Iterable, Iterator
from html import escape
from itertools import chain
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
    AssistantAgentError,
    CatalogItemServiceError,
    SequenceService,
    TypeSafeServiceError,
)
from langchain_app.ui.sequence_import import render_sequence_import

THUMBNAIL_STREAM_DELAY_SECONDS = 0.05


class AsanaThumbnail(TypedDict):
    uuid: str | None
    order: int
    image_url: str
    alt: str
    start_mirror: bool
    end_mirror: bool


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
        thumbnail_size: int = 64,
        generator_factory: Callable[[], AssistantAgent] | None = None,
        language: Language = "en",
        catalog_url: str | None = None,
    ) -> None:
        self._thumbnail_size = thumbnail_size
        self._generator_factory = generator_factory
        self._language: Language = language
        self._catalog_url = catalog_url

    def render_generated_sequence(
        self, request: str, *, key: str, sequence_uuid: str | None = None,
    ) -> SequenceView | None:
        """Generate from the original request and retain only a complete result."""
        if self._generator_factory is None:
            raise ValueError("A generator factory is required to generate sequences")
        assistant = self._generator_factory()
        return self._render_catalog_items(
            assistant.generate(request, language=self._language, sequence_uuid=sequence_uuid),
            key=key, assistant=assistant,
        )

    def _render_catalog_items(
        self, catalog_items: Iterator[CatalogItemSchema], *, key: str, assistant: AssistantAgent,
    ) -> SequenceView | None:
        """Display items as they arrive, retaining only a complete result."""
        previous_uuid = assistant.active_sequence_uuid
        try:
            first = next(catalog_items, None)
            if first is not None:
                self.render(
                    SequenceView(
                        info=create_sequence_view(self._language)["info"],
                        thumbnails=generate_thumbnails_one_by_one(chain([first], catalog_items)),
                    ),
                    key=key,
                )
        except (
            CatalogItemServiceError, TypeSafeServiceError, AssistantAgentError, ValueError
        ) as exc:
            st.error(f"Could not update sequence: {exc}")
        # A committed tool result survives failure of the subsequent text reply.
        active_uuid = assistant.active_sequence_uuid
        if active_uuid is None or active_uuid == previous_uuid:
            return None
        snapshot = assistant.get_sequence(active_uuid)
        return generated_sequence_view(snapshot) if snapshot is not None else None

    def render_messages(self, messages: list[ChatMessage]) -> None:
        """Display chat history with controls for each generated sequence."""
        for message_index, message in enumerate(messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                self._render_message_sequence(messages, message_index)

    def _render_message_sequence(
        self, messages: list[ChatMessage], message_index: int
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
            regenerate = st.button(
                "Regenerate", key=f"{key}-regenerate", disabled=request is None
            )
            if assistant is not None and selected_uuid is not None:
                if selected_uuid == assistant.active_sequence_uuid:
                    st.caption("Active sequence", width="content")
                elif st.button("Use this sequence", key=f"{key}-use"):
                    assistant.select_sequence(selected_uuid)
                    st.rerun()
            if len(variants) > 1:
                if st.button(
                    "Previous Sequence", icon=":material/chevron_left:",
                    key=f"{key}-previous", disabled=selected == 0,
                ):
                    selected -= 1
                st.caption(f"{selected + 1}/{len(variants)}", width="content")
                if st.button(
                    "Next Sequence", icon=":material/chevron_right:",
                    key=f"{key}-next", disabled=selected == len(variants) - 1,
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

        if regenerate and request is not None:
            with st.container():
                new_sequence = self.render_generated_sequence(
                    request, key=f"{key}-live",
                    sequence_uuid=selected_uuid if assistant is not None else None,
                )
            if new_sequence is not None:
                variants.append(new_sequence)
                message["selected_sequence_index"] = len(variants) - 1
                message["sequence"] = new_sequence
                new_uuid = new_sequence["info"]["uuid"]
                message["sequence_uuid"] = new_uuid
                if variant_uuids is not None and new_uuid is not None:
                    variant_uuids.append(new_uuid)
                st.rerun()

    def render_response(
        self, request: str, *, content: str, key: str, require_generation: bool = False,
    ) -> ChatMessage:
        """Display assistant text and any generated sequence in a chat message."""
        if self._generator_factory is None:
            raise ValueError("An assistant factory is required")

        assistant = self._generator_factory()
        texts = [content] if content else []
        with st.chat_message("assistant"):
            text_output = st.empty()
            if content:
                text_output.markdown(content)

            def catalog_items() -> Iterator[CatalogItemSchema]:
                for chunk in assistant.stream(request, require_generation=require_generation):
                    if isinstance(chunk, CatalogItemSchema):
                        yield chunk
                    else:
                        texts.append(chunk)
                        text_output.markdown("".join(texts))

            sequence = self._render_catalog_items(catalog_items(), key=key, assistant=assistant)

        sequence_uuid = sequence["info"]["uuid"] if sequence is not None else None
        snapshots = assistant.get_sequence_variants(sequence_uuid) if sequence_uuid else []
        variants = [generated_sequence_view(snapshot) for snapshot in snapshots]
        selected = next((index for index, snapshot in enumerate(snapshots) if str(snapshot.uuid) == sequence_uuid), 0)
        return {
            "role": "assistant",
            "content": "".join(texts),
            "sequence": sequence,
            "sequence_uuid": sequence_uuid,
            "sequence_variant_uuids": [str(snapshot.uuid) for snapshot in snapshots],
            "sequence_request": snapshots[selected].generation_request if snapshots else request,
            "sequence_variants": variants,
            "selected_sequence_index": selected,
        }

    def render_styles(self) -> None:
        """Add the styles used by all sequence thumbnails on the page."""

        st.html(
            f"""
            <style>
            .mirror-thumbnail {{
                --thumbnail-size: {self._thumbnail_size}px;
                --bracket-color: green;
                --bracket-stroke: 2px;

                position: relative;
                display: block;
                width: var(--thumbnail-size);
                height: var(--thumbnail-size);
            }}
            .mirror-thumbnail img {{
                display: block;
                width: 100%;
                height: 100%;
                object-fit: cover;
                border-radius: 10px;
            }}
            .mirror-thumbnail__bracket {{
                position: absolute;
                top: 6px;
                bottom: 6px;
                width: 8px;
                border-top: var(--bracket-stroke) solid var(--bracket-color);
                border-bottom: var(--bracket-stroke) solid var(--bracket-color);
                pointer-events: none;
            }}
            .mirror-thumbnail__bracket--start {{
                left: 4px;
                border-left: var(--bracket-stroke) solid var(--bracket-color);
            }}
            .mirror-thumbnail__bracket--end {{
                right: 4px;
                border-right: var(--bracket-stroke) solid var(--bracket-color);
            }}
            [class*="stElementToolbar"] {{
                visibility: hidden;
            }}
            </style>
            """
        )

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
        brackets = ""
        if thumbnail["start_mirror"]:
            brackets += (
                '<span class="mirror-thumbnail__bracket '
                'mirror-thumbnail__bracket--start" aria-hidden="true"></span>'
            )
        if thumbnail["end_mirror"]:
            brackets += (
                '<span class="mirror-thumbnail__bracket '
                'mirror-thumbnail__bracket--end" aria-hidden="true"></span>'
            )

        return (
            '<div class="mirror-thumbnail">'
            f'<img src="{escape(thumbnail["image_url"], quote=True)}" '
            f'alt="{escape(thumbnail["alt"], quote=True)}">'
            f"{brackets}"
            "</div>"
        )

    def _render_images(
        self,
        thumbnails: Iterable[AsanaThumbnail],
        *,
        key: str,
    ) -> None:
        with st.container(key=key, horizontal=True, wrap=True, gap="xxsmall"):
            for thumbnail in thumbnails:
                if thumbnail["image_url"]:
                    st.html(self._thumbnail_html(thumbnail), width="content")
                else:
                    st.caption(thumbnail["alt"], width="content")
