"""Streamlit rendering for public yoga sequences."""

import time
from collections.abc import Iterable, Iterator
from html import escape
from typing import TypedDict

import streamlit as st

from langchain_app.schemas import CatalogItemSchema, Language
from langchain_app.services import (
    CatalogItemService,
    SequenceService,
    TypeSafeService,
)

THUMBNAIL_STREAM_DELAY_SECONDS = 0.05


class AsanaThumbnail(TypedDict):
    image_url: str
    alt: str
    start_mirror: bool
    end_mirror: bool


class SequenceInfo(TypedDict):
    id: int
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
            "name": "New Sequence",
            "description": None,
            "language": language,
            "elements_count": 0,
        },
        "thumbnails": [],
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


def generate_thumbnails_one_by_one(
    catalog_item_service: CatalogItemService,
    type_safe_service: TypeSafeService,
    *,
    language: Language,
    request: str | None,
    count: int,
) -> Iterator[AsanaThumbnail]:
    """Select catalog items with Jev and yield their thumbnails one by one."""

    if count < 0:
        raise ValueError("count must not be negative")

    user_request: str = request or ""
    catalog_items = [
        item
        for item in catalog_item_service.search(language=language)
        if item.image_url is not None
    ]
    selected_items: list[CatalogItemSchema] = []

    for _ in range(count):
        catalog_item = type_safe_service.choose_next_catalog_item(
            catalog_items,
            selected_items=selected_items,
            request=user_request,
            elements_count=count,
        )
        selected_items.append(catalog_item)

        if catalog_item.image_url is None:
            continue

        yield {
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
    ) -> None:
        self._thumbnail_size = thumbnail_size

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

        st.caption(
            f"Sequence #{sequence_info['id']} · "
            f"{sequence_info['elements_count']} poses · "
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
                st.html(self._thumbnail_html(thumbnail), width="content")
