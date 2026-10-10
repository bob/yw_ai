"""Shared thumbnail styles, safe markup, and wrapping Streamlit layout."""

from collections.abc import Iterable
from html import escape
from typing import TypedDict

import streamlit as st

DEFAULT_THUMBNAIL_SIZE = 64


class Thumbnail(TypedDict):
    image_url: str
    alt: str


class AsanaThumbnail(Thumbnail):
    uuid: str | None
    order: int
    start_mirror: bool
    end_mirror: bool


class ThumbnailRenderer:
    """Render thumbnails with optional occurrence-specific mirror brackets."""

    def __init__(self, *, thumbnail_size: int = DEFAULT_THUMBNAIL_SIZE) -> None:
        self._thumbnail_size = thumbnail_size

    def render_styles(self) -> None:
        """Install shared styles once at the page entry point, before rendering."""
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

    @staticmethod
    def _thumbnail_html(thumbnail: Thumbnail) -> str:
        brackets = ""
        if thumbnail.get("start_mirror", False):
            brackets += (
                '<span class="mirror-thumbnail__bracket '
                'mirror-thumbnail__bracket--start" aria-hidden="true"></span>'
            )
        if thumbnail.get("end_mirror", False):
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

    def render(self, thumbnails: Iterable[Thumbnail], *, key: str) -> None:
        with st.container(key=key, horizontal=True, wrap=True, gap="xxsmall"):
            for thumbnail in thumbnails:
                if thumbnail["image_url"]:
                    st.html(self._thumbnail_html(thumbnail), width="content")
                else:
                    st.caption(thumbnail["alt"], width="content")
