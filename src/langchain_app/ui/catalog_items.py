"""Render a confirmed catalog selection using the shared thumbnail layout."""

from collections.abc import Iterable

from langchain_app.schemas.catalog import CatalogItemSchema
from langchain_app.ui.thumbnails import DEFAULT_THUMBNAIL_SIZE, ThumbnailRenderer


class CatalogItemsRenderer:
    """Show catalog images and missing-image names without sequence controls."""

    def __init__(self, *, thumbnail_size: int = DEFAULT_THUMBNAIL_SIZE) -> None:
        self._thumbnails = ThumbnailRenderer(thumbnail_size=thumbnail_size)

    def render(self, items: Iterable[CatalogItemSchema], *, key: str) -> None:
        self._thumbnails.render(
            (
                {
                    "image_url": str(item.image_url) if item.image_url else "",
                    "alt": item.name,
                }
                for item in items
            ),
            key=key,
        )
