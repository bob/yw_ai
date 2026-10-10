"""The shared assistant's tool registry."""

from langchain_app.tools.catalog import get_catalog_tag_names, search_catalog_items, show_catalog_items
from langchain_app.tools.sequence import generate_sequence, replace_sequence_element

ASSISTANT_TOOLS = (
    generate_sequence,
    search_catalog_items,
    get_catalog_tag_names,
    replace_sequence_element,
    show_catalog_items,
)

__all__ = [
    "ASSISTANT_TOOLS", "generate_sequence", "search_catalog_items",
    "get_catalog_tag_names", "replace_sequence_element",
    "show_catalog_items",
]
