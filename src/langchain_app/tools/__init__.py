"""The shared assistant's tool registry."""

from langchain_app.tools.catalog import get_catalog_tag_names, search_catalog_items, show_catalog_items
from langchain_app.tools.sequence import (
    add_sequence_elements, generate_sequence, remove_sequence_elements,
    replace_sequence_element, swap_sequence_elements,
)

ASSISTANT_TOOLS = (
    generate_sequence,
    search_catalog_items,
    get_catalog_tag_names,
    replace_sequence_element,
    swap_sequence_elements,
    add_sequence_elements,
    remove_sequence_elements,
    show_catalog_items,
)

__all__ = [
    "ASSISTANT_TOOLS", "generate_sequence", "search_catalog_items",
    "get_catalog_tag_names", "replace_sequence_element", "swap_sequence_elements", "add_sequence_elements",
    "remove_sequence_elements",
    "show_catalog_items",
]
