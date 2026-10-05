"""Cached application service dependencies."""

import os

import streamlit as st

from langchain_app.services import (
    CatalogItemService,
    SequenceAgent,
    SequenceGenerator,
    TypeSafeService,
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


@st.cache_resource
def get_sequence_generator(
    base_url: str, catalog_folder_uuid: str
) -> SequenceGenerator:
    return SequenceGenerator(
        catalog_item_service=get_catalog_item_service(
            base_url,
            catalog_folder_uuid,
        ),
        type_safe_service=get_type_safe_service(),
    )


def get_sequence_agent(
    base_url: str, catalog_folder_uuid: str
) -> SequenceAgent:
    # Initialize Streamlit-cached clients on the script thread, before the tool runs.
    generator = get_sequence_generator(base_url, catalog_folder_uuid)
    return SequenceAgent(
        generator_factory=lambda: generator,
        model=os.getenv("SEQUENCE_AGENT_MODEL", "openai:gpt-5.5"),
    )
