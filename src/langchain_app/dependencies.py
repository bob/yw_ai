"""Cached application service dependencies."""

import os

import streamlit as st

from langchain_app.model_config import DEFAULT_CHAT_MODEL
from langchain_app.services import (
    AssistantAgent,
    CatalogItemService,
    CatalogTagService,
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
def get_catalog_tag_service(
    base_url: str, catalog_folder_uuid: str
) -> CatalogTagService:
    return CatalogTagService(
        base_url=base_url, catalog_folder_uuid=catalog_folder_uuid,
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


def get_assistant_agent(
    base_url: str, catalog_folder_uuid: str
) -> AssistantAgent:
    # Initialize Streamlit-cached clients on the script thread, before the tool runs.
    generator = get_sequence_generator(base_url, catalog_folder_uuid)

    return AssistantAgent(
        generator_factory=lambda: generator,
        catalog_item_service=get_catalog_item_service(base_url, catalog_folder_uuid),
        catalog_tag_service=get_catalog_tag_service(base_url, catalog_folder_uuid),
        model=(
            os.getenv("ASSISTANT_AGENT_MODEL", "").strip()
            or os.getenv("SEQUENCE_AGENT_MODEL", "").strip()
            or DEFAULT_CHAT_MODEL
        ),
    )
