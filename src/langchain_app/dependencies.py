"""Cached application service dependencies."""

import streamlit as st

from langchain_app.services import (
    CatalogItemService,
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
