"""Shared hard constraints for selection and materialized sequence additions."""

from uuid import UUID

from langchain_app.schemas import CatalogItemSchema

MAX_SEQUENCE_ELEMENTS = 45
STARTING_TAG = "Other/Starting"
LYING_TAG_PREFIX = "Position/Lying"
STANDING_TAG = "Position/Standing"
SUPINE_TAG = "Position/Lying on the back"


def is_lying(item: CatalogItemSchema) -> bool:
    return any(tag.startswith(LYING_TAG_PREFIX) for tag in item.catalog_tag_names)


def starting_pose_allowed(item: CatalogItemSchema, position: int) -> bool:
    """A new first pose must be a start; new later poses must not be starts."""
    return (STARTING_TAG in item.catalog_tag_names) == (position == 1)


def second_pose_allowed(first: CatalogItemSchema, second: CatalogItemSchema) -> bool:
    if is_lying(first):
        return SUPINE_TAG in second.catalog_tag_names
    if STANDING_TAG in first.catalog_tag_names:
        return not is_lying(second)
    return True


def has_catalog_uuid(item: CatalogItemSchema) -> bool:
    if not item.uuid:
        return False
    try:
        UUID(item.uuid)
    except ValueError:
        return False
    return True
