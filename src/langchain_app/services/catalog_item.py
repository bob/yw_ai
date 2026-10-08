"""Client for the public English-language catalog-items API."""

from pydantic import TypeAdapter, ValidationError

from langchain_app.schemas.catalog import (
    CatalogItemSearchResult,
    CatalogItemsSearchArguments,
)
from langchain_app.services.catalog_client import CatalogClient
from langchain_app.services.catalog_errors import (
    CatalogServiceError,
)

# Preserve the exception imports used by existing generation consumers.
CatalogItemServiceError = CatalogServiceError


class CatalogItemApiError(CatalogServiceError):
    """Catalog items could not be loaded or validated."""


class CatalogItemService(CatalogClient):
    """Search items in a public folder; every request uses language=en."""

    api_error_type = CatalogItemApiError
    _ITEMS = TypeAdapter(list[CatalogItemSearchResult])

    def search(
        self,
        *,
        name: str | None = None,
        uuid: str | None = None,
        bilateral: bool | None = None,
        catalog_tag_names: list[str] | None = None,
        with_rigman: bool = False,
        with_tags: bool = True,
        limit: int = 10,
        rand: bool = False,
    ) -> list[CatalogItemSearchResult]:
        """Return filtered API data without translating catalog names or tags."""
        arguments = CatalogItemsSearchArguments(
            name=name, uuid=uuid, bilateral=bilateral,
            catalog_tag_names=catalog_tag_names if catalog_tag_names is not None else [],
            with_rigman=with_rigman, with_tags=with_tags, limit=limit, rand=rand,
        )
        params = [
            (key, str(value).lower() if isinstance(value, bool) else str(value))
            for key, value in arguments.model_dump(exclude={"catalog_tag_names"}).items()
            if value is not None
        ]
        params.extend(("catalog_tag_names[]", tag) for tag in arguments.catalog_tag_names)
        response = self._get("catalog_items/search", params)
        try:
            items = self._ITEMS.validate_python(response.json())
            if with_tags and any(item.catalog_tag_names is None for item in items):
                raise ValueError("catalog_tag_names is required when with_tags=true")
        except (ValidationError, ValueError) as exc:
            raise CatalogItemApiError(
                "Catalog items API returned an invalid response", code="invalid_api_response"
            ) from exc
        return items
