"""Client for public English-language catalog tag names."""

from pydantic import TypeAdapter, ValidationError

from langchain_app.services.catalog_client import CatalogClient
from langchain_app.services.catalog_errors import CatalogServiceError


class CatalogTagApiError(CatalogServiceError):
    """Catalog tag names could not be loaded or validated."""


class CatalogTagService(CatalogClient):
    """Fetch tag names in the order and spelling returned by the API."""

    api_error_type = CatalogTagApiError
    _NAMES = TypeAdapter(list[str])

    def get_names(self) -> list[str]:
        response = self._get("catalog_tags", [])
        try:
            return self._NAMES.validate_python(response.json(), strict=True)
        except (ValidationError, ValueError) as exc:
            raise CatalogTagApiError(
                "Catalog tags API returned an invalid response", code="invalid_api_response"
            ) from exc
