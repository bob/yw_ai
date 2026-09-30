"""Client for the public catalog-items API."""

from types import TracebackType
from typing import Self
from urllib.parse import quote

import httpx
from pydantic import TypeAdapter, ValidationError

from langchain_app.schemas import CatalogItemSchema, Language


class CatalogItemServiceError(Exception):
    """Base exception raised by :class:`CatalogItemService`."""


class CatalogFolderNotFoundError(CatalogItemServiceError):
    """Raised when the configured public catalog folder does not exist."""


class CatalogItemApiError(CatalogItemServiceError):
    """Raised when catalog items cannot be loaded from the public API."""


class CatalogItemService:
    """Fetch catalog items from a public catalog folder."""

    _LANGUAGES = {"en", "uk", "ru"}  # noqa: RUF012
    _ITEMS = TypeAdapter(list[CatalogItemSchema])

    def __init__(
        self,
        base_url: str,
        catalog_folder_uuid: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 10.0,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not catalog_folder_uuid.strip():
            raise ValueError("catalog_folder_uuid must not be empty")

        self._base_url = base_url.rstrip("/")
        self._catalog_folder_uuid = catalog_folder_uuid.strip()
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)

    def search(self, *, language: Language) -> list[CatalogItemSchema]:
        """Return localized items from the configured public catalog folder."""

        self._validate_language(language)
        folder_uuid = quote(self._catalog_folder_uuid, safe="")
        url = (
            f"{self._base_url}/api/public/catalog_folders/{folder_uuid}/"
            "catalog_items/search"
        )

        try:
            response = self._client.get(
                url,
                params={
                    "language": "en",
                    "with_tags": True,
                    "limit": 0,
                    "rand": True,
                }
            )
        except httpx.HTTPError as exc:
            raise CatalogItemApiError("Could not reach the catalog items API") from exc

        if response.status_code == httpx.codes.NOT_FOUND:
            raise CatalogFolderNotFoundError(
                f'Public catalog folder "{self._catalog_folder_uuid}" was not found'
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise CatalogItemApiError(
                f"Catalog items API returned HTTP {response.status_code}"
            ) from exc

        try:
            catalog_items = self._ITEMS.validate_python(response.json())
        except (ValidationError, ValueError) as exc:
            raise CatalogItemApiError(
                "Catalog items API returned an invalid response"
            ) from exc

        return catalog_items

    def close(self) -> None:
        """Close the internally-created HTTP client."""

        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @classmethod
    def _validate_language(cls, language: str) -> None:
        if language not in cls._LANGUAGES:
            raise ValueError("language must be one of: en, uk, ru")
