"""Shared HTTP transport for the two English-language catalog endpoints."""

from types import TracebackType
from typing import Self
from urllib.parse import quote

import httpx

from langchain_app.services.catalog_errors import (
    CatalogFolderNotFoundError,
    CatalogServiceError,
)


class CatalogClient:
    """Own a client unless one was injected, and enforce English at HTTP level."""

    api_error_type = CatalogServiceError

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
        folder_uuid = quote(catalog_folder_uuid.strip(), safe="")
        self._folder_url = (
            f"{base_url.strip().rstrip('/')}/api/public/catalog_folders/{folder_uuid}"
        )
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)

    def _get(self, endpoint: str, params: list[tuple[str, str]]) -> httpx.Response:
        try:
            response = self._client.get(
                f"{self._folder_url}/{endpoint}",
                params=[("language", "en"), *params],
            )
        except httpx.HTTPError as exc:
            raise self.api_error_type("Could not reach the catalog API") from exc
        if response.status_code == httpx.codes.NOT_FOUND:
            raise CatalogFolderNotFoundError("Public catalog folder was not found")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise self.api_error_type(
                f"Catalog API returned HTTP {response.status_code}",
                code="invalid_arguments" if response.status_code == 422
                else "catalog_api_unavailable",
            ) from exc
        return response

    def close(self) -> None:
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
