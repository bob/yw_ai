"""Errors shared by the public catalog clients."""


class CatalogServiceError(Exception):
    """A catalog request failed with a machine-readable tool error code."""

    def __init__(self, message: str, *, code: str = "catalog_api_unavailable") -> None:
        super().__init__(message)
        self.code = code


class CatalogFolderNotFoundError(CatalogServiceError):
    """The configured folder is missing or private."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="catalog_folder_not_found")
