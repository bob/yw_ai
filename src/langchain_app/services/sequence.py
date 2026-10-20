"""Client for the public sequences API."""

from types import TracebackType
from typing import Self

import httpx

from langchain_app.schemas.sequence import Language, SequenceSchema
from langchain_app.services.http_transport import ResponseTooLargeError, bounded_get


class SequenceServiceError(Exception):
    """Base exception raised by :class:`SequenceService`."""


class SequenceNotFoundError(SequenceServiceError):
    """Raised when the requested public sequence does not exist."""


class SequenceValidationError(SequenceServiceError):
    """Raised when the API rejects the lookup parameters."""


class SequenceApiError(SequenceServiceError):
    """Raised when the sequence API request fails unexpectedly."""


class SequenceService:
    """Fetch localized public sequences by ID or exact name.

    A client can be injected for testing or to share connection pooling. When
    no client is provided, the service creates and owns one.
    """

    _PATH = "/api/public/sequences"
    _LANGUAGES = {"en", "uk", "ru"}  # noqa: RUF012

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 10.0,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")

        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout,
        )

    def get_sequence(
        self,
        sequence_id: int | None = None,
        *,
        name: str | None = None,
        language: Language,
    ) -> SequenceSchema:
        """Return one public sequence selected by exactly one ID or name."""

        self._validate_lookup(sequence_id=sequence_id, name=name, language=language)

        path = f"{self._PATH}/{sequence_id}" if sequence_id is not None else self._PATH
        params = {"language": language}
        if name is not None:
            params["name"] = name

        try:
            response = bounded_get(self._client, f"{self._base_url}{path}", params=params)
        except ResponseTooLargeError as exc:
            raise SequenceApiError("Public sequences API response exceeded the size limit") from exc
        except httpx.HTTPError as exc:
            raise SequenceApiError("Could not reach the public sequences API") from exc

        if response.status_code == httpx.codes.NOT_FOUND:
            lookup = f"ID {sequence_id}" if sequence_id is not None else f'name "{name}"'
            raise SequenceNotFoundError(f"Public sequence with {lookup} was not found")
        if response.status_code == httpx.codes.UNPROCESSABLE_ENTITY:
            raise SequenceValidationError(self._error_message(response))

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise SequenceApiError(
                f"Public sequences API returned HTTP {response.status_code}"
            ) from exc

        return SequenceSchema.model_validate(response.json())

    def get_by_id(self, sequence_id: int, *, language: Language) -> SequenceSchema:
        """Return a public sequence by its numeric ID."""

        return self.get_sequence(sequence_id, language=language)

    def get_by_name(self, name: str, *, language: Language) -> SequenceSchema:
        """Return a public sequence by its exact name or translation."""

        return self.get_sequence(name=name, language=language)

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
    def _validate_lookup(
        cls,
        *,
        sequence_id: int | None,
        name: str | None,
        language: str,
    ) -> None:
        if (sequence_id is None) == (name is None):
            raise ValueError("Provide exactly one of sequence_id or name")
        if sequence_id is not None and (
            isinstance(sequence_id, bool) or sequence_id <= 0
        ):
            raise ValueError("sequence_id must be a positive integer")
        if name is not None and not name.strip():
            raise ValueError("name must not be empty")
        if language not in cls._LANGUAGES:
            raise ValueError("language must be one of: en, uk, ru")

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return "The public sequences API rejected the request"

        if isinstance(payload, dict):
            for key in ("message", "error", "detail"):
                value = payload.get(key)
                if isinstance(value, str):
                    return value

        return "The public sequences API rejected the request"
