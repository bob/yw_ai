"""Validate user requests before they reach paid services or session history."""

MAX_REQUEST_CHARACTERS = 16_000


class InvalidRequestError(ValueError):
    """A safe, user-visible validation failure without the submitted content."""


def validate_request(request: str) -> None:
    if not isinstance(request, str) or not request.strip():
        raise InvalidRequestError("Enter a nonempty request.")
    if len(request) > MAX_REQUEST_CHARACTERS:
        raise InvalidRequestError(
            f"Please shorten your request to {MAX_REQUEST_CHARACTERS:,} characters or fewer."
        )
