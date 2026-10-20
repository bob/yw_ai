"""Read upstream responses with an explicit memory budget and no redirects."""

import httpx

MAX_API_RESPONSE_BYTES = 16 * 1024 * 1024


class ResponseTooLargeError(ValueError):
    """An upstream body exceeded the caller's allowed size."""


def bounded_get(
    client: httpx.Client,
    url: str | httpx.URL,
    *,
    params: dict[str, str] | list[tuple[str, str]] | None = None,
    max_bytes: int = MAX_API_RESPONSE_BYTES,
) -> httpx.Response:
    """Limit retained decoded bytes, including chunked and compressed responses.

    Request identity encoding to avoid unnecessary decompression. An injected
    client must not silently follow redirects to a different trust boundary.
    Client ownership and its timeout/transport settings remain with the caller.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    with client.stream(
        "GET", url, params=params, follow_redirects=False,
        headers={"Accept-Encoding": "identity"},
    ) as response:
        length = response.headers.get("content-length")
        if length is not None:
            try:
                declared_length = int(length)
            except ValueError:
                declared_length = 0
            if declared_length > max_bytes:
                raise ResponseTooLargeError("Upstream response exceeded the size limit")
        body = bytearray()
        for chunk in response.iter_bytes(chunk_size=64 * 1024):
            if len(body) + len(chunk) > max_bytes:
                raise ResponseTooLargeError("Upstream response exceeded the size limit")
            body.extend(chunk)
        # iter_bytes has already decoded Content-Encoding. Reconstruct a normal
        # buffered response without decoding it a second time or retaining the
        # original (compressed) length/transfer framing.
        headers = [
            (name, value) for name, value in response.headers.multi_items()
            if name.lower() not in {"content-encoding", "content-length", "transfer-encoding"}
        ]
        return httpx.Response(
            response.status_code, headers=headers, content=bytes(body),
            request=response.request, extensions=response.extensions,
        )
