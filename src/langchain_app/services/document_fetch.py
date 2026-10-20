"""Constrained document fetching for the Gutenberg training example only."""

import httpx

from langchain_app.services.http_transport import ResponseTooLargeError, bounded_get

MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
_DOCUMENT_HOSTS = frozenset({"www.gutenberg.org", "gutenberg.org"})


def _document_url(value: str) -> httpx.URL:
    """Accept only the example's trusted HTTPS origins, with no userinfo."""
    if any(ord(char) < 32 for char in value) or "\\" in value:
        raise ValueError("Only HTTPS Project Gutenberg document URLs are allowed")
    try:
        url = httpx.URL(value)
    except httpx.InvalidURL as exc:
        raise ValueError("Only HTTPS Project Gutenberg document URLs are allowed") from exc
    if (
        url.scheme != "https" or url.host not in _DOCUMENT_HOSTS
        or url.port not in {None, 443} or url.userinfo or url.fragment
    ):
        raise ValueError("Only HTTPS Project Gutenberg document URLs are allowed")
    return url


def fetch_document_text(url: str, *, client: httpx.Client | None = None) -> str:
    """Read a bounded Gutenberg document; never fetch a model-selected host.

    Redirects are resolved explicitly and validated before every request.
    Invalid URLs and fetch errors return public messages without raw addresses,
    response bodies, credentials, or local file contents.
    """
    try:
        target = _document_url(url)
    except (ValueError, TypeError):
        return "Fetch failed: only HTTPS Project Gutenberg document URLs are allowed."
    if client is None:
        with httpx.Client(
            timeout=30.0, trust_env=False,
            headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
        ) as owned:
            return fetch_document_text(str(target), client=owned)
    try:
        for _ in range(4):
            response = bounded_get(client, target, max_bytes=MAX_DOCUMENT_BYTES)
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    return "Fetch failed: invalid document redirect."
                target = _document_url(str(target.join(location)))
                continue
            response.raise_for_status()
            return response.content.decode("utf-8", errors="replace")
    except ResponseTooLargeError:
        return "Fetch failed: document exceeds the 2 MiB size limit."
    except (httpx.HTTPError, httpx.InvalidURL, ValueError):
        return "Fetch failed: document unavailable or redirect destination is not allowed."
    return "Fetch failed: too many document redirects."
