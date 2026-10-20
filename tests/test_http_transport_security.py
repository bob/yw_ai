"""Exercise upstream resource limits without opening a network socket."""

import gzip
import unittest

import httpx

from langchain_app.services.http_transport import ResponseTooLargeError, bounded_get
from langchain_app.services.http_transport import MAX_API_RESPONSE_BYTES
from langchain_app.services.catalog_errors import CatalogServiceError
from langchain_app.services.catalog_item import CatalogItemService
from langchain_app.services.catalog_tag import CatalogTagService
from langchain_app.services.sequence import SequenceApiError, SequenceService


class ObservedStream(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False
        self.read_chunks = 0

    def __iter__(self):
        for chunk in self.chunks:
            self.read_chunks += 1
            yield chunk

    def close(self):
        self.closed = True


class BoundedResponseTests(unittest.TestCase):
    def test_all_catalog_consumers_translate_oversize_into_public_service_errors(self):
        with httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"content-length": str(MAX_API_RESPONSE_BYTES + 1)}),
        )) as client:
            for kind, method in ((CatalogItemService, "search"), (CatalogTagService, "get_names")):
                with self.subTest(kind=kind), self.assertRaises(CatalogServiceError) as error:
                    getattr(kind("https://example.com", "folder", client=client), method)()
                self.assertEqual(error.exception.code, "invalid_api_response")
            with self.assertRaises(SequenceApiError):
                SequenceService("https://example.com", client=client).get_by_id(1, language="en")

    def test_normal_and_compressed_json_preserve_decoded_content(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                payload = b'{"name":"Tree"}'
                body = gzip.compress(payload) if compressed else payload
                headers = {"content-encoding": "gzip"} if compressed else {}
                with httpx.Client(transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, content=body, headers=headers),
                )) as client:
                    response = bounded_get(client, "https://example.com/catalog")
                self.assertEqual(response.json(), {"name": "Tree"})
                self.assertNotIn("content-encoding", response.headers)

    def test_declared_oversize_is_rejected_without_reading_and_stream_is_closed(self):
        stream = ObservedStream([b"not read"])
        with httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"content-length": "1000"}, stream=stream),
        )) as client, self.assertRaises(ResponseTooLargeError):
            bounded_get(client, "https://example.com/catalog", max_bytes=10)
        self.assertEqual(stream.read_chunks, 0)
        self.assertTrue(stream.closed)

    def test_chunked_and_gzipped_oversize_are_rejected(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                payload = b"x" * (128 * 1024)
                body = gzip.compress(payload) if compressed else payload
                headers = {"content-encoding": "gzip"} if compressed else {}
                stream = ObservedStream([body, b"must not be read"])
                with httpx.Client(transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, headers=headers, stream=stream),
                )) as client, self.assertRaises(ResponseTooLargeError):
                    bounded_get(client, "https://example.com/catalog", max_bytes=1024)
                self.assertEqual(stream.read_chunks, 1)
                self.assertTrue(stream.closed)

    def test_injected_client_cannot_follow_redirect_into_an_internal_service(self):
        requests = []

        def redirect(request):
            requests.append(request)
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})

        with httpx.Client(transport=httpx.MockTransport(redirect), follow_redirects=True) as client:
            response = bounded_get(client, "https://example.com/catalog")
            self.assertFalse(client.is_closed)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].headers["accept-encoding"], "identity")
