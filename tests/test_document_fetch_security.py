"""Keep model-selected document URLs away from files and internal services."""

import pathlib
import runpy
import unittest
from unittest.mock import patch

import httpx

from langchain_app.services.document_fetch import fetch_document_text


class DocumentFetchTests(unittest.TestCase):
    def test_disallowed_urls_never_reach_the_transport(self):
        requests = []
        with httpx.Client(transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(200, text="private"),
        )) as client:
            for url in (
                "file:///etc/passwd", "http://127.0.0.1:4999/admin",
                "http://169.254.169.254/latest/meta-data/", "https://[::1]/",
                "https://www.gutenberg.org.evil.example/file",
                "https://www.gutenberg.org@127.0.0.1/file",
                "https://user:password@www.gutenberg.org/file",
                "https://www.gutenberg.org:8443/file", "ftp://www.gutenberg.org/file",
                "https://www.gutenberg.org\\@evil.example/file",
                "https://www.gutenberg.org/\nfile", "https://www.gutenberg.org/file#fragment",
            ):
                with self.subTest(url=url):
                    self.assertTrue(fetch_document_text(url, client=client).startswith("Fetch failed:"))
        self.assertEqual(requests, [])

    def test_https_document_and_same_origin_redirect_still_work(self):
        requests = []

        def respond(request):
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(302, headers={"location": "/files/64317/64317-0.txt"})
            return httpx.Response(200, text="Gatsby\nDaisy\n")

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            text = fetch_document_text("https://www.gutenberg.org/document", client=client)
        self.assertEqual(text, "Gatsby\nDaisy\n")
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(r.url.host == "www.gutenberg.org" for r in requests))

    def test_disallowed_redirect_is_rejected_before_a_second_request(self):
        for location in (
            "file:///etc/passwd", "http://169.254.169.254/",
            "https://evil.example/file", "https://[invalid",
        ):
            requests = []

            def redirect(request):
                requests.append(request)
                return httpx.Response(302, headers={"location": location})

            with self.subTest(location=location), httpx.Client(
                transport=httpx.MockTransport(redirect), follow_redirects=True,
            ) as client:
                result = fetch_document_text("https://www.gutenberg.org/document", client=client)
            self.assertTrue(result.startswith("Fetch failed:"))
            self.assertEqual(len(requests), 1)

    def test_document_size_and_redirect_chain_are_bounded(self):
        with httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"content-length": "3000000"}),
        )) as client:
            self.assertIn("2 MiB", fetch_document_text("https://www.gutenberg.org/file", client=client))
        requests = []
        with httpx.Client(transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(302, headers={"location": "/again"}),
        )) as client:
            self.assertIn("too many", fetch_document_text("https://www.gutenberg.org/file", client=client))
        self.assertEqual(len(requests), 4)

    def test_importing_the_training_example_does_not_initialize_or_call_a_model(self):
        script = pathlib.Path(__file__).parents[1] / "src/langchain_app/langchain-training/third.py"
        with patch("langchain.chat_models.init_chat_model") as model:
            runpy.run_path(str(script))
        model.assert_not_called()
