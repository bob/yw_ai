"""Verify English catalog HTTP contracts with an in-memory transport."""

import unittest
from unittest.mock import Mock, patch

import httpx
from pydantic import ValidationError

from langchain_app.services.catalog_errors import CatalogServiceError
from langchain_app.services.catalog_item import CatalogItemService
from langchain_app.services.catalog_tag import CatalogTagService


def catalog_item(**overrides):
    return {
        "id": 42, "uuid": "tree-pose", "name": "Tree pose", "bilateral": False,
        "image_url": "https://example.com/tree.png", "catalog_tag_names": ["Balance", "Standing"],
        **overrides,
    }


class CatalogServiceTests(unittest.TestCase):
    def service(self, kind=CatalogItemService, *, payload=None, status=200, content=None):
        self.requests = []

        def handle(request):
            self.requests.append(request)
            if content is not None:
                return httpx.Response(status, content=content)
            return httpx.Response(status, json=[] if payload is None else payload)

        client = httpx.Client(transport=httpx.MockTransport(handle))
        self.addCleanup(client.close)
        return kind("https://catalog.example/prefix/", " yoga/folder ", client=client)

    def test_default_search_and_tags_always_use_english_and_configured_folder(self):
        items = self.service(payload=[catalog_item()]).search()
        request = self.requests[0]
        self.assertEqual(request.url.raw_path.split(b"?")[0],
                         b"/prefix/api/public/catalog_folders/yoga%2Ffolder/catalog_items/search")
        self.assertEqual(dict(request.url.params), {
            "language": "en", "with_tags": "true", "with_rigman": "false",
            "limit": "10", "rand": "false",
        })
        self.assertEqual(items[0].name, "Tree pose")
        self.assertEqual(items[0].catalog_tag_names, ["Balance", "Standing"])
        names = ["Other/Starting", "Balance", "Standing"]
        self.assertEqual(self.service(CatalogTagService, payload=names).get_names(), names)
        self.assertEqual(dict(self.requests[0].url.params), {"language": "en"})
        self.assertTrue(self.requests[0].url.path.endswith("/catalog_tags"))

    def test_filters_preserve_false_and_repeat_tag_keys(self):
        self.service().search(
            name=" Tree ", uuid=" ABC ", bilateral=False,
            catalog_tag_names=[" Balance ", "Standing"], limit=25, rand=True,
        )
        params = self.requests[0].url.params
        self.assertEqual(params["name"], "Tree")
        self.assertEqual(params["uuid"], "ABC")
        self.assertEqual(params["bilateral"], "false")
        self.assertEqual(params.get_list("catalog_tag_names[]"), ["Balance", "Standing"])
        self.assertEqual(params["rand"], "true")
        self.assertEqual(params["limit"], "25")
        self.assertEqual(params["language"], "en")

    def test_empty_results_and_filters_are_valid(self):
        self.assertEqual(self.service().search(catalog_tag_names=[]), [])
        self.assertNotIn("catalog_tag_names[]", self.requests[0].url.params)
        self.assertNotIn("bilateral", self.requests[0].url.params)
        self.assertNotIn("name", self.requests[0].url.params)
        self.assertEqual(self.service(CatalogTagService).get_names(), [])

    def test_invalid_local_filters_fail_before_http(self):
        service = self.service()
        for arguments in (
            {"limit": True}, {"limit": 2.5}, {"limit": -1}, {"name": " "},
            {"uuid": " "}, {"catalog_tag_names": [" "]}, {"rand": "false"},
            {"bilateral": "false"},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValidationError):
                service.search(**arguments)
        self.assertEqual(self.requests, [])
        with self.assertRaises(TypeError):
            service.search(language="ru")

    def test_optional_fields_and_tags_contract(self):
        payload = catalog_item(rigman_data={"render": {"asana": True}}, second_side_rigman_data=None)
        results = self.service(payload=[payload]).search(with_rigman=True)
        self.assertEqual(results[0].rigman_data, {"render": {"asana": True}})
        payload.pop("catalog_tag_names")
        self.assertIsNone(self.service(payload=[payload]).search(with_tags=False)[0].catalog_tag_names)
        for data in (payload, {**payload, "catalog_tag_names": None}):
            with self.subTest(data=data), self.assertRaises(CatalogServiceError) as error:
                self.service(payload=[data]).search()
            self.assertEqual(error.exception.code, "invalid_api_response")
        self.assertEqual(self.service(payload=[catalog_item(catalog_tag_names=[])]).search()[0].catalog_tag_names, [])

    def test_items_with_null_uuid_or_image_are_kept_in_search(self):
        results = self.service(payload=[catalog_item(uuid=None, image_url=None)]).search()
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0].uuid)
        self.assertIsNone(results[0].image_url)

    def test_both_clients_report_http_and_invalid_response_errors(self):
        for kind, method in ((CatalogItemService, "search"), (CatalogTagService, "get_names")):
            for status, code in ((404, "catalog_folder_not_found"), (422, "invalid_arguments"),
                                 (500, "catalog_api_unavailable")):
                with self.subTest(kind=kind, status=status), self.assertRaises(CatalogServiceError) as error:
                    getattr(self.service(kind, status=status), method)()
                self.assertEqual(error.exception.code, code)
            for payload in ({"data": []}, [7]):
                with self.subTest(kind=kind, payload=payload), self.assertRaises(CatalogServiceError) as error:
                    getattr(self.service(kind, payload=payload), method)()
                self.assertEqual(error.exception.code, "invalid_api_response")
            with self.subTest(kind=kind), self.assertRaises(CatalogServiceError) as error:
                getattr(self.service(kind, content=b"not JSON"), method)()
            self.assertEqual(error.exception.code, "invalid_api_response")

    def test_network_errors_and_client_ownership(self):
        def timeout(request):
            raise httpx.ReadTimeout("timeout", request=request)

        for kind, method in ((CatalogItemService, "search"), (CatalogTagService, "get_names")):
            with self.subTest(kind=kind), httpx.Client(transport=httpx.MockTransport(timeout)) as client:
                service = kind("https://example.com", "folder", client=client)
                with self.assertRaises(CatalogServiceError) as error:
                    getattr(service, method)()
                self.assertEqual(error.exception.code, "catalog_api_unavailable")
                service.close()
                self.assertFalse(client.is_closed)
            owned_client = Mock()
            with patch("langchain_app.services.catalog_client.httpx.Client", return_value=owned_client):
                with kind("https://example.com", "folder"):
                    pass
            owned_client.close.assert_called_once()
