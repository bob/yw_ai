"""Verify explicit catalog selection through the real agent loop, without API calls."""

import json
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage

from langchain_app.tools import show_catalog_items
from langchain_app.tools.context import AssistantToolContext
from langchain_app.services.catalog_item import CatalogItemApiError
from test_assistant_agent import ToolCallingModel, assistant, pose, tool_call
from test_sequence_replacement import catalog_pose, replacement_call


def display_call(*ids):
    return tool_call("show_catalog_items", catalog_item_ids=list(ids))


def display_results(service):
    messages = service.state["messages"]
    call_ids = {
        call["id"] for message in messages if isinstance(message, AIMessage)
        for call in message.tool_calls if call["name"] == "show_catalog_items"
    }
    return [
        message for message in messages
        if isinstance(message, ToolMessage) and message.tool_call_id in call_ids
    ]


class CatalogDisplayToolTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_selection_uses_latest_search_in_argument_order_without_changing_sequences(self):
        catalog = Mock()
        selected = [pose(2), pose(3), pose(4)]
        selected[0].image_url = None
        selected[1].bilateral = True
        catalog.search.side_effect = [[pose(1)], selected]
        model = ToolCallingModel(responses=[
            tool_call("search_catalog_items", name="First"),
            tool_call("search_catalog_items", name="Second"), display_call(3, 2),
            AIMessage(content="Here are the options."),
        ])
        factory = Mock()
        service = assistant(factory, model=model, catalog_item_service=catalog)
        self.assertEqual(list(service.stream("Show options")), ["Here are the options."])
        result, = display_results(service)
        self.assertEqual(json.loads(result.content), {
            "status": "ok", "displayed_count": 2, "catalog_item_ids": [3, 2],
        })
        self.assertEqual(result.artifact["items"], [selected[1].model_dump(mode="json"), selected[0].model_dump(mode="json")])
        self.assertEqual(service.state["latest_catalog_items"], [item.model_dump(mode="json") for item in selected])
        self.assertFalse(service.state["turn_operation_committed"])
        self.assertIsNone(service.active_sequence_uuid)
        self.assertFalse(service.state.get("sequences"))
        factory.assert_not_called()

    def test_invalid_ids_do_not_consume_successful_display(self):
        for ids in ([], [0], [-1], [True], ["1"], [1, 1], [1, 99]):
            with self.subTest(ids=ids):
                catalog = Mock()
                catalog.search.return_value = [pose(1)]
                model = ToolCallingModel(responses=[
                    tool_call("search_catalog_items"), display_call(*ids), display_call(1),
                    AIMessage(content="Found."),
                ])
                service = assistant(Mock(), model=model, catalog_item_service=catalog)
                list(service.stream("Show options"))
                failed, success = display_results(service)
                self.assertEqual(failed.status, "error")
                self.assertIsNone(failed.artifact)
                self.assertEqual(success.status, "success")
                self.assertEqual(success.artifact["items"][0]["id"], 1)

    def test_display_once_per_request_even_after_another_search_and_reset_next_turn(self):
        catalog = Mock()
        catalog.search.side_effect = [[pose(1)], [pose(2)]]
        model = ToolCallingModel(responses=[
            tool_call("search_catalog_items"), display_call(1),
            tool_call("search_catalog_items"), display_call(2), AIMessage(content="Shown."),
        ])
        service = assistant(Mock(), model=model, catalog_item_service=catalog)
        list(service.stream("Show one"))
        shown, rejected = display_results(service)
        self.assertEqual(shown.artifact["items"][0]["id"], 1)
        self.assertEqual(json.loads(rejected.content)["code"], "catalog_items_already_displayed")
        self.assertIsNone(rejected.artifact)
        model.responses = [display_call(2), AIMessage(content="Another option.")]
        list(service.stream("Show another"))
        self.assertEqual(display_results(service)[-1].artifact["items"][0]["id"], 2)
        self.assertEqual(display_results(service)[0].artifact, shown.artifact)

    def test_missing_or_cleared_search_candidates_cannot_be_displayed(self):
        for failure in ("missing", "empty", "api", "value", "arguments", "new_search"):
            with self.subTest(failure=failure):
                catalog = Mock()
                catalog.search.return_value = [pose(1)]
                model = ToolCallingModel(responses=[tool_call("search_catalog_items"), AIMessage(content="Found.")])
                service = assistant(Mock(), model=model, catalog_item_service=catalog)
                if failure != "missing":
                    list(service.stream("Find options"))
                calls = []
                if failure not in ("missing",):
                    if failure == "api":
                        catalog.search.side_effect = CatalogItemApiError("Unavailable", code="api_unavailable")
                    elif failure == "value":
                        catalog.search.side_effect = ValueError("Invalid filter")
                    else:
                        catalog.search.return_value = [pose(2)] if failure == "new_search" else []
                    calls = [tool_call("search_catalog_items", **({"limit": 0} if failure == "arguments" else {}))]
                model.responses = [*calls, display_call(1), AIMessage(content="Search again.")]
                list(service.stream("Show options"))
                result = display_results(service)[-1]
                self.assertEqual(json.loads(result.content)["code"], "catalog_item_not_verified")
                self.assertIsNone(result.artifact)

    def test_snapshot_is_detached_and_invalid_catalog_data_can_be_retried(self):
        item = pose(1)
        item.catalog_tag_names = ["Standing"]
        snapshot = item.model_dump(mode="json")
        state = {"latest_catalog_items": [snapshot], "turn_operation_committed": True}
        context = AssistantToolContext(Mock(), Mock(), Mock())
        runtime = ToolRuntime(state, context, {}, Mock(), "display-call", None)
        snapshot["image_url"] = "invalid-url"
        before = deepcopy(state)
        failed = show_catalog_items.func([1], runtime)
        self.assertEqual(failed.status, "error")
        self.assertIsNone(failed.artifact)
        self.assertFalse(context.catalog_items_displayed)
        self.assertEqual(state, before)
        snapshot["image_url"] = str(item.image_url)
        shown = show_catalog_items.func([1], runtime)
        self.assertEqual(shown.status, "success")
        shown.artifact["items"][0]["catalog_tag_names"].append("Changed")
        shown.artifact["items"][0]["name"] = "Changed"
        self.assertEqual(snapshot, item.model_dump(mode="json"))
        snapshot["name"] = "Later search"
        self.assertEqual(shown.artifact["items"][0]["name"], "Changed")
        runtime.stream_writer.assert_not_called()

    def test_showing_candidates_preserves_selected_sequence_and_allows_explicit_replacement(self):
        generator = Mock()
        generator.generate.return_value = [catalog_pose(1), catalog_pose(2), catalog_pose(3)]
        catalog = Mock()
        catalog.search.return_value = [catalog_pose(9)]
        model = ToolCallingModel(responses=[
            tool_call("generate_sequence", request="Practice", count=3), AIMessage(content="Created."),
        ])
        service = assistant(lambda: generator, model=model, catalog_item_service=catalog)
        list(service.generate("Practice"))
        source = service.get_sequence(service.active_sequence_uuid)
        before = service.state["sequences"]
        model.responses = [tool_call("search_catalog_items"), display_call(9), AIMessage(content="Candidates.")]
        list(service.stream("Find candidates, do not change anything"))
        self.assertEqual(service.state["sequences"], before)
        self.assertEqual(service.active_sequence_uuid, str(source.uuid))
        self.assertFalse(service.state["turn_operation_committed"])
        model.responses = [display_call(9), replacement_call(source), AIMessage(content="Replaced.")]
        list(service.stream("Show candidate and replace third"))
        self.assertEqual(service.get_sequence(service.active_sequence_uuid).elements[2].catalog_item.id, 9)
        self.assertEqual(service.state["sequences"][str(source.uuid)], before[str(source.uuid)])
        self.assertEqual(len(service.state["sequences"]), 2)
        generator.generate.assert_called_once()

    def test_generate_and_regenerate_commands_reject_display(self):
        generator = Mock()
        generator.generate.return_value = [pose(1)]
        catalog = Mock()
        catalog.search.return_value = [pose(9)]
        model = ToolCallingModel(responses=[
            tool_call("search_catalog_items"), display_call(9),
            tool_call("generate_sequence", request="Practice", count=1), AIMessage(content="Created."),
        ])
        service = assistant(lambda: generator, model=model, catalog_item_service=catalog)
        list(service.generate("Practice"))
        source_uuid = service.active_sequence_uuid
        model.responses = [display_call(9), tool_call("generate_sequence", request="Practice", count=1), AIMessage(content="Regenerated.")]
        # Regenerate is forbidden even if its caller did not set require_generation.
        list(service.stream("Regenerate", regenerate_sequence_uuid=source_uuid))
        for result in display_results(service):
            self.assertEqual(json.loads(result.content)["code"], "generation_required")
            self.assertIsNone(result.artifact)
        self.assertEqual(generator.generate.call_count, 2)
        self.assertEqual(len(service.state["sequences"]), 2)

    def test_search_alone_has_no_display_snapshot(self):
        catalog = Mock()
        catalog.search.return_value = [pose(1)]
        model = ToolCallingModel(responses=[tool_call("search_catalog_items"), AIMessage(content="Found.")])
        service = assistant(Mock(), model=model, catalog_item_service=catalog)
        list(service.stream("Find options"))
        self.assertEqual(display_results(service), [])
        self.assertTrue(service.state["latest_catalog_items"])
        self.assertTrue(all(message.artifact is None for message in service.state["messages"] if isinstance(message, ToolMessage)))

    def test_model_schema_only_accepts_ids(self):
        self.assertEqual(set(show_catalog_items.tool_call_schema.model_fields), {"catalog_item_ids"})
