"""Verify feature 009 through the real agent loop with offline dependencies."""

import json
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

import httpx
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
from openai import APIConnectionError

from fixtures.compact_tool_results import measure, pose, run_case, source_sequence
from langchain_app.schemas.catalog import CatalogItemSearchResult
from langchain_app.services.assistant_agent import AssistantAgentError
from langchain_app.services.response_progress_middleware import _tool_result
from test_assistant_agent import ToolCallingModel, assistant, selected_snapshot, tool_call

CASES = [("generate", count) for count in (9, 36, 45)] + [
    ("regenerate", 9), ("replace", 9), ("swap", 9), ("add", 9),
]


def operation_data(sequence):
    """Ignore newly allocated identities when comparing independent executions."""
    snapshot = sequence.model_dump(mode="json", exclude={"uuid", "family_uuid"})
    for element in snapshot["elements"]:
        element.pop("element_id")
    return snapshot


class CompactToolResultTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        environment.start()
        self.addCleanup(environment.stop)

    def assert_compact(self, case):
        payload = case.payload
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["uuid"], str(case.sequence.uuid))
        self.assertEqual(payload["operation"], case.sequence.operation)
        self.assertEqual(payload["elements_count"], len(case.sequence.elements))
        forbidden = {"elements", "family_uuid", "parent_uuid", "generation_request", "operation_request", "name", "description", "language"}
        self.assertFalse(forbidden & set(payload))
        self.assertEqual(case.message.status, "success")
        for message in case.service.state["messages"]:
            if isinstance(message, ToolMessage):
                self.assertNotIn("image_url", message.content)
                self.assertNotIn("rigman", message.content)
                self.assertIsNone(message.artifact)
        self.assertEqual(case.message.tool_call_id, case.model.received_messages[-1][-2].tool_calls[0]["id"])
        self.assertEqual(selected_snapshot(case.model), case.sequence.model_context())
        self.assertTrue(case.service.state["turn_operation_committed"])
        self.assertEqual(case.service.state["sequences"][str(case.sequence.uuid)], case.sequence.model_dump(mode="json"))
        self.assertTrue(all(element.catalog_item.image_url for element in case.sequence.elements))
        commits = [event for event in case.events if event.type == "sequence_committed"]
        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0].sequence_uuid, case.sequence.uuid)
        self.assertEqual(case.service.get_sequence(str(commits[0].sequence_uuid)), case.sequence)

    def test_successes_keep_full_state_and_one_composition_in_the_next_model_call(self):
        for operation, count in CASES:
            with self.subTest(operation=operation, count=count):
                case = run_case(operation, count)
                self.assert_compact(case)
                if operation != "generate":
                    self.assertEqual(case.service.get_sequence(str(case.source.uuid)), case.source)
                    self.assertEqual(case.sequence.parent_uuid, case.source.uuid)
                    self.assertEqual(case.sequence.family_uuid, case.source.family_uuid)
                    self.assertEqual(case.sequence.generation_request, case.source.generation_request)
                if operation == "replace":
                    self.assertEqual(case.payload["replacement"], {
                        "element_id": str(case.source.elements[2].element_id), "order": 3,
                        "old_catalog_item": {"id": 3, "name": "Seated pose 3"},
                        "new_catalog_item": {"id": 80, "name": "Seated pose 80"},
                    })
                    self.assertEqual(case.service.state["messages"][-1].text, "Заменена Seated pose 3 на Seated pose 80.")
                    candidate = case.service.state["latest_catalog_items"][0]
                    self.assertEqual(candidate["image_url"], str(pose(80).image_url))
                    self.assertEqual(case.sequence.elements[2].catalog_item.model_dump(mode="json"), candidate)
                elif operation == "swap":
                    self.assertEqual([entry["name"] for entry in case.payload["swap"]], ["Seated pose 3", "Seated pose 5"])
                    self.assertEqual(case.service.state["messages"][-1].text, "Seated pose 3: 3 → 5; Seated pose 5: 5 → 3")
                elif operation == "add":
                    self.assertEqual(case.payload["addition"]["position"], 10)
                    self.assertEqual(case.service.state["messages"][-1].text, "Добавлено 3 с позиции 10; всего 12.")
                    self.assertTrue(all(set(entry) == {"element_id", "order"} for entry in case.payload["addition"]["elements"]))

    def test_baseline_and_compact_loops_keep_gpt_calls_jev_payloads_and_selection_identical(self):
        for operation, count in CASES:
            with self.subTest(operation=operation, count=count):
                before, after = run_case(operation, count, legacy=True), run_case(operation, count)
                self.assertEqual(before.jev_inputs, after.jev_inputs)
                self.assertEqual(before.jev_outputs, after.jev_outputs)
                self.assertEqual(before.catalog.mock_calls, after.catalog.mock_calls)
                self.assertEqual(before.factory.call_count, after.factory.call_count)
                self.assertEqual(operation_data(before.sequence), operation_data(after.sequence))
                self.assertEqual(len(before.model.received_messages), len(after.model.received_messages))
                self.assertEqual(len(after.model.received_messages), 3 if operation == "replace" else 2)
                self.assertEqual(before.service.state["messages"][-1].text, after.service.state["messages"][-1].text)
                for case in (before, after):
                    result = Command(update={
                        "messages": [case.message], "sequences": {str(case.sequence.uuid): case.sequence.model_dump(mode="json")},
                    })
                    self.assertEqual(_tool_result(result, case.message.name)[0], "ok")
                self.assertEqual(_tool_result(before.message, before.message.name), _tool_result(after.message, after.message.name))
                if operation in ("generate", "regenerate", "add"):
                    self.assertTrue(after.jev_inputs)
                else:
                    self.assertEqual(after.jev_inputs, [])

    def test_search_is_an_allowlist_and_gallery_keeps_full_original_candidates(self):
        items = [CatalogItemSearchResult(
            **pose(80).model_dump(mode="json"), rigman_data={"angles": [1, 2]}, second_side_rigman_data={"angles": [3]},
        ), pose(81)]
        items[1].uuid = None
        items[1].image_url = None
        catalog = Mock()
        catalog.search.return_value = items
        model = ToolCallingModel(responses=[
            tool_call("search_catalog_items", name="Seated"),
            tool_call("show_catalog_items", catalog_item_ids=[81, 80]), AIMessage(content="Вот варианты."),
        ])
        service = assistant(Mock(), catalog_item_service=catalog, model=model)
        list(service.stream("Покажи сидячие позы"))
        messages = [m for m in service.state["messages"] if isinstance(m, ToolMessage)]
        search, display = messages
        payload = json.loads(search.content)
        self.assertEqual(set(payload), {"status", "language", "returned_count", "limit", "items"})
        self.assertEqual(payload["returned_count"], 2)
        for candidate in payload["items"]:
            self.assertEqual(set(candidate), {"id", "uuid", "name", "bilateral", "catalog_tag_names"})
        self.assertIsNone(payload["items"][1]["uuid"])
        self.assertNotIn("image_url", search.content)
        self.assertNotIn("rigman", search.content)
        snapshots = [item.model_dump(mode="json", exclude={"rigman_data", "second_side_rigman_data"}) for item in items]
        self.assertEqual(service.state["latest_catalog_items"], snapshots)
        self.assertEqual(display.artifact["items"], snapshots[::-1])
        self.assertEqual(json.loads(display.content)["catalog_item_ids"], [81, 80])

    def test_legacy_history_and_an_older_selection_keep_duplicate_occurrences_and_mirrors(self):
        source = source_sequence()
        source.elements[2].catalog_item = source.elements[4].catalog_item.model_copy(deep=True)
        source.elements[2].start_mirror, source.elements[2].end_mirror = True, False
        source.elements[4].start_mirror, source.elements[4].end_mirror = False, True
        case = run_case("regenerate", source=source, legacy=True)
        old_content = case.message.content
        self.assertIn("elements", case.payload)
        calls = len(case.model.received_messages)
        case.service.select_sequence(str(source.uuid))
        self.assertEqual(len(case.model.received_messages), calls)
        case.model.responses = [tool_call(
            "swap_sequence_elements", sequence_uuid=str(source.uuid),
            first_element_id=str(source.elements[2].element_id), second_element_id=str(source.elements[4].element_id),
        ), AIMessage(content="")]
        case.model.i = 0
        list(case.service.stream("Поменяй третью и пятую"))
        self.assertEqual(selected_snapshot(case.model, -2), source.model_context())
        variant = case.service.get_sequence(case.service.active_sequence_uuid)
        self.assertEqual(variant.parent_uuid, source.uuid)
        self.assertEqual((variant.elements[2].start_mirror, variant.elements[2].end_mirror), (False, True))
        self.assertEqual((variant.elements[4].start_mirror, variant.elements[4].end_mirror), (True, False))
        payload = json.loads(case.model.received_messages[-1][-1].content)
        self.assertNotIn("elements", payload)
        self.assertEqual([entry["name"] for entry in payload["swap"]], ["Seated pose 5"] * 2)
        self.assertEqual(len({entry["element_id"] for entry in payload["swap"]}), 2)
        self.assertEqual(case.service.get_sequence(str(source.uuid)), source)
        legacy = next(m for m in case.service.state["messages"] if isinstance(m, ToolMessage) and m.id == case.message.id)
        self.assertEqual(legacy.content, old_content)

    def test_insert_and_cap_report_actual_counts_and_full_limit_does_not_commit(self):
        for size, requested, anchor, added, position in ((9, 3, 2, 3, 3), (43, 5, None, 2, 44)):
            with self.subTest(size=size):
                case = run_case("add", source=source_sequence(size), added_count=requested, anchor_index=anchor)
                self.assert_compact(case)
                change = case.payload["addition"]
                self.assertEqual((change["requested_count"], change["added_count"], change["position"]), (requested, added, position))
                self.assertEqual(change["limit_applied"], requested > added)
                self.assertEqual([e["order"] for e in change["elements"]], list(range(position, position + added)))
                self.assertEqual(case.sequence.elements_count, size + added)
        full = run_case("add", source=source_sequence(45))
        self.assertEqual(full.payload["status"], "no_change")
        self.assertEqual(full.payload["code"], "sequence_limit_reached")
        self.assertEqual(full.payload["added_count"], 0)
        self.assertTrue(full.payload["limit_applied"])
        self.assertEqual(full.sequence, full.source)
        self.assertEqual(len(full.service.state["sequences"]), 1)
        self.assertFalse(any(event.type == "sequence_committed" for event in full.events))
        full.factory.assert_not_called()
        self.assertEqual(_tool_result(full.message, full.message.name)[0], "no_change")

    def test_sequence_content_token_reduction_is_at_least_seventy_percent(self):
        report = measure()
        self.assertEqual(report["encoding"], "o200k_base")
        self.assertGreaterEqual(report["reduction_percent"], 70)
        for row in report["rows"]:
            with self.subTest(scenario=row["scenario"]):
                self.assertGreaterEqual(row["reduction_percent"], 70)

    def test_failure_before_commit_keeps_source_and_failure_after_commit_keeps_compact_success(self):
        for phase in ("before", "after"):
            with self.subTest(phase=phase):
                source = source_sequence()
                generator = Mock()
                generator.generate.return_value = [pose(80)]

                def broken_generation(*args, **kwargs):
                    yield pose(80)
                    raise ValueError("Selection failed")

                if phase == "before":
                    generator.generate.side_effect = broken_generation
                model = ToolCallingModel(responses=[
                    tool_call("generate_sequence", request="New practice", count=1), AIMessage(content=""),
                ])
                service = assistant(lambda: generator, model=model)
                service._get_agent().update_state(service._config, {
                    "sequences": {str(source.uuid): source.model_dump(mode="json")}, "active_sequence_uuid": str(source.uuid),
                })
                original_generate = model._generate

                def fail_final_reply(messages, **kwargs):
                    if isinstance(messages[-1], ToolMessage):
                        raise APIConnectionError(request=httpx.Request("POST", "https://example.com"))
                    return original_generate(messages, **kwargs)

                events = []
                with patch.object(model, "_generate", side_effect=fail_final_reply):
                    with self.assertRaises(ValueError if phase == "before" else AssistantAgentError):
                        for event in service.stream_response_events("New practice"):
                            events.append(event)
                self.assertEqual(service.get_sequence(str(source.uuid)), source)
                self.assertEqual(events[-1].type, "request_finished")
                self.assertEqual(events[-1].outcome, "error")
                message = next(m for m in reversed(service.state["messages"]) if isinstance(m, ToolMessage))
                payload = json.loads(message.content)
                self.assertNotIn("elements", payload)
                self.assertNotIn("image_url", message.content)
                if phase == "before":
                    self.assertEqual(service.active_sequence_uuid, str(source.uuid))
                    self.assertEqual(message.status, "error")
                    self.assertFalse(any(e.type == "sequence_committed" for e in events))
                    self.assertEqual(len(service.state["sequences"]), 1)
                else:
                    self.assertEqual(payload["status"], "ok")
                    self.assertEqual(payload["elements_count"], 1)
                    self.assertEqual(message.status, "success")
                    self.assertEqual(len(service.state["sequences"]), 2)
                    self.assertEqual(service.active_sequence_uuid, payload["uuid"])
                    committed = next(e for e in events if e.type == "sequence_committed")
                    self.assertEqual(str(committed.sequence_uuid), payload["uuid"])
                    self.assertEqual(service.get_sequence(payload["uuid"]).elements[0].catalog_item, pose(80))


if __name__ == "__main__":
    unittest.main()
