"""Verify replacement commits, candidate provenance, and immutable history."""

import json
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import httpx
from langchain_core.messages import AIMessage, ToolMessage
from openai import APIConnectionError

from langchain_app.schemas import GeneratedSequence
from langchain_app.services.assistant_agent import AssistantAgentError
from langchain_app.services.catalog_item import CatalogItemApiError
from langchain_app.services.sequence_editor import replace_element
from test_assistant_agent import ToolCallingModel, assistant, pose, selected_snapshot, tool_call


def catalog_pose(number, *, bilateral=False):
    item = pose(number)
    item.uuid = str(uuid4())
    item.bilateral = bilateral
    item.catalog_tag_names = ["Standing"]
    return item


def replacement_call(source, candidate_id=9, *, element_index=2, **overrides):
    return tool_call("replace_sequence_element", **{
        "sequence_uuid": str(source.uuid),
        "element_id": str(source.elements[element_index].element_id),
        "replacement_catalog_item_id": candidate_id,
        **overrides,
    })


class SequenceReplacementTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        env.start()
        self.addCleanup(env.stop)

    def generated(self):
        generator = Mock()
        generator.generate.return_value = [catalog_pose(n) for n in range(1, 4)]
        catalog = Mock()
        catalog.search.return_value = [catalog_pose(9, bilateral=True)]
        model = ToolCallingModel(responses=[
            tool_call("generate_sequence", request="Gentle practice", count=3),
            AIMessage(content="Ready"),
        ])
        service = assistant(lambda: generator, model=model, catalog_item_service=catalog)
        list(service.generate("Gentle practice"))
        source = service.get_sequence(service.active_sequence_uuid)
        return service, source, generator, catalog, model

    def test_replace_third_occurrence_commits_snapshot_and_refreshes_model_context(self):
        service, source, generator, catalog, model = self.generated()
        before = service.state["sequences"]
        model.responses = [
            tool_call("search_catalog_items", name="Tree"), replacement_call(source),
            AIMessage(content="Заменена Pose 3 на Pose 9."),
        ]
        self.assertEqual(list(service.stream("Замени третью на дерево")), ["Заменена Pose 3 на Pose 9."])
        variant = service.get_sequence(service.active_sequence_uuid)
        self.assertNotEqual(variant.uuid, source.uuid)
        self.assertEqual(variant.parent_uuid, source.uuid)
        self.assertEqual(variant.family_uuid, source.family_uuid)
        self.assertEqual(variant.operation, "replace")
        self.assertEqual(variant.generation_request, source.generation_request)
        self.assertEqual(variant.operation_request, "Замени третью на дерево")
        self.assertEqual(variant.elements_count, source.elements_count)
        self.assertEqual(variant.elements[:2], source.elements[:2])
        self.assertEqual(variant.elements[2].element_id, source.elements[2].element_id)
        self.assertEqual(variant.elements[2].order, 3)
        self.assertEqual(variant.elements[2].catalog_item, catalog.search.return_value[0])
        self.assertTrue(variant.elements[2].start_mirror)
        self.assertTrue(variant.elements[2].end_mirror)
        self.assertEqual(service.state["sequences"][str(source.uuid)], before[str(source.uuid)])
        self.assertEqual(selected_snapshot(model)["uuid"], str(variant.uuid))
        result = json.loads(model.received_messages[-1][-1].content)
        self.assertEqual(result["replacement"]["old_catalog_item"]["id"], 3)
        self.assertEqual(result["replacement"]["new_catalog_item"]["id"], 9)
        self.assertEqual([item.uuid for item in service.get_sequence_variants(str(source.uuid))], [source.uuid, variant.uuid])
        # Dependencies and returned projections cannot change committed history.
        catalog.search.return_value[0].name = "Changed externally"
        variant.elements[0].catalog_item.name = "Changed projection"
        self.assertEqual(service.get_sequence(service.active_sequence_uuid).elements[2].catalog_item.name, "Pose 9")
        self.assertEqual(service.get_sequence(str(source.uuid)), source)
        generator.generate.assert_called_once()

    def test_editor_preserves_metadata_occurrences_and_resets_mirroring_both_ways(self):
        for bilateral in (True, False):
            with self.subTest(bilateral=bilateral):
                source = GeneratedSequence.from_catalog_items([catalog_pose(1), catalog_pose(2)], request="Original")
                source.name = "Custom name"
                source.description = "Custom description"
                source.elements[0].start_mirror = not bilateral
                source.elements[0].end_mirror = not bilateral
                before = source.model_dump(mode="json")
                candidate = catalog_pose(9, bilateral=bilateral)
                variant = replace_element(
                    source, element_id=source.elements[0].element_id, replacement_catalog_item_id=9,
                    candidates=[candidate], request="Replace first",
                )
                self.assertEqual(variant.name, source.name)
                self.assertEqual(variant.description, source.description)
                self.assertEqual(variant.elements[1], source.elements[1])
                self.assertEqual(variant.elements[0].start_mirror, bilateral)
                self.assertEqual(variant.elements[0].end_mirror, bilateral)
                self.assertEqual(source.model_dump(mode="json"), before)

    def test_replacement_uses_explicitly_selected_older_family(self):
        service, source, generator, catalog, model = self.generated()
        list(service.generate("Other practice"))
        newer = service.get_sequence(service.active_sequence_uuid)
        service.select_sequence(str(source.uuid))
        model.responses = [
            tool_call("search_catalog_items", name="Tree"), replacement_call(source), AIMessage(content="Done"),
        ]
        list(service.stream("Replace third"))
        variant = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual(variant.parent_uuid, source.uuid)
        self.assertEqual(variant.family_uuid, source.family_uuid)
        self.assertNotEqual(variant.family_uuid, newer.family_uuid)
        self.assertEqual(service.get_sequence(str(newer.uuid)), newer)
        self.assertEqual(len(service.state["sequences"]), 3)
        self.assertEqual(selected_snapshot(model, -3)["uuid"], str(source.uuid))
        self.assertEqual(len(service.get_sequence_variants(str(source.uuid))), 2)
        self.assertEqual(generator.generate.call_count, 2)

    def test_invalid_replacements_keep_selected_snapshot_and_history(self):
        cases = [
            ("stale_sequence", {}, {"sequence_uuid": str(uuid4())}),
            ("sequence_element_not_found", {}, {"element_id": str(uuid4())}),
            ("catalog_item_not_verified", {}, {"replacement_catalog_item_id": 42}),
            ("catalog_item_not_usable", {"image_url": None}, {}),
            ("catalog_item_not_usable", {"uuid": None}, {}),
            ("catalog_item_not_usable", {"uuid": "  "}, {}),
            ("duplicate_catalog_item", {"id": 1}, {"replacement_catalog_item_id": 1}),
            ("duplicate_catalog_item", {"uuid": "other_occurrence"}, {}),
        ]
        for code, changes, arguments in cases:
            with self.subTest(code=code, changes=changes):
                service, source, generator, catalog, model = self.generated()
                if changes.get("uuid") == "other_occurrence":
                    changes = {"uuid": source.elements[0].catalog_item.uuid.upper()}
                catalog.search.return_value[0] = catalog.search.return_value[0].model_copy(update=changes)
                before = service.state["sequences"]
                model.responses = [
                    tool_call("search_catalog_items", name="Tree"), replacement_call(source, **arguments),
                    AIMessage(content="Cannot replace"),
                ]
                list(service.stream("Replace third"))
                result = model.received_messages[-1][-1]
                self.assertEqual(result.status, "error")
                self.assertEqual(json.loads(result.content)["code"], code)
                self.assertEqual(service.state["sequences"], before)
                self.assertEqual(service.active_sequence_uuid, str(source.uuid))
                generator.generate.assert_called_once()

    def test_replacement_with_same_id_or_uuid_returns_no_change(self):
        for match in ("id", "uuid"):
            with self.subTest(match=match):
                service, source, generator, catalog, model = self.generated()
                candidate = catalog_pose(9)
                if match == "id":
                    candidate.id = source.elements[2].catalog_item.id
                else:
                    candidate.uuid = source.elements[2].catalog_item.uuid.upper()
                catalog.search.return_value = [candidate]
                before = service.state["sequences"]
                model.responses = [
                    tool_call("search_catalog_items", name="Same"), replacement_call(source, candidate.id),
                    AIMessage(content="Already selected"),
                ]
                list(service.stream("Replace third"))
                self.assertEqual(json.loads(model.received_messages[-1][-1].content)["status"], "no_change")
                self.assertEqual(service.state["sequences"], before)
                self.assertEqual(service.active_sequence_uuid, str(source.uuid))
                self.assertFalse(service.state["turn_operation_committed"])

    def test_candidates_persist_for_user_selection_on_a_later_turn(self):
        service, source, generator, catalog, model = self.generated()
        before = service.state["sequences"]
        model.responses = [tool_call("search_catalog_items", name="Tree"), AIMessage(content="Use Pose 9?")]
        list(service.stream("Find alternatives"))
        self.assertEqual(service.state["sequences"], before)
        self.assertEqual(service.active_sequence_uuid, str(source.uuid))
        model.responses = [replacement_call(source), AIMessage(content="Replaced")]
        list(service.stream("Yes, replace the third with Pose 9"))
        self.assertEqual(service.get_sequence(service.active_sequence_uuid).elements[2].catalog_item.id, 9)
        catalog.search.assert_called_once()
        generator.generate.assert_called_once()

    def test_new_search_replaces_previous_candidates_including_empty_results(self):
        for results in ([catalog_pose(10)], []):
            with self.subTest(results=results):
                service, source, generator, catalog, model = self.generated()
                catalog.search.side_effect = [[catalog_pose(9)], results]
                model.responses = [
                    tool_call("search_catalog_items", name="Tree"),
                    tool_call("search_catalog_items", name="Mountain"), replacement_call(source),
                    AIMessage(content="Search again"),
                ]
                list(service.stream("Replace third"))
                self.assertEqual(json.loads(model.received_messages[-1][-1].content)["code"], "catalog_item_not_verified")
                self.assertEqual(len(service.state["sequences"]), 1)
                self.assertEqual(len(service.state["latest_catalog_items"]), len(results))

    def test_failed_search_clears_candidates_even_when_arguments_fail_validation(self):
        for failure in ("api", "value", "validation"):
            with self.subTest(failure=failure):
                service, source, generator, catalog, model = self.generated()
                model.responses = [tool_call("search_catalog_items", name="Tree"), AIMessage(content="Found")]
                list(service.stream("Find a tree pose"))
                self.assertTrue(service.state["latest_catalog_items"])
                if failure == "api":
                    catalog.search.side_effect = CatalogItemApiError("Unavailable", code="api_unavailable")
                elif failure == "value":
                    catalog.search.side_effect = ValueError("Invalid filter")
                arguments = {"limit": 0} if failure == "validation" else {"name": "Other"}
                model.responses = [
                    tool_call("search_catalog_items", **arguments), replacement_call(source), AIMessage(content="Search again"),
                ]
                list(service.stream("Find another and replace"))
                self.assertEqual(service.state["latest_catalog_items"], [])
                self.assertEqual(json.loads(model.received_messages[-1][-1].content)["code"], "catalog_item_not_verified")
                self.assertEqual(service.active_sequence_uuid, str(source.uuid))
                self.assertEqual(len(service.state["sequences"]), 1)

    def test_only_one_commit_per_turn_including_generation_after_replacement(self):
        for second_tool in ("replace", "generate"):
            with self.subTest(second_tool=second_tool):
                service, source, generator, catalog, model = self.generated()
                repeated = replacement_call(source) if second_tool == "replace" else tool_call(
                    "generate_sequence", request="Unexpected generation", count=3,
                )
                model.responses = [
                    tool_call("search_catalog_items", name="Tree"), replacement_call(source), repeated,
                    AIMessage(content="Done"),
                ]
                list(service.stream("Replace third"))
                self.assertEqual(len(service.state["sequences"]), 2)
                self.assertEqual(json.loads(model.received_messages[-1][-1].content)["code"], "operation_already_started")
                generator.generate.assert_called_once()

    def test_second_occurrence_is_blocked_until_next_user_request(self):
        service, source, generator, catalog, model = self.generated()
        catalog.search.return_value.append(catalog_pose(10))
        first_uuid = uuid4()
        model.responses = [
            tool_call("search_catalog_items", name="Standing"), replacement_call(source),
            replacement_call(source, 10, element_index=1, sequence_uuid=str(first_uuid)),
            AIMessage(content="One element replaced"),
        ]
        with patch("langchain_app.services.sequence_editor.uuid4", return_value=first_uuid):
            list(service.stream("Replace the third pose"))
        first = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual(first.uuid, first_uuid)
        self.assertEqual([item.catalog_item.id for item in first.elements], [1, 2, 9])
        self.assertEqual(len(service.state["sequences"]), 2)
        self.assertEqual(json.loads(model.received_messages[-1][-1].content)["code"], "operation_already_started")

        model.responses = [replacement_call(first, 10, element_index=1), AIMessage(content="Second element replaced")]
        list(service.stream("Now replace the second pose with Pose 10"))
        second = service.get_sequence(service.active_sequence_uuid)
        self.assertNotEqual(second.uuid, first.uuid)
        self.assertEqual(second.parent_uuid, first.uuid)
        self.assertEqual(second.family_uuid, source.family_uuid)
        self.assertEqual([item.catalog_item.id for item in second.elements], [1, 10, 9])
        self.assertEqual(second.elements[0], first.elements[0])
        self.assertEqual(second.elements[2], first.elements[2])
        self.assertEqual(service.get_sequence(str(first.uuid)), first)
        self.assertEqual(service.get_sequence(str(source.uuid)), source)
        self.assertEqual(len(service.state["sequences"]), 3)
        catalog.search.assert_called_once()
        generator.generate.assert_called_once()

    def test_generation_command_cannot_be_satisfied_by_replacement(self):
        service, source, generator, catalog, model = self.generated()
        before = service.state["sequences"]
        model.responses = [replacement_call(source), AIMessage(content="Could not generate")]
        with self.assertRaises(AssistantAgentError):
            list(service.generate("Generate again"))
        self.assertEqual(json.loads(model.received_messages[-1][-1].content)["code"], "generation_required")
        self.assertEqual(service.state["sequences"], before)
        self.assertEqual(service.active_sequence_uuid, str(source.uuid))

    def test_parallel_tool_batches_are_rejected_without_creating_variants(self):
        for tools in ("replace", "search"):
            with self.subTest(tools=tools):
                service, source, generator, catalog, model = self.generated()
                model.responses = [tool_call("search_catalog_items", name="Tree"), AIMessage(content="Found")]
                list(service.stream("Find a tree pose"))
                before = service.state["sequences"]
                calls = [replacement_call(source), replacement_call(source, element_index=1)] if tools == "replace" else [
                    tool_call("search_catalog_items", name="Tree"), tool_call("search_catalog_items", name="Mountain"),
                ]
                batch = AIMessage(content="", tool_calls=[call.tool_calls[0] for call in calls])
                model.responses = [batch, AIMessage(content="Will call sequentially")]
                list(service.stream("Replace third"))
                results = model.received_messages[-1][-2:]
                self.assertEqual([json.loads(message.content)["code"] for message in results], [
                    "sequential_tool_calls_required", "sequential_tool_calls_required",
                ])
                self.assertEqual(service.state["sequences"], before)
                self.assertEqual(service.active_sequence_uuid, str(source.uuid))
                catalog.search.assert_called_once()
                generator.generate.assert_called_once()

    def test_invalid_tool_arguments_do_not_modify_sequence(self):
        for arguments in ({"element_id": "bad-uuid"}, {"replacement_catalog_item_id": 0}):
            with self.subTest(arguments=arguments):
                service, source, generator, catalog, model = self.generated()
                before = service.state["sequences"]
                model.responses = [replacement_call(source, **arguments), AIMessage(content="Invalid request")]
                list(service.stream("Replace third"))
                self.assertEqual(model.received_messages[-1][-1].status, "error")
                self.assertEqual(service.state["sequences"], before)
                self.assertEqual(service.active_sequence_uuid, str(source.uuid))

    def test_committed_replacement_survives_failure_of_followup_model_reply(self):
        service, source, generator, catalog, model = self.generated()
        model.responses = [
            tool_call("search_catalog_items", name="Tree"), replacement_call(source), AIMessage(content="Done"),
        ]
        original_generate = model.__class__._generate

        def fail_after_replace(model_instance, messages, **kwargs):
            if isinstance(messages[-1], ToolMessage) and messages[-1].name == "replace_sequence_element":
                raise APIConnectionError(request=httpx.Request("POST", "https://example.com"))
            return original_generate(model_instance, messages, **kwargs)

        with patch.object(ToolCallingModel, "_generate", new=fail_after_replace):
            with self.assertRaises(AssistantAgentError):
                list(service.stream("Replace third"))
        variant = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual(variant.parent_uuid, source.uuid)
        self.assertEqual(variant.elements[2].catalog_item.id, 9)
        self.assertEqual(service.get_sequence(str(source.uuid)), source)
        list(service.stream("What is selected?"))
        self.assertEqual(selected_snapshot(model)["uuid"], str(variant.uuid))


if __name__ == "__main__":
    unittest.main()
