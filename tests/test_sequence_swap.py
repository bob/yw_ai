"""Verify occurrence swaps, detached variants, and snapshot compatibility."""

import json
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch
from uuid import uuid4

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from langchain_app.schemas import CatalogItemSchema, GeneratedSequence
from langchain_app.services.assistant_agent import AssistantAgentError
from langchain_app.services.sequence_editor import SequenceEditorError, swap_elements
from langchain_app.tools import ASSISTANT_TOOLS, swap_sequence_elements
from langchain_app.tools.context import AssistantToolContext
from test_assistant_agent import ToolCallingModel, assistant, selected_snapshot, tool_call
from test_sequence_replacement import catalog_pose, replacement_call


def sequence(count=5):
    source = GeneratedSequence.from_catalog_items([
        CatalogItemSchema(
            id=number, uuid=str(uuid4()), name=f"Pose {number}",
            bilateral=number % 2 == 0,
            image_url=f"https://example.com/{number}.png",
            catalog_tag_names=["Standing", f"Tag {number}"],
        )
        for number in range(1, count + 1)
    ], request="Original generation request")
    source.name = "Custom practice"
    source.description = "Custom description"
    return source


def swap_call(source, first=2, second=4, **overrides):
    return tool_call("swap_sequence_elements", **{
        "sequence_uuid": str(source.uuid),
        "first_element_id": str(source.elements[first].element_id),
        "second_element_id": str(source.elements[second].element_id),
        **overrides,
    })


class SequenceSwapTests(unittest.TestCase):
    def test_nonadjacent_swap_leaves_intermediate_positions_unchanged(self):
        source = sequence()
        before = source.model_dump(mode="json")
        variant = swap_elements(
            source, first_element_id=source.elements[2].element_id,
            second_element_id=source.elements[4].element_id,
            request="Поменяй третью и пятую асаны местами",
        )
        self.assertEqual([item.catalog_item.id for item in variant.elements], [1, 2, 5, 4, 3])
        self.assertEqual([item.order for item in variant.elements], [1, 2, 3, 4, 5])
        for index in (0, 1, 3):
            self.assertEqual(variant.elements[index], source.elements[index])
        self.assertEqual(source.model_dump(mode="json"), before)

    def test_adjacent_extreme_and_two_element_swaps_in_both_argument_orders(self):
        cases = [
            (5, 1, 2, [1, 3, 2, 4, 5]),
            (5, 0, 4, [5, 2, 3, 4, 1]),
            (2, 0, 1, [2, 1]),
        ]
        for count, first, second, expected in cases:
            source = sequence(count)
            results = []
            for first_index, second_index in ((first, second), (second, first)):
                with self.subTest(count=count, first=first_index, second=second_index):
                    variant = swap_elements(
                        source, first_element_id=source.elements[first_index].element_id,
                        second_element_id=source.elements[second_index].element_id,
                        request="Swap the pair",
                    )
                    self.assertEqual([item.catalog_item.id for item in variant.elements], expected)
                    self.assertEqual([item.order for item in variant.elements], list(range(1, count + 1)))
                    results.append(variant.elements)
            self.assertEqual(results[0], results[1])

    def test_ids_full_catalog_items_and_mirror_flags_follow_each_occurrence(self):
        source = sequence(4)
        flags = [(True, False), (False, True), (True, True), (False, False)]
        for element, (start, end) in zip(source.elements, flags):
            element.start_mirror, element.end_mirror = start, end
        before = source.model_dump(mode="json")
        variant = swap_elements(
            source, first_element_id=source.elements[0].element_id,
            second_element_id=source.elements[1].element_id, request="Swap first and second",
        )
        self.assertEqual(
            [element.element_id for element in variant.elements],
            [source.elements[index].element_id for index in (1, 0, 2, 3)],
        )
        for actual, original in zip(variant.elements, [source.elements[index] for index in (1, 0, 2, 3)]):
            self.assertEqual(actual.catalog_item, original.catalog_item)
            self.assertEqual(actual.start_mirror, original.start_mirror)
            self.assertEqual(actual.end_mirror, original.end_mirror)
        self.assertEqual(source.model_dump(mode="json"), before)

    def test_variant_preserves_metadata_and_uses_selected_variant_as_parent(self):
        original = sequence()
        source = GeneratedSequence.from_catalog_items(
            [element.catalog_item for element in original.elements],
            request="Previous regeneration request", parent=original,
        )
        source.name = "Selected variant"
        source.description = "Selected description"
        before = source.model_dump(mode="json")
        request = "Поменяй первую и последнюю позы местами"
        variant = swap_elements(
            source, first_element_id=source.elements[0].element_id,
            second_element_id=source.elements[-1].element_id, request=request,
        )
        self.assertNotEqual(variant.uuid, source.uuid)
        self.assertNotEqual(variant.uuid, original.uuid)
        self.assertEqual(variant.parent_uuid, source.uuid)
        self.assertEqual(variant.family_uuid, source.family_uuid)
        self.assertEqual(variant.operation, "swap")
        self.assertEqual(variant.operation_request, request)
        for field in ("generation_request", "name", "description", "language", "elements_count"):
            self.assertEqual(getattr(variant, field), getattr(source, field))
        self.assertEqual(source.model_dump(mode="json"), before)

    def test_variant_and_parent_share_no_mutable_sequence_or_catalog_objects(self):
        source = sequence()
        before = source.model_dump(mode="json")
        variant = swap_elements(
            source, first_element_id=source.elements[2].element_id,
            second_element_id=source.elements[4].element_id, request="Swap third and fifth",
        )
        self.assertIsNot(variant.elements, source.elements)
        originals = {element.element_id: element for element in source.elements}
        for element in variant.elements:
            original = originals[element.element_id]
            self.assertIsNot(element, original)
            self.assertIsNot(element.catalog_item, original.catalog_item)
            self.assertIsNot(element.catalog_item.catalog_tag_names, original.catalog_item.catalog_tag_names)
            element.catalog_item.name = "Changed variant"
            element.catalog_item.catalog_tag_names.append("Changed tag")
            element.start_mirror = not element.start_mirror
        variant.elements.pop()
        self.assertEqual(source.model_dump(mode="json"), before)
        variant_before = variant.model_dump(mode="json")
        source.elements[0].catalog_item.catalog_tag_names.clear()
        source.elements.clear()
        self.assertEqual(variant.model_dump(mode="json"), variant_before)

    def test_distinct_occurrences_of_same_catalog_item_create_a_variant(self):
        source = sequence(2)
        source.elements[1].catalog_item = source.elements[0].catalog_item.model_copy(deep=True)
        source.elements[0].start_mirror, source.elements[0].end_mirror = True, False
        source.elements[1].start_mirror, source.elements[1].end_mirror = False, True
        before = source.model_dump(mode="json")
        variant = swap_elements(
            source, first_element_id=source.elements[0].element_id,
            second_element_id=source.elements[1].element_id, request="Swap the two occurrences",
        )
        self.assertNotEqual(variant.uuid, source.uuid)
        self.assertEqual([element.element_id for element in variant.elements], [
            source.elements[1].element_id, source.elements[0].element_id,
        ])
        self.assertEqual(variant.elements[0].catalog_item, variant.elements[1].catalog_item)
        self.assertEqual([(element.start_mirror, element.end_mirror) for element in variant.elements], [
            (False, True), (True, False),
        ])
        self.assertEqual(source.model_dump(mode="json"), before)

    def test_same_existing_occurrence_returns_none_without_creating_uuid(self):
        for count in (1, 5):
            with self.subTest(count=count):
                source = sequence(count)
                before = source.model_dump(mode="json")
                element_id = source.elements[0].element_id
                with patch("langchain_app.services.sequence_editor.uuid4") as new_uuid:
                    self.assertIsNone(swap_elements(
                        source, first_element_id=element_id, second_element_id=element_id,
                        request="Swap the same occurrence",
                    ))
                new_uuid.assert_not_called()
                self.assertEqual(source.model_dump(mode="json"), before)

    def test_unknown_occurrences_fail_before_creating_uuid_or_changing_source(self):
        source = sequence()
        known = source.elements[0].element_id
        unknown, other_unknown = uuid4(), uuid4()
        before = source.model_dump(mode="json")
        for first, second in ((unknown, known), (known, unknown), (unknown, other_unknown), (unknown, unknown)):
            with self.subTest(first=first, second=second):
                with patch("langchain_app.services.sequence_editor.uuid4") as new_uuid:
                    with self.assertRaises(SequenceEditorError) as error:
                        swap_elements(
                            source, first_element_id=first, second_element_id=second,
                            request="Swap an unknown occurrence",
                        )
                self.assertEqual(error.exception.code, "sequence_element_not_found")
                new_uuid.assert_not_called()
                self.assertEqual(source.model_dump(mode="json"), before)

    def test_orders_are_normalized_to_the_actual_list_positions(self):
        source = sequence(3)
        for element, order in zip(source.elements, (30, 10, 20)):
            element.order = order
        variant = swap_elements(
            source, first_element_id=source.elements[0].element_id,
            second_element_id=source.elements[2].element_id, request="Swap first and last",
        )
        self.assertEqual([element.catalog_item.id for element in variant.elements], [3, 2, 1])
        self.assertEqual([element.order for element in variant.elements], [1, 2, 3])
        self.assertEqual([element.order for element in source.elements], [30, 10, 20])

    def test_swap_snapshot_round_trip_preserves_order_ids_and_flags(self):
        source = sequence()
        source.elements[2].start_mirror, source.elements[2].end_mirror = True, False
        variant = swap_elements(
            source, first_element_id=source.elements[2].element_id,
            second_element_id=source.elements[4].element_id, request="Swap third and fifth",
        )
        snapshot = json.loads(variant.model_dump_json())
        restored = GeneratedSequence.from_snapshot(snapshot)
        self.assertEqual(restored.operation, "swap")
        self.assertEqual(restored, variant)
        self.assertEqual(restored.model_dump(mode="json"), snapshot)

    def test_previous_operation_snapshots_remain_readable(self):
        snapshot = sequence().model_dump(mode="json")
        for operation in ("generate", "regenerate", "replace"):
            with self.subTest(operation=operation):
                previous = {**snapshot, "operation": operation}
                restored = GeneratedSequence.from_snapshot(previous)
                self.assertEqual(restored.operation, operation)
                self.assertEqual(restored.model_dump(mode="json"), previous)


class SequenceSwapToolTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        env.start()
        self.addCleanup(env.stop)

    def generated(self):
        generator = Mock()
        generator.generate.return_value = [element.catalog_item for element in sequence().elements]
        factory, catalog, tags = Mock(return_value=generator), Mock(), Mock()
        model = ToolCallingModel(responses=[
            tool_call("generate_sequence", request="Original practice", count=5),
            AIMessage(content="Ready"),
        ])
        service = assistant(factory, model=model, catalog_item_service=catalog, catalog_tag_service=tags)
        list(service.generate("Original practice"))
        source = service.get_sequence(service.active_sequence_uuid)
        return service, source, model, factory, catalog, tags

    def runtime(self, source):
        state = {
            "sequences": {str(source.uuid): source.model_dump(mode="json")},
            "active_sequence_uuid": str(source.uuid), "turn_operation_committed": False,
        }
        context = AssistantToolContext(Mock(), Mock(), Mock(), user_request="Поменяй третью и пятую местами")
        return ToolRuntime(state, context, {}, Mock(), "swap-call", None)

    def test_registered_tool_exposes_only_three_uuid_arguments(self):
        self.assertIn(swap_sequence_elements, ASSISTANT_TOOLS)
        schema = swap_sequence_elements.tool_call_schema.model_json_schema()
        fields = {"sequence_uuid", "first_element_id", "second_element_id"}
        self.assertEqual(set(schema["properties"]), fields)
        self.assertEqual(set(schema["required"]), fields)
        for field in fields:
            self.assertEqual(schema["properties"][field]["type"], "string")
            self.assertEqual(schema["properties"][field]["format"], "uuid")

    def test_command_contains_full_snapshot_and_swap_positions_in_argument_order(self):
        source = sequence()
        source.elements[2].start_mirror, source.elements[2].end_mirror = True, False
        source.elements[4].start_mirror, source.elements[4].end_mirror = False, True
        for indices in ((2, 4), (4, 2)):
            for candidates in ([], ["Unrelated, unvalidated catalog data"]):
                with self.subTest(indices=indices, candidates=candidates):
                    runtime = self.runtime(source)
                    runtime.state["latest_catalog_items"] = candidates
                    before = deepcopy(runtime.state)
                    first, second = (source.elements[index] for index in indices)
                    result = swap_sequence_elements.func(source.uuid, first.element_id, second.element_id, runtime)
                    self.assertIsInstance(result, Command)
                    update = result.update
                    self.assertEqual(set(update), {
                        "sequences", "active_sequence_uuid", "turn_operation_committed", "messages",
                    })
                    variant = GeneratedSequence.from_snapshot(update["sequences"][update["active_sequence_uuid"]])
                    self.assertEqual(len(update["sequences"]), 1)
                    self.assertTrue(update["turn_operation_committed"])
                    self.assertEqual(variant.operation_request, runtime.context.user_request)
                    self.assertEqual(variant.elements[2].element_id, source.elements[4].element_id)
                    self.assertEqual(variant.elements[4].element_id, source.elements[2].element_id)
                    self.assertEqual((variant.elements[2].start_mirror, variant.elements[2].end_mirror), (False, True))
                    self.assertEqual((variant.elements[4].start_mirror, variant.elements[4].end_mirror), (True, False))
                    message, = update["messages"]
                    self.assertEqual(message.tool_call_id, "swap-call")
                    self.assertEqual(json.loads(message.content), {
                        "status": "ok", "uuid": str(variant.uuid),
                        "operation": "swap", "elements_count": variant.elements_count,
                        "swap": [
                            {"element_id": str(first.element_id), "name": first.catalog_item.name,
                             "old_order": first.order, "new_order": second.order},
                            {"element_id": str(second.element_id), "name": second.catalog_item.name,
                             "old_order": second.order, "new_order": first.order},
                        ],
                    })
                    self.assertEqual(runtime.state, before)
                    self.assertFalse(runtime.context.generation_started)
                    runtime.context.generator_factory.assert_not_called()
                    self.assertEqual(runtime.context.catalog_item_service.mock_calls, [])
                    self.assertEqual(runtime.context.catalog_tag_service.mock_calls, [])
                    runtime.stream_writer.assert_not_called()

    def test_guards_precede_snapshot_and_occurrence_validation(self):
        source = sequence()
        for require_generation, generation_started, committed, code in (
            (True, False, False, "generation_required"),
            (True, True, True, "generation_required"),
            (False, True, False, "operation_already_started"),
            (False, False, True, "operation_already_started"),
        ):
            with self.subTest(require_generation=require_generation, generation_started=generation_started, committed=committed):
                runtime = self.runtime(source)
                runtime.context.require_generation = require_generation
                runtime.context.generation_started = generation_started
                runtime.state = {"turn_operation_committed": committed}
                before = deepcopy(runtime.state)
                result = swap_sequence_elements.func(uuid4(), uuid4(), uuid4(), runtime)
                self.assertIsInstance(result, ToolMessage)
                self.assertEqual(result.status, "error")
                self.assertEqual(result.tool_call_id, "swap-call")
                self.assertEqual(json.loads(result.content)["code"], code)
                self.assertEqual(runtime.state, before)

    def test_missing_active_snapshot_does_not_reconstruct_history(self):
        source = sequence()
        for active_uuid in (None, str(uuid4()), str(source.uuid)):
            with self.subTest(active_uuid=active_uuid):
                runtime = self.runtime(source)
                runtime.state["active_sequence_uuid"] = active_uuid
                if active_uuid == str(source.uuid):
                    runtime.state["sequences"] = {}
                before = deepcopy(runtime.state)
                result = swap_sequence_elements.func(
                    source.uuid, source.elements[2].element_id, source.elements[4].element_id, runtime,
                )
                self.assertEqual(json.loads(result.content)["code"], "sequence_not_found")
                self.assertEqual(result.status, "error")
                self.assertEqual(result.tool_call_id, "swap-call")
                self.assertEqual(runtime.state, before)

    def test_real_tool_cycle_commits_once_and_refreshes_selected_snapshot(self):
        service, source, model, factory, catalog, tags = self.generated()
        before = service.state["sequences"]
        request = "Поменяй третью и пятую асаны местами"
        model.responses = [swap_call(source), AIMessage(content="Pose 3 и Pose 5 поменяны местами.")]
        self.assertEqual(list(service.stream(request)), ["Pose 3 и Pose 5 поменяны местами."])
        variant = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual([element.catalog_item.id for element in variant.elements], [1, 2, 5, 4, 3])
        self.assertEqual(variant.operation, "swap")
        self.assertEqual(variant.parent_uuid, source.uuid)
        self.assertEqual(variant.family_uuid, source.family_uuid)
        self.assertEqual(variant.generation_request, source.generation_request)
        self.assertEqual(variant.operation_request, request)
        self.assertTrue(service.state["turn_operation_committed"])
        self.assertEqual(len(service.state["sequences"]), 2)
        self.assertEqual(service.state["sequences"][str(source.uuid)], before[str(source.uuid)])
        self.assertEqual(selected_snapshot(model), variant.model_context())
        message = model.received_messages[-1][-2]
        self.assertEqual(message.tool_call_id, model.received_messages[-1][-3].tool_calls[0]["id"])
        payload = json.loads(message.content)
        self.assertEqual({k: v for k, v in payload.items() if k not in ("status", "swap")}, {
            "uuid": str(variant.uuid), "operation": "swap", "elements_count": variant.elements_count,
        })
        self.assertEqual([entry["new_order"] for entry in payload["swap"]], [5, 3])
        factory.assert_called_once()
        self.assertEqual(catalog.mock_calls, [])
        self.assertEqual(tags.mock_calls, [])

    def test_swap_uses_explicitly_selected_older_family_and_rejects_stored_stale_variant(self):
        service, source, model, factory, catalog, tags = self.generated()
        list(service.generate("Other practice"))
        newer = service.get_sequence(service.active_sequence_uuid)
        service.select_sequence(str(source.uuid))
        model.responses = [swap_call(newer), AIMessage(content="Use the selected variant")]
        list(service.stream("Swap third and fifth"))
        self.assertEqual(json.loads(model.received_messages[-1][-2].content)["code"], "stale_sequence")
        self.assertEqual(service.active_sequence_uuid, str(source.uuid))
        self.assertEqual(len(service.state["sequences"]), 2)
        model.responses = [swap_call(source), AIMessage(content="Done")]
        list(service.stream("Swap third and fifth in the selected sequence"))
        variant = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual(variant.parent_uuid, source.uuid)
        self.assertEqual(variant.family_uuid, source.family_uuid)
        self.assertEqual(service.get_sequence(str(newer.uuid)), newer)
        self.assertEqual(service.get_sequence(str(source.uuid)), source)
        self.assertEqual([item.uuid for item in service.get_sequence_variants(str(source.uuid))], [source.uuid, variant.uuid])

    def test_invalid_uuids_and_unknown_occurrences_keep_state_and_return_matching_tool_message(self):
        unknown = str(uuid4())
        for arguments, code in (
            ({"sequence_uuid": str(uuid4())}, "stale_sequence"),
            ({"first_element_id": unknown}, "sequence_element_not_found"),
            ({"second_element_id": unknown}, "sequence_element_not_found"),
            ({"first_element_id": unknown, "second_element_id": unknown}, "sequence_element_not_found"),
            ({"sequence_uuid": "invalid-uuid"}, None),
            ({"first_element_id": "invalid-uuid"}, None),
            ({"second_element_id": "invalid-uuid"}, None),
        ):
            with self.subTest(arguments=arguments):
                service, source, model, factory, catalog, tags = self.generated()
                before = service.state["sequences"]
                model.responses = [swap_call(source, **arguments), AIMessage(content="Cannot swap")]
                with patch("langchain_app.services.sequence_editor.uuid4") as new_uuid:
                    list(service.stream("Swap third and fifth"))
                new_uuid.assert_not_called()
                result = model.received_messages[-1][-2]
                self.assertEqual(result.status, "error")
                self.assertEqual(result.tool_call_id, model.received_messages[-1][-3].tool_calls[0]["id"])
                if code is not None:
                    self.assertEqual(json.loads(result.content)["code"], code)
                self.assertEqual(service.state["sequences"], before)
                self.assertEqual(service.active_sequence_uuid, str(source.uuid))
                self.assertFalse(service.state["turn_operation_committed"])

    def test_same_occurrence_returns_only_no_change_without_consuming_commit(self):
        service, source, model, factory, catalog, tags = self.generated()
        before = service.state["sequences"]
        model.responses = [swap_call(source, second=2), AIMessage(content="The same occurrence")]
        with patch("langchain_app.services.sequence_editor.uuid4") as new_uuid:
            list(service.stream("Swap the third occurrence with itself"))
        new_uuid.assert_not_called()
        result = model.received_messages[-1][-2]
        self.assertEqual(json.loads(result.content), {"status": "no_change", "uuid": str(source.uuid)})
        self.assertEqual(result.tool_call_id, model.received_messages[-1][-3].tool_calls[0]["id"])
        self.assertEqual(service.state["sequences"], before)
        self.assertEqual(service.active_sequence_uuid, str(source.uuid))
        self.assertFalse(service.state["turn_operation_committed"])

    def test_error_or_no_change_allows_corrected_swap_in_same_request(self):
        for failure in ("unknown", "same", "invalid"):
            with self.subTest(failure=failure):
                service, source, model, factory, catalog, tags = self.generated()
                arguments = {
                    "first_element_id": str(uuid4()) if failure == "unknown" else "invalid-uuid",
                } if failure != "same" else {"second_element_id": str(source.elements[2].element_id)}
                model.responses = [swap_call(source, **arguments), swap_call(source), AIMessage(content="Done")]
                list(service.stream("Swap third and fifth"))
                self.assertEqual(service.get_sequence(service.active_sequence_uuid).operation, "swap")
                self.assertEqual(len(service.state["sequences"]), 2)
                self.assertTrue(service.state["turn_operation_committed"])

    def test_swap_blocks_other_operations_with_current_uuid_and_resets_next_request(self):
        for second_tool in ("swap", "replace", "generate"):
            with self.subTest(second_tool=second_tool):
                service, source, model, factory, catalog, tags = self.generated()
                variant_uuid = uuid4()
                repeated = {
                    "swap": swap_call(source, sequence_uuid=str(variant_uuid)),
                    "replace": replacement_call(source, sequence_uuid=str(variant_uuid)),
                    "generate": tool_call("generate_sequence", request="Another practice", count=5),
                }[second_tool]
                model.responses = [swap_call(source), repeated, AIMessage(content="One operation completed")]
                with patch("langchain_app.services.sequence_editor.uuid4", return_value=variant_uuid):
                    list(service.stream("Swap third and fifth"))
                self.assertEqual(service.active_sequence_uuid, str(variant_uuid))
                self.assertEqual(len(service.state["sequences"]), 2)
                self.assertEqual(json.loads(model.received_messages[-1][-2].content)["code"], "operation_already_started")
                variant = service.get_sequence(service.active_sequence_uuid)
                model.responses = [swap_call(variant), AIMessage(content="Swapped again")]
                list(service.stream("Now swap third and fifth again"))
                self.assertEqual(len(service.state["sequences"]), 3)
                self.assertEqual(service.get_sequence(service.active_sequence_uuid).parent_uuid, variant_uuid)
                factory.assert_called_once()

    def test_swap_after_replacement_or_generation_is_blocked(self):
        for operation in ("replace", "generate"):
            with self.subTest(operation=operation):
                service, source, model, factory, catalog, tags = self.generated()
                variant_uuid = uuid4()
                if operation == "replace":
                    catalog.search.return_value = [catalog_pose(9)]
                    preceding = [tool_call("search_catalog_items", name="Tree"), replacement_call(source)]
                else:
                    preceding = [tool_call("generate_sequence", request="Other practice", count=5)]
                model.responses = [
                    *preceding, swap_call(source, sequence_uuid=str(variant_uuid)), AIMessage(content="One operation completed"),
                ]
                if operation == "replace":
                    with patch("langchain_app.services.sequence_editor.uuid4", return_value=variant_uuid):
                        list(service.stream("Change the sequence"))
                else:
                    list(service.stream("Change the sequence"))
                self.assertEqual(json.loads(model.received_messages[-1][-2].content)["code"], "operation_already_started")
                self.assertEqual(len(service.state["sequences"]), 2)
                committed = json.loads(model.received_messages[-1][-4].content)
                self.assertEqual(service.active_sequence_uuid, committed["uuid"])
                self.assertEqual(service.get_sequence(service.active_sequence_uuid).operation, operation)

    def test_generate_and_regenerate_commands_cannot_be_satisfied_by_swap(self):
        for regenerate in (False, True):
            with self.subTest(regenerate=regenerate):
                service, source, model, factory, catalog, tags = self.generated()
                before = service.state["sequences"]
                model.responses = [swap_call(source), AIMessage(content="Generation required")]
                with self.assertRaises(AssistantAgentError):
                    list(service.generate("Generate again", sequence_uuid=str(source.uuid) if regenerate else None))
                self.assertEqual(json.loads(model.received_messages[-1][-2].content)["code"], "generation_required")
                self.assertEqual(service.state["sequences"], before)
                self.assertEqual(service.active_sequence_uuid, str(source.uuid))
                factory.assert_called_once()


if __name__ == "__main__":
    unittest.main()
