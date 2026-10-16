"""Verify removal validation, atomic commits, and the shared operation limit."""

import json
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch
from uuid import uuid4

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
from pydantic import ValidationError

from langchain_app.schemas import GeneratedSequence
from langchain_app.tools import ASSISTANT_TOOLS, remove_sequence_elements
from langchain_app.tools.context import AssistantToolContext
from test_assistant_agent import ToolCallingModel, assistant, selected_snapshot, tool_call
from test_sequence_removal import sequence


def removal_call(source, **overrides):
    return tool_call("remove_sequence_elements", **{
        "sequence_uuid": str(source.uuid),
        "element_ids": [str(source.elements[1].element_id)],
        **overrides,
    })


class SequenceRemovalToolTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        env.start()
        self.addCleanup(env.stop)

    def runtime(self, source):
        context = AssistantToolContext(
            Mock(), Mock(), Mock(), user_request="  Удали вторую и четвёртую асаны\n",
        )
        return ToolRuntime({
            "sequences": {str(source.uuid): source.model_dump(mode="json")},
            "active_sequence_uuid": str(source.uuid), "turn_operation_committed": False,
            "latest_catalog_items": ["Unrelated, unvalidated catalog data"],
        }, context, {}, Mock(), "removal-call", None)

    def call(self, source, runtime, **overrides):
        return remove_sequence_elements.func(**{
            "sequence_uuid": source.uuid, "element_ids": [source.elements[0].element_id],
            "runtime": runtime, **overrides,
        })

    def assert_no_external_calls(self, runtime):
        self.assertEqual(runtime.context.generator_factory.mock_calls, [])
        self.assertEqual(runtime.context.catalog_item_service.mock_calls, [])
        self.assertEqual(runtime.context.catalog_tag_service.mock_calls, [])
        runtime.stream_writer.assert_not_called()

    def assert_rejected(self, source, runtime, code, **overrides):
        before = deepcopy(runtime.state)
        context_before = vars(runtime.context).copy()
        with patch("langchain_app.services.sequence_editor.uuid4") as new_uuid:
            result = self.call(source, runtime, **overrides)
        self.assertIsInstance(result, ToolMessage)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.tool_call_id, "removal-call")
        payload = json.loads(result.content)
        self.assertEqual(payload, {"status": "error", "code": code, "message": payload["message"]})
        self.assertTrue(payload["message"])
        self.assertEqual(runtime.state, before)
        self.assertEqual(vars(runtime.context), context_before)
        new_uuid.assert_not_called()
        self.assert_no_external_calls(runtime)

    def test_registered_tool_exposes_only_required_sequence_and_occurrence_uuids(self):
        self.assertEqual(ASSISTANT_TOOLS.count(remove_sequence_elements), 1)
        schema = remove_sequence_elements.tool_call_schema.model_json_schema()
        self.assertEqual(set(schema["properties"]), {"sequence_uuid", "element_ids"})
        self.assertEqual(set(schema["required"]), {"sequence_uuid", "element_ids"})
        self.assertEqual(schema["properties"]["sequence_uuid"]["format"], "uuid")
        self.assertEqual(schema["properties"]["element_ids"]["type"], "array")
        self.assertEqual(schema["properties"]["element_ids"]["items"]["format"], "uuid")
        self.assertEqual(schema["properties"]["element_ids"]["minItems"], 1)

    def test_standard_validation_rejects_missing_null_empty_or_invalid_arguments(self):
        source = sequence()
        valid = {"sequence_uuid": str(source.uuid), "element_ids": [str(source.elements[0].element_id)]}
        invalid = [
            {}, {"sequence_uuid": valid["sequence_uuid"]}, {"element_ids": valid["element_ids"]},
            *[{**valid, "element_ids": value} for value in (
                None, [], "third", 1, True, {}, (), tuple(valid["element_ids"]),
                [None], [1], [True], ["not-a-uuid"], [{}], [valid["element_ids"]],
                [valid["element_ids"][0], "not-a-uuid"],
            )],
            *[{**valid, "sequence_uuid": value} for value in (None, 1, True, "not-a-uuid", [])],
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                runtime = self.runtime(source)
                before = deepcopy(runtime.state)
                with patch("langchain_app.services.sequence_editor.uuid4") as new_uuid:
                    with self.assertRaises(ValidationError):
                        remove_sequence_elements.invoke({**arguments, "runtime": runtime})
                new_uuid.assert_not_called()
                self.assertEqual(runtime.state, before)
                self.assertFalse(runtime.context.generation_started)
                self.assert_no_external_calls(runtime)

    def test_generation_and_committed_guards_precede_snapshot_and_target_checks(self):
        source = sequence()
        for required, started, committed, code in (
            (True, True, True, "generation_required"),
            (True, False, False, "generation_required"),
            (False, True, False, "operation_already_started"),
            (False, False, True, "operation_already_started"),
        ):
            with self.subTest(required=required, started=started, committed=committed):
                runtime = self.runtime(source)
                runtime.context.require_generation = required
                runtime.context.generation_started = started
                runtime.state = {"turn_operation_committed": committed}
                self.assert_rejected(source, runtime, code, sequence_uuid=uuid4(), element_ids=[])

    def test_missing_active_snapshot_does_not_use_another_variant_from_history(self):
        source = sequence()
        for active in (None, str(uuid4()), str(source.uuid)):
            with self.subTest(active=active):
                runtime = self.runtime(source)
                runtime.state["active_sequence_uuid"] = active
                if active == str(source.uuid):
                    del runtime.state["sequences"][active]
                self.assert_rejected(source, runtime, "sequence_not_found", element_ids=[])

    def test_stored_stale_uuid_is_rejected_before_empty_source_or_target_checks(self):
        source, stale = sequence(), sequence()
        runtime = self.runtime(source)
        runtime.state["sequences"][str(stale.uuid)] = stale.model_dump(mode="json")
        runtime.state["sequences"][str(source.uuid)]["elements"] = []
        self.assert_rejected(source, runtime, "stale_sequence", sequence_uuid=stale.uuid, element_ids=[])

    def test_empty_source_precedes_empty_duplicate_and_unknown_targets(self):
        source = sequence()
        for targets in ([], [uuid4()], [source.elements[0].element_id] * 2):
            with self.subTest(targets=targets):
                runtime = self.runtime(source)
                runtime.state["sequences"][str(source.uuid)]["elements"] = []
                self.assert_rejected(source, runtime, "empty_sequence", element_ids=targets)

    def test_empty_and_duplicate_targets_are_invalid_even_when_uuid_is_unknown(self):
        source = sequence()
        unknown = uuid4()
        for targets in ([], [source.elements[0].element_id] * 2, [unknown, unknown]):
            with self.subTest(targets=targets):
                self.assert_rejected(source, self.runtime(source), "invalid_arguments", element_ids=targets)

    def test_any_unknown_target_rejects_whole_removal_before_empty_result_check(self):
        source = sequence()
        ids = [element.element_id for element in source.elements]
        for targets in ([uuid4()], [ids[0], uuid4()], [*ids, uuid4()]):
            with self.subTest(targets=targets):
                self.assert_rejected(source, self.runtime(source), "sequence_element_not_found", element_ids=targets)

    def test_removing_all_occurrences_or_only_remaining_one_is_rejected(self):
        for size in (1, 5):
            with self.subTest(size=size):
                source = sequence(size)
                self.assert_rejected(source, self.runtime(source), "would_empty_sequence",
                                     element_ids=[element.element_id for element in source.elements])

    def test_one_command_commits_full_variant_and_compact_removal_in_source_order(self):
        source = sequence()
        source.elements[2].start_mirror, source.elements[2].end_mirror = True, False
        source.elements[1].catalog_item.name = "Поза дерева"
        runtime = self.runtime(source)
        before = deepcopy(runtime.state)
        result = self.call(source, runtime, element_ids=[source.elements[i].element_id for i in (3, 1)])
        self.assertIsInstance(result, Command)
        update = result.update
        self.assertEqual(set(update), {"sequences", "active_sequence_uuid", "turn_operation_committed", "messages"})
        self.assertEqual(len(update["sequences"]), 1)
        variant = GeneratedSequence.from_snapshot(update["sequences"][update["active_sequence_uuid"]])
        self.assertTrue(update["turn_operation_committed"])
        self.assertNotEqual(variant.uuid, source.uuid)
        self.assertEqual(variant.parent_uuid, source.uuid)
        self.assertEqual(variant.family_uuid, source.family_uuid)
        self.assertEqual(variant.operation, "remove")
        self.assertEqual(variant.operation_request, runtime.context.user_request)
        for field in ("generation_request", "name", "description", "language"):
            self.assertEqual(getattr(variant, field), getattr(source, field))
        self.assertEqual([element.model_dump() for element in variant.elements], [
            source.elements[index].model_copy(update={"order": order}).model_dump()
            for order, index in enumerate((0, 2, 4), 1)
        ])
        message, = update["messages"]
        self.assertEqual(message.status, "success")
        self.assertEqual(message.tool_call_id, "removal-call")
        self.assertIsNone(message.artifact)
        self.assertEqual(json.loads(message.content), {
            "status": "ok", "uuid": str(variant.uuid), "operation": "remove", "elements_count": 3,
            "removal": {"removed_count": 2, "elements": [
                {"element_id": str(source.elements[i].element_id),
                 "name": source.elements[i].catalog_item.name, "old_order": i + 1}
                for i in (1, 3)
            ]},
        })
        self.assertEqual(runtime.state, before)
        self.assertFalse(runtime.context.generation_started)
        self.assert_no_external_calls(runtime)

    def test_sizes_use_actual_list_allow_one_survivor_and_ignore_generation_limit(self):
        for size, removed_count in ((2, 1), (5, 4), (50, 1)):
            with self.subTest(size=size, removed_count=removed_count):
                source = sequence(size)
                runtime = self.runtime(source)
                runtime.state["sequences"][str(source.uuid)]["elements_count"] = 999
                result = self.call(source, runtime, element_ids=[e.element_id for e in source.elements[:removed_count]])
                payload = json.loads(result.update["messages"][0].content)
                variant = GeneratedSequence.from_snapshot(result.update["sequences"][payload["uuid"]])
                self.assertEqual(payload["elements_count"], size - removed_count)
                self.assertEqual(payload["removal"]["removed_count"], removed_count)
                self.assertEqual(variant.elements_count, size - removed_count)
                self.assert_no_external_calls(runtime)

    def test_failed_removal_allows_a_corrected_call_without_consuming_operation(self):
        source = sequence()
        runtime = self.runtime(source)
        self.assert_rejected(source, runtime, "sequence_element_not_found", element_ids=[uuid4()])
        self.assertIsInstance(self.call(source, runtime), Command)

    def generated(self):
        generator = Mock()
        generator.generate.return_value = [element.catalog_item for element in sequence().elements]
        generator.add_elements.return_value = [sequence(100).elements[-1].catalog_item]
        factory, catalog, tags = Mock(return_value=generator), Mock(), Mock()
        model = ToolCallingModel(responses=[
            tool_call("generate_sequence", request="Original practice", count=5), AIMessage(content=""),
        ])
        service = assistant(factory, model=model, catalog_item_service=catalog, catalog_tag_service=tags)
        list(service.generate("Original practice"))
        factory.reset_mock()
        return service, service.get_sequence(service.active_sequence_uuid), model, factory, catalog, tags

    def test_real_tool_loop_merges_variant_and_model_receives_committed_snapshot(self):
        service, source, model, factory, catalog, tags = self.generated()
        before = service.state["sequences"]
        request = "Удали вторую и четвёртую асаны"
        model.responses = [removal_call(source, element_ids=[
            str(source.elements[i].element_id) for i in (3, 1)
        ]), AIMessage(content="")]
        self.assertEqual(list(service.stream(request)), [])
        variant = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual([e.catalog_item.id for e in variant.elements], [1, 3, 5])
        self.assertEqual(variant.operation_request, request)
        self.assertEqual(service.state["sequences"][str(source.uuid)], before[str(source.uuid)])
        self.assertEqual(len(service.state["sequences"]), 2)
        self.assertTrue(service.state["turn_operation_committed"])
        self.assertEqual(selected_snapshot(model), variant.model_context())
        message = model.received_messages[-1][-2]
        self.assertEqual(message.tool_call_id, model.received_messages[-1][-3].tool_calls[0]["id"])
        self.assertEqual(json.loads(message.content)["removal"]["removed_count"], 2)
        self.assertIn("remove_sequence_elements", model.tool_names)
        factory.assert_not_called()
        self.assertEqual(catalog.mock_calls, [])
        self.assertEqual(tags.mock_calls, [])

    def test_selected_older_variant_is_parent_and_stale_call_can_be_corrected(self):
        service, source, model, factory, catalog, tags = self.generated()
        model.responses = [removal_call(source), AIMessage(content="")]
        list(service.stream("Remove second"))
        newer = service.get_sequence(service.active_sequence_uuid)
        service.select_sequence(str(source.uuid))
        before = service.state["sequences"]
        model.responses = [removal_call(newer), removal_call(source), AIMessage(content="")]
        list(service.stream("Remove from the selected older variant"))
        messages = model.received_messages[-1]
        errors = [m for m in messages if isinstance(m, ToolMessage) and m.status == "error"]
        self.assertEqual(json.loads(errors[-1].content)["code"], "stale_sequence")
        variant = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual(variant.parent_uuid, source.uuid)
        self.assertEqual(variant.family_uuid, source.family_uuid)
        self.assertEqual(len(service.state["sequences"]), 3)
        for key, snapshot in before.items():
            self.assertEqual(service.state["sequences"][key], snapshot)
        factory.assert_not_called()

    def test_removed_occurrence_is_unknown_in_new_variant_and_retry_can_succeed(self):
        service, source, model, factory, catalog, tags = self.generated()
        model.responses = [removal_call(source), AIMessage(content="")]
        list(service.stream("Remove second"))
        variant = service.get_sequence(service.active_sequence_uuid)
        model.responses = [removal_call(source, sequence_uuid=str(variant.uuid)),
                           removal_call(variant), AIMessage(content="")]
        list(service.stream("Remove an occurrence from this variant"))
        errors = [m for m in model.received_messages[-1] if isinstance(m, ToolMessage) and m.status == "error"]
        self.assertEqual(json.loads(errors[-1].content)["code"], "sequence_element_not_found")
        self.assertEqual(service.get_sequence(service.active_sequence_uuid).elements_count, 3)
        self.assertEqual(len(service.state["sequences"]), 3)
        factory.assert_not_called()

    def test_removal_blocks_every_second_operation_even_with_the_new_uuid(self):
        for name in ("remove_sequence_elements", "generate_sequence", "replace_sequence_element",
                     "swap_sequence_elements", "add_sequence_elements"):
            with self.subTest(name=name):
                service, source, model, factory, catalog, tags = self.generated()
                next_uuid = uuid4()
                calls = {
                    "remove_sequence_elements": removal_call(source, sequence_uuid=str(next_uuid),
                                                             element_ids=[str(source.elements[0].element_id)]),
                    "generate_sequence": tool_call("generate_sequence", request="New practice", count=5),
                    "replace_sequence_element": tool_call(
                        "replace_sequence_element", sequence_uuid=str(next_uuid),
                        element_id=str(source.elements[0].element_id), replacement_catalog_item_id=100,
                    ),
                    "swap_sequence_elements": tool_call(
                        "swap_sequence_elements", sequence_uuid=str(next_uuid),
                        first_element_id=str(source.elements[0].element_id), second_element_id=str(source.elements[2].element_id),
                    ),
                    "add_sequence_elements": tool_call(
                        "add_sequence_elements", sequence_uuid=str(next_uuid), count=1, request="Add one",
                    ),
                }
                model.responses = [removal_call(source), calls[name], AIMessage(content="")]
                with patch("langchain_app.services.sequence_editor.uuid4", return_value=next_uuid) as new_uuid:
                    list(service.stream("Remove then change again"))
                new_uuid.assert_called_once_with()
                self.assertEqual(json.loads(model.received_messages[-1][-2].content)["code"], "operation_already_started")
                self.assertEqual(service.active_sequence_uuid, str(next_uuid))
                self.assertEqual(len(service.state["sequences"]), 2)
                factory.assert_not_called()
                self.assertEqual(catalog.mock_calls, [])

    def test_removal_after_each_other_successful_operation_is_rejected(self):
        for operation in ("generate", "replace", "swap", "add"):
            with self.subTest(operation=operation):
                service, source, model, factory, catalog, tags = self.generated()
                args = {"sequence_uuid": str(source.uuid)}
                if operation == "generate":
                    calls = [tool_call("generate_sequence", request="New practice", count=5)]
                elif operation == "swap":
                    calls = [tool_call("swap_sequence_elements", **args,
                        first_element_id=str(source.elements[0].element_id), second_element_id=str(source.elements[1].element_id))]
                elif operation == "add":
                    calls = [tool_call("add_sequence_elements", **args, count=1, request="Add one")]
                else:
                    catalog.search.return_value = [sequence(100).elements[-1].catalog_item]
                    calls = [tool_call("search_catalog_items", name="Pose 100"), tool_call(
                        "replace_sequence_element", **args, element_id=str(source.elements[0].element_id),
                        replacement_catalog_item_id=100)]
                model.responses = [*calls, removal_call(source), AIMessage(content="")]
                list(service.stream("Change then remove"))
                self.assertEqual(json.loads(model.received_messages[-1][-2].content)["code"], "operation_already_started")
                self.assertEqual(len(service.state["sequences"]), 2)
                self.assertEqual(service.get_sequence(service.active_sequence_uuid).operation, operation)

    def test_generate_and_regenerate_reject_removal_and_allow_required_generation(self):
        for regenerate in (False, True):
            with self.subTest(regenerate=regenerate):
                service, source, model, factory, catalog, tags = self.generated()
                model.responses = [removal_call(source), tool_call(
                    "generate_sequence", request="Original practice", count=5,
                ), AIMessage(content="")]
                list(service.generate("Generate practice", sequence_uuid=str(source.uuid) if regenerate else None))
                errors = [m for m in model.received_messages[-1] if isinstance(m, ToolMessage) and m.status == "error"]
                self.assertEqual(json.loads(errors[-1].content)["code"], "generation_required")
                self.assertEqual(len(service.state["sequences"]), 2)
                self.assertEqual(service.get_sequence(service.active_sequence_uuid).operation,
                                 "regenerate" if regenerate else "generate")
                factory.assert_called_once_with()

    def test_parallel_removals_are_rejected_without_a_variant(self):
        service, source, model, factory, catalog, tags = self.generated()
        before = service.state["sequences"]
        model.responses = [AIMessage(content="", tool_calls=[
            removal_call(source).tool_calls[0], removal_call(source).tool_calls[0],
        ]), AIMessage(content="")]
        list(service.stream("Remove twice"))
        errors = [m for m in model.received_messages[-1] if isinstance(m, ToolMessage) and m.status == "error"]
        self.assertEqual([json.loads(m.content)["code"] for m in errors[-2:]], ["sequential_tool_calls_required"] * 2)
        self.assertEqual(service.state["sequences"], before)
        self.assertFalse(service.state["turn_operation_committed"])
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
