"""Verify addition guards, exact block commits, and the real assistant tool loop."""

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
from langchain_app.services.catalog_errors import CatalogFolderNotFoundError
from langchain_app.services.catalog_item import CatalogItemApiError
from langchain_app.services.sequence_generator import SequenceGenerator
from langchain_app.services.type_safe import TypeSafeSelectionError, TypeSafeServiceError
from langchain_app.tools import ASSISTANT_TOOLS, add_sequence_elements, swap_sequence_elements
from langchain_app.tools.context import AssistantToolContext
from test_assistant_agent import ToolCallingModel, assistant, selected_snapshot, tool_call
from test_sequence_insertion import pose, source


def addition_call(sequence, **overrides):
    return tool_call("add_sequence_elements", **{
        "sequence_uuid": str(sequence.uuid), "count": 1, "request": "Add one pose",
        **overrides,
    })


class SequenceAdditionToolTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        env.start()
        self.addCleanup(env.stop)

    def runtime(self, sequence, items=None):
        generator = Mock()
        generator.add_elements.return_value = [pose(100)] if items is None else items
        context = AssistantToolContext(
            Mock(return_value=generator), Mock(), Mock(), user_request="Добавь асаны с моими пожеланиями",
        )
        return ToolRuntime({
            "sequences": {str(sequence.uuid): sequence.model_dump(mode="json")},
            "active_sequence_uuid": str(sequence.uuid), "turn_operation_committed": False,
            "latest_catalog_items": ["Unrelated, unvalidated catalog data"],
        }, context, {}, Mock(), "addition-call", None)

    def call(self, sequence, runtime, **overrides):
        return add_sequence_elements.func(**{
            "sequence_uuid": sequence.uuid, "count": 1, "request": "Add one pose",
            "runtime": runtime, **overrides,
        })

    def assert_error(self, result, code):
        self.assertIsInstance(result, ToolMessage)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.tool_call_id, "addition-call")
        payload = json.loads(result.content)
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["code"], code)
        self.assertTrue(payload["message"])

    def assert_preflight_rejected(self, sequence, runtime, code, **overrides):
        before = deepcopy(runtime.state)
        started = runtime.context.generation_started
        self.assert_error(self.call(sequence, runtime, **overrides), code)
        self.assertEqual(runtime.state, before)
        self.assertEqual(runtime.context.generation_started, started)
        runtime.context.generator_factory.assert_not_called()
        runtime.context.catalog_item_service.search.assert_not_called()
        runtime.context.catalog_tag_service.get_names.assert_not_called()
        runtime.stream_writer.assert_not_called()

    def test_registered_export_exposes_only_model_arguments_and_unbounded_positive_count(self):
        self.assertIn(add_sequence_elements, ASSISTANT_TOOLS)
        schema = add_sequence_elements.tool_call_schema.model_json_schema()
        self.assertEqual(set(schema["properties"]), {
            "sequence_uuid", "count", "request", "insert_before_element_id",
        })
        self.assertEqual(set(schema["required"]), {"sequence_uuid", "count", "request"})
        self.assertEqual(schema["properties"]["sequence_uuid"]["format"], "uuid")
        self.assertEqual(schema["properties"]["count"]["exclusiveMinimum"], 0)
        self.assertNotIn("maximum", schema["properties"]["count"])
        self.assertIsNone(schema["properties"]["insert_before_element_id"]["default"])
        validated = add_sequence_elements.tool_call_schema.model_validate({
            "sequence_uuid": str(uuid4()), "count": 100, "request": "Add 100",
        })
        self.assertEqual(validated.count, 100)
        self.assertIsNone(validated.insert_before_element_id)

    def test_standard_argument_validation_rejects_nonpositive_or_noninteger_counts(self):
        sequence = source()
        for count in (0, -1, True, False, 1.5, 2.0, "2", None):
            with self.subTest(count=count):
                runtime = self.runtime(sequence)
                before = deepcopy(runtime.state)
                with self.assertRaises(ValidationError):
                    add_sequence_elements.invoke({
                        "sequence_uuid": str(sequence.uuid), "count": count,
                        "request": "Add poses", "runtime": runtime,
                    })
                runtime.context.generator_factory.assert_not_called()
                self.assertEqual(runtime.state, before)

    def test_standard_argument_validation_rejects_invalid_uuids_and_empty_request(self):
        sequence = source()
        for overrides in (
            {"sequence_uuid": "not-a-uuid"}, {"insert_before_element_id": "third"},
            {"request": ""},
        ):
            with self.subTest(overrides=overrides):
                runtime = self.runtime(sequence)
                with self.assertRaises(ValidationError):
                    add_sequence_elements.invoke({
                        "sequence_uuid": str(sequence.uuid), "count": 1,
                        "request": "Add", "runtime": runtime, **overrides,
                    })
                runtime.context.generator_factory.assert_not_called()

    def test_operation_guards_precede_snapshot_request_and_anchor_checks(self):
        sequence = source()
        for required, started, committed, code in (
            (True, True, True, "generation_required"),
            (True, False, False, "generation_required"),
            (False, True, False, "operation_already_started"),
            (False, False, True, "operation_already_started"),
        ):
            with self.subTest(required=required, started=started, committed=committed):
                runtime = self.runtime(sequence)
                runtime.context.require_generation = required
                runtime.context.generation_started = started
                runtime.state = {"turn_operation_committed": committed}
                self.assert_preflight_rejected(
                    sequence, runtime, code, sequence_uuid=uuid4(), request=" ",
                    insert_before_element_id=uuid4(),
                )

    def test_missing_active_snapshot_does_not_reconstruct_from_history(self):
        sequence = source()
        for active_uuid in (None, str(uuid4()), str(sequence.uuid)):
            with self.subTest(active_uuid=active_uuid):
                runtime = self.runtime(sequence)
                runtime.state["active_sequence_uuid"] = active_uuid
                if active_uuid == str(sequence.uuid):
                    runtime.state["sequences"] = {}
                self.assert_preflight_rejected(sequence, runtime, "sequence_not_found", request=" ")

    def test_stored_stale_uuid_is_rejected_before_request_validation(self):
        sequence, stale = source(), source()
        runtime = self.runtime(sequence)
        runtime.state["sequences"][str(stale.uuid)] = stale.model_dump(mode="json")
        self.assert_preflight_rejected(
            sequence, runtime, "stale_sequence", sequence_uuid=stale.uuid, request=" ",
        )

    def test_blank_request_and_empty_sequence_precede_anchor_and_limit(self):
        sequence = source()
        self.assert_preflight_rejected(sequence, self.runtime(sequence), "invalid_arguments", request=" \n\t ")
        sequence.elements.clear()
        self.assert_preflight_rejected(sequence, self.runtime(sequence), "invalid_arguments", request=" ")
        self.assert_preflight_rejected(
            sequence, self.runtime(sequence), "empty_sequence", insert_before_element_id=uuid4(),
        )

    def test_unknown_anchor_is_rejected_even_at_or_above_limit(self):
        for count in (4, 45, 46):
            with self.subTest(count=count):
                sequence = source(count)
                self.assert_preflight_rejected(
                    sequence, self.runtime(sequence), "sequence_element_not_found",
                    insert_before_element_id=uuid4(),
                )

    def test_full_and_oversized_sequences_return_no_change_without_consuming_operation(self):
        for count in (45, 46):
            with self.subTest(count=count):
                sequence = source(count)
                runtime = self.runtime(sequence)
                runtime.state["sequences"][str(sequence.uuid)]["elements_count"] = 1
                before = deepcopy(runtime.state)
                result = self.call(sequence, runtime, count=100)
                self.assertEqual(result.tool_call_id, "addition-call")
                payload = json.loads(result.content)
                self.assertEqual(payload, {
                    "status": "no_change", "code": "sequence_limit_reached",
                    "message": payload["message"], "uuid": str(sequence.uuid),
                    "requested_count": 100, "added_count": 0, "limit_applied": True,
                })
                self.assertIn("45", payload["message"])
                self.assertEqual(runtime.state, before)
                self.assertFalse(runtime.context.generation_started)
                runtime.context.generator_factory.assert_not_called()
                result = swap_sequence_elements.func(
                    sequence.uuid, sequence.elements[0].element_id,
                    sequence.elements[-1].element_id, runtime,
                )
                self.assertIsInstance(result, Command)

    def test_commit_contains_whole_variant_and_exact_addition_for_each_insertion_position(self):
        for index, count in ((None, 1), (None, 2), (0, 2), (2, 2)):
            with self.subTest(index=index, count=count):
                sequence = source()
                sequence.elements[2].start_mirror, sequence.elements[2].end_mirror = True, False
                items = [pose(100 + offset, bilateral=offset % 2 == 0) for offset in range(count)]
                if index == 0:
                    items[0].catalog_tag_names = ["Other/Starting", "Position/Sitting"]
                runtime = self.runtime(sequence, items)
                before = deepcopy(runtime.state)
                anchor = sequence.elements[index].element_id if index is not None else None
                insertion = len(sequence.elements) if index is None else index
                result = self.call(sequence, runtime, count=count, insert_before_element_id=anchor)
                self.assertIsInstance(result, Command)
                update = result.update
                self.assertEqual(set(update), {"sequences", "active_sequence_uuid", "turn_operation_committed", "messages"})
                self.assertEqual(len(update["sequences"]), 1)
                variant = GeneratedSequence.from_snapshot(update["sequences"][update["active_sequence_uuid"]])
                self.assertTrue(update["turn_operation_committed"])
                self.assertNotEqual(variant.uuid, sequence.uuid)
                self.assertEqual(variant.parent_uuid, sequence.uuid)
                self.assertEqual(variant.family_uuid, sequence.family_uuid)
                self.assertEqual(variant.operation, "add")
                self.assertEqual(variant.operation_request, runtime.context.user_request)
                for field in ("generation_request", "name", "description", "language"):
                    self.assertEqual(getattr(variant, field), getattr(sequence, field))
                self.assertEqual([element.catalog_item.id for element in variant.elements], [
                    *range(1, insertion + 1), *range(100, 100 + count), *range(insertion + 1, 5),
                ])
                message, = update["messages"]
                self.assertEqual(message.tool_call_id, "addition-call")
                self.assertEqual(json.loads(message.content), {
                    "status": "ok", "uuid": str(variant.uuid),
                    "operation": "add", "elements_count": variant.elements_count,
                    "addition": {
                        "requested_count": count, "added_count": count, "limit_applied": False,
                        "position": insertion + 1, "insert_before_element_id": str(anchor) if anchor else None,
                        "elements": [
                            {"element_id": str(element.element_id), "order": element.order}
                            for element in variant.elements[insertion:insertion + count]
                        ],
                    },
                })
                generator = runtime.context.generator_factory.return_value
                runtime.context.generator_factory.assert_called_once_with()
                generator.add_elements.assert_called_once_with(
                    sequence, "Add one pose", language="en", count=count,
                    insert_before_element_id=anchor, progress_observer=runtime.stream_writer,
                )
                generator.generate.assert_not_called()
                self.assertFalse(runtime.context.generation_started)
                self.assertEqual(runtime.state, before)
                runtime.context.catalog_item_service.search.assert_not_called()
                runtime.stream_writer.assert_not_called()

    def test_requested_count_is_capped_by_actual_elements_before_selection(self):
        for size, requested, added in ((1, 44, 44), (9, 36, 36), (43, 5, 2), (43, 100, 2), (44, 1, 1)):
            with self.subTest(size=size, requested=requested):
                sequence = source(size)
                runtime = self.runtime(sequence, [pose(100 + offset) for offset in range(added)])
                runtime.state["sequences"][str(sequence.uuid)]["elements_count"] = 999
                result = self.call(sequence, runtime, count=requested)
                payload = json.loads(result.update["messages"][0].content)
                self.assertEqual(payload["elements_count"], 45)
                self.assertEqual(payload["addition"]["requested_count"], requested)
                self.assertEqual(payload["addition"]["added_count"], added)
                self.assertEqual(payload["addition"]["limit_applied"], requested > added)
                self.assertEqual(runtime.context.generator_factory.return_value.add_elements.call_args.kwargs["count"], added)

    def test_invalid_substituted_blocks_never_commit(self):
        sequence = source()
        invalid = pose(100).model_copy(update={"bilateral": "invalid"})
        cases = [
            [], [pose(100), pose(101)], None, ["pose"], [invalid],
            [pose(100).model_copy(update={"uuid": "invalid"})],
            [pose(100).model_copy(update={"image_url": None})],
            [sequence.elements[0].catalog_item],
            [pose(100).model_copy(update={"uuid": sequence.elements[-1].catalog_item.uuid})],
            [pose(100, ["Other/Starting"])],
        ]
        for items in cases:
            with self.subTest(items=items):
                runtime = self.runtime(sequence)
                runtime.context.generator_factory.return_value.add_elements.return_value = items
                before = deepcopy(runtime.state)
                self.assert_error(self.call(sequence, runtime), "invalid_addition_result")
                self.assertEqual(runtime.state, before)
                self.assertFalse(runtime.context.generation_started)

    def test_substituted_block_cannot_bypass_first_second_pose_constraints(self):
        sequence = source()
        sequence.elements[0].catalog_item.catalog_tag_names = ["Position/Lying on the back"]
        runtime = self.runtime(sequence, [pose(100, ["Position/Standing"])])
        self.assert_error(self.call(
            sequence, runtime, insert_before_element_id=sequence.elements[1].element_id,
        ), "invalid_addition_result")

    def test_expected_catalog_and_selection_failures_do_not_consume_operation(self):
        sequence = source()
        for error, code in (
            (CatalogItemApiError("Catalog unavailable"), "catalog_search_failed"),
            (CatalogFolderNotFoundError("Folder missing"), "catalog_search_failed"),
            (TypeSafeSelectionError("No candidates"), "addition_selection_failed"),
            (TypeSafeServiceError("Classifier failed"), "addition_selection_failed"),
        ):
            with self.subTest(error=error):
                runtime = self.runtime(sequence)
                generator = runtime.context.generator_factory.return_value
                generator.add_elements.side_effect = error
                before = deepcopy(runtime.state)
                self.assert_error(self.call(sequence, runtime), code)
                self.assertEqual(runtime.state, before)
                self.assertFalse(runtime.context.generation_started)
                generator.add_elements.side_effect = None
                self.assertIsInstance(self.call(sequence, runtime, request="Corrected preferences"), Command)

    def test_lazy_partial_block_failure_and_unexpected_errors_never_commit(self):
        sequence = source()

        def partial_block():
            yield pose(100)
            raise TypeSafeSelectionError("Second selection failed")

        runtime = self.runtime(sequence)
        runtime.context.generator_factory.return_value.add_elements.return_value = partial_block()
        before = deepcopy(runtime.state)
        self.assert_error(self.call(sequence, runtime, count=2), "addition_selection_failed")
        self.assertEqual(runtime.state, before)
        runtime.context.generator_factory.return_value.add_elements.side_effect = RuntimeError("Unexpected failure")
        with self.assertRaisesRegex(RuntimeError, "Unexpected failure"):
            self.call(sequence, runtime)
        self.assertEqual(runtime.state, before)

    def generated(self, size=4):
        generator = Mock()
        generator.generate.return_value = [element.catalog_item for element in source(size).elements]
        generator.add_elements.return_value = [pose(100)]
        factory, catalog = Mock(return_value=generator), Mock()
        model = ToolCallingModel(responses=[
            tool_call("generate_sequence", request="Original practice", count=size), AIMessage(content="Ready"),
        ])
        service = assistant(factory, model=model, catalog_item_service=catalog)
        list(service.generate("Original practice"))
        factory.reset_mock()
        generator.generate.reset_mock()
        return service, service.get_sequence(service.active_sequence_uuid), model, generator, factory, catalog

    def test_real_tool_cycle_merges_variant_and_refreshes_selected_context(self):
        service, sequence, model, generator, factory, catalog = self.generated()
        before = service.state["sequences"]
        generator.add_elements.return_value = [pose(100), pose(101, bilateral=True)]
        request = "Добавь две асаны перед третьей"
        model.responses = [addition_call(
            sequence, count=2, request=request, insert_before_element_id=str(sequence.elements[2].element_id),
        ), AIMessage(content="Добавлено две асаны, теперь их шесть.")]
        self.assertEqual(list(service.stream(request)), ["Добавлено две асаны, теперь их шесть."])
        variant = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual([element.catalog_item.id for element in variant.elements], [1, 2, 100, 101, 3, 4])
        self.assertEqual(variant.operation_request, request)
        self.assertEqual(service.state["sequences"][str(sequence.uuid)], before[str(sequence.uuid)])
        self.assertEqual(len(service.state["sequences"]), 2)
        self.assertTrue(service.state["turn_operation_committed"])
        self.assertEqual(selected_snapshot(model), variant.model_context())
        message = model.received_messages[-1][-2]
        self.assertEqual(message.tool_call_id, model.received_messages[-1][-3].tool_calls[0]["id"])
        self.assertEqual(json.loads(message.content)["addition"]["position"], 3)
        factory.assert_called_once_with()
        generator.generate.assert_not_called()
        self.assertEqual(catalog.mock_calls, [])

    def test_real_tool_cycle_uses_existing_generator_catalog_and_selection_service(self):
        service, sequence, model, generator, factory, catalog = self.generated()
        catalog.search.return_value = [pose(100), pose(101), pose(102)]
        selector = Mock()
        selector.choose_addition_catalog_item.side_effect = lambda candidates, **kwargs: candidates[0]
        factory.return_value = SequenceGenerator(catalog, selector)
        model.responses = [addition_call(sequence, count=2), AIMessage(content="Added two")]
        list(service.stream("Add two"))
        variant = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual(variant.elements_count, 6)
        self.assertEqual(selector.choose_addition_catalog_item.call_count, 2)
        catalog.search.assert_called_once_with(with_tags=True, with_rigman=False, limit=0, rand=True)
        generator.generate.assert_not_called()

    def test_selected_older_variant_is_parent_and_stored_stale_variant_is_rejected(self):
        service, sequence, model, generator, factory, catalog = self.generated()
        list(service.generate("Newer practice"))
        newer = service.get_sequence(service.active_sequence_uuid)
        service.select_sequence(str(sequence.uuid))
        factory.reset_mock()
        model.responses = [addition_call(newer), AIMessage(content="Select the current variant")]
        list(service.stream("Add to selected"))
        self.assertEqual(json.loads(model.received_messages[-1][-2].content)["code"], "stale_sequence")
        factory.assert_not_called()
        self.assertEqual(service.active_sequence_uuid, str(sequence.uuid))
        model.responses = [addition_call(sequence), AIMessage(content="Added")]
        list(service.stream("Add to this older variant"))
        variant = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual(variant.parent_uuid, sequence.uuid)
        self.assertEqual(variant.family_uuid, sequence.family_uuid)
        self.assertEqual(len(service.state["sequences"]), 3)

    def test_addition_blocks_every_second_operation_and_next_request_resets_limit(self):
        for name in ("add_sequence_elements", "generate_sequence", "replace_sequence_element", "swap_sequence_elements"):
            with self.subTest(name=name):
                service, sequence, model, generator, factory, catalog = self.generated()
                calls = {
                    "add_sequence_elements": addition_call(sequence),
                    "generate_sequence": tool_call("generate_sequence", request="Regenerate", count=4),
                    "replace_sequence_element": tool_call(
                        "replace_sequence_element", sequence_uuid=str(sequence.uuid),
                        element_id=str(sequence.elements[0].element_id), replacement_catalog_item_id=200,
                    ),
                    "swap_sequence_elements": tool_call(
                        "swap_sequence_elements", sequence_uuid=str(sequence.uuid),
                        first_element_id=str(sequence.elements[0].element_id), second_element_id=str(sequence.elements[1].element_id),
                    ),
                }
                model.responses = [addition_call(sequence), calls[name], AIMessage(content="One operation")]
                list(service.stream("Add then change again"))
                self.assertEqual(json.loads(model.received_messages[-1][-2].content)["code"], "operation_already_started")
                self.assertEqual(len(service.state["sequences"]), 2)
                factory.assert_called_once_with()
                variant = service.get_sequence(service.active_sequence_uuid)
                generator.add_elements.return_value = [pose(101)]
                model.responses = [addition_call(variant), AIMessage(content="Added again")]
                list(service.stream("Add another in the next request"))
                self.assertEqual(service.get_sequence(service.active_sequence_uuid).elements_count, 6)
                self.assertEqual(len(service.state["sequences"]), 3)

    def test_addition_after_generation_replacement_and_swap_is_rejected(self):
        for operation in ("generate", "replace", "swap"):
            with self.subTest(operation=operation):
                service, sequence, model, generator, factory, catalog = self.generated()
                if operation == "generate":
                    first_calls = [tool_call("generate_sequence", request="New practice", count=4)]
                elif operation == "swap":
                    first_calls = [tool_call(
                        "swap_sequence_elements", sequence_uuid=str(sequence.uuid),
                        first_element_id=str(sequence.elements[0].element_id), second_element_id=str(sequence.elements[1].element_id),
                    )]
                else:
                    catalog.search.return_value = [pose(200)]
                    first_calls = [tool_call("search_catalog_items", name="Pose 200"), tool_call(
                        "replace_sequence_element", sequence_uuid=str(sequence.uuid),
                        element_id=str(sequence.elements[0].element_id), replacement_catalog_item_id=200,
                    )]
                model.responses = [*first_calls, addition_call(sequence), AIMessage(content="One operation")]
                list(service.stream("Change then add"))
                self.assertEqual(json.loads(model.received_messages[-1][-2].content)["code"], "operation_already_started")
                generator.add_elements.assert_not_called()
                self.assertEqual(len(service.state["sequences"]), 2)

    def test_failed_selection_allows_corrected_call_and_preserves_latest_catalog_items(self):
        service, sequence, model, generator, factory, catalog = self.generated()
        catalog.search.return_value = [pose(200)]
        generator.add_elements.side_effect = [TypeSafeSelectionError("No suitable candidates"), [pose(100)]]
        model.responses = [
            tool_call("search_catalog_items", name="Pose 200"),
            addition_call(sequence, request="Only unsuitable poses"),
            addition_call(sequence, request="Corrected preferences"), AIMessage(content="Added"),
        ]
        list(service.stream("Add with corrected preferences"))
        errors = [message for message in model.received_messages[-1] if isinstance(message, ToolMessage) and message.status == "error"]
        self.assertEqual(json.loads(errors[-1].content)["code"], "addition_selection_failed")
        self.assertEqual(len(service.state["sequences"]), 2)
        self.assertEqual(service.state["latest_catalog_items"], [pose(200).model_dump(mode="json")])
        self.assertEqual(generator.add_elements.call_count, 2)
        catalog.search.assert_called_once()

    def test_real_no_change_allows_swap_in_same_request(self):
        service, sequence, model, generator, factory, catalog = self.generated(45)
        model.responses = [addition_call(sequence), tool_call(
            "swap_sequence_elements", sequence_uuid=str(sequence.uuid),
            first_element_id=str(sequence.elements[0].element_id), second_element_id=str(sequence.elements[-1].element_id),
        ), AIMessage(content="Swapped")]
        list(service.stream("Add or swap at the limit"))
        self.assertEqual(service.get_sequence(service.active_sequence_uuid).operation, "swap")
        self.assertEqual(len(service.state["sequences"]), 2)
        generator.add_elements.assert_not_called()
        factory.assert_not_called()

    def test_parallel_tool_calls_are_rejected_before_selection(self):
        service, sequence, model, generator, factory, catalog = self.generated()
        model.responses = [AIMessage(content="", tool_calls=[
            addition_call(sequence).tool_calls[0], addition_call(sequence).tool_calls[0],
        ]), AIMessage(content="Call one at a time")]
        list(service.stream("Add in two places"))
        errors = [message for message in model.received_messages[-1] if isinstance(message, ToolMessage)]
        self.assertEqual([json.loads(message.content)["code"] for message in errors[-2:]], ["sequential_tool_calls_required"] * 2)
        self.assertEqual(len(service.state["sequences"]), 1)
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
