"""Keep removal facts compact and current occurrences authoritative after summarization."""

import json
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_app.services.assistant_history import HistoryBudget, compact_history
from langchain_app.services.sequence_editor import remove_elements
from test_assistant_agent import ToolCallingModel, assistant, selected_context, selected_snapshot
from test_assistant_history import assert_tool_pairs, seed_history, summary_reply
from test_sequence_removal import sequence
from test_sequence_removal_tool import removal_call


class SequenceRemovalHistoryTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        env.start()
        self.addCleanup(env.stop)

    def history(self):
        source = sequence()
        removed = [source.elements[i] for i in (1, 3)]
        variant = remove_elements(source, element_ids=[e.element_id for e in removed], request="Remove second and fourth")
        payload = {
            "status": "ok", "uuid": str(variant.uuid), "operation": "remove", "elements_count": 3,
            "removal": {"removed_count": 2, "elements": [
                {"element_id": str(e.element_id), "name": e.catalog_item.name, "old_order": e.order}
                for e in removed
            ]},
        }
        return source, variant, payload

    def test_projection_preserves_removal_facts_and_errors_without_full_snapshots(self):
        source, variant, compact = self.history()
        for named in (True, False):
            with self.subTest(named=named):
                call = removal_call(source)
                result = ToolMessage(
                    content=json.dumps({"status": "ok", **variant.model_dump(mode="json"), "removal": compact["removal"]}),
                    tool_call_id=call.tool_calls[0]["id"],
                    name="remove_sequence_elements" if named else None,
                    artifact={"snapshot": variant.model_dump(mode="json")},
                )
                messages = [HumanMessage(content="Remove second and fourth"), call, result, AIMessage(content="")]
                original = deepcopy(messages)
                projected = compact_history(messages)
                self.assertEqual(json.loads(projected[2].content), compact)
                self.assertIsNone(projected[2].artifact)
                self.assertEqual(projected[2].tool_call_id, result.tool_call_id)
                for field in ("image_url", "catalog_item", "catalog_tag_names", "generation_request"):
                    self.assertNotIn(field, projected[2].content)
                self.assertEqual(messages, original)
                assert_tool_pairs(self, projected)
                current = ToolMessage(content=json.dumps(compact), name="remove_sequence_elements", tool_call_id="current")
                error = ToolMessage(content='{"status":"error","code":"would_empty_sequence"}',
                                    name="remove_sequence_elements", tool_call_id="failed", status="error")
                self.assertEqual(compact_history([current, error]), [current, error])

    def service_with_history(self, *, failing_summary=False):
        source, variant, payload = self.history()
        model = ToolCallingModel(responses=[AIMessage(content="")])
        summary = ToolCallingModel(responses=[summary_reply(relevant_facts=[
            "The original practice contained Pose 1, Pose 2, Pose 3, Pose 4 and Pose 5.",
            "Pose 2 and Pose 4 were removed from one historical variant.",
        ])])
        if failing_summary:
            summary = Mock(model_name="", invoke=Mock(side_effect=TimeoutError("offline")))
        factory = Mock(side_effect=AssertionError("Removal must not initialize a generator"))
        service = assistant(factory, model=model, summary_model=summary, history_budget=HistoryBudget(1600, 1000, 256))
        call = removal_call(source, element_ids=[str(source.elements[i].element_id) for i in (1, 3)])
        legacy = {"status": "ok", **variant.model_dump(mode="json"), "removal": payload["removal"]}
        messages = [HumanMessage(content="Remove second and fourth"), call,
                    ToolMessage(content=json.dumps(legacy), name="remove_sequence_elements", tool_call_id=call.tool_calls[0]["id"]),
                    AIMessage(content="")]
        service._get_agent().update_state(service._config, {
            "messages": messages, "history_turn_starts": [0],
            "sequences": {str(s.uuid): s.model_dump(mode="json") for s in (source, variant)},
            "active_sequence_uuid": str(variant.uuid),
        })
        seed_history(service, turns=16)
        return service, source, variant, model, summary, payload

    def test_summary_receives_compact_removal_and_next_edit_uses_current_occurrences(self):
        service, source, variant, model, summary, payload = self.service_with_history()
        original = deepcopy(service.state["messages"])
        model.responses = [removal_call(variant), AIMessage(content="")]
        list(service.stream("Remove the second remaining pose"))
        self.assertEqual(len(summary.received_messages), 1)
        summary_input = json.loads(summary.received_messages[0][-1].content)
        covered = [m for turn in summary_input["newly_covered_turns"] for m in turn]
        result = next(m for m in covered if m["role"] == "tool")
        self.assertEqual(json.loads(result["content"]), payload)
        self.assertEqual(service.state["messages"][:len(original)], original)
        self.assertEqual(selected_snapshot(model, 0), variant.model_context())
        self.assertFalse(selected_context(model, 0)["turn_operation_committed"])
        edited = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual([e.catalog_item.id for e in edited.elements], [1, 5])
        self.assertEqual(selected_snapshot(model), edited.model_context())
        self.assertTrue(selected_context(model)["turn_operation_committed"])
        self.assertEqual((edited.parent_uuid, edited.generation_request), (variant.uuid, source.generation_request))
        self.assertEqual(service.get_sequence(str(source.uuid)), source)
        self.assertEqual(service.get_sequence(str(variant.uuid)), variant)
        self.assertEqual(service.state["history_projection_version"], 2)
        for outgoing in model.received_messages:
            assert_tool_pairs(self, outgoing)
        service._generator_factory.assert_not_called()
        self.assertEqual(service._catalog_item_service.mock_calls + service._catalog_tag_service.mock_calls, [])

    def test_summary_failure_keeps_selected_removal_and_explicit_parent_selection(self):
        service, source, variant, model, summary, _ = self.service_with_history(failing_summary=True)
        list(service.stream("Explain the current sequence"))
        self.assertTrue(service.state["history_context_incomplete"])
        self.assertEqual(summary.invoke.call_count, 1)
        self.assertEqual(selected_snapshot(model), variant.model_context())
        self.assertEqual(service.get_sequence(str(variant.uuid)), variant)
        service.select_sequence(str(source.uuid))
        list(service.stream("Explain the selected original"))
        self.assertEqual(selected_snapshot(model), source.model_context())
        self.assertEqual(service.active_sequence_uuid, str(source.uuid))
        self.assertEqual(service.get_sequence(str(variant.uuid)), variant)
        self.assertEqual(len(service.state["sequences"]), 2)
        for outgoing in model.received_messages:
            assert_tool_pairs(self, outgoing)
        service._generator_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
