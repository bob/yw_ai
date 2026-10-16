"""Exercise removal cards, variant selection and the exact Save component payload."""

import unittest
from copy import deepcopy
from unittest.mock import ANY, Mock, patch
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from streamlit.testing.v1 import AppTest

from langchain_app.services.assistant_agent import AssistantAgentError
from langchain_app.services.sequence_editor import remove_elements
from langchain_app.ui.sequence import generated_sequence_view
from test_assistant_agent import last_message
from test_sequence_removal import sequence
from test_sequence_removal_tool import removal_call
from test_sequence_variants import APP, ToolCallingModel, selected_snapshot, tool_model


class ErrorAfterRemovalModel(ToolCallingModel):
    def _generate(self, messages, **kwargs):
        if isinstance(last_message(messages), ToolMessage) and last_message(messages).name == "remove_sequence_elements":
            raise AssistantAgentError("Model unavailable")
        return super()._generate(messages, **kwargs)


class SequenceRemovalUITests(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        env.start()
        self.addCleanup(env.stop)
        self.generator = Mock()
        self.generator.generate.return_value = [e.catalog_item for e in sequence().elements]
        self.model = tool_model(count=5)
        self.save = Mock()
        for target, value in (
            ("langchain_app.dependencies.get_sequence_generator", self.generator),
            ("langchain_app.services.assistant_agent.init_chat_model", self.model),
            ("langchain_app.ui.sequence_import._save_component", self.save),
        ):
            resource = patch(target, return_value=value)
            resource.start()
            self.addCleanup(resource.stop)
        editor = patch("langchain_app.tools.sequence.remove_elements", wraps=remove_elements)
        self.editor = editor.start()
        self.addCleanup(editor.stop)

    def app_with_source(self):
        app = AppTest.from_file(str(APP)).run()
        app.chat_input[0].set_value("Generate a yoga sequence with 5 poses.").run()
        self.assertFalse(app.exception)
        service = app.session_state.assistant_agent
        source = service.get_sequence(service.active_sequence_uuid)
        # Every removal must use only the selected snapshot.
        service._generator_factory = Mock(side_effect=AssertionError("Unexpected generator initialization"))
        service._catalog_item_service = Mock(side_effect=AssertionError("Unexpected catalog access"))
        service._catalog_tag_service = Mock(side_effect=AssertionError("Unexpected tag access"))
        return app, service, source

    def remove(self, app, source, **overrides):
        self.model.responses = [removal_call(source, **overrides), AIMessage(content="")]
        self.model.i = 0
        app.chat_input[0].set_value("Удали выбранные асаны").run()
        self.assertFalse(app.exception)

    def saved(self):
        return deepcopy(self.save.call_args.kwargs["data"]["sequence"])

    def assert_payload(self, saved, variant):
        self.assertEqual(saved, generated_sequence_view(variant))
        self.assertEqual(saved["info"]["uuid"], str(variant.uuid))
        self.assertEqual(saved["info"]["elements_count"], len(variant.elements))
        self.assertEqual([e["uuid"] for e in saved["thumbnails"]], [e.catalog_item.uuid for e in variant.elements])
        self.assertEqual([e["order"] for e in saved["thumbnails"]], list(range(1, len(variant.elements) + 1)))
        self.assertEqual([e["start_mirror"] for e in saved["thumbnails"]], [e.start_mirror for e in variant.elements])
        self.assertEqual([e["end_mirror"] for e in saved["thumbnails"]], [e.end_mirror for e in variant.elements])

    def assert_no_edit_services(self, service):
        service._generator_factory.assert_not_called()
        self.assertEqual(service._catalog_item_service.mock_calls + service._catalog_tag_service.mock_calls, [])

    def test_card_history_rerun_and_save_preserve_only_remaining_occurrences(self):
        app, service, source = self.app_with_source()
        source.elements[0].start_mirror, source.elements[0].end_mirror = True, False
        source.elements[2].start_mirror, source.elements[2].end_mirror = False, True
        service._get_agent().update_state(service._config, {"sequences": {str(source.uuid): source.model_dump(mode="json")}})
        self.remove(app, source, element_ids=[str(source.elements[i].element_id) for i in (3, 1)])
        variant = service.get_sequence(service.active_sequence_uuid)
        message = app.session_state.messages[-1]
        self.assertEqual(message["content"], "")
        self.assertEqual(message["sequence_uuid"], str(variant.uuid))
        self.assertEqual(message["sequence_variant_uuids"], [str(source.uuid), str(variant.uuid)])
        self.assertEqual(message["sequence_request"], source.generation_request)
        self.assertEqual(message["execution"].sequence_uuid, variant.uuid)
        update = next(s for s in message["execution"].steps if s.code == "updating_sequence")
        self.assertEqual((update.title, update.outcome, update.summary),
                         ("Removing sequence elements...", "ok", "Removed 2 sequence elements; 3 poses in total."))
        self.assertEqual(update.progress, [])
        saved = self.saved()
        self.assert_payload(saved, variant)
        self.assertEqual([e["alt"] for e in saved["thumbnails"]], ["Pose 1", "Pose 3", "Pose 5"])
        self.assertIn("Sequence #0 · 3 poses · EN", [caption.value for caption in app.caption])
        calls = len(self.model.received_messages)
        app.button(key="assistant-sequence-3-previous").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(service.active_sequence_uuid, str(source.uuid))
        self.assert_payload(self.saved(), source)
        app.button(key="assistant-sequence-3-next").click().run()
        app.session_state.messages[-1]["sequence_variants"] = []
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(service.active_sequence_uuid, str(variant.uuid))
        self.assertEqual(self.saved(), saved)
        self.assertEqual(len(self.model.received_messages), calls)
        self.editor.assert_called_once()
        self.assertEqual(service.get_sequence(str(source.uuid)), source)
        self.assert_no_edit_services(service)

    def test_parent_selection_creates_a_branch_and_use_restores_original_without_tools(self):
        app, service, source = self.app_with_source()
        self.remove(app, source)
        newer = service.get_sequence(service.active_sequence_uuid)
        app.button(key="assistant-sequence-3-previous").click().run()
        self.assertFalse(app.exception)
        self.remove(app, source, element_ids=[str(source.elements[3].element_id)])
        branch = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual((branch.parent_uuid, branch.family_uuid), (source.uuid, source.family_uuid))
        self.assertEqual([e.catalog_item.id for e in branch.elements], [1, 2, 3, 5])
        self.assertEqual(selected_snapshot(self.model, -2), source.model_context())
        self.assertEqual(service.get_sequence(str(newer.uuid)), newer)
        self.assertEqual(app.session_state.messages[-1]["sequence_variant_uuids"],
                         [str(source.uuid), str(newer.uuid), str(branch.uuid)])
        calls = len(self.model.received_messages)
        app.button(key="assistant-sequence-1-use").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(service.active_sequence_uuid, str(source.uuid))
        self.assertEqual(len(self.model.received_messages), calls)
        self.assertEqual(self.editor.call_count, 2)
        self.assertEqual(service.get_sequence(str(source.uuid)), source)
        self.assert_no_edit_services(service)

    def test_regenerate_uses_original_request_and_can_restore_previously_removed_poses(self):
        app, service, source = self.app_with_source()
        self.remove(app, source)
        removed = service.get_sequence(service.active_sequence_uuid)
        self.model.responses = [tool_model(source.generation_request, count=5).responses[0], AIMessage(content="")]
        self.model.i = 0
        service._generator_factory = Mock(return_value=self.generator)
        app.button(key="assistant-sequence-3-regenerate").click().run()
        self.assertFalse(app.exception)
        result = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual((result.operation, result.parent_uuid, result.family_uuid), ("regenerate", removed.uuid, source.family_uuid))
        self.assertEqual(result.generation_request, source.generation_request)
        self.assertEqual(result.elements_count, 5)
        self.assertEqual([e.catalog_item.id for e in result.elements], [1, 2, 3, 4, 5])
        self.generator.generate.assert_called_with(source.generation_request, language="en", count=5,
                                                  starting_catalog_item_id=None, progress_observer=ANY)
        self.editor.assert_called_once()
        self.assert_payload(self.saved(), result)

    def test_failed_removal_keeps_original_save_without_creating_a_card(self):
        app, service, source = self.app_with_source()
        for targets in ([str(source.elements[1].element_id), str(uuid4())], [str(e.element_id) for e in source.elements]):
            with self.subTest(targets=targets):
                self.remove(app, source, element_ids=targets)
                message = app.session_state.messages[-1]
                self.assertIsNone(message["sequence"])
                self.assertIsNone(message["execution"].sequence_uuid)
                update = next(s for s in message["execution"].steps if s.code == "updating_sequence")
                self.assertEqual((update.title, update.outcome), ("Removing sequence elements...", "error"))
                self.assertEqual(service.active_sequence_uuid, str(source.uuid))
                self.assertEqual(len(service.state["sequences"]), 1)
                self.assert_payload(self.saved(), source)
        self.assert_no_edit_services(service)

    def test_committed_removal_card_and_save_survive_failure_of_final_model_call(self):
        self.model = ErrorAfterRemovalModel(responses=tool_model(count=5).responses)
        with patch("langchain_app.services.assistant_agent.init_chat_model", return_value=self.model):
            app, service, source = self.app_with_source()
            self.remove(app, source)
            self.assertEqual(len(app.error), 1)
            variant = service.get_sequence(service.active_sequence_uuid)
            message = app.session_state.messages[-1]
            self.assertEqual((variant.operation, variant.elements_count), ("remove", 4))
            self.assertEqual(message["sequence_uuid"], str(variant.uuid))
            self.assertEqual(message["execution"].outcome, "error")
            self.assert_payload(self.saved(), variant)
            calls = len(self.model.received_messages)
            saved = self.saved()
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(service.active_sequence_uuid, str(variant.uuid))
            self.assertEqual(self.saved(), saved)
            self.assertEqual(len(self.model.received_messages), calls)
            self.editor.assert_called_once()
            self.assert_no_edit_services(service)

    def test_imported_long_snapshot_passes_complete_reduced_payload_to_save(self):
        app, service, _ = self.app_with_source()
        source = sequence(50)
        service._get_agent().update_state(service._config, {
            "sequences": {str(source.uuid): source.model_dump(mode="json")}, "active_sequence_uuid": str(source.uuid),
        })
        app.session_state.messages[1].update({
            "sequence": generated_sequence_view(source), "sequence_uuid": str(source.uuid),
            "sequence_variant_uuids": [str(source.uuid)], "selected_sequence_index": 0,
            "sequence_request": source.generation_request,
        })
        self.remove(app, source)
        variant = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual(variant.elements_count, 49)
        self.assert_payload(self.saved(), variant)
        self.assertEqual(len(self.saved()["thumbnails"]), 49)
        self.assert_no_edit_services(service)


if __name__ == "__main__":
    unittest.main()
