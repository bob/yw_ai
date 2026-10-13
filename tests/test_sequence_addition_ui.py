"""Exercise addition cards, branching, failures and Save through the real app."""

import unittest
from unittest.mock import ANY, Mock, patch

from langchain_core.messages import AIMessage, ToolMessage
from streamlit.testing.v1 import AppTest

from langchain_app.services.sequence_generator import SequenceGenerator
from langchain_app.services.type_safe import TypeSafeSelectionError
from test_sequence_addition_tool import addition_call
from test_sequence_insertion import pose
from test_sequence_variants import APP, ToolCallingModel, selected_snapshot, tool_model


class ErrorAfterAdditionModel(ToolCallingModel):
    def _generate(self, messages, **kwargs):
        if isinstance(messages[-1], ToolMessage) and messages[-1].name == "add_sequence_elements":
            from langchain_app.services.assistant_agent import AssistantAgentError
            raise AssistantAgentError("Model unavailable")
        return super()._generate(messages, **kwargs)


class SequenceAdditionUITests(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        env.start()
        self.addCleanup(env.stop)
        self.generator = Mock()
        self.generator.generate.return_value = [pose(n, bilateral=n == 3) for n in range(1, 6)]
        self.generator.add_elements.return_value = [pose(100, bilateral=True), pose(101)]
        self.model = tool_model(count=5)
        self.save = Mock()
        self.catalog = Mock()
        for target, value in (
            ("langchain_app.dependencies.get_sequence_generator", self.generator),
            ("langchain_app.dependencies.get_catalog_item_service", self.catalog),
            ("langchain_app.services.assistant_agent.init_chat_model", self.model),
            ("langchain_app.ui.sequence_import._save_component", self.save),
        ):
            resource = patch(target, return_value=value)
            resource.start()
            self.addCleanup(resource.stop)

    def app_with_source(self):
        app = AppTest.from_file(str(APP)).run()
        app.chat_input[0].set_value("Generate a yoga sequence with 5 poses.").run()
        self.assertFalse(app.exception)
        service = app.session_state.assistant_agent
        return app, service, service.get_sequence(service.active_sequence_uuid)

    def add(self, app, source, *, count=2, **kwargs):
        self.model.responses = [addition_call(source, count=count, **kwargs), AIMessage(content="Добавлены две асаны.")]
        app.chat_input[0].set_value("Добавь две асаны перед третьей").run()
        self.assertFalse(app.exception)

    def test_insert_card_history_rerun_and_save_keep_full_order_and_mirror_flags(self):
        app, service, source = self.app_with_source()
        # Imported flags may differ from the catalog's bilateral setting.
        source.elements[1].start_mirror, source.elements[1].end_mirror = True, False
        service._get_agent().update_state(service._config, {"sequences": {str(source.uuid): source.model_dump(mode="json")}})
        self.add(app, source, insert_before_element_id=str(source.elements[2].element_id))
        variant = service.get_sequence(service.active_sequence_uuid)
        message = app.session_state.messages[-1]
        self.assertEqual(message["sequence_uuid"], str(variant.uuid))
        self.assertEqual(message["sequence_variant_uuids"], [str(source.uuid), str(variant.uuid)])
        self.assertEqual(message["sequence_request"], source.generation_request)
        self.assertEqual(message["execution"].sequence_uuid, variant.uuid)
        update = next(s for s in message["execution"].steps if s.code == "updating_sequence")
        self.assertEqual((update.title, update.outcome), ("Adding sequence elements...", "ok"))
        saved = self.save.call_args.kwargs["data"]["sequence"]
        self.assertEqual(saved["info"]["uuid"], str(variant.uuid))
        self.assertEqual(saved["info"]["elements_count"], 7)
        self.assertEqual([x["uuid"] for x in saved["thumbnails"]], [e.catalog_item.uuid for e in variant.elements])
        self.assertEqual([x["alt"] for x in saved["thumbnails"]], ["Pose 1", "Pose 2", "Pose 100", "Pose 101", "Pose 3", "Pose 4", "Pose 5"])
        self.assertEqual([x["order"] for x in saved["thumbnails"]], list(range(1, 8)))
        self.assertEqual([x["start_mirror"] for x in saved["thumbnails"]], [e.start_mirror for e in variant.elements])
        self.assertEqual([x["end_mirror"] for x in saved["thumbnails"]], [e.end_mirror for e in variant.elements])
        calls = len(self.model.received_messages)
        app.button(key="assistant-sequence-3-previous").click().run()
        self.assertEqual(service.active_sequence_uuid, str(source.uuid))
        self.assertEqual(self.save.call_args.kwargs["data"]["sequence"]["info"]["uuid"], str(source.uuid))
        app.button(key="assistant-sequence-3-next").click().run()
        app.session_state.messages[-1]["sequence_variants"] = []
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(service.active_sequence_uuid, str(variant.uuid))
        self.assertEqual(self.save.call_args.kwargs["data"]["sequence"], saved)
        self.assertEqual(len(self.model.received_messages), calls)
        self.generator.add_elements.assert_called_once()
        self.catalog.search.assert_not_called()
        self.assertEqual(service.get_sequence(str(source.uuid)), source)

    def test_addition_branches_from_older_selected_variant_and_use_selects_without_tools(self):
        app, service, source = self.app_with_source()
        self.add(app, source)
        newer = service.get_sequence(service.active_sequence_uuid)
        app.button(key="assistant-sequence-3-previous").click().run()
        self.assertEqual(service.active_sequence_uuid, str(source.uuid))
        self.generator.add_elements.return_value = [pose(102)]
        self.add(app, source, count=1)
        branch = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual((branch.parent_uuid, branch.family_uuid), (source.uuid, source.family_uuid))
        self.assertEqual(branch.elements_count, 6)
        self.assertEqual(service.get_sequence(str(newer.uuid)), newer)
        self.assertEqual(app.session_state.messages[-1]["sequence_variant_uuids"], [str(source.uuid), str(newer.uuid), str(branch.uuid)])
        self.assertEqual(selected_snapshot(self.model, -2)["uuid"], str(source.uuid))
        use = next(button for button in app.button if button.label == "Use this sequence")
        calls = len(self.model.received_messages)
        use.click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(self.model.received_messages), calls)
        self.assertEqual(self.generator.add_elements.call_count, 2)

    def test_regenerate_addition_reuses_original_generation_request(self):
        app, service, source = self.app_with_source()
        self.add(app, source)
        added = service.get_sequence(service.active_sequence_uuid)
        self.model.responses = [tool_model(source.generation_request, count=5).responses[0], AIMessage(content="Regenerated")]
        self.generator.generate.return_value = [pose(n) for n in range(10, 15)]
        app.button(key="assistant-sequence-3-regenerate").click().run()
        self.assertFalse(app.exception)
        result = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual((result.operation, result.parent_uuid, result.family_uuid), ("regenerate", added.uuid, source.family_uuid))
        self.assertEqual(result.generation_request, source.generation_request)
        self.assertEqual(result.elements_count, 5)
        self.generator.generate.assert_called_with(source.generation_request, language="en", count=5,
                                                  starting_catalog_item_id=None, progress_observer=ANY)
        self.generator.add_elements.assert_called_once()

    def test_failure_after_partial_selection_or_reached_limit_keeps_source_without_card(self):
        for mode in ("selection", "full"):
            with self.subTest(mode=mode):
                self.model.i = 0
                self.model.responses = tool_model(count=45 if mode == "full" else 5).responses
                self.generator.generate.return_value = [pose(n) for n in range(1, 46 if mode == "full" else 6)]
                app, service, source = self.app_with_source()
                selector = Mock()
                selector.choose_addition_catalog_item.side_effect = [pose(100), TypeSafeSelectionError("offline")]
                self.catalog.search.return_value = [pose(n) for n in range(100, 105)]
                service._generator_factory = lambda: SequenceGenerator(self.catalog, selector)
                self.add(app, source)
                message = app.session_state.messages[-1]
                self.assertIsNone(message["sequence"])
                self.assertIsNone(message["execution"].sequence_uuid)
                update = next(s for s in message["execution"].steps if s.code == "updating_sequence")
                self.assertEqual(update.outcome, "no_change" if mode == "full" else "error")
                self.assertEqual(service.active_sequence_uuid, str(source.uuid))
                self.assertEqual(len(service.state["sequences"]), 1)
                self.assertEqual(self.save.call_args.kwargs["data"]["sequence"]["info"]["uuid"], str(source.uuid))
                if mode == "selection":
                    self.assertEqual(update.progress[-1].completed, 1)
                    self.assertEqual(selector.choose_addition_catalog_item.call_count, 2)

    def test_committed_addition_and_save_survive_failure_of_final_text(self):
        model = ErrorAfterAdditionModel(responses=tool_model(count=5).responses)
        with patch("langchain_app.services.assistant_agent.init_chat_model", return_value=model):
            app = AppTest.from_file(str(APP)).run()
            app.chat_input[0].set_value("Generate a yoga sequence with 5 poses.").run()
            service = app.session_state.assistant_agent
            source = service.get_sequence(service.active_sequence_uuid)
            model.responses = [addition_call(source, count=2)]
            app.chat_input[0].set_value("Add two poses").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.error), 1)
            variant = service.get_sequence(service.active_sequence_uuid)
            message = app.session_state.messages[-1]
            self.assertEqual((variant.operation, variant.elements_count), ("add", 7))
            self.assertEqual(message["sequence_uuid"], str(variant.uuid))
            self.assertEqual(message["execution"].outcome, "error")
            saved = self.save.call_args.kwargs["data"]["sequence"]
            self.assertEqual(saved["info"]["uuid"], str(variant.uuid))
            calls = len(model.received_messages)
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(service.active_sequence_uuid, str(variant.uuid))
            self.assertEqual(self.save.call_args.kwargs["data"]["sequence"], saved)
            self.assertEqual(len(model.received_messages), calls)


if __name__ == "__main__":
    unittest.main()
