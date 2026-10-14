"""Exercise variant history and navigation without external generation calls."""

import json
import unittest
from pathlib import Path
from unittest.mock import ANY, Mock, call, patch
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from openai import APIConnectionError
import httpx
from pydantic import Field
from streamlit.testing.v1 import AppTest

from langchain_app.schemas import CatalogItemSchema
from langchain_app.services.sequence_generator import SequenceGenerator
from test_assistant_agent import last_message, tool_call
from test_sequence_replacement import catalog_pose, replacement_call
from test_sequence_swap import swap_call

APP = Path(__file__).resolve().parents[1] / "src/langchain_app/assistant.py"


class ToolCallingModel(FakeMessagesListChatModel):
    received_requests: list[str] = Field(default_factory=list)
    received_messages: list[list] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received_messages.append(list(messages))
        if isinstance(last_message(messages), HumanMessage):
            self.received_requests.append(last_message(messages).content)
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        for generation in result.generations:
            generation.message = generation.message.model_copy(deep=True)
            generation.message.id = str(uuid4())
            for tool in generation.message.tool_calls:
                tool["id"] = str(uuid4())
        return result


class ErrorAfterGenerationModel(ToolCallingModel):
    fail_on_tool_result: bool = True

    def _generate(self, messages, **kwargs):
        if self.fail_on_tool_result and isinstance(last_message(messages), ToolMessage):
            self.fail_on_tool_result = False
            raise APIConnectionError(request=httpx.Request("POST", "https://example.com"))
        return super()._generate(messages, **kwargs)


class ErrorAfterReplacementModel(ToolCallingModel):
    def _generate(self, messages, **kwargs):
        if isinstance(last_message(messages), ToolMessage) and last_message(messages).name == "replace_sequence_element":
            raise APIConnectionError(request=httpx.Request("POST", "https://example.com"))
        return super()._generate(messages, **kwargs)


class ErrorAfterSwapModel(ToolCallingModel):
    def _generate(self, messages, **kwargs):
        if isinstance(last_message(messages), ToolMessage) and last_message(messages).name == "swap_sequence_elements":
            raise APIConnectionError(request=httpx.Request("POST", "https://example.com"))
        return super()._generate(messages, **kwargs)


def selected_snapshot(model, call_index=-1):
    context = model.received_messages[call_index][-1].content_blocks[-1]["text"]
    return json.loads(next(line for line in context.splitlines() if line.startswith('{"selected_sequence":')))["selected_sequence"]


def tool_model(request="Generate a yoga sequence with 9 poses.", count=9):
    return ToolCallingModel(responses=[AIMessage(content="", tool_calls=[{
        "name": "generate_sequence", "args": {"request": request, "count": count},
        "id": "sequence-call", "type": "tool_call",
    }]), AIMessage(content="")])


def pose(number):
    return CatalogItemSchema(
        id=number, uuid=None, name=f"Pose {number}", bilateral=False,
        image_url=f"https://example.com/{number}.png", catalog_tag_names=[],
    )


class SequenceVariantTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.model = patch(
            "langchain_app.services.assistant_agent.init_chat_model",
            side_effect=lambda *args, **kwargs: tool_model(),
        )
        self.model.start()
        self.addCleanup(self.model.stop)

    def test_generate_clears_prompt_and_regenerate_reuses_request_after_count_changes(self):
        request = "Generate a yoga sequence with 12 poses."
        generator = Mock()
        generator.generate.side_effect = [[pose(1)], [pose(2)]]
        language_model = tool_model(request, count=12)
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator), patch(
            "langchain_app.services.assistant_agent.init_chat_model",
            return_value=language_model,
        ) as model:
            app = AppTest.from_file(str(APP)).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.text_area), 0)
            self.assertEqual(len(app.chat_input), 1)
            model.assert_not_called()
            generator.generate.assert_not_called()
            app.selectbox(key="sequence-elements-count").set_value(12)
            app.button(key="generate-sequence").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.chat_input(key="sequence-prompt").proto.value, "")
            self.assertTrue(app.chat_input(key="sequence-prompt").proto.set_value)
            self.assertEqual(app.session_state.messages[0]["content"], request)
            self.assertEqual(app.session_state.messages[1]["sequence_request"], request)
            app.selectbox(key="sequence-elements-count").set_value(24).run()
            app.button(key="assistant-sequence-1-regenerate").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.chat_input(key="sequence-prompt").proto.value, "")
            self.assertEqual(generator.generate.call_args_list, [
                call(request, language="en", count=12, starting_catalog_item_id=None, progress_observer=ANY),
                call(request, language="en", count=12, starting_catalog_item_id=None, progress_observer=ANY),
            ])
            self.assertEqual(language_model.received_requests, [request, request])

    def test_initial_chat_request_uses_model_count_instead_of_dropdown(self):
        request = "Сгенерируй вечернюю практику из 15 асан"
        generator = Mock()
        generator.generate.return_value = [pose(1)]
        language_model = tool_model(request, count=15)
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator), patch(
            "langchain_app.services.assistant_agent.init_chat_model", return_value=language_model,
        ):
            app = AppTest.from_file(str(APP)).run()
            self.assertEqual(app.selectbox(key="sequence-elements-count").value, 9)
            app.chat_input[0].set_value(request).run()
            self.assertFalse(app.exception)
            self.assertEqual(language_model.received_requests, [request])
            self.assertEqual(app.session_state.messages[-1]["sequence_request"], request)
            self.assertEqual(app.session_state.messages[-1]["sequence"]["thumbnails"][0]["alt"], "Pose 1")
            generator.generate.assert_called_once_with(request, language="en", count=15, starting_catalog_item_id=None, progress_observer=ANY)

    def test_twelve_pose_request_produces_twelve_thumbnails_and_caption(self):
        request = "Generate a yoga sequence with 12 poses."
        poses = [pose(number) for number in range(1, 16)]
        poses[0].catalog_tag_names = ["Other/Starting"]
        poses[-1].catalog_tag_names = ["Other/Finishing"]
        catalog = Mock()
        catalog.search.return_value = poses
        selector = Mock()
        selector.choose_next_catalog_item.side_effect = lambda candidates, **kwargs: candidates[0]
        generator = SequenceGenerator(catalog, selector)

        def get_generator(*args):
            from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
            self.assertIsNotNone(get_script_run_ctx(suppress_warning=True))
            return generator

        with patch("langchain_app.dependencies.get_sequence_generator", side_effect=get_generator), patch(
            "langchain_app.services.assistant_agent.init_chat_model",
            return_value=tool_model(request, count=12),
        ):
            app = AppTest.from_file(str(APP)).run()
            app.chat_input[0].set_value(request).run()
            self.assertFalse(app.exception)
            sequence = app.session_state.messages[-1]["sequence"]
            self.assertEqual(len(sequence["thumbnails"]), 12)
            self.assertEqual(sequence["info"]["elements_count"], 12)
            self.assertIn("Sequence #0 · 12 poses · EN", [caption.value for caption in app.caption])
            self.assertEqual(selector.choose_next_catalog_item.call_count, 11)

    def test_model_failure_is_displayed_without_saving_a_sequence(self):
        with patch(
            "langchain_app.services.assistant_agent.init_chat_model",
            return_value=ToolCallingModel(responses=[AIMessage(content="No tool call")]),
        ):
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertEqual(app.session_state.messages[0], {
                "role": "user", "content": "Generate a yoga sequence with 9 poses."
            })
            self.assertEqual(len(app.session_state.messages), 2)
            self.assertIsNone(app.session_state.messages[1]["sequence"])
            self.assertEqual(app.session_state.messages[1]["execution"].outcome, "error")

    def test_regenerate_appends_and_navigation_does_not_generate(self):
        generator = Mock()
        generator.generate.side_effect = [[pose(1)], [pose(2)], [pose(3)]]
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator):
            app = AppTest.from_file(str(APP)).run()
            self.assertEqual(len(app.chat_input), 1)
            app.button(key="generate-sequence").click().run()
            self.assertEqual(len(app.chat_input), 1)
            self.assertFalse(app.exception)
            app.button(key="assistant-sequence-1-regenerate").click().run()
            self.assertFalse(app.exception)
            message = app.session_state.messages[1]
            self.assertEqual(len(message["sequence_variants"]), 2)
            self.assertEqual(message["selected_sequence_index"], 1)
            app.button(key="assistant-sequence-1-previous").click().run()
            self.assertEqual(app.session_state.messages[1]["sequence"]["thumbnails"][0]["alt"], "Pose 1")
            app.button(key="assistant-sequence-1-regenerate").click().run()
            self.assertEqual(len(app.session_state.messages[1]["sequence_variants"]), 3)
            app.button(key="assistant-sequence-1-previous").click().run()
            app.button(key="assistant-sequence-1-next").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(generator.generate.call_args_list, [
                call("Generate a yoga sequence with 9 poses.", language="en", count=9, starting_catalog_item_id=None, progress_observer=ANY)
            ] * 3)

    def test_generation_error_keeps_existing_variant(self):
        generator = Mock()
        generator.generate.side_effect = [[pose(1)], ValueError("generation failed")]
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator):
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            app.button(key="assistant-sequence-1-regenerate").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            message = app.session_state.messages[1]
            self.assertEqual(len(message["sequence_variants"]), 1)
            self.assertEqual(message["selected_sequence_index"], 0)
            self.assertEqual(message["sequence"]["thumbnails"][0]["alt"], "Pose 1")

    def test_text_reply_in_user_language_is_rendered_and_saved_without_a_sequence(self):
        generator = Mock()
        reply = "Какую практику вы хотите?"
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator), patch(
            "langchain_app.services.assistant_agent.init_chat_model",
            return_value=ToolCallingModel(responses=[AIMessage(content=reply)]),
        ):
            app = AppTest.from_file(str(APP)).run()
            app.chat_input[0].set_value("Помоги выбрать практику").run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(app.session_state.messages[-1]["content"], reply)
            self.assertIsNone(app.session_state.messages[-1]["sequence"])
            self.assertIn(reply, [text.value for text in app.markdown])
            self.assertFalse(any("New Sequence" in text.value for text in app.markdown))
            generator.generate.assert_not_called()

    def test_navigation_changes_next_model_context_and_reruns_rebuild_projections(self):
        request = "Generate a yoga sequence with 9 poses."
        first_pose = pose(1)
        first_pose.uuid = str(uuid4())
        first_pose.catalog_tag_names = ["Standing"]
        generator = Mock()
        generator.generate.side_effect = [[first_pose], [pose(2)]]
        generation_call = tool_model(request).responses[0]
        model = ToolCallingModel(responses=[
            generation_call, AIMessage(content="First ready"),
            generation_call, AIMessage(content="Second ready"),
            AIMessage(content="The selected pose is Pose 1"),
        ])
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator), patch(
            "langchain_app.services.assistant_agent.init_chat_model", return_value=model,
        ), patch("langchain_app.ui.sequence.render_sequence_import") as save:
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            message = app.session_state.messages[-1]
            first_uuid = message["sequence_uuid"]
            app.button(key="assistant-sequence-1-regenerate").click().run()
            calls_before_navigation = len(model.received_messages)
            app.button(key="assistant-sequence-1-previous").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state.assistant_agent.active_sequence_uuid, first_uuid)
            self.assertEqual(len(model.received_messages), calls_before_navigation)
            # Stored thumbnails are projections, not the source of catalog context.
            app.session_state.messages[1]["sequence"]["thumbnails"][0]["alt"] = "Incorrect cached thumbnail"
            app.selectbox(key="sequence-elements-count").set_value(24).run()
            self.assertEqual(app.session_state.messages[1]["sequence"]["thumbnails"][0]["alt"], "Pose 1")
            self.assertEqual(app.session_state.assistant_agent.active_sequence_uuid, first_uuid)
            self.assertEqual(save.call_args.args[0]["info"]["uuid"], first_uuid)
            self.assertEqual(save.call_args.args[0]["thumbnails"][0]["uuid"], first_pose.uuid)
            app.chat_input[0].set_value("What is the first pose?").run()
            self.assertFalse(app.exception)
            self.assertEqual(selected_snapshot(model)["uuid"], first_uuid)
            self.assertEqual(selected_snapshot(model)["elements"][0]["catalog_item"]["catalog_tag_names"], ["Standing"])
            self.assertEqual(generator.generate.call_count, 2)
            self.assertIsNone(app.session_state.messages[-1]["sequence"])
            self.assertEqual(app.session_state.messages[-1]["content"], "The selected pose is Pose 1")

    def test_using_an_older_card_is_not_overridden_by_rendering_a_later_card(self):
        generator = Mock()
        generator.generate.side_effect = [[pose(1)], [pose(2)]]
        generation_call = tool_model().responses[0]
        model = ToolCallingModel(responses=[
            generation_call, AIMessage(content="Ready"),
            generation_call, AIMessage(content="Ready"),
            AIMessage(content="Pose 1 is selected"),
        ])
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator), patch(
            "langchain_app.services.assistant_agent.init_chat_model", return_value=model,
        ):
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            first_uuid = app.session_state.messages[1]["sequence_uuid"]
            app.button(key="generate-sequence").click().run()
            self.assertNotEqual(app.session_state.assistant_agent.active_sequence_uuid, first_uuid)
            calls_before_selection = len(model.received_messages)
            app.button(key="assistant-sequence-1-use").click().run()
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state.assistant_agent.active_sequence_uuid, first_uuid)
            self.assertEqual(len(model.received_messages), calls_before_selection)
            self.assertEqual([caption.value for caption in app.caption].count("Active sequence"), 1)
            app.chat_input[0].set_value("What is selected?").run()
            self.assertEqual(selected_snapshot(model)["uuid"], first_uuid)
            self.assertEqual(generator.generate.call_count, 2)

    def test_regenerate_older_card_uses_its_source_and_family(self):
        first_request = "Generate a yoga sequence with 9 poses."
        second_request = "Generate a yoga sequence with 12 poses."
        generator = Mock()
        generator.generate.side_effect = [[pose(1)], [pose(2)], [pose(3)]]
        model = ToolCallingModel(responses=[
            tool_model(first_request).responses[0], AIMessage(content="Ready"),
            tool_model(second_request, count=12).responses[0], AIMessage(content="Ready"),
            tool_model(first_request).responses[0], AIMessage(content="Ready"),
        ])
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator), patch(
            "langchain_app.services.assistant_agent.init_chat_model", return_value=model,
        ):
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            first_uuid = app.session_state.messages[1]["sequence_uuid"]
            service = app.session_state.assistant_agent
            first = service.get_sequence(first_uuid)
            app.selectbox(key="sequence-elements-count").set_value(12)
            app.button(key="generate-sequence").click().run()
            second_uuid = service.active_sequence_uuid
            app.button(key="assistant-sequence-1-regenerate").click().run()
            self.assertFalse(app.exception)
            regenerated = service.get_sequence(service.active_sequence_uuid)
            self.assertEqual(regenerated.parent_uuid, first.uuid)
            self.assertEqual(regenerated.family_uuid, first.family_uuid)
            self.assertNotEqual(regenerated.family_uuid, service.get_sequence(second_uuid).family_uuid)
            self.assertEqual(model.received_requests, [first_request, second_request, first_request])
            self.assertEqual(selected_snapshot(model, 4)["uuid"], first_uuid)
            self.assertEqual(len(service.state["sequences"]), 3)

    def test_committed_snapshot_is_shown_even_when_model_reply_fails(self):
        generator = Mock()
        generator.generate.return_value = [pose(1)]
        model = ErrorAfterGenerationModel(responses=[
            tool_model().responses[0], AIMessage(content="Pose 1 remains selected"),
        ])
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator), patch(
            "langchain_app.services.assistant_agent.init_chat_model", return_value=model,
        ):
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state.messages), 2)
            sequence_uuid = app.session_state.messages[-1]["sequence_uuid"]
            self.assertEqual(app.session_state.assistant_agent.active_sequence_uuid, sequence_uuid)
            self.assertEqual(app.session_state.messages[-1]["sequence"]["thumbnails"][0]["alt"], "Pose 1")
            app.chat_input[0].set_value("What is selected?").run()
            self.assertFalse(app.exception)
            self.assertEqual(selected_snapshot(model)["uuid"], sequence_uuid)
            generator.generate.assert_called_once()

    def test_streamlit_sessions_do_not_share_assistant_state(self):
        generator = Mock()
        generator.generate.return_value = [pose(1)]
        first_model = tool_model()
        second_model = ToolCallingModel(responses=[AIMessage(content="No selected sequence")])
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator), patch(
            "langchain_app.services.assistant_agent.init_chat_model", side_effect=[first_model, second_model],
        ):
            first_app = AppTest.from_file(str(APP)).run()
            first_app.button(key="generate-sequence").click().run()
            second_app = AppTest.from_file(str(APP)).run()
            second_app.chat_input[0].set_value("What is selected?").run()
            self.assertFalse(first_app.exception)
            self.assertFalse(second_app.exception)
            self.assertIsNot(first_app.session_state.assistant_agent, second_app.session_state.assistant_agent)
            self.assertIsNone(second_app.session_state.assistant_agent.active_sequence_uuid)
            self.assertIsNone(selected_snapshot(second_model))
            self.assertEqual(len(first_app.session_state.assistant_agent.state["sequences"]), 1)

    def test_failed_regenerate_of_an_older_card_keeps_active_sequence_and_history(self):
        def fail():
            yield pose(3)
            raise ValueError("Partial generation failed")

        generator = Mock()
        generator.generate.side_effect = [[pose(1)], [pose(2)], fail()]
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator):
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            app.button(key="generate-sequence").click().run()
            service = app.session_state.assistant_agent
            active_before = service.active_sequence_uuid
            snapshots_before = service.state["sequences"]
            app.button(key="assistant-sequence-1-regenerate").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertEqual(service.active_sequence_uuid, active_before)
            self.assertEqual(service.state["sequences"], snapshots_before)
            self.assertEqual(len(app.session_state.messages[1]["sequence_variants"]), 1)

    def test_replacement_of_selected_older_variant_has_history_save_and_original_regenerate_request(self):
        request = "Generate a yoga sequence with 9 poses."
        generator = Mock()
        generator.generate.side_effect = [
            [catalog_pose(n) for n in (1, 2, 3)],
            [catalog_pose(n) for n in (4, 5, 6)],
            [catalog_pose(n) for n in (7, 8, 10)],
        ]
        catalog = Mock()
        candidate = catalog_pose(9, bilateral=True)
        catalog.search.return_value = [candidate]
        model = tool_model(request)
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator), patch(
            "langchain_app.dependencies.get_catalog_item_service", return_value=catalog,
        ), patch("langchain_app.services.assistant_agent.init_chat_model", return_value=model), patch(
            "langchain_app.ui.sequence.render_sequence_import",
        ) as save:
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            service = app.session_state.assistant_agent
            source = service.get_sequence(service.active_sequence_uuid)
            app.button(key="assistant-sequence-1-regenerate").click().run()
            regenerated_uuid = service.active_sequence_uuid
            app.button(key="assistant-sequence-1-previous").click().run()
            self.assertEqual(service.active_sequence_uuid, str(source.uuid))
            model.responses = [
                tool_call("search_catalog_items", name="Tree"), replacement_call(source),
                AIMessage(content="Заменена Pose 3 на Pose 9."),
            ]
            app.chat_input[0].set_value("Замени третью на дерево").run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            message = app.session_state.messages[-1]
            variant = service.get_sequence(service.active_sequence_uuid)
            variant_uuid = str(variant.uuid)
            self.assertEqual(variant.parent_uuid, source.uuid)
            self.assertEqual(message["sequence_variant_uuids"], [str(source.uuid), regenerated_uuid, variant_uuid])
            self.assertEqual(message["selected_sequence_index"], 2)
            self.assertEqual(message["sequence_uuid"], variant_uuid)
            self.assertEqual(message["sequence_request"], request)
            self.assertEqual(message["sequence"]["thumbnails"][2]["alt"], "Pose 9")
            self.assertEqual(service.get_sequence(str(source.uuid)), source)
            self.assertEqual(generator.generate.call_count, 2)
            self.assertEqual(message["content"], "Заменена Pose 3 на Pose 9.")
            saved = save.call_args.args[0]
            self.assertEqual(saved["info"]["uuid"], variant_uuid)
            self.assertEqual(saved["thumbnails"][2]["uuid"], candidate.uuid)
            self.assertTrue(saved["thumbnails"][2]["start_mirror"])
            self.assertTrue(saved["thumbnails"][2]["end_mirror"])
            self.assertEqual([item["order"] for item in saved["thumbnails"]], [1, 2, 3])
            calls_before_navigation = len(model.received_messages)
            app.button(key="assistant-sequence-3-previous").click().run()
            app.button(key="assistant-sequence-3-previous").click().run()
            self.assertEqual(service.active_sequence_uuid, str(source.uuid))
            self.assertEqual(app.session_state.messages[-1]["sequence"]["thumbnails"][2]["alt"], "Pose 3")
            self.assertEqual(save.call_args.args[0]["info"]["uuid"], str(source.uuid))
            app.button(key="assistant-sequence-3-next").click().run()
            app.button(key="assistant-sequence-3-next").click().run()
            app.session_state.messages[-1]["sequence_variants"] = []
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state.messages[-1]["sequence_uuid"], variant_uuid)
            self.assertEqual(save.call_args.args[0]["info"]["uuid"], variant_uuid)
            self.assertEqual(len(model.received_messages), calls_before_navigation)
            self.assertEqual(len(service.state["sequences"]), 3)
            model.responses = [tool_model(request).responses[0], AIMessage(content="Regenerated")]
            app.button(key="assistant-sequence-3-regenerate").click().run()
            self.assertFalse(app.exception)
            regenerated = service.get_sequence(service.active_sequence_uuid)
            self.assertEqual(regenerated.parent_uuid, variant.uuid)
            self.assertEqual(regenerated.family_uuid, source.family_uuid)
            self.assertEqual(regenerated.generation_request, request)
            generator.generate.assert_called_with(request, language="en", count=9, starting_catalog_item_id=None, progress_observer=ANY)

    def test_committed_replacement_is_retained_in_ui_when_text_reply_fails(self):
        generator = Mock()
        generator.generate.return_value = [catalog_pose(n) for n in (1, 2, 3)]
        catalog = Mock()
        catalog.search.return_value = [catalog_pose(9)]
        model = ErrorAfterReplacementModel(responses=tool_model().responses)
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator), patch(
            "langchain_app.dependencies.get_catalog_item_service", return_value=catalog,
        ), patch("langchain_app.services.assistant_agent.init_chat_model", return_value=model):
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            service = app.session_state.assistant_agent
            source = service.get_sequence(service.active_sequence_uuid)
            model.responses = [tool_call("search_catalog_items", name="Tree"), replacement_call(source)]
            app.chat_input[0].set_value("Replace third").run()
            self.assertFalse(app.exception)
            variant_uuid = service.active_sequence_uuid
            self.assertNotEqual(variant_uuid, str(source.uuid))
            self.assertEqual(app.session_state.messages[-1]["sequence_uuid"], variant_uuid)
            self.assertEqual(app.session_state.messages[-1]["sequence"]["thumbnails"][2]["alt"], "Pose 9")
            self.assertEqual(app.session_state.messages[-1]["sequence_variant_uuids"], [str(source.uuid), variant_uuid])
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(service.active_sequence_uuid, variant_uuid)
            self.assertEqual(app.session_state.messages[-1]["sequence_uuid"], variant_uuid)
            self.assertEqual(len(service.state["sequences"]), 2)
            generator.generate.assert_called_once()

    def test_no_change_replacement_does_not_create_a_variant_or_sequence_card(self):
        generator = Mock()
        generator.generate.return_value = [catalog_pose(n) for n in (1, 2, 3)]
        catalog = Mock()
        model = tool_model()
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator), patch(
            "langchain_app.dependencies.get_catalog_item_service", return_value=catalog,
        ), patch("langchain_app.services.assistant_agent.init_chat_model", return_value=model):
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            service = app.session_state.assistant_agent
            source = service.get_sequence(service.active_sequence_uuid)
            catalog.search.return_value = [source.elements[2].catalog_item]
            model.responses = [
                tool_call("search_catalog_items", name="Pose 3"), replacement_call(source, 3),
                AIMessage(content="Pose 3 is already selected."),
            ]
            app.chat_input[0].set_value("Replace third with Pose 3").run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state.messages[-1]["sequence"])
            self.assertEqual(app.session_state.messages[-1]["content"], "Pose 3 is already selected.")
            execution = app.session_state.messages[-1]["execution"]
            self.assertEqual(execution.outcome, "complete")
            self.assertIsNone(execution.sequence_uuid)
            update = next(step for step in execution.steps if step.code == "updating_sequence")
            self.assertEqual(update.outcome, "no_change")
            self.assertEqual(len(service.state["sequences"]), 1)
            self.assertEqual(service.active_sequence_uuid, str(source.uuid))

    def test_swap_card_replays_progress_history_and_save_with_moved_mirror_flags(self):
        generator = Mock()
        generator.generate.return_value = [catalog_pose(n, bilateral=n == 5) for n in range(1, 6)]
        model = tool_model()
        save = Mock()
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator), patch(
            "langchain_app.services.assistant_agent.init_chat_model", return_value=model,
        ), patch("langchain_app.ui.sequence_import._save_component", return_value=save):
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            service = app.session_state.assistant_agent
            source = service.get_sequence(service.active_sequence_uuid)
            model.responses = [swap_call(source), AIMessage(content="Поменяла третью и пятую позы местами.")]
            app.chat_input[0].set_value("Поменяй третью и пятую местами").run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            variant = service.get_sequence(service.active_sequence_uuid)
            variant_uuid = str(variant.uuid)
            message = app.session_state.messages[-1]
            execution = message["execution"]
            self.assertEqual(execution.sequence_uuid, variant.uuid)
            self.assertEqual(execution.source_sequence_uuid, source.uuid)
            update = next(step for step in execution.steps if step.code == "updating_sequence")
            self.assertEqual((update.title, update.outcome, update.summary), (
                "Swapping sequence elements...", "ok", "Sequence elements swapped.",
            ))
            self.assertTrue(any("Sequence elements swapped." in element.value for element in app.markdown))
            self.assertEqual(message["sequence_uuid"], variant_uuid)
            self.assertEqual(message["sequence_variant_uuids"], [str(source.uuid), variant_uuid])
            self.assertEqual(message["selected_sequence_index"], 1)
            self.assertEqual(message["content"], "Поменяла третью и пятую позы местами.")
            self.assertEqual(message["sequence_request"], source.generation_request)
            saved = save.call_args.kwargs["data"]["sequence"]
            self.assertEqual(saved, message["sequence"])
            self.assertEqual(saved["info"]["uuid"], variant_uuid)
            moved = [source.elements[index] for index in (0, 1, 4, 3, 2)]
            self.assertEqual([item["alt"] for item in saved["thumbnails"]], [element.catalog_item.name for element in moved])
            self.assertEqual([item["uuid"] for item in saved["thumbnails"]], [element.catalog_item.uuid for element in moved])
            self.assertEqual([item["order"] for item in saved["thumbnails"]], [1, 2, 3, 4, 5])
            self.assertEqual([item["start_mirror"] for item in saved["thumbnails"]], [element.start_mirror for element in moved])
            self.assertEqual([item["end_mirror"] for item in saved["thumbnails"]], [element.end_mirror for element in moved])
            calls_before_navigation = len(model.received_messages)
            app.button(key="assistant-sequence-3-previous").click().run()
            self.assertEqual(service.active_sequence_uuid, str(source.uuid))
            self.assertEqual(save.call_args.kwargs["data"]["sequence"]["info"]["uuid"], str(source.uuid))
            self.assertEqual([item["alt"] for item in app.session_state.messages[-1]["sequence"]["thumbnails"]],
                             [element.catalog_item.name for element in source.elements])
            app.button(key="assistant-sequence-3-next").click().run()
            app.session_state.messages[-1]["sequence_variants"] = []
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state.messages[-1]["sequence_uuid"], variant_uuid)
            self.assertEqual(app.session_state.messages[-1]["execution"], execution)
            self.assertEqual(save.call_args.kwargs["data"]["sequence"], saved)
            self.assertEqual(len(model.received_messages), calls_before_navigation)
            self.assertEqual(service.get_sequence(str(source.uuid)), source)
            self.assertEqual(len(service.state["sequences"]), 2)
            generator.generate.assert_called_once()

    def test_regenerate_swap_uses_original_request_without_repeating_the_swap(self):
        request = "Generate a yoga sequence with 5 poses for balance."
        generator = Mock()
        generator.generate.side_effect = [
            [catalog_pose(n, bilateral=n == 5) for n in range(1, 6)],
            [catalog_pose(n) for n in range(6, 11)],
        ]
        model = tool_model(request, count=5)
        save = Mock()
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator), patch(
            "langchain_app.services.assistant_agent.init_chat_model", return_value=model,
        ), patch("langchain_app.ui.sequence_import._save_component", return_value=save):
            app = AppTest.from_file(str(APP)).run()
            app.chat_input[0].set_value(request).run()
            service = app.session_state.assistant_agent
            source = service.get_sequence(service.active_sequence_uuid)
            model.responses = [swap_call(source), AIMessage(content="Поменяла третью и пятую позы местами.")]
            app.chat_input[0].set_value("Поменяй третью и пятую местами").run()
            self.assertFalse(app.exception)
            swapped = service.get_sequence(service.active_sequence_uuid)
            swapped_message = app.session_state.messages[-1]
            swapped_execution = swapped_message["execution"].model_copy(deep=True)
            model.responses = [tool_model(request, count=5).responses[0], AIMessage(content="Regenerated")]
            app.button(key="assistant-sequence-3-regenerate").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            regenerated = service.get_sequence(service.active_sequence_uuid)
            self.assertEqual(regenerated.operation, "regenerate")
            self.assertEqual(regenerated.parent_uuid, swapped.uuid)
            self.assertEqual(regenerated.family_uuid, source.family_uuid)
            self.assertEqual(regenerated.generation_request, request)
            self.assertEqual(regenerated.operation_request, request)
            self.assertEqual(model.received_requests, [request, "Поменяй третью и пятую местами", request])
            self.assertEqual(selected_snapshot(model, -2)["uuid"], str(swapped.uuid))
            self.assertEqual([element.catalog_item.id for element in regenerated.elements], list(range(6, 11)))
            self.assertEqual(service.get_sequence(str(source.uuid)), source)
            self.assertEqual(service.get_sequence(str(swapped.uuid)), swapped)
            self.assertEqual(len(service.state["sequences"]), 3)
            self.assertEqual(swapped_message["execution_history"], [swapped_execution])
            self.assertEqual(swapped_message["sequence_variant_uuids"], [
                str(source.uuid), str(swapped.uuid), str(regenerated.uuid),
            ])
            saved = save.call_args.kwargs["data"]["sequence"]
            self.assertEqual(saved["info"]["uuid"], str(regenerated.uuid))
            self.assertEqual([item["alt"] for item in saved["thumbnails"]], [f"Pose {n}" for n in range(6, 11)])
            generator.generate.assert_called_with(
                request, language="en", count=5, starting_catalog_item_id=None, progress_observer=ANY,
            )
            self.assertEqual(generator.generate.call_count, 2)
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(service.active_sequence_uuid, str(regenerated.uuid))
            self.assertEqual(len(model.received_requests), 3)

    def test_committed_swap_retains_progress_card_and_save_when_final_reply_fails(self):
        generator = Mock()
        generator.generate.return_value = [catalog_pose(n) for n in range(1, 6)]
        model = ErrorAfterSwapModel(responses=tool_model().responses)
        save = Mock()
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator), patch(
            "langchain_app.services.assistant_agent.init_chat_model", return_value=model,
        ), patch("langchain_app.ui.sequence_import._save_component", return_value=save):
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            service = app.session_state.assistant_agent
            source = service.get_sequence(service.active_sequence_uuid)
            model.responses = [swap_call(source)]
            app.chat_input[0].set_value("Swap third and fifth").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.error), 1)
            variant_uuid = service.active_sequence_uuid
            self.assertNotEqual(variant_uuid, str(source.uuid))
            message = app.session_state.messages[-1]
            execution = message["execution"]
            self.assertEqual(execution.outcome, "error")
            self.assertEqual(str(execution.sequence_uuid), variant_uuid)
            update = next(step for step in execution.steps if step.code == "updating_sequence")
            self.assertEqual((update.outcome, update.summary), ("ok", "Sequence elements swapped."))
            self.assertEqual(message["sequence_uuid"], variant_uuid)
            saved = save.call_args.kwargs["data"]["sequence"]
            self.assertEqual(saved["info"]["uuid"], variant_uuid)
            self.assertEqual([item["alt"] for item in saved["thumbnails"]], ["Pose 1", "Pose 2", "Pose 5", "Pose 4", "Pose 3"])
            calls_before_rerun = len(model.received_messages)
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.error), 1)
            self.assertEqual(app.session_state.messages[-1]["execution"], execution)
            self.assertEqual(service.active_sequence_uuid, variant_uuid)
            self.assertEqual(save.call_args.kwargs["data"]["sequence"], saved)
            self.assertEqual(len(model.received_messages), calls_before_rerun)
            self.assertEqual(len(service.state["sequences"]), 2)
            generator.generate.assert_called_once()

    def test_no_change_and_failed_swaps_replay_progress_without_a_new_sequence_card(self):
        for outcome in ("no_change", "error"):
            with self.subTest(outcome=outcome):
                generator = Mock()
                generator.generate.return_value = [catalog_pose(n) for n in range(1, 6)]
                model = tool_model()
                with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator), patch(
                    "langchain_app.services.assistant_agent.init_chat_model", return_value=model,
                ):
                    app = AppTest.from_file(str(APP)).run()
                    app.button(key="generate-sequence").click().run()
                    service = app.session_state.assistant_agent
                    source = service.get_sequence(service.active_sequence_uuid)
                    second_id = str(source.elements[2].element_id) if outcome == "no_change" else str(uuid4())
                    model.responses = [swap_call(source, second_element_id=second_id), AIMessage(content="No swap applied.")]
                    app.chat_input[0].set_value("Swap the pair").run()
                    self.assertFalse(app.exception)
                    self.assertFalse(app.error)
                    message = app.session_state.messages[-1]
                    self.assertIsNone(message["sequence"])
                    self.assertIsNone(message["sequence_uuid"])
                    self.assertEqual(message["sequence_variant_uuids"], [])
                    execution = message["execution"]
                    self.assertEqual(execution.outcome, "complete")
                    self.assertIsNone(execution.sequence_uuid)
                    update = next(step for step in execution.steps if step.code == "updating_sequence")
                    self.assertEqual(update.title, "Swapping sequence elements...")
                    self.assertEqual(update.outcome, outcome)
                    if outcome == "no_change":
                        self.assertEqual(update.summary, "Both references identify the same sequence element; no change was needed.")
                    calls_before_rerun = len(model.received_messages)
                    app.run()
                    self.assertFalse(app.exception)
                    self.assertEqual(app.session_state.messages[-1]["execution"], execution)
                    self.assertIsNone(app.session_state.messages[-1]["sequence"])
                    self.assertEqual(service.active_sequence_uuid, str(source.uuid))
                    self.assertEqual(len(service.state["sequences"]), 1)
                    self.assertEqual(len(model.received_messages), calls_before_rerun)
                    generator.generate.assert_called_once()
