"""Exercise variant history and navigation without external generation calls."""

import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import Field
from streamlit.testing.v1 import AppTest

from langchain_app.schemas import CatalogItemSchema
from langchain_app.services.sequence_generator import SequenceGenerator

APP = Path(__file__).resolve().parents[1] / "src/langchain_app/generate_sequence.py"


class ToolCallingModel(FakeMessagesListChatModel):
    received_requests: list[str] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received_requests.append(messages[-1].content)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def tool_model(request="Generate a yoga sequence with 9 poses.", count=9):
    return ToolCallingModel(responses=[AIMessage(content="", tool_calls=[{
        "name": "generate_sequence", "args": {"request": request, "count": count},
        "id": "sequence-call", "type": "tool_call",
    }])])


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
            "langchain_app.services.sequence_agent.init_chat_model",
            side_effect=lambda *args, **kwargs: tool_model(),
        )
        self.model.start()
        self.addCleanup(self.model.stop)

    def test_generate_fills_prompt_and_regenerate_reuses_it_after_count_changes(self):
        request = "Generate a yoga sequence with 12 poses."
        generator = Mock()
        generator.generate.side_effect = [[pose(1)], [pose(2)]]
        language_model = tool_model(request, count=12)
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator), patch(
            "langchain_app.services.sequence_agent.init_chat_model",
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
            self.assertEqual(app.chat_input(key="sequence-prompt").proto.value, request)
            self.assertTrue(app.chat_input(key="sequence-prompt").proto.set_value)
            self.assertEqual(app.session_state.messages[0]["content"], request)
            self.assertEqual(app.session_state.messages[1]["sequence_request"], request)
            app.selectbox(key="sequence-elements-count").set_value(24).run()
            app.button(key="assistant-sequence-1-regenerate").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(generator.generate.call_args_list, [
                call(request, language="en", count=12),
                call(request, language="en", count=12),
            ])
            self.assertEqual(language_model.received_requests, [request, request])

    def test_initial_chat_request_uses_model_count_instead_of_dropdown(self):
        request = "Сгенерируй вечернюю практику из 15 асан"
        generator = Mock()
        generator.generate.return_value = [pose(1)]
        language_model = tool_model(request, count=15)
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator), patch(
            "langchain_app.services.sequence_agent.init_chat_model", return_value=language_model,
        ):
            app = AppTest.from_file(str(APP)).run()
            self.assertEqual(app.selectbox(key="sequence-elements-count").value, 9)
            app.chat_input[0].set_value(request).run()
            self.assertFalse(app.exception)
            self.assertEqual(language_model.received_requests, [request])
            self.assertEqual(app.session_state.messages[-1]["sequence_request"], request)
            self.assertEqual(app.session_state.messages[-1]["sequence"]["thumbnails"][0]["alt"], "Pose 1")
            generator.generate.assert_called_once_with(request, language="en", count=15)

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
            "langchain_app.services.sequence_agent.init_chat_model",
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
            "langchain_app.services.sequence_agent.init_chat_model",
            return_value=ToolCallingModel(responses=[AIMessage(content="No tool call")]),
        ):
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertEqual(app.session_state.messages, [{
                "role": "user", "content": "Generate a yoga sequence with 9 poses."
            }])

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
                call("Generate a yoga sequence with 9 poses.", language="en", count=9)
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
