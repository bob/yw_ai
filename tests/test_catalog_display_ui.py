"""Feature 003 acceptance with scripted models, real tools, and Streamlit reruns."""

import unittest
from copy import deepcopy
from unittest.mock import Mock, patch
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from streamlit.testing.v1 import AppTest

from langchain_app.schemas.response_progress import (
    CatalogItemsDisplayedEvent, RequestFinishedEvent, RequestStartedEvent,
)
from langchain_app.services.catalog_item import CatalogItemApiError
from langchain_app.ui.catalog_items import CatalogItemsRenderer
from test_assistant_agent import last_message, ToolCallingModel, pose, tool_call, tool_model
from test_catalog_display_tool import display_call
from test_sequence_replacement import catalog_pose, replacement_call
from test_sequence_variants import APP


def catalog_html(app):
    return [node.proto.body for node in app.get("html") if '<div class="mirror-thumbnail">' in node.proto.body]


class CatalogDisplayUITests(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        env.start()
        self.addCleanup(env.stop)

    def start_app(self, model, catalog=None, generator=None):
        catalog = catalog or Mock()
        generator = generator or Mock()
        self.enterContext(patch("langchain_app.services.assistant_agent.init_chat_model", return_value=model))
        self.enterContext(patch("langchain_app.dependencies.get_catalog_item_service", return_value=catalog))
        tags = Mock()
        tags.get_names.return_value = ["Standing"]
        self.enterContext(patch("langchain_app.dependencies.get_catalog_tag_service", return_value=tags))
        self.enterContext(patch("langchain_app.dependencies.get_sequence_generator", return_value=generator))
        return AppTest.from_file(str(APP)).run(), generator

    def assert_complete(self, app, outcome="complete"):
        self.assertFalse(app.exception)
        message = app.session_state.messages[-1]
        self.assertEqual(message["execution"].outcome, outcome)
        self.assertIsNone(app.session_state.pending_response)
        self.assertFalse(app.chat_input[0].disabled)
        return message

    def test_browsing_and_available_pose_question_show_only_selected_latest_results(self):
        for request in ("Покажи асаны сидя в наклоне", "Какие есть позы лежа"):
            with self.subTest(request=request):
                items = [pose(2), pose(3), pose(4)]
                items[1].bilateral = True
                catalog = Mock()
                catalog.search.side_effect = [[pose(1)], items]
                model = ToolCallingModel(responses=[
                    tool_call("search_catalog_items", name="First"),
                    tool_call("search_catalog_items", name="Second"), display_call(3, 2),
                    AIMessage(content="Вот подходящие варианты."),
                ])
                app, factory = self.start_app(model, catalog)
                app.chat_input[0].set_value(request).run()
                message = self.assert_complete(app)
                self.assertEqual(message["catalog_items"], [items[1], items[0]])
                self.assertIsNone(message["sequence"])
                self.assertEqual([step.code for step in message["execution"].steps].count("showing_catalog_items"), 1)
                self.assertEqual(len(catalog_html(app)), 2)
                self.assertTrue(all("bracket" not in body for body in catalog_html(app)))
                self.assertEqual([button.label for button in app.button], ["Generate"])
                self.assertEqual([caption.value for caption in app.caption], [])
                self.assertFalse(app.get("status")[-1].get("html"))
                public = app.chat_message[-1]
                self.assertEqual([node.type for node in public.children.values()], ["status", "flex_container", "markdown"])
                self.assertEqual(model.received_requests, [request])
                # Instructions reach the model; routing itself is scripted offline.
                system = str(model.received_messages[0][0].content)
                for instruction in ("show_catalog_items", "The word 'show'", "do not change anything yet"):
                    self.assertIn(instruction, system)
                factory.generate.assert_not_called()

    def test_more_options_and_later_search_preserve_each_message_and_rerun_uses_snapshots(self):
        first, second = pose(1), pose(2)
        first.catalog_tag_names = ["Original"]
        catalog = Mock()
        catalog.search.side_effect = [[first], [second], [pose(3)]]
        model = ToolCallingModel(responses=[
            tool_call("search_catalog_items"), display_call(1), AIMessage(content="Первая подборка."),
            tool_call("search_catalog_items"), display_call(2),
            tool_call("search_catalog_items"), display_call(3), AIMessage(content="Ещё варианты."),
        ])
        app, factory = self.start_app(model, catalog)
        app.chat_input[0].set_value("Покажи варианты").run()
        original = deepcopy(app.session_state.messages[-1]["catalog_items"])
        app.chat_input[0].set_value("Покажи ещё варианты").run()
        message = self.assert_complete(app)
        self.assertEqual(message["catalog_items"], [second])
        shows = [step for step in message["execution"].steps if step.code == "showing_catalog_items"]
        self.assertEqual([step.outcome for step in shows], ["ok", "error"])
        self.assertEqual(app.session_state.messages[1]["catalog_items"], original)
        first.name = "Changed service data"
        first.catalog_tag_names.append("Changed service tags")
        calls = len(model.received_messages)
        html = catalog_html(app)
        app.run()
        self.assert_complete(app)
        self.assertEqual(catalog_html(app), html)
        self.assertEqual(app.session_state.messages[1]["catalog_items"], original)
        self.assertEqual(len(model.received_messages), calls)
        self.assertEqual(catalog.search.call_count, 3)
        factory.generate.assert_not_called()
        self.assertNotIn('"catalog_items":', str(model.received_messages[-1][0].content))

    def test_candidates_without_changes_preserve_selected_variant_and_history(self):
        generator, catalog = Mock(), Mock()
        generator.generate.return_value = [catalog_pose(1), catalog_pose(2), catalog_pose(3)]
        catalog.search.return_value = [catalog_pose(9, bilateral=True)]
        model = tool_model("Practice", count=3)
        app, _ = self.start_app(model, catalog, generator)
        app.button(key="generate-sequence").click().run()
        agent = app.session_state.assistant_agent
        before = agent.state["sequences"]
        selected = agent.active_sequence_uuid
        card = deepcopy(app.session_state.messages[-1])
        model.responses = [tool_call("search_catalog_items"), display_call(9), AIMessage(content="Кандидаты для замены.")]
        app.chat_input[0].set_value("Найди кандидатов для замены, пока ничего не меняй").run()
        message = self.assert_complete(app)
        self.assertEqual(message["catalog_items"], catalog.search.return_value)
        self.assertIsNone(message["sequence"])
        self.assertEqual(agent.active_sequence_uuid, selected)
        self.assertEqual(agent.state["sequences"], before)
        self.assertFalse(agent.state["turn_operation_committed"])
        self.assertEqual(app.session_state.messages[1], card)
        self.assertEqual(generator.generate.call_count, 1)
        self.assertNotIn("replacing_sequence_element", [step.code for step in message["execution"].steps])

    def test_ordinary_answers_tags_and_search_alone_do_not_create_gallery(self):
        cases = [
            ("Как выполнять наклон сидя?", []),
            ("Какие есть теги?", [tool_call("get_catalog_tag_names")]),
            ("Справка по позе", [tool_call("search_catalog_items")]),
        ]
        for request, calls in cases:
            with self.subTest(request=request):
                catalog = Mock()
                catalog.search.return_value = [pose(1)]
                model = ToolCallingModel(responses=[*calls, AIMessage(content="Ответ.")])
                app, _ = self.start_app(model, catalog)
                app.chat_input[0].set_value(request).run()
                self.assertNotIn("catalog_items", self.assert_complete(app))
                self.assertFalse(catalog_html(app))

    def test_empty_failed_and_invalid_display_keep_gallery_absent(self):
        for failure in ("empty", "search_error", "unknown", "duplicates", "empty_ids"):
            with self.subTest(failure=failure):
                catalog = Mock()
                catalog.search.return_value = [] if failure == "empty" else [pose(1)]
                if failure == "search_error":
                    catalog.search.side_effect = CatalogItemApiError("PRIVATE_API_ERROR")
                ids = {"unknown": [99], "duplicates": [1, 1], "empty_ids": []}.get(failure, [1])
                model = ToolCallingModel(responses=[
                    tool_call("search_catalog_items"), display_call(*ids), AIMessage(content="Не удалось показать подборку."),
                ])
                app, _ = self.start_app(model, catalog)
                app.chat_input[0].set_value("Покажи варианты").run()
                self.assertNotIn("catalog_items", self.assert_complete(app))
                self.assertFalse(catalog_html(app))

    def test_model_failure_preserves_only_successfully_displayed_items_after_rerun(self):
        for shown in (False, True):
            with self.subTest(shown=shown):
                catalog = Mock()
                catalog.search.return_value = [pose(1)]
                calls = [tool_call("search_catalog_items")]
                if shown:
                    calls.append(display_call(1))
                model = ToolCallingModel(responses=calls)
                original_generate = model._generate

                def fail_on_final(messages, **kwargs):
                    if len(model.received_messages) == len(calls):
                        raise ValueError("PRIVATE_MODEL_ERROR")
                    return original_generate(messages, **kwargs)

                with patch.object(ToolCallingModel, "_generate", side_effect=fail_on_final):
                    app, _ = self.start_app(model, catalog)
                    app.chat_input[0].set_value("Покажи варианты").run()
                message = self.assert_complete(app, "error")
                self.assertEqual(bool(message.get("catalog_items")), shown)
                self.assertEqual(len(catalog_html(app)), int(shown))
                self.assertEqual(message["content"], "")
                self.assertNotIn("PRIVATE_MODEL_ERROR", app.error[0].value)
                app.run()
                self.assert_complete(app, "error")
                self.assertEqual(len(catalog_html(app)), int(shown))
                self.assertEqual(len(model.received_messages), len(calls))
                catalog.search.assert_called_once()

    def test_display_is_rendered_before_followup_model_and_needs_no_final_text(self):
        catalog = Mock()
        catalog.search.return_value = [pose(1)]
        model = ToolCallingModel(responses=[tool_call("search_catalog_items"), display_call(1), AIMessage(content="")])
        generate = model._generate
        rendered = []
        render = CatalogItemsRenderer.render

        def record_render(renderer, items, **kwargs):
            rendered.append(list(items))
            return render(renderer, rendered[-1], **kwargs)

        def check_live(messages, **kwargs):
            if isinstance(last_message(messages), ToolMessage) and last_message(messages).artifact:
                self.assertEqual(rendered, [[pose(1)]])
            return generate(messages, **kwargs)

        with patch.object(CatalogItemsRenderer, "render", autospec=True, side_effect=record_render), patch.object(
            ToolCallingModel, "_generate", side_effect=check_live,
        ):
            app, _ = self.start_app(model, catalog)
            app.chat_input[0].set_value("Покажи варианты").run()
        message = self.assert_complete(app)
        self.assertEqual(message["catalog_items"], [pose(1)])
        self.assertEqual(message["content"], "")
        self.assertEqual(len(catalog_html(app)), 1)

    def test_generation_regeneration_and_replacement_have_no_catalog_gallery(self):
        save = self.enterContext(patch("langchain_app.ui.sequence.render_sequence_import"))
        generator, catalog = Mock(), Mock()
        generator.generate.return_value = [catalog_pose(1, bilateral=True), catalog_pose(2), catalog_pose(3)]
        catalog.search.return_value = [catalog_pose(9, bilateral=True)]
        model = tool_model("Сгенерируй секвенцию из 12 поз", count=12)
        app, _ = self.start_app(model, catalog, generator)
        app.chat_input[0].set_value("Сгенерируй секвенцию из 12 поз").run()
        self.assertNotIn("catalog_items", self.assert_complete(app))
        app.button(key="assistant-sequence-1-regenerate").click().run()
        self.assertNotIn("catalog_items", self.assert_complete(app))
        agent = app.session_state.assistant_agent
        source = agent.get_sequence(agent.active_sequence_uuid)
        model.responses = [tool_call("search_catalog_items"), replacement_call(source, element_index=1), AIMessage(content="Заменено.")]
        app.chat_input[0].set_value("Замени вторую позу на позу сидя в наклоне").run()
        message = self.assert_complete(app)
        self.assertNotIn("catalog_items", message)
        self.assertNotEqual(message["sequence_uuid"], str(source.uuid))
        self.assertEqual(len(message["sequence_variant_uuids"]), 3)
        self.assertIn("mirror-thumbnail__bracket", "".join(catalog_html(app)))
        self.assertTrue(app.button(key="assistant-sequence-3-regenerate"))
        self.assertEqual(save.call_args.args[0], message["sequence"])
        generator.generate.assert_called()
        self.assertEqual(generator.generate.call_count, 2)

    def test_explicit_display_and_replacement_render_separate_catalog_and_sequence_areas(self):
        generator, catalog = Mock(), Mock()
        generator.generate.return_value = [catalog_pose(1), catalog_pose(2), catalog_pose(3)]
        candidate = catalog_pose(9, bilateral=True)
        catalog.search.return_value = [candidate]
        model = tool_model("Practice", count=3)
        app, _ = self.start_app(model, catalog, generator)
        app.button(key="generate-sequence").click().run()
        agent = app.session_state.assistant_agent
        source = agent.get_sequence(agent.active_sequence_uuid)
        model.responses = [
            tool_call("search_catalog_items"), display_call(9), replacement_call(source), AIMessage(content="Показано и заменено."),
        ]
        app.chat_input[0].set_value("Покажи кандидата и замени третью позу").run()
        message = self.assert_complete(app)
        self.assertEqual(message["catalog_items"], [candidate])
        self.assertEqual(len(message["sequence"]["thumbnails"]), 3)
        gallery = catalog_html(app)[3]
        self.assertIn('alt="Pose 9"', gallery)
        self.assertNotIn("bracket", gallery)
        self.assertIn("bracket", catalog_html(app)[-1])
        self.assertNotEqual(agent.active_sequence_uuid, str(source.uuid))

    def test_foreign_duplicate_out_of_order_and_terminal_events_cannot_change_gallery(self):
        def script():
            import streamlit as st
            from unittest.mock import Mock
            from langchain_app.ui.sequence import SequenceRenderer
            from langchain_app.ui.response_progress import PendingResponse

            agent = Mock()
            agent.stream_response_events.return_value = iter(st.session_state.events)
            renderer = SequenceRenderer(generator_factory=lambda: agent)
            renderer.render_styles()
            if "saved" not in st.session_state:
                st.session_state.saved = renderer.render_response(
                    "Show", content="", key="live", pending=PendingResponse("Show", request_id=st.session_state.request_id),
                )
            else:
                renderer.render_messages([st.session_state.saved])

        request_id = uuid4()
        step_id = uuid4()
        displayed = CatalogItemsDisplayedEvent(
            request_id=request_id, event_index=2, elapsed_seconds=0.1,
            step_id=step_id, tool_call_id="display", items=[pose(1)],
        )
        app = AppTest.from_function(script)
        app.session_state.request_id = request_id
        app.session_state.events = [
            RequestStartedEvent(request_id=request_id, event_index=0, elapsed_seconds=0),
            displayed.model_copy(update={"request_id": uuid4(), "event_index": 100, "items": [pose(2)]}),
            displayed,
            displayed.model_copy(update={"items": [pose(2)]}),
            displayed.model_copy(update={"event_index": 1, "items": [pose(2)]}),
            displayed.model_copy(update={"event_index": 3, "items": [pose(2)]}),
            RequestFinishedEvent(request_id=request_id, event_index=4, elapsed_seconds=0.2, duration_seconds=0.2, outcome="complete"),
            displayed.model_copy(update={"event_index": 5, "items": [pose(2)]}),
        ]
        app.run()
        self.assertFalse(app.exception)
        displayed.items[0].name = "Mutated producer data"
        self.assertEqual(app.session_state.saved["catalog_items"], [pose(1)])
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(catalog_html(app)), 1)
        self.assertIn('alt="Pose 1"', catalog_html(app)[0])

    def test_catalog_renderer_has_shared_styles_safe_fallback_and_no_sequence_metadata(self):
        def script():
            from langchain_app.schemas import CatalogItemSchema
            from langchain_app.ui import CatalogItemsRenderer, SequenceRenderer

            SequenceRenderer(thumbnail_size=80).render_styles()
            CatalogItemsRenderer(thumbnail_size=80).render([
                CatalogItemSchema(id=1, uuid=None, catalog_tag_names=[], name='Tree "<pose>"', bilateral=True, image_url='https://example.com/a.png?x=1&y=2'),
                CatalogItemSchema(id=2, uuid=None, catalog_tag_names=[], name="Missing image <pose>", bilateral=True, image_url=None),
            ], key="catalog")

        app = AppTest.from_function(script).run()
        self.assertFalse(app.exception)
        html = [node.proto.body for node in app.get("html")]
        styles = [body for body in html if "<style>" in body]
        self.assertEqual(len(styles), 1)
        self.assertIn("--thumbnail-size: 80px", styles[0])
        image, = catalog_html(app)
        self.assertIn('alt="Tree &quot;&lt;pose&gt;&quot;"', image)
        self.assertIn("x=1&amp;y=2", image)
        self.assertNotIn("bracket", image)
        self.assertEqual([caption.value for caption in app.caption], ["Missing image <pose>"])
        self.assertFalse(app.button)
        self.assertFalse(app.markdown)
