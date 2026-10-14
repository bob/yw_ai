"""Feature 002 UI acceptance checks with real agent events and no paid calls."""

import time
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from langchain_app.schemas.response_progress import (
    AssistantMessageEvent, RequestStartedEvent,
    SequenceCommittedEvent, StepStartedEvent,
)
from langchain_app.services.response_execution import ResponseExecution
from langchain_app.services.assistant_agent import AssistantAgent
from langchain_app.services.catalog_item import CatalogItemApiError
from langchain_app.services.sequence_generator import SequenceGenerator
from langchain_app.ui.response_progress import ResponseProgressRenderer
from test_assistant_agent import tool_call
from test_catalog_display_tool import display_call
from test_sequence_variants import (
    APP, ErrorAfterGenerationModel, ToolCallingModel, catalog_pose, pose, tool_model,
)


def response_layout_app():
    """Keep the live response visible before testing its history projection."""
    import streamlit as st
    from langchain_app.ui.sequence import SequenceRenderer

    renderer = SequenceRenderer(generator_factory=lambda: st.session_state.assistant)
    if "message" not in st.session_state:
        st.session_state.message = renderer.render_response(
            "Show the requested poses", content="", key="live",
        )
    else:
        renderer.render_messages([st.session_state.message])


class ResponseProgressUITests(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        env.start()
        self.addCleanup(env.stop)

    def assert_terminal(self, app, *, outcome="complete", index=-1):
        self.assertFalse(app.exception)
        snapshot = app.session_state.messages[index]["execution"]
        self.assertEqual(snapshot.outcome, outcome)
        statuses = app.get("status")
        self.assertEqual(statuses[-1].proto.expanded, outcome == "error")
        self.assertEqual(statuses[-1].proto.state, 3 if outcome == "error" else 2)
        self.assertEqual(app.session_state.pending_response, None)
        self.assertFalse(app.chat_input[0].disabled)
        self.assertEqual(app.chat_input[0].proto.value, "")
        self.assertFalse(app.button(key="generate-sequence").disabled)
        return snapshot

    def test_plain_chat_saves_collapsed_history_and_rerun_does_not_call_model(self):
        model = ToolCallingModel(responses=[AIMessage(content="How can I help?")])
        with patch("langchain_app.services.assistant_agent.init_chat_model", return_value=model):
            app = AppTest.from_file(str(APP)).run()
            app.chat_input[0].set_value("Hello").run()
            snapshot = self.assert_terminal(app)
            self.assertEqual([step.code for step in snapshot.steps], ["thinking", "preparing_response"])
            self.assertIsNone(snapshot.sequence_uuid)
            self.assertEqual(app.session_state.messages[-1]["content"], "How can I help?")
            self.assertEqual([item.value for item in app.markdown].count("How can I help?"), 1)
            calls = len(model.received_messages)
            app.run()
            self.assertEqual(self.assert_terminal(app), snapshot)
            self.assertEqual(len(model.received_messages), calls)

    def test_useful_text_stays_below_thumbnails_live_and_in_history(self):
        explanation = "Only 45 poses fit within the sequence limit."
        for output in ("sequence", "catalog", "both"):
            with self.subTest(output=output):
                generator, catalog = Mock(), Mock()
                generator.generate.return_value = [pose(1)]
                catalog.search.return_value = [pose(2)]
                calls = []
                if output in ("sequence", "both"):
                    calls.append(tool_call("generate_sequence", request="Practice", count=1))
                if output in ("catalog", "both"):
                    calls.extend([tool_call("search_catalog_items"), display_call(2)])
                model = ToolCallingModel(responses=[*calls, AIMessage(content=explanation)])
                service = AssistantAgent(
                    lambda: generator, model=model,
                    catalog_item_service=catalog, catalog_tag_service=Mock(),
                )
                app = AppTest.from_function(response_layout_app)
                app.session_state.assistant = service
                for phase in ("live", "history"):
                    with self.subTest(phase=phase):
                        app.run()
                        self.assertFalse(app.exception)
                        nodes = list(app.chat_message[-1])
                        text_position = next(
                            index for index, node in enumerate(nodes)
                            if node.type == "markdown" and node.value == explanation
                        )
                        thumbnails = [
                            index for index, node in enumerate(nodes)
                            if node.type == "html" and 'class="mirror-thumbnail"' in node.proto.body
                        ]
                        self.assertEqual(len(thumbnails), 2 if output == "both" else 1)
                        self.assertTrue(all(index < text_position for index in thumbnails))
                        self.assertEqual(app.session_state.message["content"], explanation)
                self.assertEqual(len(model.received_messages), len(calls) + 1)

    def test_regeneration_without_text_clears_previous_reply(self):
        previous_reply = "Only 45 poses fit within the sequence limit."
        model = tool_model()
        model.responses[-1] = AIMessage(content=previous_reply)
        generator = Mock()
        generator.generate.side_effect = [[pose(1)], [pose(2)]]
        with patch("langchain_app.services.assistant_agent.init_chat_model", return_value=model), patch(
            "langchain_app.dependencies.get_sequence_generator", return_value=generator,
        ):
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            self.assert_terminal(app)
            self.assertEqual(app.session_state.messages[-1]["content"], previous_reply)
            model.responses[-1] = AIMessage(content="")
            app.button(key="assistant-sequence-1-regenerate").click().run()
            self.assert_terminal(app)
            self.assertEqual(app.session_state.messages[-1]["content"], "")
            self.assertEqual(app.session_state.messages[-1]["sequence"]["thumbnails"][0]["alt"], "Pose 2")
            self.assertNotIn(previous_reply, [node.value for node in app.markdown])
            app.run()
            self.assertEqual(app.session_state.messages[-1]["content"], "")
            self.assertEqual(generator.generate.call_count, 2)

    def test_tags_and_repeated_search_keep_public_messages_and_actual_order(self):
        first = tool_call("get_catalog_tag_names")
        first.content = [
            {"type": "text", "text": "I'll check the catalog."},
            {"type": "reasoning", "reasoning": "PRIVATE_REASONING"},
        ]
        model = ToolCallingModel(responses=[
            first, tool_call("search_catalog_items", name="Tree"),
            tool_call("search_catalog_items", name="Balance"), AIMessage(content="No matching poses."),
        ])
        catalog, tags = Mock(), Mock()
        catalog.search.return_value = []
        tags.get_names.return_value = ["Balance"]
        with patch("langchain_app.services.assistant_agent.init_chat_model", return_value=model), patch(
            "langchain_app.dependencies.get_catalog_item_service", return_value=catalog,
        ), patch("langchain_app.dependencies.get_catalog_tag_service", return_value=tags):
            app = AppTest.from_file(str(APP)).run()
            app.chat_input[0].set_value("Find balancing poses").run()
            snapshot = self.assert_terminal(app)
            self.assertEqual([step.code for step in snapshot.steps], [
                "thinking", "loading_catalog_tags", "thinking", "searching_catalog_items",
                "thinking", "searching_catalog_items", "thinking", "preparing_response",
            ])
            self.assertEqual(len({step.step_id for step in snapshot.steps}), 8)
            searches = [step for step in snapshot.steps if step.code == "searching_catalog_items"]
            self.assertTrue(all(step.outcome == "ok" and "0 catalog items" in step.summary for step in searches))
            self.assertEqual(snapshot.intermediate_messages[0].text, "I'll check the catalog.")
            self.assertEqual(app.session_state.messages[-1]["content"], "No matching poses.")
            self.assertNotIn("PRIVATE_REASONING", str(snapshot))
            self.assertNotIn("PRIVATE_REASONING", str([item.value for item in app.markdown]))
            self.assertIn("I'll check the catalog.", [item.value for item in app.get("status")[0].markdown])

    def test_initialization_failure_is_visible_first_saved_and_releases_controls(self):
        with patch("streamlit.status", wraps=st.status) as statuses, patch(
            "streamlit.chat_input", wraps=st.chat_input,
        ) as chat, patch("streamlit.form_submit_button", wraps=st.form_submit_button) as generate:
            def failed_factory(*args):
                self.assertEqual(statuses.call_args.args[0], "Thinking...")
                self.assertTrue(statuses.call_args.kwargs["expanded"])
                self.assertTrue(chat.call_args.kwargs["disabled"])
                self.assertTrue(generate.call_args.kwargs["disabled"])
                time.sleep(0.02)
                raise ValueError("PRIVATE_CONFIGURATION")

            with patch("langchain_app.dependencies.get_assistant_agent", side_effect=failed_factory) as factory:
                app = AppTest.from_file(str(APP)).run()
                app.chat_input[0].set_value("Hello").run()
                snapshot = self.assert_terminal(app, outcome="error")
                self.assertGreaterEqual(snapshot.duration_seconds, 0.02)
                self.assertEqual(snapshot.steps, [])
                self.assertEqual(app.session_state.messages[-1]["content"], "")
                self.assertNotIn("PRIVATE_CONFIGURATION", app.error[0].value)
                app.run()
                self.assertEqual(self.assert_terminal(app, outcome="error"), snapshot)
                factory.assert_called_once()
        model = ToolCallingModel(responses=[AIMessage(content="Recovered")])
        with patch("langchain_app.services.assistant_agent.init_chat_model", return_value=model):
            app.chat_input[0].set_value("Try again").run()
            self.assert_terminal(app)
            self.assertEqual(app.session_state.messages[-1]["content"], "Recovered")
            self.assertEqual(len(app.session_state.messages), 4)

    def test_recoverable_catalog_failure_remains_a_failed_step_in_completed_request(self):
        model = ToolCallingModel(responses=[
            tool_call("search_catalog_items", name="Tree"), AIMessage(content="Please try another search."),
        ])
        catalog = Mock()
        catalog.search.side_effect = CatalogItemApiError("PRIVATE_RESPONSE")
        with patch("langchain_app.services.assistant_agent.init_chat_model", return_value=model), patch(
            "langchain_app.dependencies.get_catalog_item_service", return_value=catalog,
        ):
            app = AppTest.from_file(str(APP)).run()
            app.chat_input[0].set_value("Find Tree").run()
            snapshot = self.assert_terminal(app)
            search = next(step for step in snapshot.steps if step.code == "searching_catalog_items")
            self.assertEqual(search.outcome, "error")
            self.assertNotIn("PRIVATE_RESPONSE", str(snapshot))
            self.assertFalse(app.error)
            self.assertEqual(app.session_state.messages[-1]["content"], "Please try another search.")

    def test_generator_records_search_and_counters_and_replays_committed_card(self):
        items = [catalog_pose(n) for n in (1, 2, 3)]
        items[0].catalog_tag_names = ["Other/Starting"]
        items[2].catalog_tag_names = ["Other/Finishing"]
        catalog, selector = Mock(), Mock()
        catalog.search.return_value = items
        selector.choose_next_catalog_item.side_effect = lambda candidates, **kwargs: candidates[0]
        generator = SequenceGenerator(catalog, selector)
        model = tool_model(count=3)
        with patch("langchain_app.services.assistant_agent.init_chat_model", return_value=model), patch(
            "langchain_app.dependencies.get_sequence_generator", return_value=generator,
        ):
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            snapshot = self.assert_terminal(app)
            generation = next(step for step in snapshot.steps if step.code == "generating_sequence")
            search = next(step for step in snapshot.steps if step.code == "searching_catalog_items")
            self.assertEqual(search.parent_step_id, generation.step_id)
            self.assertEqual([update.completed for update in generation.progress], [0, 0, 1, 2, 3])
            self.assertTrue(all(update.total == 3 for update in generation.progress))
            message = app.session_state.messages[-1]
            self.assertEqual(str(snapshot.sequence_uuid), message["sequence_uuid"])
            self.assertEqual(len(message["sequence"]["thumbnails"]), 3)
            calls = len(model.received_messages)
            app.run()
            self.assertEqual(self.assert_terminal(app), snapshot)
            self.assertEqual(len(model.received_messages), calls)

    def test_regenerate_fail_then_retry_preserves_each_attempt_and_source(self):
        def partial_failure(*args, **kwargs):
            yield pose(2)
            raise ValueError("PRIVATE_FAILURE")

        generator = Mock()
        generator.generate.side_effect = [[pose(1)], partial_failure(), [pose(3)]]
        model = tool_model()
        with patch("langchain_app.services.assistant_agent.init_chat_model", return_value=model), patch(
            "langchain_app.dependencies.get_sequence_generator", return_value=generator,
        ), patch("langchain_app.ui.sequence.render_sequence_import") as save:
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            first = self.assert_terminal(app)
            source = app.session_state.messages[-1]["sequence_uuid"]
            app.button(key="assistant-sequence-1-regenerate").click().run()
            failed = self.assert_terminal(app, outcome="error")
            message = app.session_state.messages[-1]
            self.assertEqual(message["execution_history"], [first])
            self.assertEqual(str(failed.source_sequence_uuid), source)
            self.assertIsNone(failed.sequence_uuid)
            self.assertEqual(message["sequence_uuid"], source)
            self.assertEqual(message["sequence"]["thumbnails"][0]["alt"], "Pose 1")
            self.assertNotIn("Pose 2", str([element.value for element in app.get("html")]))
            self.assertEqual(save.call_args.args[0]["info"]["uuid"], source)
            app.run()
            self.assertEqual(self.assert_terminal(app, outcome="error"), failed)
            self.assertEqual(generator.generate.call_count, 2)
            # The interrupted turn never consumed its fake final reply.
            model.i = 0
            app.button(key="assistant-sequence-1-regenerate").click().run()
            last = self.assert_terminal(app)
            message = app.session_state.messages[-1]
            self.assertEqual(message["execution_history"], [first, failed])
            self.assertEqual(len({first.request_id, failed.request_id, last.request_id}), 3)
            self.assertEqual(str(last.source_sequence_uuid), source)
            self.assertEqual(len(message["sequence_variants"]), 2)
            self.assertEqual(message["selected_sequence_index"], 1)
            self.assertEqual(save.call_args.args[0]["info"]["uuid"], str(last.sequence_uuid))

    def test_error_after_commit_retains_card_save_and_error_after_rerun(self):
        generator = Mock()
        generator.generate.return_value = [pose(1)]
        model = ErrorAfterGenerationModel(responses=tool_model().responses)
        with patch("langchain_app.services.assistant_agent.init_chat_model", return_value=model), patch(
            "langchain_app.dependencies.get_sequence_generator", return_value=generator,
        ), patch("langchain_app.ui.sequence.render_sequence_import") as save:
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            snapshot = self.assert_terminal(app, outcome="error")
            self.assertIsNotNone(snapshot.sequence_uuid)
            self.assertEqual(save.call_args.args[0]["info"]["uuid"], str(snapshot.sequence_uuid))
            self.assertEqual(snapshot.steps[-1].outcome, "error")
            self.assertTrue(any(step.code == "generating_sequence" and step.outcome == "ok" for step in snapshot.steps))
            app.run()
            self.assertEqual(self.assert_terminal(app, outcome="error"), snapshot)
            self.assertEqual(len(app.error), 1)
            generator.generate.assert_called_once()

    def test_two_sessions_keep_distinct_execution_and_legacy_messages_still_render(self):
        models = [ToolCallingModel(responses=[AIMessage(content=reply)]) for reply in ("First", "Second")]
        with patch("langchain_app.services.assistant_agent.init_chat_model", side_effect=models):
            first = AppTest.from_file(str(APP)).run()
            second = AppTest.from_file(str(APP)).run()
            first.chat_input[0].set_value("Hello").run()
            second.chat_input[0].set_value("Hello").run()
            self.assertNotEqual(self.assert_terminal(first).request_id, self.assert_terminal(second).request_id)
            first.session_state.messages.insert(0, {"role": "assistant", "content": "Legacy reply"})
            first.run()
            self.assertFalse(first.exception)
            self.assertIn("Legacy reply", [element.value for element in first.markdown])
            self.assertEqual(second.session_state.messages[-1]["content"], "Second")
            self.assertTrue(all(len(model.received_messages) == 1 for model in models))

    def test_regenerate_and_variant_controls_are_disabled_during_another_request(self):
        generator = Mock()
        generator.generate.side_effect = [[pose(1)], [pose(2)], [pose(3)]]
        model = tool_model()
        with patch("langchain_app.services.assistant_agent.init_chat_model", return_value=model), patch(
            "langchain_app.dependencies.get_sequence_generator", return_value=generator,
        ):
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            app.button(key="assistant-sequence-1-regenerate").click().run()
            app.button(key="generate-sequence").click().run()
            model.responses = [AIMessage(content="Done")]
            model.i = 0
            original = ToolCallingModel._generate
            with patch("streamlit.button", wraps=st.button) as buttons:
                def check_controls(instance, messages, **kwargs):
                    for label in ("Regenerate", "Use this sequence", "Previous Sequence", "Next Sequence"):
                        calls = [call for call in buttons.call_args_list if call.args[0] == label]
                        self.assertTrue(calls, label)
                        self.assertTrue(all(call.kwargs["disabled"] for call in calls), label)
                    return original(instance, messages, **kwargs)

                with patch.object(ToolCallingModel, "_generate", new=check_controls):
                    app.chat_input[0].set_value("Hello").run()
            self.assert_terminal(app)
            self.assertEqual(app.session_state.messages[-1]["content"], "Done")


class ResponseExecutionTests(unittest.TestCase):
    def test_foreign_events_and_duplicate_messages_cannot_change_this_snapshot(self):
        request_id, foreign_id, step_id = uuid4(), uuid4(), uuid4()
        execution = ResponseExecution(request_id)
        execution.accept(RequestStartedEvent(request_id=request_id, event_index=0, elapsed_seconds=0.0))
        self.assertFalse(execution.accept(SequenceCommittedEvent(
            request_id=foreign_id, event_index=100, elapsed_seconds=0.0,
            step_id=uuid4(), sequence_uuid=uuid4(),
        )))
        execution.accept(StepStartedEvent(
            request_id=request_id, event_index=1, elapsed_seconds=0.0,
            step_id=step_id, code="thinking", title="Thinking...",
        ))
        for index in (2, 3):
            execution.accept(AssistantMessageEvent(
                request_id=request_id, event_index=index, elapsed_seconds=0.1,
                message_id="same", text="Once", purpose="final",
            ))
        execution.fail(0.2)
        execution.fail(0.3)
        snapshot = execution.snapshot()
        self.assertEqual(execution.final_texts, ["Once"])
        self.assertIsNone(snapshot.sequence_uuid)
        self.assertEqual(snapshot.duration_seconds, 0.2)
        self.assertEqual(snapshot.steps[0].outcome, "error")
        self.assertEqual(snapshot.steps[0].duration_seconds, 0.2)

    def test_live_updates_preserve_manual_collapse_until_terminal_state(self):
        request_id, step_id = uuid4(), uuid4()
        with patch("langchain_app.ui.response_progress.st.status") as status, patch("streamlit.error"):
            renderer = ResponseProgressRenderer(request_id, time.monotonic())
            renderer.accept(StepStartedEvent(
                request_id=request_id, event_index=0, elapsed_seconds=0.0,
                step_id=step_id, code="thinking", title="Thinking...",
            ))
            self.assertNotIn("expanded", status.return_value.update.call_args.kwargs)
            renderer.finish()
            self.assertEqual(status.return_value.update.call_args.kwargs["state"], "error")
            self.assertTrue(status.return_value.update.call_args.kwargs["expanded"])


if __name__ == "__main__":
    unittest.main()
