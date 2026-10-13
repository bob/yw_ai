"""Feature 008: structured API errors reach live and replayed UI, offline."""

import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import httpx
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai.chat_models.base import OpenAIRateLimitError
from openai import BadRequestError, InternalServerError, RateLimitError
from streamlit.testing.v1 import AppTest

from langchain_app.schemas.response_progress import (
    RequestFinishedEvent,
    RequestStartedEvent,
    SequenceCommittedEvent,
    StepFinishedEvent,
    StepStartedEvent,
)
from langchain_app.services.assistant_agent import AssistantAgentError
from langchain_app.services.request_errors import (
    PUBLIC_CREDITS_EXHAUSTED_ERROR,
    PUBLIC_QUOTA_EXCEEDED_ERROR,
    PUBLIC_REQUEST_ERROR,
    public_request_error,
)
from langchain_app.services.response_execution import ResponseExecution
from test_assistant_agent import ToolCallingModel, assistant, pose, tool_model
from test_sequence_variants import APP


def api_error(body=None, error_class=RateLimitError, *, status_code=429):
    return error_class(
        "PRIVATE_PROVIDER_DETAILS",
        response=httpx.Response(status_code, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")),
        body=body,
    )


def credit_error(error_class=RateLimitError):
    return api_error({"code": "credit_balance_exhausted", "type": "insufficient_quota"}, error_class)


class PublicRequestErrorTests(unittest.TestCase):
    def test_exact_credit_code_survives_langchain_exception_conversion(self):
        for error_class in (RateLimitError, OpenAIRateLimitError):
            with self.subTest(error_class=error_class):
                self.assertEqual(public_request_error(credit_error(error_class)), PUBLIC_CREDITS_EXHAUSTED_ERROR)

    def test_legacy_quota_reports_credits_and_limits_without_claiming_empty_balance(self):
        for body in (
            {"code": "insufficient_quota", "type": "insufficient_quota"},
            {"code": "insufficient_quota"},
            {"type": "insufficient_quota"},
        ):
            with self.subTest(body=body):
                self.assertEqual(public_request_error(api_error(body)), PUBLIC_QUOTA_EXCEEDED_ERROR)

    def test_other_429_errors_never_claim_credits_have_run_out(self):
        for body in (
            {"code": "rate_limit_exceeded", "type": "rate_limit_error"},
            {"code": "slow_down", "type": "rate_limit_error"},
            {"code": "organization_spend_limit_exceeded", "type": "insufficient_quota"},
            {"code": "project_spend_limit_exceeded", "type": "insufficient_quota"},
            {"code": "organization_usage_limit_exceeded", "type": "insufficient_quota"},
            {"code": "unknown", "type": "insufficient_quota"},
            {"code": ["unexpected"], "type": "insufficient_quota"},
            {}, None, "invalid response",
        ):
            with self.subTest(body=body):
                self.assertEqual(public_request_error(api_error(body)), PUBLIC_REQUEST_ERROR)

    def test_matching_raw_text_is_not_used_to_recognize_credit_errors(self):
        for error in (
            RuntimeError("credit_balance_exhausted: PRIVATE_PROVIDER_DETAILS"),
            RateLimitError(
                "credit_balance_exhausted: PRIVATE_PROVIDER_DETAILS",
                response=httpx.Response(429, request=httpx.Request("POST", "https://example.com")),
                body=None,
            ),
        ):
            self.assertEqual(public_request_error(error), PUBLIC_REQUEST_ERROR)

    def test_credit_and_quota_codes_require_http_429_rate_limit_errors(self):
        for error_class, status_code in (
            (RateLimitError, 400), (RateLimitError, 500),
            (BadRequestError, 400), (InternalServerError, 500),
        ):
            for code in ("credit_balance_exhausted", "insufficient_quota", None):
                with self.subTest(error_class=error_class, status_code=status_code, code=code):
                    error = api_error(
                        {"code": code, "type": "insufficient_quota"},
                        error_class, status_code=status_code,
                    )
                    self.assertEqual(public_request_error(error), PUBLIC_REQUEST_ERROR)

    def test_exception_chain_handles_wrappers_context_suppression_and_cycles(self):
        error = credit_error()
        wrapped = AssistantAgentError("Wrapped error")
        wrapped.__cause__ = error
        self.assertEqual(public_request_error(wrapped), PUBLIC_CREDITS_EXHAUSTED_ERROR)
        outer = RuntimeError("Outer error")
        outer.__context__ = wrapped
        self.assertEqual(public_request_error(outer), PUBLIC_CREDITS_EXHAUSTED_ERROR)
        outer.__suppress_context__ = True
        self.assertEqual(public_request_error(outer), PUBLIC_REQUEST_ERROR)
        first, second = RuntimeError("First"), RuntimeError("Second")
        first.__cause__, second.__cause__ = second, first
        self.assertEqual(public_request_error(first), PUBLIC_REQUEST_ERROR)


class ApiCreditErrorExecutionTests(unittest.TestCase):
    def test_fallback_preserves_credit_message_commit_and_finished_steps(self):
        request_id, source_id, sequence_id = uuid4(), uuid4(), uuid4()
        finished_id, active_id = uuid4(), uuid4()
        execution = ResponseExecution(request_id)
        execution.accept(RequestStartedEvent(
            request_id=request_id, event_index=0, elapsed_seconds=0.0,
            source_sequence_uuid=source_id,
        ))
        execution.accept(StepStartedEvent(
            request_id=request_id, event_index=1, elapsed_seconds=0.0,
            step_id=finished_id, code="generating_sequence", title="Generating sequence...",
        ))
        execution.accept(SequenceCommittedEvent(
            request_id=request_id, event_index=2, elapsed_seconds=0.1,
            step_id=finished_id, sequence_uuid=sequence_id,
        ))
        execution.accept(StepFinishedEvent(
            request_id=request_id, event_index=3, elapsed_seconds=0.1,
            step_id=finished_id, outcome="ok", duration_seconds=0.1, summary="Sequence created.",
        ))
        execution.accept(StepStartedEvent(
            request_id=request_id, event_index=4, elapsed_seconds=0.1,
            step_id=active_id, code="thinking", title="Thinking...",
        ))
        execution.fail(0.2, error_message=PUBLIC_CREDITS_EXHAUSTED_ERROR)
        snapshot = execution.snapshot()
        execution.fail(0.3)
        self.assertEqual(execution.snapshot(), snapshot)
        self.assertEqual(snapshot.error_message, PUBLIC_CREDITS_EXHAUSTED_ERROR)
        self.assertEqual(snapshot.source_sequence_uuid, source_id)
        self.assertEqual(snapshot.sequence_uuid, sequence_id)
        self.assertEqual([step.outcome for step in snapshot.steps], ["ok", "error"])
        self.assertAlmostEqual(snapshot.steps[-1].duration_seconds, 0.1)

    def test_fallback_cannot_replace_an_existing_terminal_result(self):
        for outcome, message in (("complete", None), ("error", PUBLIC_QUOTA_EXCEEDED_ERROR)):
            with self.subTest(outcome=outcome):
                execution = ResponseExecution(uuid4())
                terminal = RequestFinishedEvent(
                    request_id=execution.request_id, event_index=0, elapsed_seconds=0.1,
                    duration_seconds=0.1, outcome=outcome, error_message=message,
                )
                execution.accept(terminal)
                snapshot = execution.snapshot()
                execution.fail(0.2, error_message=PUBLIC_CREDITS_EXHAUSTED_ERROR)
                self.assertIs(execution.result, terminal)
                self.assertEqual(execution.snapshot(), snapshot)


class ApiCreditErrorStreamTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        env.start()
        self.addCleanup(env.stop)

    def collect_failure(self, service, request="Hello", **kwargs):
        events = []
        with self.assertRaises(AssistantAgentError) as caught:
            events.extend(service.stream_response_events(request, **kwargs))
        self.assertEqual(public_request_error(caught.exception), PUBLIC_CREDITS_EXHAUSTED_ERROR)
        self.assertEqual(events[-1].type, "request_finished")
        self.assertEqual(events[-1].outcome, "error")
        self.assertEqual(events[-1].error_message, PUBLIC_CREDITS_EXHAUSTED_ERROR)
        self.assertEqual(sum(event.type == "request_finished" for event in events), 1)
        self.assertNotIn("PRIVATE_PROVIDER_DETAILS", str(events))
        self.assertEqual(
            [event.step_id for event in events if event.type == "step_started"],
            [event.step_id for event in events if event.type == "step_finished"],
        )
        return events

    def test_model_initialization_credit_failure_emits_public_terminal_event(self):
        factory = Mock()
        service = assistant(factory)
        with patch("langchain_app.services.assistant_agent.init_chat_model", side_effect=credit_error()):
            events = self.collect_failure(service)
        self.assertEqual([event.type for event in events], ["request_started", "request_finished"])
        factory.assert_not_called()

    def test_credit_failure_during_model_step_never_runs_tools(self):
        factory = Mock()
        model = tool_model("practice")
        service = assistant(factory, model=model)
        with patch.object(ToolCallingModel, "_generate", side_effect=credit_error(OpenAIRateLimitError)):
            events = self.collect_failure(service, "practice")
        self.assertEqual([event.code for event in events if event.type == "step_started"], ["thinking"])
        self.assertFalse(any(event.type == "sequence_committed" for event in events))
        self.assertIsNone(service.active_sequence_uuid)
        factory.assert_not_called()


class ApiCreditErrorUITests(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        env.start()
        self.addCleanup(env.stop)

    def assert_credit_failure(self, app):
        self.assertFalse(app.exception)
        snapshot = app.session_state.messages[-1]["execution"]
        self.assertEqual(snapshot.outcome, "error")
        self.assertEqual(snapshot.error_message, PUBLIC_CREDITS_EXHAUSTED_ERROR)
        self.assertEqual([alert.value for alert in app.error], [PUBLIC_CREDITS_EXHAUSTED_ERROR])
        self.assertNotIn("PRIVATE_PROVIDER_DETAILS", str(snapshot))
        self.assertIsNone(app.session_state.pending_response)
        self.assertFalse(app.chat_input[0].disabled)
        self.assertFalse(app.button(key="generate-sequence").disabled)
        return snapshot

    def test_chat_and_generate_display_error_and_replay_without_retrying(self):
        for generate in (False, True):
            with self.subTest(generate=generate):
                model = tool_model("practice")
                with patch("langchain_app.services.assistant_agent.init_chat_model", return_value=model), patch.object(
                    ToolCallingModel, "_generate", side_effect=credit_error(OpenAIRateLimitError),
                ) as model_call, patch("langchain_app.dependencies.get_sequence_generator") as factory:
                    app = AppTest.from_file(str(APP)).run()
                    if generate:
                        app.button(key="generate-sequence").click().run()
                    else:
                        app.chat_input[0].set_value("Hello").run()
                    snapshot = self.assert_credit_failure(app)
                    self.assertIsNone(snapshot.sequence_uuid)
                    app.run()
                    self.assertEqual(self.assert_credit_failure(app), snapshot)
                    model_call.assert_called_once()
                    factory.assert_called_once()
                    factory.return_value.generate.assert_not_called()
                    factory.return_value.add_elements.assert_not_called()
                model.responses, model.i = [AIMessage(content="Recovered")], 0
                app.chat_input[0].set_value("Try again").run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state.messages[-1]["execution"].outcome, "complete")
                self.assertEqual(app.session_state.messages[-1]["content"], "Recovered")

    def test_dependency_initialization_uses_same_public_message(self):
        with patch("langchain_app.dependencies.get_assistant_agent", side_effect=credit_error()) as factory:
            app = AppTest.from_file(str(APP)).run()
            app.chat_input[0].set_value("Hello").run()
            snapshot = self.assert_credit_failure(app)
            self.assertEqual(snapshot.steps, [])
            app.run()
            self.assertEqual(self.assert_credit_failure(app), snapshot)
            factory.assert_called_once()

    def test_quota_and_temporary_rate_limit_remain_distinct_in_ui(self):
        for code, expected in (
            ("insufficient_quota", PUBLIC_QUOTA_EXCEEDED_ERROR),
            ("rate_limit_exceeded", PUBLIC_REQUEST_ERROR),
        ):
            with self.subTest(code=code), patch(
                "langchain_app.services.assistant_agent.init_chat_model",
                side_effect=api_error({"code": code}),
            ):
                app = AppTest.from_file(str(APP)).run()
                app.chat_input[0].set_value("Hello").run()
                self.assertFalse(app.exception)
                self.assertEqual([alert.value for alert in app.error], [expected])
                self.assertEqual(app.session_state.messages[-1]["execution"].error_message, expected)

    def test_failed_regenerate_keeps_source_save_and_attempt_history(self):
        generator = Mock()
        generator.generate.return_value = [pose(1)]
        model = tool_model("practice", count=1)
        with patch("langchain_app.services.assistant_agent.init_chat_model", return_value=model), patch(
            "langchain_app.dependencies.get_sequence_generator", return_value=generator,
        ), patch("langchain_app.ui.sequence.render_sequence_import") as save:
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            message = app.session_state.messages[-1]
            original, source = message["execution"], message["sequence_uuid"]
            with patch.object(ToolCallingModel, "_generate", side_effect=credit_error()):
                app.button(key="assistant-sequence-1-regenerate").click().run()
                snapshot = self.assert_credit_failure(app)
            self.assertEqual(str(snapshot.source_sequence_uuid), source)
            self.assertIsNone(snapshot.sequence_uuid)
            self.assertEqual(message["sequence_uuid"], source)
            self.assertEqual(message["execution_history"], [original])
            self.assertEqual(save.call_args.args[0]["info"]["uuid"], source)
            app.run()
            self.assertEqual(self.assert_credit_failure(app), snapshot)
            generator.generate.assert_called_once()

    def test_credit_failure_after_commit_keeps_created_sequence_and_save(self):
        generator = Mock()
        generator.generate.return_value = [pose(1)]
        model = tool_model("practice", count=1)
        original_generate = ToolCallingModel._generate

        def fail_final_reply(instance, messages, **kwargs):
            if isinstance(messages[-1], ToolMessage):
                raise credit_error(OpenAIRateLimitError)
            return original_generate(instance, messages, **kwargs)

        with patch("langchain_app.services.assistant_agent.init_chat_model", return_value=model), patch(
            "langchain_app.dependencies.get_sequence_generator", return_value=generator,
        ), patch("langchain_app.ui.sequence.render_sequence_import") as save, patch.object(
            ToolCallingModel, "_generate", new=fail_final_reply,
        ):
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            snapshot = self.assert_credit_failure(app)
            self.assertIsNotNone(snapshot.sequence_uuid)
            message = app.session_state.messages[-1]
            self.assertEqual(message["sequence_uuid"], str(snapshot.sequence_uuid))
            self.assertEqual(save.call_args.args[0]["info"]["uuid"], str(snapshot.sequence_uuid))
            self.assertTrue(any(step.code == "generating_sequence" and step.outcome == "ok" for step in snapshot.steps))
            app.run()
            self.assertEqual(self.assert_credit_failure(app), snapshot)
            generator.generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
