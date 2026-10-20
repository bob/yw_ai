"""Reject oversized requests before creating a model or altering a sequence."""

import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from langchain_app.schemas.response_progress import RequestFinishedEvent, RequestStartedEvent
from langchain_app.services.assistant_agent import AssistantAgent
from langchain_app.services.request_limits import (
    MAX_REQUEST_CHARACTERS, InvalidRequestError, validate_request,
)
from test_assistant_agent import ToolCallingModel, assistant, tool_call
from test_sequence_variants import APP


class RequestLimitTests(unittest.TestCase):
    def test_boundary_keeps_original_content_and_invalid_inputs_are_safe(self):
        validate_request("я" * MAX_REQUEST_CHARACTERS)
        for request in ("", " \n ", "private content " * MAX_REQUEST_CHARACTERS):
            with self.subTest(length=len(request)), self.assertRaises(InvalidRequestError) as error:
                validate_request(request)
            self.assertNotIn("private content", str(error.exception))

    def test_service_rejects_before_model_initialization_and_keeps_state_empty(self):
        for request in (" ", "x" * (MAX_REQUEST_CHARACTERS + 1)):
            service = AssistantAgent(Mock(), catalog_item_service=Mock(), catalog_tag_service=Mock())
            events = []
            with self.subTest(length=len(request)), patch(
                "langchain_app.services.assistant_agent.init_chat_model",
            ) as model, self.assertRaises(InvalidRequestError):
                events.extend(service.stream_response_events(request))
            model.assert_not_called()
            self.assertEqual(service.state, {})
            self.assertIsInstance(events[0], RequestStartedEvent)
            self.assertIsInstance(events[-1], RequestFinishedEvent)
            self.assertEqual(events[-1].outcome, "error")
            if request.strip():
                self.assertNotIn(request, events[-1].error_message)

    def test_model_cannot_forward_an_oversized_generation_request_to_typesafe(self):
        factory = Mock()
        model = ToolCallingModel(responses=[
            tool_call("generate_sequence", request="x" * (MAX_REQUEST_CHARACTERS + 1), count=9),
            AIMessage(content="Please shorten the request."),
        ])
        service = assistant(factory, model=model)
        list(service.stream("Generate a short sequence"))
        factory.assert_not_called()
        self.assertIsNone(service.active_sequence_uuid)

    def test_browser_submission_guard_rejects_forged_oversize_without_initializing_dependencies(self):
        with patch("langchain_app.dependencies.get_assistant_agent") as factory:
            app = AppTest.from_file(str(APP)).run()
            self.assertEqual(app.chat_input[0].proto.max_chars, MAX_REQUEST_CHARACTERS)
            app.chat_input[0].set_value("x" * (MAX_REQUEST_CHARACTERS + 1)).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertEqual(app.session_state.messages, [])
            self.assertIsNone(app.session_state.pending_response)
            factory.assert_not_called()
