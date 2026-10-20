"""Verify that rejected or interrupted requests cannot leak capacity slots."""

import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from langchain_app.services.request_capacity import RequestCapacity, RequestCapacityError
from test_assistant_agent import ToolCallingModel, assistant


class RequestCapacityTests(unittest.TestCase):
    def test_full_capacity_rejects_without_initializing_a_model(self):
        capacity = RequestCapacity(1)
        self.assertTrue(capacity.acquire())
        service = assistant(lambda: None)
        events = []
        with patch("langchain_app.services.assistant_agent.get_request_capacity", return_value=capacity), patch(
            "langchain_app.services.assistant_agent.init_chat_model",
        ) as model, self.assertRaises(RequestCapacityError):
            events.extend(service.stream_response_events("Hello"))
        model.assert_not_called()
        self.assertEqual(events[-1].outcome, "error")
        self.assertEqual(events[-1].error_message, "The assistant is busy. Please try again shortly.")
        self.assertEqual(service.state, {})
        # Rejecting a request must not release somebody else's slot.
        self.assertFalse(capacity.acquire())
        capacity.release()
        self.assertTrue(capacity.acquire())
        capacity.release()

    def test_consumer_cancellation_releases_its_slot(self):
        capacity = RequestCapacity(1)
        model = ToolCallingModel(responses=[AIMessage(content="Hello")])
        service = assistant(lambda: None, model=model)
        with patch("langchain_app.services.assistant_agent.get_request_capacity", return_value=capacity):
            stream = service.stream_response_events("Hello")
            next(stream)  # request_started precedes admission
            next(stream)  # a model/progress event occurs while the slot is held
            self.assertFalse(capacity.acquire())
            stream.close()
        self.assertTrue(capacity.acquire())
        capacity.release()

    def test_model_initialization_failure_releases_its_slot(self):
        capacity = RequestCapacity(1)
        service = assistant(lambda: None)
        with patch("langchain_app.services.assistant_agent.get_request_capacity", return_value=capacity), patch(
            "langchain_app.services.assistant_agent.init_chat_model", side_effect=ValueError("invalid config"),
        ), self.assertRaises(ValueError):
            list(service.stream_response_events("Hello"))
        self.assertTrue(capacity.acquire())
        capacity.release()
