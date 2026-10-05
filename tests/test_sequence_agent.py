"""Exercise the real agent/tool loop without external model calls."""

import unittest
from threading import Event
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from openai import APIConnectionError
import httpx
from pydantic import Field

from langchain_app.schemas import CatalogItemSchema
from langchain_app.services.sequence_agent import SequenceAgent, SequenceAgentError


class ToolCallingModel(FakeMessagesListChatModel):
    received_requests: list[str] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received_requests.append(messages[-1].content)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def tool_model(request, count=9):
    return ToolCallingModel(responses=[AIMessage(content="", tool_calls=[{
        "name": "generate_sequence",
        "args": {"request": request, "count": count},
        "id": "sequence-call",
        "type": "tool_call",
    }])])


def pose(number):
    return CatalogItemSchema(
        id=number, uuid=None, name=f"Pose {number}", bilateral=False,
        image_url=f"https://example.com/{number}.png", catalog_tag_names=[],
    )


class SequenceAgentTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_tool_streams_poses_before_generation_finishes(self):
        resume = Event()
        self.addCleanup(resume.set)
        poses = [pose(1), pose(2)]

        def generate(request, **kwargs):
            yield poses[0]
            if not resume.wait(timeout=3):
                raise RuntimeError("First pose was buffered until tool completion")
            yield poses[1]

        generator = Mock()
        generator.generate.side_effect = generate
        factory = Mock(return_value=generator)
        request = "gentle practice with 12 poses"
        model = tool_model(request, count=12)
        service = SequenceAgent(factory, model=model)
        sequence = service.generate(request, language="uk")
        factory.assert_not_called()
        self.assertEqual(next(sequence), poses[0])
        resume.set()
        self.assertEqual(list(sequence), poses[1:])
        factory.assert_called_once()
        generator.generate.assert_called_once_with(
            request, language="uk", count=12
        )
        self.assertEqual(model.received_requests, [request])

    def test_text_only_model_response_does_not_bypass_tool(self):
        factory = Mock()
        service = SequenceAgent(factory, model=ToolCallingModel(
            responses=[AIMessage(content="Here is a yoga sequence")]
        ))
        with self.assertRaisesRegex(SequenceAgentError, "did not call"):
            list(service.generate("practice", language="en"))
        factory.assert_not_called()

    def test_partial_generation_failure_is_propagated(self):
        def generate(*args, **kwargs):
            yield pose(1)
            raise ValueError("generation failed")

        generator = Mock()
        generator.generate.side_effect = generate
        service = SequenceAgent(lambda: generator, model=tool_model("practice"))
        sequence = service.generate("practice", language="en")
        self.assertEqual(next(sequence), pose(1))
        with self.assertRaisesRegex(ValueError, "generation failed"):
            list(sequence)

    def test_model_connection_error_is_wrapped(self):
        factory = Mock()
        error = APIConnectionError(request=httpx.Request("POST", "https://example.com"))
        with patch(
            "langchain_app.services.sequence_agent.init_chat_model", side_effect=error
        ):
            with self.assertRaisesRegex(SequenceAgentError, "Language model request failed"):
                list(SequenceAgent(factory).generate("practice", language="en"))
        factory.assert_not_called()

    def test_invalid_tool_count_does_not_start_generation(self):
        factory = Mock()
        service = SequenceAgent(factory, model=tool_model("practice", count=0))
        with self.assertRaises(SequenceAgentError):
            list(service.generate("practice", language="en"))
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
