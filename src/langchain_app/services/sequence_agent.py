"""Route sequence requests through a LangChain agent and stream tool results."""

from collections.abc import Callable, Iterator
from typing import Annotated

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.errors import GraphRecursionError
from openai import OpenAIError
from pydantic import Field

from langchain_app.schemas import CatalogItemSchema, Language
from langchain_app.services.sequence_generator import SequenceGenerator


class SequenceAgentError(Exception):
    """The language model could not generate a sequence through its tool."""


class SequenceAgent:
    """Expose the existing generator as the agent's only tool."""

    def __init__(
        self,
        generator_factory: Callable[[], SequenceGenerator],
        *,
        model: str | BaseChatModel = "openai:gpt-5.5",
    ) -> None:
        self._generator_factory = generator_factory
        self._model = model

    def generate(
        self, request: str, *, language: Language
    ) -> Iterator[CatalogItemSchema]:
        """Yield poses while the LLM-selected generation tool is executing."""
        completed = False

        @tool(return_direct=True)
        def generate_sequence(
            request: str,
            count: Annotated[int, Field(gt=0)],
        ) -> str:
            """Generate a yoga sequence using the user's full sequence request.

            Args:
                request: The user's request, preserving all preferences and constraints.
                count: Number of poses inferred from an explicit count or duration,
                    using 40 seconds per pose.
            """
            nonlocal completed
            writer = get_stream_writer()
            generator = self._generator_factory()
            for item in generator.generate(request, language=language, count=count):
                writer(item)
            completed = True
            return "The generated sequence has been displayed to the user."

        try:
            model = (
                init_chat_model(
                    self._model,
                    timeout=300,
                    model_kwargs={"parallel_tool_calls": False},
                )
                if isinstance(self._model, str)
                else self._model
            )
            agent = create_agent(
                model=model,
                tools=[generate_sequence],
                system_prompt=(
                    "You generate yoga sequences using the generate_sequence tool. "
                    "For every request, call this tool exactly once. Pass the user's "
                    "full request, preserving all preferences and constraints. "
                    "Infer the number of poses from the user's request and pass it "
                    "as the count argument. The user may specify either an explicit "
                    "pose count or a sequence duration in minutes. If only a duration "
                    "is specified, assume each pose takes 40 seconds and calculate "
                    "count = duration_in_minutes * 60 / 40. Round to the nearest "
                    "whole number, rounding halves up, with a minimum of 1 pose. "
                    "For example, 10 minutes means 15 poses. If both a pose count "
                    "and a duration are specified, use the explicit pose count. "
                    "If neither a count nor a duration is specified, use 9 poses. "
                    f"The sequence language is {language}, already supplied to the tool. "
                    "Do not invent poses or answer with a sequence in text."
                ),
            )
            for chunk in agent.stream(
                {"messages": [{"role": "user", "content": request}]},
                config={"recursion_limit": 6},
                stream_mode="custom",
                version="v2",
            ):
                if chunk["type"] == "custom":
                    yield chunk["data"]
        except (OpenAIError, GraphRecursionError) as exc:
            raise SequenceAgentError(f"Language model request failed: {exc}") from exc

        if not completed:
            raise SequenceAgentError("The language model did not call the generation tool")
