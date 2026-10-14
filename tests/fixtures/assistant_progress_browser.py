"""Offline browser fixture for feature 002; run with Streamlit, never production.

Set PROGRESS_CHECK_DIR to an empty directory. Create files named dependencies,
model, search, selection and reply to release each deliberately blocked call.
This runs the real entrypoint, event producer, generator and UI with fake services.
"""

import os
import runpy
import time
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langchain_app.schemas import CatalogItemSchema
from langchain_app.services.assistant_agent import AssistantAgent
from langchain_app.services.sequence_generator import SequenceGenerator

os.environ["LANGSMITH_TRACING"] = "false"
CHECK_DIR = Path(os.environ["PROGRESS_CHECK_DIR"])


def wait_for(phase: str) -> None:
    deadline = time.monotonic() + 300
    while not (CHECK_DIR / phase).exists():
        if time.monotonic() > deadline:
            raise ValueError(f"Browser fixture timed out waiting for {phase}")
        time.sleep(0.05)


class BrowserModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        wait_for("model" if isinstance(messages[-2], HumanMessage) else "reply")
        result = super()._generate(messages, **kwargs)
        for generation in result.generations:
            generation.message = generation.message.model_copy(deep=True)
            generation.message.id = str(uuid4())
            for call in generation.message.tool_calls:
                call["id"] = str(uuid4())
        return result


def make_assistant(*args) -> AssistantAgent:
    wait_for("dependencies")
    items = [CatalogItemSchema(
        id=number, uuid=str(uuid4()), name=f"Fixture pose {number}", bilateral=False,
        image_url=f"http://127.0.0.1:8512/app/static/pose.svg?pose={number}",
        catalog_tag_names=["Other/Starting"] if number == 1 else ["Other/Finishing"] if number == 3 else [],
    ) for number in range(1, 4)]
    catalog, selector = Mock(), Mock()

    def search(**kwargs):
        wait_for("search")
        return items

    def select(candidates, **kwargs):
        wait_for("selection")
        return candidates[0]

    catalog.search.side_effect = search
    selector.choose_next_catalog_item.side_effect = select
    generator = SequenceGenerator(catalog, selector)
    model = BrowserModel(responses=[
        AIMessage(content="I'll find poses for your sequence.", tool_calls=[{
            "id": "generate", "name": "generate_sequence", "type": "tool_call",
            "args": {"request": "Generate a yoga sequence with 3 poses.", "count": 3},
        }]),
        AIMessage(content="Generated 3 poses."),
    ])
    return AssistantAgent(
        generator_factory=lambda: generator, model=model,
        catalog_item_service=catalog, catalog_tag_service=Mock(),
    )


with patch("langchain_app.dependencies.get_assistant_agent", side_effect=make_assistant):
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/langchain_app/assistant.py"), run_name="__main__")
