"""Offline browser acceptance for catalog thumbnails and live/replayed output.

Run on port 8513 with static serving enabled. Generate creates three poses;
chat browsing displays 50 catalog items, including a missing-image fallback.
If CATALOG_DISPLAY_CHECK_DIR is set, the final browsing reply waits for a file
named reply in that directory so live thumbnails can be inspected first.
"""

import json
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

os.environ["LANGSMITH_TRACING"] = "false"


class BrowserModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        if isinstance(messages[-2], HumanMessage):
            request = str(messages[-2].content)
            if request.startswith("Generate"):
                name, args = "generate_sequence", {"request": request, "count": 3}
            else:
                name, args = "search_catalog_items", {"limit": 50}
            response = AIMessage(content="", tool_calls=[{
                "name": name, "args": args, "id": str(uuid4()), "type": "tool_call",
            }])
        else:
            last_call = next(message for message in reversed(messages) if isinstance(message, AIMessage)).tool_calls[0]
            if last_call["name"] == "search_catalog_items":
                ids = [item["id"] for item in json.loads(messages[-2].content)["items"]]
                response = AIMessage(content="", tool_calls=[{
                    "name": "show_catalog_items", "args": {"catalog_item_ids": ids},
                    "id": str(uuid4()), "type": "tool_call",
                }])
            else:
                if last_call["name"] == "show_catalog_items" and (directory := os.getenv("CATALOG_DISPLAY_CHECK_DIR")):
                    deadline = time.monotonic() + 300
                    while not (Path(directory) / "reply").exists():
                        if time.monotonic() > deadline:
                            raise ValueError("Browser fixture reply timed out")
                        time.sleep(0.05)
                response = AIMessage(content="Here are the catalog options." if last_call["name"] == "show_catalog_items" else "Generated 3 poses.")
        response.id = str(uuid4())
        self.responses = [response]
        self.i = 0
        return super()._generate(messages, **kwargs)


def make_assistant(*args):
    items = [CatalogItemSchema(
        id=number, uuid=str(uuid4()), name=f'Fixture pose {number} "<real name>"', bilateral=True,
        image_url=None if number == 17 else f"http://127.0.0.1:8513/app/static/pose.svg?pose={number}&fixture=catalog",
        catalog_tag_names=["Fixture"],
    ) for number in range(1, 51)]
    catalog, generator = Mock(), Mock()
    catalog.search.return_value = items
    generator.generate.return_value = items[:3]
    return AssistantAgent(
        generator_factory=lambda: generator, model=BrowserModel(responses=[]),
        catalog_item_service=catalog, catalog_tag_service=Mock(),
    )


with patch("langchain_app.dependencies.get_assistant_agent", side_effect=make_assistant):
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/langchain_app/assistant.py"), run_name="__main__")
