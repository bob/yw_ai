"""Offline browser acceptance for feature 006, using the real Streamlit entrypoint.

uv run streamlit run tests/fixtures/sequence_swap_browser.py --server.port 8516
Submit 'Generate a yoga sequence with 5 poses.', then
'Поменяй третью и пятую местами'. Save opens a local postMessage receiver on
8517 instead of writing to Asana Catalog. SWAP_CHECK_DIR (default /tmp/006-swap-browser)
receives state.json and each captured import payload. No external services run.
"""

import json
import os
import runpy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import streamlit as st
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

from langchain_app.schemas import CatalogItemSchema
from langchain_app.services.assistant_agent import AssistantAgent

CHECK_DIR = Path(os.getenv("SWAP_CHECK_DIR", "/tmp/006-swap-browser"))
CHECK_DIR.mkdir(parents=True, exist_ok=True)
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["ASANA_SERVICE_BASE_URL"] = "http://localhost:8517"

IMPORT_PAGE = b"""<!doctype html><html><title>006 test import receiver</title>
<h1>006 test import receiver</h1><p id="status">Waiting for the sequence</p><pre id="payload"></pre>
<script>
const origin = new URLSearchParams(location.search).get('origin');
if (origin === 'http://localhost:8516') {
  addEventListener('message', async (event) => {
    if (event.origin !== origin || event.source !== opener ||
        event.data?.type !== 'asana-catalog:sequence-import:save') return;
    const response = await fetch('/payload', {method: 'POST', body: JSON.stringify(event.data.sequence)});
    if (!response.ok) throw new Error('Could not capture payload');
    document.querySelector('#payload').textContent = JSON.stringify(event.data.sequence, null, 2);
    document.querySelector('#status').textContent = 'Captured test sequence ' + event.data.sequence.info.uuid;
    opener.postMessage({type: 'asana-catalog:sequence-import:saved'}, origin);
  });
  opener.postMessage({type: 'asana-catalog:sequence-import:ready'}, origin);
}
</script></html>"""


@st.cache_resource
def import_receiver():
    class Receiver(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/sequences/import":
                content, content_type = IMPORT_PAGE, "text/html; charset=utf-8"
            elif url.path == "/pose.svg":
                number = int(parse_qs(url.query).get("pose", [1])[0])
                content = (
                    '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">'
                    f'<rect width="64" height="64" rx="8" fill="hsl({number * 50},50%,80%)"/>'
                    f'<text x="32" y="43" text-anchor="middle" font-size="32">{number}</text></svg>'
                ).encode()
                content_type = "image/svg+xml"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            if self.path != "/payload":
                self.send_error(404)
                return
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            (CHECK_DIR / f"import-{uuid4()}.json").write_text(json.dumps(payload, indent=2) + "\n")
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 8517), Receiver)
    Thread(target=server.serve_forever, daemon=True).start()
    return server


class BrowserModel(FakeMessagesListChatModel):
    requests: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        if isinstance(messages[-2], HumanMessage):
            request = messages[-2].content
            self.requests.append(request)
            if request.startswith("Generate"):
                name, args = "generate_sequence", {"request": request, "count": 5}
            elif request == "Поменяй третью и пятую местами":
                context = messages[-1].content_blocks[-1]["text"]
                source = json.loads(next(line for line in context.splitlines()
                                         if line.startswith('{"selected_sequence":')))["selected_sequence"]
                name, args = "swap_sequence_elements", {
                    "sequence_uuid": source["uuid"],
                    "first_element_id": source["elements"][2]["element_id"],
                    "second_element_id": source["elements"][4]["element_id"],
                }
            else:
                raise ValueError("Use one of the documented fixture requests")
            self.tool_names.append(name)
            message = AIMessage(content="", tool_calls=[{
                "name": name, "args": args, "id": str(uuid4()), "type": "tool_call",
            }])
        else:
            payload = json.loads(messages[-2].content)
            message = AIMessage(content=(
                "Tree pose перемещена с позиции 3 на 5, Corpse pose — с позиции 5 на 3."
                if "swap" in payload else "Generated 5 poses."
            ))
        self.responses = [message]
        result = super()._generate(messages, **kwargs)
        result.generations[0].message.id = str(uuid4())
        return result


def make_assistant(*args):
    names = ["Mountain pose", "Cat pose", "Tree pose", "Cobra pose", "Corpse pose"]
    generator = Mock()
    generator.generate.return_value = [CatalogItemSchema(
        id=number, uuid=str(uuid4()), name=name, bilateral=number == 3,
        image_url=f"http://localhost:8517/pose.svg?pose={number}", catalog_tag_names=[],
    ) for number, name in enumerate(names, 1)]
    return AssistantAgent(
        generator_factory=Mock(return_value=generator),
        model=BrowserModel(responses=[AIMessage(content="")]),
        catalog_item_service=Mock(), catalog_tag_service=Mock(),
    )


import_receiver()
with patch("langchain_app.dependencies.get_assistant_agent", side_effect=make_assistant):
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/langchain_app/assistant.py"), run_name="__main__")
if service := st.session_state.get("assistant_agent"):
    (CHECK_DIR / "state.json").write_text(json.dumps({
        "sequences": service.state.get("sequences", {}), "active_sequence_uuid": service.active_sequence_uuid,
        "requests": service._model.requests, "tool_names": service._model.tool_names,
    }, ensure_ascii=False, indent=2) + "\n")
