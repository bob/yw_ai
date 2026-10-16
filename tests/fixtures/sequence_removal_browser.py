"""Browser acceptance for feature 015 with real UI/tools and a local Save receiver.

uv run streamlit run tests/fixtures/sequence_removal_browser.py --server.port 8520
Open http://localhost:8520, then submit 'Удали вторую и четвёртую асаны'.
Save captures the real postMessage payload on 8521 without writing to the catalog.
Use ?count=50 for a long imported snapshot or ?fail_final=1 for a final-model error.
REMOVAL_CHECK_DIR (default /tmp/015-removal-browser) holds state and import payloads.
"""

import json
import os
import runpy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse
from uuid import NAMESPACE_URL, uuid4, uuid5

import streamlit as st
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

from langchain_app.schemas import CatalogItemSchema, GeneratedSequence
from langchain_app.services.assistant_agent import AssistantAgent, AssistantAgentError
from langchain_app.ui.sequence import generated_sequence_view

CHECK_DIR = Path(os.getenv("REMOVAL_CHECK_DIR", "/tmp/015-removal-browser"))
CHECK_DIR.mkdir(parents=True, exist_ok=True)
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["ASANA_SERVICE_BASE_URL"] = "http://localhost:8521"

IMPORT_PAGE = b"""<!doctype html><html><title>015 test import receiver</title>
<h1>015 test import receiver</h1><p id="status">Waiting for the sequence</p><pre id="payload"></pre>
<script>
const origin = new URLSearchParams(location.search).get('origin');
if (origin === 'http://localhost:8520') {
  addEventListener('message', async (event) => {
    if (event.origin !== origin || event.source !== opener ||
        event.data?.type !== 'asana-catalog:sequence-import:save') return;
    const response = await fetch('/payload', {method: 'POST', body: JSON.stringify(event.data.sequence)});
    if (!response.ok) {
      document.querySelector('#status').textContent = 'Rejected invalid test payload';
      opener.postMessage({type: 'asana-catalog:sequence-import:error', error: 'Expected 1..75 poses.'}, origin);
      return;
    }
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
            count = len(payload["thumbnails"])
            valid = 1 <= count <= 75 and payload["info"]["elements_count"] == count
            (CHECK_DIR / f"{'import' if valid else 'rejected'}-{uuid4()}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
            self.send_response(200 if valid else 422)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 8521), Receiver)
    Thread(target=server.serve_forever, daemon=True).start()
    return server


class BrowserModel(FakeMessagesListChatModel):
    requests: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    fail_final: bool = False

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        last = messages[-2]
        if isinstance(last, HumanMessage):
            request = last.text
            self.requests.append(request)
            context = messages[-1].content_blocks[-1]["text"]
            source = json.loads(next(line for line in context.splitlines()
                                     if line.startswith('{"selected_sequence":')))["selected_sequence"]
            if request == "Удали вторую и четвёртую асаны":
                targets = [source["elements"][i]["element_id"] for i in (3, 1)]
            elif request == "Удали вторую асану":
                targets = [source["elements"][1]["element_id"]]
            elif request == "Оставь только первую асану":
                targets = [e["element_id"] for e in source["elements"][1:]]
            elif request == "Проверь ошибку удаления":
                targets = [source["elements"][1]["element_id"], str(uuid4())]
            else:
                raise ValueError("Use one of the documented fixture requests")
            self.tool_names.append("remove_sequence_elements")
            message = AIMessage(content="", tool_calls=[{
                "name": "remove_sequence_elements", "args": {"sequence_uuid": source["uuid"], "element_ids": targets},
                "id": str(uuid4()), "type": "tool_call",
            }])
        else:
            if self.fail_final:
                raise AssistantAgentError("Fixture model unavailable after commit")
            payload = json.loads(last.text)
            message = AIMessage(content="Неизвестное вхождение: секвенция не изменена." if payload["status"] == "error" else "")
        self.responses = [message]
        result = super()._generate(messages, **kwargs)
        result.generations[0].message.id = str(uuid4())
        return result


def make_assistant():
    count = int(st.query_params.get("count", "5"))
    names = ["Mountain pose", "Cat pose", "Tree pose", "Cobra pose", "Corpse pose"]
    source = GeneratedSequence.from_catalog_items([
        CatalogItemSchema(
            id=number, uuid=str(uuid5(NAMESPACE_URL, f"015-browser-pose/{number}")),
            name=names[number - 1] if number <= len(names) else f"Pose {number}", bilateral=number == 3,
            image_url=f"http://localhost:8521/pose.svg?pose={number}", catalog_tag_names=[],
        ) for number in range(1, count + 1)
    ], request=f"Original imported practice with {count} poses")
    source.elements[0].start_mirror, source.elements[0].end_mirror = True, False
    source.elements[2].start_mirror, source.elements[2].end_mirror = False, True
    forbidden = Mock(side_effect=AssertionError("Removal must not call external services"))
    service = AssistantAgent(
        forbidden, model=BrowserModel(responses=[AIMessage(content="")], fail_final=st.query_params.get("fail_final") == "1"),
        catalog_item_service=forbidden, catalog_tag_service=forbidden,
    )
    service._get_agent().update_state(service._config, {
        "sequences": {str(source.uuid): source.model_dump(mode="json")}, "active_sequence_uuid": str(source.uuid),
    })
    st.session_state.messages = [{
        "role": "assistant", "content": "", "sequence": generated_sequence_view(source),
        "sequence_uuid": str(source.uuid), "sequence_variant_uuids": [str(source.uuid)],
        "sequence_request": source.generation_request, "selected_sequence_index": 0,
    }]
    return service


import_receiver()
if "assistant_agent" not in st.session_state:
    st.session_state.assistant_agent = make_assistant()
with patch("langchain_app.dependencies.get_assistant_agent", return_value=st.session_state.assistant_agent):
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/langchain_app/assistant.py"), run_name="__main__")
service = st.session_state.assistant_agent
state_json = json.dumps({
    "sequences": service.state.get("sequences", {}), "active_sequence_uuid": service.active_sequence_uuid,
    "requests": service._model.requests, "tool_names": service._model.tool_names,
    "external_calls": len(service._generator_factory.mock_calls),
}, ensure_ascii=False, indent=2) + "\n"
(CHECK_DIR / "state.json").write_text(state_json)
(CHECK_DIR / f"state-{service.active_sequence_uuid}.json").write_text(state_json)
