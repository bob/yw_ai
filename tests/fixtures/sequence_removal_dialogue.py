"""Opt-in live-model checks for feature 015 instructions, outside unittest discovery.

uv run --env-file .env python tests/fixtures/sequence_removal_dialogue.py --output /tmp/removal-dialogue.json
Only the assistant LLM is live. Catalog, generator and TypeSafe are forbidden.
Progress is recorded; bounded-history and UI/Save have separate acceptance checks.
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from sequence_swap_dialogue import snapshot
from langchain_app.model_config import DEFAULT_CHAT_MODEL, chat_model_options
from langchain_app.services.assistant_agent import AssistantAgent, SYSTEM_PROMPT
from langchain_app.services.prompt_cache import prefix_version
from langchain_app.services.sequence_context import GENERATION_INSTRUCTION, SELECTED_SEQUENCE_INSTRUCTIONS
from langchain_app.tools import ASSISTANT_TOOLS


# name, request, zero-based targets (None means no tools), source mode
CASES = [
    ("single_ru", "Удали третью асану", (2,), "normal"),
    ("single_en", "Remove the third pose", (2,), "normal"),
    ("list_ru", "Удали вторую и четвёртую асаны", (1, 3), "normal"),
    ("list_en", "Remove the second, fourth and fifth poses", (1, 3, 4), "normal"),
    ("range_ru", "Удали асаны со второй по четвёртую включительно", (1, 2, 3), "normal"),
    ("range_en", "Remove poses two through four, inclusive", (1, 2, 3), "normal"),
    ("boundaries_ru", "Убери первую и последнюю позы", (0, 4), "normal"),
    ("boundaries_en", "Remove the first and last poses", (0, 4), "normal"),
    ("last_two_ru", "Убери последние две асаны", (3, 4), "normal"),
    ("last_two_en", "Remove the last two poses", (3, 4), "normal"),
    ("name", "Удали Tree pose", (2,), "normal"),
    ("polite_ru", "Можешь удалить третью?", (2,), "normal"),
    ("polite_en", "Can you remove the third pose?", (2,), "normal"),
    ("duplicate_reference", "Удали вторую, четвёртую и ещё раз вторую асаны", (1, 3), "normal"),
    ("same_occurrence_by_name", "Удали третью асану и Tree pose", (2,), "normal"),
    ("distinct_repeated_items", "Удали вторую и третью асаны", (1, 2), "repeated"),
    ("leave_one", "Удали первые четыре асаны", (0, 1, 2, 3), "normal"),
    ("older_selection", "Удали вторую и четвёртую асаны", (1, 3), "older"),
    ("requested_explanation", "Удали третью асану и объясни, как изменится порядок остальных", (2,), "explain"),
    ("hypothetical_ru", "Что изменится, если удалить третью асану?", None, "normal"),
    ("hypothetical_en", "What would change if we removed the third pose?", None, "normal"),
    ("repeated_name", "Удали Tree pose", None, "repeated"),
    ("ambiguous_pronoun", "Удали её", None, "normal"),
    ("missing_target", "Удали асану", None, "normal"),
    ("shorten_count", "Сократи секвенцию до трёх асан", None, "normal"),
    ("shorten_duration", "Сделай эту секвенцию короче на две минуты", None, "normal"),
    ("out_of_range", "Удали вторую и шестую асаны", None, "normal"),
    ("zero_position", "Удали нулевую и вторую асаны", None, "normal"),
    ("partly_unresolved", "Удали вторую асану и Warrior pose", None, "normal"),
    ("no_selection", "Удали третью асану", None, "missing"),
    ("remove_all_ru", "Удали все асаны", None, "normal"),
    ("remove_all_en", "Remove all poses", None, "normal"),
    ("only_pose", "Удали единственную оставшуюся асану", None, "single"),
    ("remove_and_swap", "Удали вторую и поменяй первую с третьей местами", None, "normal"),
    ("remove_and_replace", "Удали вторую и замени четвёртую на Warrior pose", None, "normal"),
    ("remove_and_add", "Удали вторую и добавь две позы перед последней", None, "normal"),
    ("remove_and_generate", "Удали вторую и сгенерируй новую секвенцию из пяти асан", None, "normal"),
]


def run_case(model, case):
    name, request, targets, mode = case
    forbidden = Mock(side_effect=AssertionError("Removal must not call external services"))
    catalog, tags = Mock(), Mock()
    catalog.search = forbidden
    tags.get_names = forbidden
    service = AssistantAgent(
        forbidden, model=model, catalog_item_service=catalog, catalog_tag_service=tags,
    )
    source = snapshot(repeated=mode == "repeated")
    if mode == "single":
        source.elements = source.elements[:1]
    sources = [] if mode == "missing" else [source]
    history = []
    if mode == "older":
        newer = source.model_copy(deep=True, update={
            "uuid": uuid4(), "parent_uuid": source.uuid, "operation": "swap",
        })
        newer.elements[0], newer.elements[-1] = newer.elements[-1], newer.elements[0]
        for order, element in enumerate(newer.elements, 1):
            element.order = order
        sources.append(newer)
        history = [
            HumanMessage(content="Поменяй первую и последнюю асаны местами"),
            AIMessage(content="", tool_calls=[{
                "name": "swap_sequence_elements", "args": {
                    "sequence_uuid": str(source.uuid),
                    "first_element_id": str(source.elements[0].element_id),
                    "second_element_id": str(source.elements[-1].element_id),
                }, "id": "previous-swap", "type": "tool_call",
            }]),
            ToolMessage(content=json.dumps({
                "status": "ok", "uuid": str(newer.uuid), "operation": "swap", "elements_count": 5,
            }), tool_call_id="previous-swap", name="swap_sequence_elements"),
            AIMessage(content=""),
        ]
    service._get_agent().update_state(service._config, {
        "messages": history,
        "sequences": {str(item.uuid): item.model_dump(mode="json") for item in sources},
        "active_sequence_uuid": str(source.uuid) if sources else None,
    })
    before = service.state
    events, problems = [], []
    try:
        events.extend(service.stream_response_events(request))
    except Exception as exc:
        # Preserve attempted calls and partial events when a forbidden service raises.
        problems.append(f"{type(exc).__name__}: {exc}")
    messages = service.state["messages"][len(history):]
    calls = [call for message in messages if isinstance(message, AIMessage) for call in message.tool_calls]
    tool_results = []
    for message in messages:
        if isinstance(message, ToolMessage):
            try:
                tool_results.append(json.loads(message.text))
            except ValueError:
                tool_results.append({"status": message.status, "message": message.text})
    reply = messages[-1].text if isinstance(messages[-1], AIMessage) else ""
    if targets is None:
        if calls or service.state["sequences"] != before["sequences"]:
            problems.append("Expected a text reply without any tools or changes")
        if service.active_sequence_uuid != before.get("active_sequence_uuid"):
            problems.append("Active selection changed without an operation")
        if service.state.get("turn_operation_committed"):
            problems.append("Unexpected committed operation")
    else:
        expected_ids = {str(source.elements[index].element_id) for index in targets}
        if len(calls) != 1 or calls[0]["name"] != "remove_sequence_elements":
            problems.append("Expected exactly one removal call and no other tools")
        else:
            arguments = calls[0]["args"]
            actual_ids = arguments.get("element_ids", [])
            if (arguments.get("sequence_uuid") != str(source.uuid)
                    or set(actual_ids) != expected_ids or len(actual_ids) != len(expected_ids)):
                problems.append("Expected unique occurrence UUIDs from the selected snapshot")
        variant = service.get_sequence(service.active_sequence_uuid)
        expected = [element for index, element in enumerate(source.elements) if index not in targets]
        if (len(service.state["sequences"]) != len(sources) + 1
                or variant is None or variant.operation != "remove"
                or not service.state.get("turn_operation_committed")):
            problems.append("Expected one committed removal variant")
        elif ([element.element_id for element in variant.elements] != [element.element_id for element in expected]
              or variant.parent_uuid != source.uuid or variant.family_uuid != source.family_uuid
              or variant.operation_request != request):
            problems.append("Wrong remaining occurrences, selected parent, family or request")
        if len(tool_results) != 1 or tool_results[0].get("status") != "ok":
            problems.append("Expected one successful tool result")
    if targets is None or mode == "explain":
        if not reply.strip():
            problems.append("Expected a useful explanation or clarification")
    elif reply != "":
        problems.append("Expected exactly zero final text after ordinary success")
    if reply.strip():
        # Synthetic catalog names are all English, so Cyrillic belongs to the reply.
        has_cyrillic = bool(re.search(r"[А-Яа-яЁё]", reply))
        expects_cyrillic = not name.endswith("_en")
        if has_cyrillic != expects_cyrillic:
            problems.append("Reply language does not match the current request")
        if mode == "explain" and reply.startswith(("Асана удалена", "Готово", "Done")):
            problems.append("Requested explanation includes a redundant completion preamble")
    if any(event.type == "request_finished" and event.outcome == "error" for event in events):
        problems.append("Request failed")
    if forbidden.mock_calls or catalog.mock_calls or tags.mock_calls:
        problems.append("Unexpected catalog, tag or generator call")
    for item in sources:
        if service.get_sequence(str(item.uuid)) != item:
            problems.append("An existing variant changed")
    return {"case": name, "request": request, "passed": not problems, "problems": problems,
            "selected_sequence": source.model_context() if sources else None,
            "tool_calls": calls, "tool_results": tool_results, "reply": reply, "usage": service.model_usage,
            "events": [event.model_dump(mode="json") for event in events]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("ASSISTANT_AGENT_MODEL", "").strip() or DEFAULT_CHAT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", help="Run only a named case; may be repeated")
    args = parser.parse_args()
    os.environ["LANGSMITH_TRACING"] = "false"
    model = init_chat_model(args.model, timeout=60, max_retries=0, disable_streaming=True,
                            model_kwargs={"parallel_tool_calls": False}, **chat_model_options(args.model))
    report = {"model": args.model, "started_at": datetime.now(timezone.utc).isoformat(),
              "system_prompt_sha256": sha256(SYSTEM_PROMPT.encode()).hexdigest(),
              "context_instructions_sha256": sha256(SELECTED_SEQUENCE_INSTRUCTIONS.encode()).hexdigest(),
              "prefix_version": prefix_version(SYSTEM_PROMPT, ASSISTANT_TOOLS,
                                                context_instructions=SELECTED_SEQUENCE_INSTRUCTIONS + GENERATION_INSTRUCTION),
              "conditions": "Live LLM; synthetic snapshots; no external services, retries or tracing",
              "cases": []}
    for case in CASES:
        if args.case and case[0] not in args.case:
            continue
        try:
            result = run_case(model, case)
        except Exception as exc:
            result = {"case": case[0], "request": case[1], "passed": False,
                      "problems": [str(exc)], "error_type": type(exc).__name__}
        report["cases"].append(result)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"{case[0]}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    if not report["cases"] or any(not case["passed"] for case in report["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
