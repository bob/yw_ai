"""Opt-in live-model acceptance for feature 006; never run by unittest discovery.

uv run --env-file .env python tests/fixtures/sequence_swap_dialogue.py --output /tmp/swap-dialogue.json
Only the configured assistant LLM is live. Catalog, generator and TypeSafe are
forbidden; snapshots are synthetic and each case has an isolated checkpoint.
The JSON captures replies for manual review as well as structural assertions.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage

from langchain_app.schemas import CatalogItemSchema, GeneratedSequence
from langchain_app.services.assistant_agent import AssistantAgent


def snapshot(*, repeated=False):
    names = ["Mountain pose", "Cat pose", "Tree pose", "Cobra pose", "Corpse pose"]
    if repeated:
        names[1] = "Tree pose"
    result = GeneratedSequence.from_catalog_items([
        CatalogItemSchema(
            id=index, uuid=str(uuid4()), name=name, bilateral=index == 3,
            image_url=f"https://example.com/{index}.svg", catalog_tag_names=[],
        ) for index, name in enumerate(names, 1)
    ], request="Generate 5 poses for balance.")
    result.elements[2].start_mirror, result.elements[2].end_mirror = True, False
    result.elements[4].start_mirror, result.elements[4].end_mirror = False, True
    return result


CASES = [
    ("positions", "Поменяй третью и пятую асаны местами", (2, 4), "normal"),
    ("boundaries", "Поменяй первую и последнюю позы", (0, 4), "normal"),
    ("names", "Поменяй местами Tree pose и Mountain pose", (2, 0), "normal"),
    ("polite_action", "Можешь поменять третью и пятую местами?", (2, 4), "normal"),
    ("explanation", "Что изменится, если поменять третью и пятую местами?", None, "normal"),
    ("repeated_name", "Поменяй местами Tree pose и Mountain pose", None, "repeated"),
    ("ambiguous_pronoun", "Поменяй её местами с последней позой", None, "normal"),
    ("missing_second", "Поменяй местами третью позу", None, "normal"),
    ("out_of_range", "Поменяй третью и шестую позы местами", None, "normal"),
    ("no_selection", "Поменяй третью и пятую местами", None, "missing"),
    ("multiple_swaps", "Поменяй первую и вторую, а также третью и пятую местами", None, "normal"),
    ("swap_and_replace", "Поменяй третью и пятую местами и замени вторую на Warrior pose", None, "normal"),
    ("same_occurrence", "Поменяй третью позу местами с третьей", (2, 2), "normal"),
    ("older_selection", "Поменяй первую и последнюю позы", (0, 4), "older"),
    ("english_reply", "Swap the third and fifth poses", (2, 4), "normal"),
    ("duplicate_positions", "Поменяй вторую и третью позы местами", (1, 2), "repeated"),
]


def run_case(model, case):
    name, request, pair, mode = case
    forbidden = Mock(side_effect=AssertionError("Swap must not call external services"))
    catalog, tags = Mock(), Mock()
    catalog.search = forbidden
    tags.get_names = forbidden
    service = AssistantAgent(
        forbidden, model=model, catalog_item_service=catalog, catalog_tag_service=tags,
    )
    source = snapshot(repeated=mode == "repeated")
    sources = [source] if mode != "missing" else []
    history = []
    if mode == "older":
        newer = snapshot()
        newer.elements.reverse()
        for order, element in enumerate(newer.elements, 1):
            element.order = order
        sources.append(newer)
        history = [
            AIMessage(content="", tool_calls=[{
                "name": "generate_sequence", "args": {"request": "Newer practice", "count": 5},
                "id": "older-generation", "type": "tool_call",
            }]),
            ToolMessage(content=json.dumps({"status": "ok", **newer.model_dump(mode="json")}),
                        tool_call_id="older-generation", name="generate_sequence"),
            AIMessage(content="Generated a newer practice."),
        ]
    service._get_agent().update_state(service._config, {
        "messages": history,
        "sequences": {str(item.uuid): item.model_dump(mode="json") for item in sources},
        "active_sequence_uuid": str(source.uuid) if sources else None,
    })
    before = service.state
    events = list(service.stream_response_events(request))
    messages = service.state["messages"][len(history):]
    calls = [call for message in messages if isinstance(message, AIMessage) for call in message.tool_calls]
    reply = "\n".join(event.text for event in events if event.type == "assistant_message" and event.purpose == "final")
    commits = [event for event in events if event.type == "sequence_committed"]
    problems = []
    if pair is None:
        if calls or commits or service.state["sequences"] != before["sequences"]:
            problems.append("Expected a text reply without tools or changes")
    else:
        expected_ids = {str(source.elements[index].element_id) for index in pair}
        if len(calls) != 1 or calls[0]["name"] != "swap_sequence_elements":
            problems.append("Expected exactly one swap call")
        elif calls[0]["args"].get("sequence_uuid") != str(source.uuid) or {
            calls[0]["args"].get("first_element_id"), calls[0]["args"].get("second_element_id"),
        } != expected_ids:
            problems.append("Expected UUIDs from the selected snapshot")
        if pair[0] == pair[1]:
            if commits or service.state["sequences"] != before["sequences"]:
                problems.append("The same occurrence must not create a variant")
        else:
            expected = list(source.elements)
            expected[pair[0]], expected[pair[1]] = expected[pair[1]], expected[pair[0]]
            variant = service.get_sequence(service.active_sequence_uuid)
            if len(commits) != 1 or variant is None or variant.operation != "swap":
                problems.append("Expected one committed swap variant")
            elif [item.element_id for item in variant.elements] != [item.element_id for item in expected]:
                problems.append("Wrong swapped order")
            elif variant.parent_uuid != source.uuid or variant.family_uuid != source.family_uuid:
                problems.append("Wrong selected parent or family")
    if commits:
        if reply.strip():
            problems.append("Expected no completion text for the displayed swap")
    elif not reply.strip():
        problems.append("Expected an explanation or clarification")
    if forbidden.mock_calls or catalog.mock_calls or tags.mock_calls:
        problems.append("Unexpected catalog, tag or generator call")
    for item in sources:
        if service.get_sequence(str(item.uuid)) != item:
            problems.append("An existing variant changed")
    return {
        "case": name, "request": request, "passed": not problems, "problems": problems,
        "tool_calls": calls, "reply": reply, "commit_count": len(commits),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=(os.getenv("ASSISTANT_AGENT_MODEL", "").strip()
                        or os.getenv("SEQUENCE_AGENT_MODEL", "").strip() or "openai:gpt-5.5"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", help="Run only a named case; may be repeated")
    args = parser.parse_args()
    os.environ["LANGSMITH_TRACING"] = "false"
    report = {"model": args.model, "started_at": datetime.now(timezone.utc).isoformat(), "cases": []}
    for case in CASES:
        if args.case and case[0] not in args.case:
            continue
        result = run_case(args.model, case)
        report["cases"].append(result)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"{case[0]}: {'PASS' if result['passed'] else 'FAIL'} — {result['reply']}", flush=True)
    if not report["cases"] or any(not case["passed"] for case in report["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
