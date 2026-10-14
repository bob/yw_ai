"""Opt-in live LLM acceptance for 007; catalog and TypeSafe remain offline.

uv run --env-file .env python tests/fixtures/sequence_addition_dialogue.py --output /tmp/addition-dialogue.json
Each case gets an isolated checkpoint. Review replies in the JSON alongside
the structural checks; this does not verify live TypeSafe pose quality.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID, uuid4

from langchain_core.messages import AIMessage, ToolMessage

from langchain_app.schemas import CatalogItemSchema, GeneratedSequence
from langchain_app.services.assistant_agent import AssistantAgent
from langchain_app.services.sequence_generator import GeneratorAdditionProgress, GeneratorSearchProgress


def pose(number, name=None):
    return CatalogItemSchema(
        id=number, uuid=str(UUID(int=number)), name=name or f"Pose {number}",
        bilateral=number % 2 == 0, image_url=f"https://example.com/{number}.svg",
        catalog_tag_names=["Other/Starting", "Position/Sitting"] if number == 100 else [],
    )


def snapshot(count=5, repeated=False):
    names = ["Mountain pose", "Cat pose", "Tree pose", "Cobra pose", "Corpse pose"]
    if repeated:
        names[1] = "Tree pose"
    result = GeneratedSequence.from_catalog_items([
        pose(n, names[n - 1] if n <= 5 else None) for n in range(1, count + 1)
    ], request=f"Generate {count} poses for balance.")
    result.elements[2].start_mirror, result.elements[2].end_mirror = True, False
    return result


# name, request, requested delta (None means no edit), insertion index, source mode
CASES = [
    ("default_one", "Добавь асану", 1, None, "normal"),
    ("append_many", "Добавь ещё три асаны стоя", 3, None, "normal"),
    ("before_third", "Добавь две асаны перед третьей", 2, 2, "normal"),
    ("after_second", "Добавь две асаны после второй", 2, 2, "normal"),
    ("prepend", "Добавь одну асану в начало", 1, 0, "normal"),
    ("after_last", "Добавь одну асану после последней", 1, None, "normal"),
    ("target_20", "Дополни секвенцию до 20 элементов", 15, None, "normal"),
    ("another_20", "Добавь ещё 20 элементов", 20, None, "normal"),
    ("fill_limit", "Дополни до лимита", 40, None, "normal"),
    ("truncated", "Добавь ещё пять асан", 5, None, "near_full"),
    ("full", "Добавь одну асану", 1, None, "full"),
    ("already_target", "Дополни до 5 элементов", None, None, "normal"),
    ("lower_target", "Дополни до 3 элементов", None, None, "normal"),
    ("polite_action", "Можешь добавить две асаны после второй?", 2, 2, "normal"),
    ("english", "Add two poses before the third pose", 2, 2, "normal"),
    ("english_default", "Add a pose", 1, None, "normal"),
    ("unique_name", "Добавь одну асану перед Tree pose", 1, 2, "normal"),
    ("older_selection", "Добавь две асаны перед третьей", 2, 2, "older"),
    ("repeated_name", "Добавь асану перед Tree pose", None, None, "repeated"),
    ("ambiguous_pronoun", "Добавь асану после неё", None, None, "normal"),
    ("out_of_range", "Добавь асану после шестой", None, None, "normal"),
    ("no_selection", "Добавь две асаны", None, None, "missing"),
    ("multiple_locations", "Добавь одну асану в начало и две после третьей", None, None, "normal"),
    ("add_and_replace", "Добавь две асаны и замени вторую на Warrior pose", None, None, "normal"),
    ("add_and_swap", "Добавь две асаны и поменяй первую и последнюю местами", None, None, "normal"),
    ("explanation", "Что изменится, если добавить две асаны перед третьей?", None, None, "normal"),
    ("duration_only", "Добавь асаны ещё на пять минут", None, None, "normal"),
    ("candidates", "Найди кандидатов для добавления, пока ничего не меняй", None, None, "browse"),
]


def run_case(model, case):
    name, request, delta, index, mode = case
    forbidden = Mock(side_effect=AssertionError("Unexpected generation or tags call"))
    generator, catalog, tags = Mock(), Mock(), Mock()
    generator.generate = forbidden
    tags.get_names.return_value = ["Position/Standing", "Position/Sitting"]
    catalog.search.return_value = [pose(110, "Seated forward bend"), pose(111, "Standing balance")]

    def add(sequence, request, *, count, progress_observer, **kwargs):
        progress_observer(GeneratorAdditionProgress(0, count))
        progress_observer(GeneratorSearchProgress("started"))
        progress_observer(GeneratorSearchProgress("finished", candidate_count=50))
        items = [pose(n) for n in range(100 if index == 0 else 110, (100 if index == 0 else 110) + count)]
        for completed in range(1, count + 1):
            progress_observer(GeneratorAdditionProgress(completed, count))
        return items

    generator.add_elements.side_effect = add
    factory = Mock(return_value=generator)
    service = AssistantAgent(factory, model=model, catalog_item_service=catalog, catalog_tag_service=tags)
    source = snapshot(count=43 if mode == "near_full" else 45 if mode == "full" else 5, repeated=mode == "repeated")
    sources = [] if mode == "missing" else [source]
    history = []
    if mode == "older":
        newer = snapshot()
        newer.elements.reverse()
        for order, element in enumerate(newer.elements, 1):
            element.order = order
        sources.append(newer)
        history = [
            AIMessage(content="", tool_calls=[{"name": "generate_sequence", "args": {"request": "Newer practice", "count": 5},
                                              "id": "older-generation", "type": "tool_call"}]),
            ToolMessage(content=json.dumps({"status": "ok", **newer.model_dump(mode="json")}),
                        tool_call_id="older-generation", name="generate_sequence"),
            AIMessage(content="Generated a newer practice."),
        ]
    service._get_agent().update_state(service._config, {
        "messages": history, "sequences": {str(s.uuid): s.model_dump(mode="json") for s in sources},
        "active_sequence_uuid": str(source.uuid) if sources else None,
    })
    events = list(service.stream_response_events(request))
    messages = service.state["messages"][len(history):]
    calls = [call for message in messages if isinstance(message, AIMessage) for call in message.tool_calls]
    replies = [e.text for e in events if e.type == "assistant_message" and e.purpose == "final"]
    results = [json.loads(m.text) for m in messages if isinstance(m, ToolMessage)]
    commits = [e for e in events if e.type == "sequence_committed"]
    problems = []
    if delta is None:
        if commits or any(c["name"] in ("add_sequence_elements", "generate_sequence", "replace_sequence_element", "swap_sequence_elements") for c in calls):
            problems.append("Expected no sequence operation")
        if mode != "browse" and calls:
            problems.append("Expected clarification/explanation without tools")
        if mode == "browse" and not any(c["name"] == "show_catalog_items" for c in calls):
            problems.append("Expected catalog display without addition")
        factory.assert_not_called()
    else:
        expected_anchor = str(source.elements[index].element_id) if index is not None else None
        if len(calls) != 1 or calls[0]["name"] != "add_sequence_elements":
            problems.append("Expected exactly one addition call")
        elif (calls[0]["args"].get("sequence_uuid"), calls[0]["args"].get("count"),
              calls[0]["args"].get("insert_before_element_id")) != (str(source.uuid), delta, expected_anchor):
            problems.append("Wrong selected UUID, delta or insertion anchor")
        k = min(delta, 45 - len(source.elements))
        if k == 0:
            if commits or len(results) != 1 or results[0].get("code") != "sequence_limit_reached":
                problems.append("Expected reached-limit no_change")
            factory.assert_not_called()
        else:
            variant = service.get_sequence(service.active_sequence_uuid)
            if len(commits) != 1 or variant.operation != "add" or variant.elements_count != len(source.elements) + k:
                problems.append("Expected one complete addition variant")
            elif variant.parent_uuid != source.uuid or variant.family_uuid != source.family_uuid:
                problems.append("Wrong selected parent or family")
            if not results or results[0].get("addition", {}).get("added_count") != k:
                problems.append("Wrong addition result")
            if generator.add_elements.call_count != 1:
                problems.append("Expected one selection call")
        if catalog.mock_calls or tags.mock_calls:
            problems.append("Unexpected external catalog lookup")
    needs_text = (not commits and mode != "browse") or any(
        payload.get("addition", {}).get("limit_applied") for payload in results
    )
    if needs_text and not "".join(replies).strip():
        problems.append("Expected an explanation or clarification")
    elif not needs_text and "".join(replies).strip():
        problems.append("Expected no completion text for the displayed result")
    if forbidden.mock_calls:
        problems.append("Unexpected generation")
    for item in sources:
        if service.get_sequence(str(item.uuid)) != item:
            problems.append("An existing snapshot changed")
    return {"case": name, "request": request, "passed": not problems, "problems": problems,
            "tool_calls": calls, "reply": "\n".join(replies), "commit_count": len(commits),
            "addition": next((p["addition"] for p in results if "addition" in p), None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("ASSISTANT_AGENT_MODEL", "").strip()
                        or os.getenv("SEQUENCE_AGENT_MODEL", "").strip() or "openai:gpt-5.5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append")
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
    if not report["cases"] or any(not c["passed"] for c in report["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
