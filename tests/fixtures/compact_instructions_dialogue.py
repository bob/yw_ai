"""Opt-in live prompt regression; catalog, generator, and TypeSafe stay offline.

Reuses swap/addition acceptance cases and checks generation, browsing, and
replacement with the current assistant model. No provider retries or tracing.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock

from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures.sequence_addition_dialogue import CASES as ADD_CASES, run_case as run_add
from fixtures.sequence_swap_dialogue import CASES as SWAP_CASES, run_case as run_swap, snapshot
from langchain_app.model_config import DEFAULT_CHAT_MODEL, chat_model_options
from langchain_app.services.assistant_agent import AssistantAgent, SYSTEM_PROMPT
from langchain_app.services.prompt_cache import prefix_version
from langchain_app.services.sequence_context import GENERATION_INSTRUCTION, SELECTED_SEQUENCE_INSTRUCTIONS
from langchain_app.tools import ASSISTANT_TOOLS


# name, request, expected operation (None means browsing), generator count/duration
CASES = [
    ("generate_default", "Создай секвенцию", "generate", 9, None),
    ("generate_duration", "Создай секвенцию на десять минут", "generate", 15, 600),
    ("generate_explicit_priority", "Создай секвенцию из трёх асан на десять минут", "generate", 3, None),
    ("generate_limit", "Создай секвенцию из 50 асан", "generate", 45, None),
    ("generate_duration_limit", "Создай секвенцию на 60 минут", "generate", 45, 3600),
    ("generate_first_pose", "Создай три асаны, первой поставь Tree pose", "generate", 3, None),
    ("browse_candidates", "Найди кандидатов для замены третьей асаны, пока ничего не меняй", None, None, None),
    ("browse_available", "Какие есть позы сидя?", None, None, None),
    ("replace", "Замени вторую асану на Warrior pose", "replace", None, None),
    ("starting_replacement_forbidden", "Замени вторую асану на Starting pose", "blocked", None, None),
    ("starting_swap_allowed", "Поменяй первую и последнюю асаны местами", "swap", None, None),
]


def run_extra(model, case):
    name, request, operation, count, duration = case
    source = snapshot()
    source.elements[0].catalog_item.catalog_tag_names = ["Other/Starting", "Position/Sitting"]
    candidate = source.elements[2].catalog_item.model_copy(deep=True)
    if operation != "generate":
        candidate.id = 100
        candidate.uuid = str(source.elements[0].element_id)
        candidate.name = "Starting pose" if operation == "blocked" else "Warrior pose"
        candidate.catalog_tag_names = ["Other/Starting"] if operation == "blocked" else ["Position/Standing"]
        if operation is None:
            candidate.name = "Seated forward bend"
            candidate.catalog_tag_names = ["Position/Sitting", "Bend/Forward"]
    catalog, tags, generator = Mock(), Mock(), Mock()
    catalog.search.return_value = [candidate]
    tags.get_names.return_value = ["Position/Sitting", "Position/Standing", "Other/Starting"]
    generator.generate.side_effect = lambda request, **kwargs: [
        candidate.model_copy(update={"id": candidate.id + index, "uuid": None, "bilateral": False}, deep=True)
        for index in range(kwargs["count"])
    ]
    factory = Mock(return_value=generator)
    service = AssistantAgent(factory, model=model, catalog_item_service=catalog, catalog_tag_service=tags)
    service._get_agent().update_state(service._config, {
        "sequences": {str(source.uuid): source.model_dump(mode="json")},
        "active_sequence_uuid": str(source.uuid),
    })
    events = list(service.stream_response_events(request))
    calls = [call for message in service.state["messages"] if isinstance(message, AIMessage) for call in message.tool_calls]
    commits = [event for event in events if event.type == "sequence_committed"]
    reply = "\n".join(event.text for event in events if event.type == "assistant_message" and event.purpose == "final")
    changes = [call for call in calls if call["name"] in {
        "generate_sequence", "replace_sequence_element", "swap_sequence_elements", "add_sequence_elements",
    }]
    problems = []
    limited = name in {"generate_limit", "generate_duration_limit"}
    if operation in {None, "blocked"}:
        if changes or commits or service.state["sequences"] != {str(source.uuid): source.model_dump(mode="json")}:
            problems.append("Expected no sequence changes or attempted operations")
        if operation is None and sum(call["name"] == "show_catalog_items" for call in calls) != 1:
            problems.append("Expected one catalog display")
        factory.assert_not_called()
    else:
        if len(changes) != 1 or changes[0]["name"] != {
            "generate": "generate_sequence", "replace": "replace_sequence_element", "swap": "swap_sequence_elements",
        }[operation] or len(commits) != 1:
            problems.append("Expected exactly one requested sequence operation and commit")
        if service.get_sequence(str(source.uuid)) != source:
            problems.append("The original snapshot changed")
        variant = service.get_sequence(service.active_sequence_uuid)
        if operation == "generate":
            if generator.generate.call_count != 1:
                problems.append("Expected one generator call")
            else:
                kwargs = generator.generate.call_args.kwargs
                if (kwargs["count"], kwargs.get("duration_seconds")) != (count, duration):
                    problems.append("Wrong computed generation count/duration")
                if name == "generate_first_pose" and kwargs["starting_catalog_item_id"] != candidate.id:
                    problems.append("Requested first pose was not resolved")
                if duration is not None and changes[0]["args"].get("count") is not None:
                    problems.append("Model calculated count for a duration-only request")
        elif operation == "replace" and variant.elements[1].catalog_item.id != candidate.id:
            problems.append("Wrong replaced occurrence or candidate")
        elif operation == "swap" and [e.element_id for e in variant.elements] != [
            source.elements[index].element_id for index in (4, 1, 2, 3, 0)
        ]:
            problems.append("Wrong swap order")
    needs_text = operation == "blocked" or limited
    if bool(reply.strip()) != needs_text:
        problems.append("Expected useful explanation" if needs_text else "Expected no final recap")
    if limited and "45" not in reply:
        problems.append("Expected an explanation of the 45-element limit")
    return {"case": name, "request": request, "passed": not problems, "problems": problems,
            "tool_calls": calls, "tool_results": [json.loads(m.text) for m in service.state["messages"]
                                                    if isinstance(m, ToolMessage)],
            "reply": reply, "commit_count": len(commits), "usage": service.model_usage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("ASSISTANT_AGENT_MODEL", "").strip() or DEFAULT_CHAT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    os.environ["LANGSMITH_TRACING"] = "false"
    model = ChatOpenAI(model=args.model.removeprefix("openai:"), max_retries=0, timeout=60,
                       disable_streaming=True, model_kwargs={"parallel_tool_calls": False},
                       **chat_model_options(args.model))
    report = {"model": args.model, "started_at": datetime.now(timezone.utc).isoformat(),
              "system_prompt_sha256": sha256(SYSTEM_PROMPT.encode()).hexdigest(),
              "context_instructions_sha256": sha256(SELECTED_SEQUENCE_INSTRUCTIONS.encode()).hexdigest(),
              "prefix_version": prefix_version(SYSTEM_PROMPT, ASSISTANT_TOOLS,
                                                context_instructions=SELECTED_SEQUENCE_INSTRUCTIONS + GENERATION_INSTRUCTION),
              "conditions": "Live LLM; offline catalog/generator/TypeSafe; no retries, prewarm, or tracing", "cases": []}
    for prefix, runner, cases in (("swap", run_swap, SWAP_CASES), ("add", run_add, ADD_CASES), ("extra", run_extra, CASES)):
        for case in cases:
            case_name = f"{prefix}_{case[0]}"
            if args.case and case_name not in args.case:
                continue
            try:
                result = runner(model, case)
            except Exception as exc:
                result = {"request": case[1], "passed": False, "problems": [str(exc)],
                          "error_type": type(exc).__name__, "reply": "", "tool_calls": []}
            result["case"] = case_name
            report["cases"].append(result)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(f"{case_name}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    if not report["cases"] or any(not case["passed"] for case in report["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
