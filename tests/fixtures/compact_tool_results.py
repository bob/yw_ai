"""Offline agent-loop fixtures and token comparison for feature 009."""

import argparse
import json
import sys
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import NAMESPACE_URL, uuid5

from langchain_core.messages import AIMessage, ToolMessage

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_app.schemas import CatalogItemSchema, GeneratedSequence
from langchain_app.services.sequence_generator import SequenceGenerator
from langchain_app.services.type_safe import TypeSafeService
from langchain_app.tools.results import catalog_item_result, commit_sequence
from test_assistant_agent import ToolCallingModel, assistant, tool_call

SEQUENCE_TOOLS = {
    "generate_sequence", "replace_sequence_element", "swap_sequence_elements", "add_sequence_elements",
}


def pose(number):
    return CatalogItemSchema(
        id=number, uuid=str(uuid5(NAMESPACE_URL, f"catalog/{number}")),
        name=f"Seated pose {number}", bilateral=number % 3 == 0,
        image_url=f"https://example.com/catalog/poses/{number}/thumbnail.png",
        catalog_tag_names=["Position/Sitting", "Bend/Forward"],
    )


def source_sequence(count=9):
    sequence = GeneratedSequence.from_catalog_items(
        [pose(number) for number in range(1, count + 1)],
        request="A gentle seated practice with a twist and forward bends",
    )
    sequence.uuid = uuid5(NAMESPACE_URL, f"sequence/{count}")
    sequence.family_uuid = uuid5(NAMESPACE_URL, f"family/{count}")
    for element in sequence.elements:
        element.element_id = uuid5(NAMESPACE_URL, f"occurrence/{count}/{element.order}")
    return sequence


def legacy_payload(sequence, compact, source=None):
    """Reconstruct the exact pre-009 payload from the same committed snapshot."""
    payload = {"status": "ok", **sequence.model_dump(mode="json")}
    if "replacement" in compact:
        replacement = deepcopy(compact["replacement"])
        target = replacement["element_id"]
        old = next(e for e in source.elements if str(e.element_id) == target)
        new = next(e for e in sequence.elements if str(e.element_id) == target)
        replacement["old_catalog_item"] = old.catalog_item.model_dump(mode="json")
        replacement["new_catalog_item"] = new.catalog_item.model_dump(mode="json")
        payload["replacement"] = replacement
    if "swap" in compact:
        payload["swap"] = [{k: v for k, v in entry.items() if k != "name"} for entry in compact["swap"]]
    if "addition" in compact:
        payload["addition"] = deepcopy(compact["addition"])
    return payload


def legacy_commit(runtime, sequence, **facts):
    result = commit_sequence(runtime, sequence, **facts)
    message = result.update["messages"][0]
    active = runtime.state.get("active_sequence_uuid")
    snapshot = runtime.state.get("sequences", {}).get(active)
    source = GeneratedSequence.from_snapshot(snapshot) if snapshot else None
    message.content = json.dumps(legacy_payload(sequence, json.loads(message.content), source), ensure_ascii=False)
    return result


class ResultReplyModel(ToolCallingModel):
    """Scripted final replies read operation facts, including legacy swap results."""

    def _generate(self, messages, **kwargs):
        last = messages[-1]
        if isinstance(last, ToolMessage) and last.name in SEQUENCE_TOOLS:
            payload = json.loads(last.content)
            if payload["status"] != "ok":
                reply = f"Без изменений: {payload.get('code', payload['status'])}."
            elif "replacement" in payload:
                change = payload["replacement"]
                reply = f"Заменена {change['old_catalog_item']['name']} на {change['new_catalog_item']['name']}."
            elif "swap" in payload:
                context = messages[0].content_blocks[-1]["text"]
                selected = json.loads(next(line for line in context.splitlines()
                                          if line.startswith('{"selected_sequence":')))["selected_sequence"]
                names = {e["element_id"]: e["catalog_item"]["name"] for e in selected["elements"]}
                reply = "; ".join(
                    f"{entry.get('name', names[entry['element_id']])}: {entry['old_order']} → {entry['new_order']}"
                    for entry in payload["swap"]
                )
            elif "addition" in payload:
                change = payload["addition"]
                reply = f"Добавлено {change['added_count']} с позиции {change['position']}; всего {payload['elements_count']}."
                if change["limit_applied"]:
                    reply += f" Лимит 45, запрошено {change['requested_count']}."
            else:
                reply = f"Создано {payload['elements_count']} поз."
            self.responses = [AIMessage(content=reply)]
            self.i = 0
        return super()._generate(messages, **kwargs)


@patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
def run_case(operation, count=9, *, source=None, legacy=False, added_count=3, anchor_index=None):
    source = source or source_sequence()
    request = "Добавь мягкие позы" if operation == "add" else "A gentle seated practice"
    catalog = Mock()
    catalog_items = [pose(number) for number in range(1, 91)]
    catalog_items[0].catalog_tag_names.append("Other/Starting")
    catalog_items[-1].catalog_tag_names.append("Other/Finishing")
    catalog.search.side_effect = lambda **kwargs: [pose(80)] if kwargs.get("name") else catalog_items
    jev_inputs, jev_outputs = [], []

    def classify(payload):
        question = payload["questions"]["next_yoga_pose"]
        jev_inputs.append(deepcopy({
            "state": payload["state"], "questions": {"next_yoga_pose": question.model_dump(mode="json")},
        }))
        choice = next(iter(question.criteria))
        jev_outputs.append(choice)
        return SimpleNamespace(choices={"next_yoga_pose": SimpleNamespace(choice=choice)})

    classifier = Mock()
    classifier.invoke.side_effect = classify
    generator = SequenceGenerator(catalog, TypeSafeService(classifier=classifier))
    factory = Mock(return_value=generator)
    if operation in ("generate", "regenerate"):
        if operation == "regenerate":
            request = source.generation_request
        calls = [tool_call("generate_sequence", request=request, count=count)]
    elif operation == "replace":
        calls = [tool_call("search_catalog_items", name="Seated pose 80"), tool_call(
            "replace_sequence_element", sequence_uuid=str(source.uuid),
            element_id=str(source.elements[2].element_id), replacement_catalog_item_id=80,
        )]
    elif operation == "swap":
        calls = [tool_call("swap_sequence_elements", sequence_uuid=str(source.uuid),
                           first_element_id=str(source.elements[2].element_id),
                           second_element_id=str(source.elements[4].element_id))]
    else:
        calls = [tool_call("add_sequence_elements", sequence_uuid=str(source.uuid), count=added_count,
                           request=request, insert_before_element_id=(
                               str(source.elements[anchor_index].element_id) if anchor_index is not None else None))]
    model = ResultReplyModel(responses=[*calls, AIMessage(content="")])
    service = assistant(factory, model=model, catalog_item_service=catalog)
    if operation != "generate":
        service._get_agent().update_state(service._config, {
            "sequences": {str(source.uuid): source.model_dump(mode="json")}, "active_sequence_uuid": str(source.uuid),
        })
    with patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}), patch(
        "langchain_app.services.sequence_generator.sample", side_effect=lambda items, k: items[:k],
    ), patch(
        "langchain_app.tools.sequence.commit_sequence", side_effect=legacy_commit if legacy else commit_sequence,
    ), patch(
        "langchain_app.tools.catalog.catalog_item_result",
        side_effect=(lambda item: item.model_dump(mode="json")) if legacy else catalog_item_result,
    ):
        events = list(service.stream_response_events(
            request, regenerate_sequence_uuid=str(source.uuid) if operation == "regenerate" else None,
        ))
    sequence = service.get_sequence(service.active_sequence_uuid)
    message = next(m for m in reversed(service.state["messages"]) if isinstance(m, ToolMessage))
    return SimpleNamespace(
        source=source, sequence=sequence, service=service, model=model, factory=factory,
        catalog=catalog, jev_inputs=jev_inputs, jev_outputs=jev_outputs,
        events=events, message=message, payload=json.loads(message.content),
    )


def measure():
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    rows = []
    cases = [(f"Generate {count}", "generate", count) for count in (9, 36, 45)]
    cases += [("Regenerate 9", "regenerate", 9), ("Replace", "replace", 9),
              ("Swap", "swap", 9), ("Add 3", "add", 9)]
    for label, operation, count in cases:
        case = run_case(operation, count)
        # Both serializations use the very same full snapshot and occurrence IDs.
        before = json.dumps(legacy_payload(case.sequence, case.payload, case.source), ensure_ascii=False)
        after = case.message.content
        baseline_tokens, after_tokens = len(encoding.encode(before)), len(encoding.encode(after))
        rows.append({
            "scenario": label, "baseline_tokens": baseline_tokens, "after_tokens": after_tokens,
            "reduction_percent": round(100 * (1 - after_tokens / baseline_tokens), 2),
            "gpt_calls": len(case.model.received_messages), "jev_calls": len(case.jev_inputs),
        })
    baseline_total = sum(row["baseline_tokens"] for row in rows)
    after_total = sum(row["after_tokens"] for row in rows)
    return {
        "tokenizer": f"tiktoken {version('tiktoken')}", "encoding": "o200k_base",
        "scope": "JSON sequence tool content only; provider overhead excluded",
        "rows": rows, "baseline_total": baseline_total, "after_total": after_total,
        "reduction_percent": round(100 * (1 - after_total / baseline_total), 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = measure()
    if arguments.output:
        arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
