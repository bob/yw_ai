"""Reproducible 50-turn comparison and opt-in live semantic check for 010."""

import argparse
import hashlib
import json
import sys
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path
from unittest.mock import Mock, patch

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import Field

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures.compact_tool_results import source_sequence
from langchain_app.services.assistant_history import HistoryBudget, HistoryTokenCounter
from langchain_app.tools import ASSISTANT_TOOLS
from test_assistant_agent import ToolCallingModel, assistant
from test_assistant_history import summary_reply

COUNTER_MODEL = ChatOpenAI(model="gpt-5.5", api_key="offline-token-counter")
COUNTER = HistoryTokenCounter(COUNTER_MODEL)


class ScriptedGPT(ToolCallingModel):
    """Offline outputs with the real GPT tokenizer, never a provider request."""

    model_name: str = "gpt-5.5"

    def get_num_tokens_from_messages(self, messages, **kwargs):
        return COUNTER.messages(messages)


class CapturingGPT(ChatOpenAI):
    received_messages: list[list] = Field(default_factory=list)
    replies: list[AIMessage] = Field(default_factory=list)

    def invoke(self, messages, config=None, **kwargs):
        # Graph callbacks can automatically select streaming, bypassing
        # _generate. Capture at the public invoke boundary in both cases.
        self.received_messages.append(list(messages))
        result = super().invoke(messages, config=config, **kwargs)
        self.replies.append(result)
        return result


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


@patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
def measure():
    """Include every normal input/output and every summarizer input/output."""
    requests = [f"Turn {i + 1}: " + "Discuss gentle seated practice and the known forward bends. " * 28
                for i in range(50)]
    reply = "Use the selected practice as context; the known seated poses stay available. " * 28
    tools_tokens = COUNTER.text(json.dumps([convert_to_openai_tool(t) for t in ASSISTANT_TOOLS]))
    source = source_sequence(9)
    rows = {}
    for name, bounded in (("baseline_009", False), ("after_010", True)):
        model = ScriptedGPT(responses=[AIMessage(content=reply)])
        summarizer = ScriptedGPT(responses=[summary_reply().model_copy(update={"usage_metadata": None})])
        with (nullcontext() if bounded else patch(
            "langchain_app.services.assistant_agent.BoundedHistoryMiddleware", return_value=AgentMiddleware(),
        )):
            service = assistant(Mock(), model=model, summary_model=summarizer)
            service._get_agent().update_state(service._config, {
                "sequences": {str(source.uuid): source.model_dump(mode="json")},
                "active_sequence_uuid": str(source.uuid),
            })
        per_turn = []
        for request in requests:
            summary_before = len(summarizer.received_messages)
            list(service.stream(request))
            outgoing = model.received_messages[-1]
            history_tokens = COUNTER.messages(outgoing[1:-1])
            summarized = len(summarizer.received_messages) > summary_before
            if bounded:
                assert history_tokens <= 8192
                if summarized:
                    assert history_tokens <= 4096
            per_turn.append({
                "turn": len(per_turn) + 1, "completed_history_tokens": history_tokens,
                "request_tokens_including_tools": COUNTER.messages(outgoing) + tools_tokens,
                "summarized": summarized,
            })
        normal_input = sum(row["request_tokens_including_tools"] for row in per_turn)
        normal_output = 50 * COUNTER.text(reply)
        summary_input = sum(COUNTER.messages(messages) for messages in summarizer.received_messages)
        summary_output = len(summarizer.received_messages) * COUNTER.text(summarizer.responses[0].text)
        rows[name] = {
            "normal_calls": len(model.received_messages), "summary_calls": len(summarizer.received_messages),
            "normal_input_tokens": normal_input, "normal_output_tokens": normal_output,
            "summary_input_tokens": summary_input, "summary_output_tokens": summary_output,
            "all_gpt_input_tokens": normal_input + summary_input,
            "all_gpt_output_tokens": normal_output + summary_output,
            "all_gpt_tokens": normal_input + normal_output + summary_input + summary_output,
            "cache_read_tokens": None, "actual_cost": None,
            "last_history_tokens": per_turn[-1]["completed_history_tokens"],
            "last_request_tokens_including_tools": per_turn[-1]["request_tokens_including_tools"],
            "canonical_messages": len(service.state["messages"]),
            "transcript_sha256": digest([(m.type, m.content) for m in service.state["messages"]]),
            "snapshots_sha256": digest(service.state["sequences"]),
            "turns": per_turn,
        }
    assert rows["baseline_009"]["transcript_sha256"] == rows["after_010"]["transcript_sha256"]
    assert rows["baseline_009"]["snapshots_sha256"] == rows["after_010"]["snapshots_sha256"]
    assert rows["baseline_009"]["last_history_tokens"] > 8192
    reference = COUNTER.messages(service.state["messages"])
    fallback = HistoryTokenCounter(ToolCallingModel(responses=[AIMessage(content="")])).messages(service.state["messages"])
    return {
        "mode": "offline estimates, not billed provider usage",
        "tokenizer": f"tiktoken {version('tiktoken')}; gpt-5.5/o200k_base",
        "scope": "All 50 normal requests (system, snapshot, messages, serialized tool schemas), "
                 "all normal outputs, all summarizer inputs and outputs. Provider framing is estimated.",
        "budgets": {"max_tokens": 8192, "target_tokens": 4096, "summary_max_tokens": 768},
        "cache_note": "Offline fixture has no actual cached input; cache-prefix effects and billed cost are unmeasured.",
        "fallback_calibration": {
            "corpus": "Same complete 50-turn canonical transcript, without system or tool schemas",
            "gpt_framing_estimate": reference, "local_fallback_estimate": fallback,
            "difference_percent": round(100 * (fallback / reference - 1), 2),
            "note": "Observed difference, not a guaranteed error bound for unknown model tokenizers.",
        },
        "rows": rows,
        "total_token_reduction_percent": round(100 * (
            1 - rows["after_010"]["all_gpt_tokens"] / rows["baseline_009"]["all_gpt_tokens"]
        ), 2),
    }


@patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
def live_semantics(output=None, scenario=None):
    """GPT only; catalog, generator and Jev are offline and must stay unused."""
    scenarios = [
        ("preserved_preference", ["В будущих практиках избегай стоячих поз. Отвечай по-русски."],
         "Какие мои постоянные пожелания ты помнишь?"),
        ("cancellation_and_one_off", [
            "В будущих практиках избегай стоячих поз. Отвечай по-русски.",
            "Сегодня, только в этом запросе, не используй скручивания.",
            "Отменяю прежний запрет на стоячие позы: их можно включать в будущие практики.",
         ], "Какие ограничения действуют для будущих практик?"),
        ("missing_reference", ["Обсудим мягкую практику. Отвечай по-русски."],
         "Замени её на другую."),
    ]
    rows = []
    if scenario and output and output.exists():
        rows = [row for row in json.loads(output.read_text())["rows"] if row["scenario"] != scenario]
    for name, preferences, request in scenarios:
        if scenario and name != scenario:
            continue
        model = CapturingGPT(model="gpt-5.5", timeout=60, max_retries=0, reasoning_effort="none")
        summarizer = CapturingGPT(model="gpt-5.5", timeout=60, max_retries=0)
        factory, catalog, tags = Mock(), Mock(), Mock()
        catalog.search.return_value, tags.get_names.return_value = [], []
        service = assistant(factory, catalog_item_service=catalog, catalog_tag_service=tags,
                            model=model, summary_model=summarizer, history_budget=HistoryBudget(1200, 800, 256))
        messages, boundaries = [], []
        for text in preferences + ["Объясни, что такое мягкая практика. " * 24 for _ in range(12)]:
            boundaries.append(len(messages))
            messages.extend([HumanMessage(content=text), AIMessage(content="Продолжаем обсуждение мягкой практики. " * 20)])
        state = {
            "messages": messages, "history_turn_starts": boundaries,
        }
        if name == "missing_reference":
            source = source_sequence(9)
            state.update({"sequences": {str(source.uuid): source.model_dump(mode="json")},
                          "active_sequence_uuid": str(source.uuid)})
        service._get_agent().update_state(service._config, state)
        reply = "".join(chunk for chunk in service.stream(request) if isinstance(chunk, str))
        factory.assert_not_called()
        catalog.search.assert_not_called()
        tags.get_names.assert_not_called()
        rows.append({
            "scenario": name, "user_request": request, "summary": json.loads(service.state["history_summary"]),
            "reply": reply, "normal_calls": len(model.received_messages),
            "summary_calls": len(summarizer.received_messages),
            "normal_usage": [message.usage_metadata for message in model.replies],
            "summary_usage": service.state["history_summary_calls"],
            "catalog_calls": 0, "generator_calls": 0, "jev_calls": 0,
            "selected_sequence_uuid": service.active_sequence_uuid,
        })
        report = {"mode": "live GPT semantic check with offline services", "model": "gpt-5.5", "rows": rows}
        if output:
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        assert len(summarizer.received_messages) == 1
        assert service.state["history_summary_calls"][-1]["status"] == "ok"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--scenario", choices=["preserved_preference", "cancellation_and_one_off", "missing_reference"])
    arguments = parser.parse_args()
    report = live_semantics(arguments.output, arguments.scenario) if arguments.live else measure()
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(arguments.output), "mode": report["mode"],
                      "reduction_percent": report.get("total_token_reduction_percent")}, ensure_ascii=False))
