"""Live feature 012 comparison with an offline catalog/generator and no Jev calls."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from unittest.mock import Mock, patch

from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures.compact_tool_results import pose, source_sequence
from langchain_app.model_config import DEFAULT_OPENAI_MODEL, chat_model_options
from langchain_app.schemas.assistant import AssistantState
from langchain_app.services.assistant_history import HistoryBudget
from langchain_app.services.prompt_cache import PromptCacheSettings, aggregate_usage, usage_cost
from langchain_app.services.sequence_context import selected_sequence_context
from test_assistant_agent import assistant
from test_assistant_history import seed_history


@wrap_model_call(state_schema=AssistantState)
def baseline_snapshot(request, handler):
    """Reproduce 010: changing snapshot in the initial system message, no key."""
    def move_to_system(projected):
        blocks = list(projected.system_message.content_blocks)
        blocks.extend(projected.messages[-1].content_blocks)
        return handler(projected.override(system_message=SystemMessage(content=blocks),
                                          messages=projected.messages[:-1]))
    return selected_sequence_context.wrap_model_call(request, move_to_system)


def run_comparison(output, *, model_name, rates, pricing_source):
    report = {
        "mode": "live OpenAI Chat Completions; offline catalog/generator; no Jev",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(), "model": model_name,
        "versions": {package: version(package) for package in ("langchain", "langchain-openai", "openai", "tiktoken")},
        "rates_per_million_usd": rates, "pricing_source": pricing_source,
        "conditions": "No prewarm, padding, extra cache-hit attempts or provider retries. "
                      "First and later calls are shown separately, without assuming they are cold/warm. "
                      "Baseline keeps automatic provider caching but sends no cache key/retention. "
                      "012 uses per-session keys and model-compatible cache settings. No profiles or routing from canceled 011.",
        "rows": [],
    }

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    for scenario in ("generate_add_select", "browse_then_add", "repeated_summary"):
        for mode in ("baseline_010", "after_012"):
            generator, catalog, tags = Mock(), Mock(), Mock()
            generator.generate.return_value = [pose(i) for i in range(1, 10)]
            generator.add_elements.return_value = [pose(i) for i in range(10, 13)]
            catalog.search.return_value = [pose(20), pose(21)]
            tags.get_names.return_value = ["Position/Sitting", "Bend/Forward"]
            model = ChatOpenAI(model=model_name, timeout=60, max_retries=0, disable_streaming=True,
                               model_kwargs={"parallel_tool_calls": False}, **chat_model_options(model_name))
            summarizer = ChatOpenAI(model=model_name, timeout=60, max_retries=0, disable_streaming=True,
                                    **chat_model_options(model_name))
            service = assistant(lambda: generator, catalog_item_service=catalog, catalog_tag_service=tags,
                                model=model, summary_model=summarizer,
                                history_budget=HistoryBudget(1600, 1000, 256) if scenario == "repeated_summary" else None,
                                prompt_cache_settings=PromptCacheSettings(mode == "after_012"))
            if mode == "baseline_010":
                with patch("langchain_app.services.assistant_agent.selected_sequence_context", baseline_snapshot):
                    service._get_agent()
            else:
                service._get_agent()
            row = {"scenario": scenario, "mode": mode, "steps": [], "records": [], "jev_calls": 0}
            report["rows"].append(row)

            def act(request):
                start = len(service.model_usage)
                reply = "".join(chunk for chunk in service.stream(request) if isinstance(chunk, str))
                records = service.model_usage[start:]
                row["steps"].append({
                    "request": request, "reply": reply, "usage": aggregate_usage(records),
                    "cost_usd": usage_cost(records, **rates),
                    "selected_count": (service.get_sequence(service.active_sequence_uuid).elements_count
                                       if service.active_sequence_uuid else None),
                })
                row["records"] = service.model_usage
                row["usage"] = aggregate_usage(row["records"])
                row["cost_usd"] = usage_cost(row["records"], **rates)
                row["normal_calls"] = sum(r["call_kind"] == "assistant" for r in row["records"])
                row["summary_calls"] = sum(r["call_kind"] == "history_summary" for r in row["records"])
                save()
                print(json.dumps({"scenario": scenario, "mode": mode, "step": len(row["steps"]),
                                  "usage": aggregate_usage(records)}, ensure_ascii=False), flush=True)

            try:
                if scenario == "generate_add_select":
                    act("Создай секвенцию из 9 поз.")
                    source = service.active_sequence_uuid
                    act("Добавь 3 позы в конец.")
                    assert service.get_sequence(service.active_sequence_uuid).elements_count == 12
                    service.select_sequence(source)
                    act("Сколько поз в выбранном варианте? Ответь одним предложением.")
                elif scenario == "browse_then_add":
                    source = source_sequence()
                    service._get_agent().update_state(service._config, {
                        "sequences": {str(source.uuid): source.model_dump(mode="json")},
                        "active_sequence_uuid": str(source.uuid),
                    })
                    act("Покажи доступные сидячие наклоны вперёд, ничего не меняй.")
                    act("Добавь 3 позы в конец выбранной секвенции.")
                    assert service.get_sequence(service.active_sequence_uuid).elements_count == 12
                else:
                    seed_history(service)
                    act("Объясни мягкую практику коротко. Учитывай этот контекст: " + "мягкая практика " * 1000)
                    act("Какие пожелания ты помнишь? Коротко.")
                    assert row["summary_calls"] == 2
                row["status"] = "ok"
            except Exception as exc:
                row.update(status="error", error_type=type(exc).__name__, records=service.model_usage)
                save()
                raise
            save()
    report["totals_by_mode"] = {}
    for mode in ("baseline_010", "after_012"):
        records = [record for row in report["rows"] if row["mode"] == mode for record in row["records"]]
        report["totals_by_mode"][mode] = {
            "normal_calls": sum(record["call_kind"] == "assistant" for record in records),
            "summary_calls": sum(record["call_kind"] == "history_summary" for record in records),
            **aggregate_usage(records), "cost_usd": usage_cost(records, **rates),
        }
    report["interpretation"] = (
        "Compare totals_by_mode for reported usage and calculated cost. Baseline automatic caching "
        "and provider routing/retention differ from fresh per-session 012 keys; this short sequential "
        "run is not a controlled estimate of long-term savings. No additional requests were made "
        "to obtain hits."
    )
    save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--input-rate", type=float, required=True)
    parser.add_argument("--cached-input-rate", type=float, required=True)
    parser.add_argument("--output-rate", type=float, required=True)
    parser.add_argument("--pricing-source", required=True)
    args = parser.parse_args()
    os.environ["LANGSMITH_TRACING"] = "false"
    run_comparison(args.output, model_name=args.model,
                   rates={"input_rate": args.input_rate, "cached_input_rate": args.cached_input_rate,
                          "output_rate": args.output_rate}, pricing_source=args.pricing_source)
