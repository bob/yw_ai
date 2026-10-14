"""Feature 012 acceptance, including actual OpenAI SDK request serialization."""

import json
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch
from uuid import uuid4

import httpx
from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from fixtures.compact_tool_results import pose, source_sequence
from langchain_app.services.assistant_history import HistoryBudget
from langchain_app.services.prompt_cache import (
    PromptCacheSession, PromptCacheSettings, aggregate_usage, prefix_version,
    reported_usage, usage_cost,
)
from langchain_app.tools import ASSISTANT_TOOLS
from test_assistant_agent import ToolCallingModel, assistant, tool_call
from test_assistant_history import seed_history, summary_reply


class OpenAIStub:
    """Real ChatOpenAI + SDK, with only HTTP replaced; no external API calls."""

    def __init__(self, replies):
        self.requests = []
        self.replies = iter(replies)
        self.model = ChatOpenAI(
            model="gpt-4.1", api_key="test", max_retries=0, disable_streaming=True,
            http_client=httpx.Client(transport=httpx.MockTransport(self.respond)),
        )

    def respond(self, request):
        self.requests.append(json.loads(request.content))
        message = next(self.replies)
        content = {"role": "assistant", "content": message.text or None}
        if message.tool_calls:
            content["tool_calls"] = [{
                "id": call["id"], "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["args"])},
            } for call in message.tool_calls]
        return httpx.Response(200, json={
            "id": f"chatcmpl-{len(self.requests)}", "object": "chat.completion",
            "created": 0, "model": "gpt-4.1-2025-04-14",
            "choices": [{"index": 0, "finish_reason": "tool_calls" if message.tool_calls else "stop", "message": content}],
            "usage": {"prompt_tokens": 2000, "completion_tokens": 100, "total_tokens": 2100,
                      "prompt_tokens_details": {"cached_tokens": 0 if len(self.requests) == 1 else 1024}},
        })


def snapshot(payload):
    content = payload["messages"][-1]["content"]
    if isinstance(content, list):
        content = content[0]["text"]
    return json.loads(next(line for line in content.splitlines() if line.startswith('{"selected_sequence":')))["selected_sequence"]


class PromptCacheTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict("os.environ", {
            "LANGSMITH_TRACING": "false", "OPENAI_API_BASE": "https://api.openai.com/v1",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "ASSISTANT_PROMPT_CACHE_ENABLED": "true", "ASSISTANT_PROMPT_CACHE_RETENTION": "",
        })
        environment.start()
        self.addCleanup(environment.stop)

    def test_sdk_generate_add_and_old_selection_keep_prefix_key_and_whole_snapshot(self):
        generator = Mock()
        generator.generate.return_value = [pose(i) for i in range(1, 10)]
        stub = OpenAIStub([tool_call("generate_sequence", request="Generate 9", count=9),
                           AIMessage(content="Создано"), AIMessage(content="Выбрано")])
        service = assistant(lambda: generator, model=stub.model)
        first_id = uuid4()
        events = list(service.stream_response_events("Generate 9", request_id=first_id))
        source = service.get_sequence(service.active_sequence_uuid)
        generator.add_elements.return_value = [pose(i) for i in range(10, 13)]
        stub.replies = iter([tool_call("add_sequence_elements", sequence_uuid=str(source.uuid),
                                      count=3, request="Добавь 3"), AIMessage(content="Добавлено")])
        list(service.stream("Добавь 3"))
        newer = service.get_sequence(service.active_sequence_uuid)
        calls = len(service.model_usage)
        service.select_sequence(str(source.uuid))
        self.assertEqual(len(service.model_usage), calls)
        stub.replies = iter([AIMessage(content="Старый вариант")])
        list(service.stream("Что выбрано?"))
        self.assertEqual([snapshot(payload)["elements_count"] if snapshot(payload) else None
                          for payload in stub.requests], [None, 9, 9, 12, 9])
        for payload in stub.requests:
            self.assertEqual(payload["messages"][0], stub.requests[0]["messages"][0])
            self.assertEqual(payload["tools"], stub.requests[0]["tools"])
            self.assertEqual(payload["prompt_cache_key"], stub.requests[0]["prompt_cache_key"])
            self.assertEqual(payload["prompt_cache_retention"], "in_memory")
            self.assertNotIn("prompt_cache_options", payload)
            self.assertEqual(payload["messages"][-1]["role"], "system")
            self.assertNotIn('{"selected_sequence":', str(payload["messages"][0]["content"]))
            self.assertEqual(sum("Current selected sequence (" in str(m["content"]) for m in payload["messages"]), 1)
            pending = set()
            for message in payload["messages"]:
                if message["role"] == "tool":
                    pending.remove(message["tool_call_id"])
                else:
                    self.assertFalse(pending)
                    pending.update(call["id"] for call in message.get("tool_calls", []))
            self.assertFalse(pending)
        self.assertEqual(snapshot(stub.requests[-1]), source.model_context())
        self.assertNotIn("image_url", json.dumps(snapshot(stub.requests[-1])))
        self.assertFalse(any(isinstance(m, SystemMessage) for m in service.state["messages"]))
        self.assertEqual(service.get_sequence(str(newer.uuid)), newer)
        self.assertEqual(len(service.model_usage), 5)
        self.assertEqual([r["request_id"] for r in service.model_usage[:2]], [str(first_id)] * 2)
        self.assertEqual([r["cached_input_tokens"] for r in service.model_usage], [0, 1024, 1024, 1024, 1024])
        self.assertEqual(events[-1].type, "request_finished")
        records = service.model_usage
        records[0]["input_tokens"] = -1
        self.assertEqual(service.model_usage[0]["input_tokens"], 2000)

    def test_summary_has_its_own_stable_key_and_usage_belongs_to_the_operation(self):
        normal, summary = OpenAIStub([AIMessage(content="Продолжим")] * 2), OpenAIStub([summary_reply()] * 2)
        service = assistant(Mock(), model=normal.model, summary_model=summary.model,
                            history_budget=HistoryBudget(1600, 1000, 256))
        original = seed_history(service)
        request_id = uuid4()
        list(service.stream_response_events("Продолжим " * 1000, request_id=request_id))
        list(service.stream("Что помнишь?"))
        self.assertEqual(len(summary.requests), 2)
        self.assertEqual(summary.requests[0]["prompt_cache_key"], summary.requests[1]["prompt_cache_key"])
        self.assertIn(":history-summary:", summary.requests[0]["prompt_cache_key"])
        self.assertNotEqual(normal.requests[0]["prompt_cache_key"], summary.requests[0]["prompt_cache_key"])
        self.assertEqual(normal.requests[0]["messages"][0], normal.requests[1]["messages"][0])
        self.assertNotIn("selected_sequence", summary.requests[0]["messages"][-1]["content"])
        self.assertEqual(service.state["messages"][:len(original)], original)
        records = service.model_usage
        self.assertEqual([r["call_kind"] for r in records], ["history_summary", "assistant"] * 2)
        self.assertEqual([r["request_id"] for r in records[:2]], [str(request_id)] * 2)
        self.assertEqual(aggregate_usage(records)["input_tokens"], 8000)

    def test_revision_order_and_session_identity(self):
        version = prefix_version("prompt", ASSISTANT_TOOLS)
        self.assertEqual(version, prefix_version("prompt", ASSISTANT_TOOLS))
        self.assertNotEqual(version, prefix_version("edited prompt", ASSISTANT_TOOLS))
        self.assertNotEqual(version, prefix_version("prompt", ASSISTANT_TOOLS[::-1]))
        schema = {"type": "function", "function": {"name": "test", "parameters": {"type": "object"}}}
        edited = deepcopy(schema)
        edited["function"]["description"] = "changed"
        self.assertNotEqual(prefix_version("prompt", [schema]), prefix_version("prompt", [edited]))
        model = OpenAIStub([]).model
        one, two = PromptCacheSession(PromptCacheSettings()), PromptCacheSession(PromptCacheSettings())
        key = one.key(model, version, "assistant")
        self.assertNotEqual(key, two.key(model, version, "assistant"))
        self.assertNotEqual(key, one.key(model, version, "history-summary"))
        self.assertNotEqual(key, one.key(model, "revision", "assistant"))
        self.assertLessEqual(len(key), 128)

    def test_settings_compatibility_disabled_fake_and_non_openai(self):
        for name, retention in (("gpt-4.1", "in_memory"), ("gpt-4.1-2025-04-14", "in_memory"),
                                ("gpt-5.1", "in_memory"), ("gpt-5.5", "24h"), ("gpt-5.6-sol", None)):
            model = ChatOpenAI(model=name, api_key="test")
            settings = PromptCacheSettings().model_settings(model, "key")
            self.assertEqual(settings.get("prompt_cache_retention"), retention)
            self.assertEqual(settings["prompt_cache_key"], "key")
        model = ChatOpenAI(model="gpt-4.1", api_key="test")
        self.assertEqual(PromptCacheSettings(retention="24h").model_settings(model, "key")["prompt_cache_retention"], "24h")
        for settings in (PromptCacheSettings(False), PromptCacheSettings()):
            fake = ToolCallingModel(responses=[AIMessage(content="Готово")])
            self.assertEqual(settings.model_settings(fake, "key"), {})
        external = ChatOpenAI(model="gpt-4.1", api_key="test", base_url="https://other.example/v1")
        self.assertEqual(PromptCacheSettings().model_settings(external, "key"), {})
        with patch.dict("os.environ", {"OPENAI_API_BASE": "", "OPENAI_BASE_URL": "https://other.example/v1"}):
            external = ChatOpenAI(model="gpt-4.1", api_key="test", base_url=None)
            self.assertEqual(PromptCacheSettings().model_settings(external, "key"), {})
        stub = OpenAIStub([AIMessage(content="Готово")])
        service = assistant(Mock(), model=stub.model, prompt_cache_settings=PromptCacheSettings(False))
        list(service.stream("Привет"))
        self.assertNotIn("prompt_cache_key", stub.requests[0])
        self.assertNotIn("prompt_cache_retention", stub.requests[0])
        self.assertIsNone(snapshot(stub.requests[0]))
        self.assertEqual(len(service.model_usage), 1)

    def test_bad_config_fails_before_model_initialization_and_has_no_fallback_request(self):
        for settings, model in ((PromptCacheSettings(retention="24h"), "openai:gpt-4.1-mini"),
                                (PromptCacheSettings(retention="in_memory"), "openai:gpt-5.5"),
                                (PromptCacheSettings(retention="24h"), "openai:gpt-5.6-sol")):
            with self.subTest(model=model), patch("langchain_app.services.assistant_agent.init_chat_model") as init:
                with self.assertRaisesRegex(ValueError, "Unsupported"):
                    assistant(Mock(), model=model, prompt_cache_settings=settings)
                init.assert_not_called()
        for name, value in (("ASSISTANT_PROMPT_CACHE_RETENTION", "bad"), ("ASSISTANT_PROMPT_CACHE_ENABLED", "maybe")):
            with patch.dict("os.environ", {name: value}), self.assertRaises(ValueError):
                assistant(Mock())
        session = PromptCacheSession(PromptCacheSettings())
        handler = Mock(side_effect=RuntimeError("provider rejected settings"))
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            session.call(OpenAIStub([]).model, "v1", "assistant", "request", handler)
        handler.assert_called_once()
        self.assertEqual(session.records[0]["status"], "error")
        self.assertIsNone(session.records[0]["input_tokens"])

    def test_summary_failure_and_main_failure_keep_usage_without_extra_calls(self):
        model = ToolCallingModel(responses=[AIMessage(content="unused")])
        summary = ToolCallingModel(responses=[AIMessage(content="invalid summary JSON")])
        service = assistant(Mock(), model=model, summary_model=summary,
                            history_budget=HistoryBudget(1600, 1000, 256))
        seed_history(service)
        request_id = uuid4()
        with patch.object(model, "_generate", side_effect=RuntimeError("model failed")):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                list(service.stream_response_events("Продолжим", request_id=request_id))
        self.assertEqual(len(summary.received_messages), 1)
        self.assertEqual(service.state["history_summary_calls"][0]["status"], "error")
        self.assertEqual([r["call_kind"] for r in service.model_usage], ["history_summary", "assistant"])
        self.assertEqual([r["status"] for r in service.model_usage], ["ok", "error"])
        for record in service.model_usage:
            self.assertEqual(record["request_id"], str(request_id))
            self.assertIsNone(record["input_tokens"])
            self.assertNotIn("prompt", record)
            self.assertNotIn("selected_sequence", record)

    def test_normalized_and_raw_usage_unknowns_reasoning_totals_and_cost(self):
        raw = {"prompt_tokens": 1000, "completion_tokens": 80,
               "prompt_tokens_details": {"cached_tokens": 500},
               "completion_tokens_details": {"reasoning_tokens": 20}}
        normalized = {"input_tokens": 100, "output_tokens": 30, "total_tokens": 130,
                      "input_token_details": {"cache_read": 0}, "output_token_details": {"reasoning": 10}}
        first = reported_usage(AIMessage(content="", usage_metadata=normalized, response_metadata={"token_usage": raw}))
        self.assertEqual(first["input_tokens"], 100)
        self.assertEqual(first["cached_input_tokens"], 0)
        second = reported_usage(AIMessage(content="", response_metadata={"token_usage": raw}))
        totals = aggregate_usage([first, second])
        self.assertEqual(totals, {"calls": 2, "input_tokens": 1100, "cached_input_tokens": 500,
                                  "uncached_input_tokens": 600, "output_tokens": 110, "reasoning_tokens": 30})
        self.assertAlmostEqual(usage_cost([first, second], input_rate=2, cached_input_rate=.5, output_rate=8), .00233)
        unknown = reported_usage(AIMessage(content=""))
        self.assertIsNone(aggregate_usage([first, unknown])["cached_input_tokens"])
        self.assertIsNone(usage_cost([unknown], input_rate=2, cached_input_rate=.5, output_rate=8))
        response_usage = {"input_tokens": 1000, "output_tokens": 80,
                          "input_tokens_details": {"cached_tokens": 500},
                          "output_tokens_details": {"reasoning_tokens": 20}}
        self.assertEqual(second, reported_usage(AIMessage(content="", response_metadata={"usage": response_usage})))
        invalid = reported_usage(AIMessage(content="", response_metadata={"usage": {
            "input_tokens": 10, "input_tokens_details": {"cached_tokens": 20},
        }}))
        self.assertEqual(invalid["usage_error"], "cached_input_exceeds_input")
        self.assertIsNone(invalid["cached_input_tokens"])


if __name__ == "__main__":
    unittest.main()
