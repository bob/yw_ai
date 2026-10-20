"""Bounded-history acceptance through the actual checkpointer and tool loop."""

import json
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from fixtures.compact_tool_results import legacy_payload, source_sequence
from langchain_app.services.assistant_history import (
    HistoryBudget, HistorySummary, HistoryTokenCounter, compact_history, summary_context,
)
from langchain_app.services.request_limits import MAX_REQUEST_CHARACTERS
from test_assistant_agent import ToolCallingModel, assistant, selected_snapshot, tool_call


class SummaryModel(ToolCallingModel):
    """Allow provider failures to be injected separately from the main model."""


def summary_reply(**changes):
    facts = {
        "ongoing_preferences": ["Avoid standing poses in future practices."],
        "current_goal": "A gentle practice", "pending_clarification": None,
        "relevant_facts": [], "response_language": "Russian",
        **changes,
    }
    return AIMessage(content=json.dumps(facts, ensure_ascii=False), usage_metadata={
        "input_tokens": 123, "output_tokens": 45, "total_tokens": 168,
        "input_token_details": {"cache_read": 32},
    })


def seed_history(service, turns=16, text=None, *, tool_turns=False):
    """Explicit lifecycle offsets, including turns with no final AI text."""
    previous = service.state
    offset = len(previous.get("messages", []))
    messages, boundaries = [], list(previous.get("history_turn_starts", []))
    for index in range(turns):
        boundaries.append(offset + len(messages))
        messages.append(HumanMessage(content=f"turn-{index}: " + (text or "gentle seated practice " * 30)))
        if tool_turns:
            messages.extend([
                AIMessage(content="", tool_calls=[
                    {"name": "get_catalog_tag_names", "args": {}, "id": f"tags-{index}", "type": "tool_call"},
                    {"name": "search_catalog_items", "args": {}, "id": f"search-{index}", "type": "tool_call"},
                ]),
                ToolMessage(content='{"tag_names":["Position/Sitting"]}', tool_call_id=f"tags-{index}"),
                ToolMessage(content='{"status":"error"}', tool_call_id=f"search-{index}", status="error"),
            ])
        else:
            messages.append(AIMessage(content=f"reply-{index}: " + "A seated practice with a forward bend. " * 15))
    service._get_agent().update_state(service._config, {
        "messages": messages, "history_turn_starts": boundaries,
    })
    return deepcopy(service.state["messages"])


def assert_tool_pairs(test, messages):
    pending = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            test.assertIn(message.tool_call_id, pending)
            pending.remove(message.tool_call_id)
        else:
            test.assertFalse(pending, "A tool result was lost at a message boundary")
            if isinstance(message, AIMessage):
                pending.update(call["id"] for call in message.tool_calls)
    test.assertFalse(pending)


class AssistantHistoryTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        environment.start()
        self.addCleanup(environment.stop)

    def service(self, responses=None, *, budget=None, summaries=None):
        model = ToolCallingModel(responses=responses or [AIMessage(content="Готово.")])
        summarizer = SummaryModel(responses=summaries or [summary_reply()])
        service = assistant(Mock(), model=model, summary_model=summarizer,
                            history_budget=budget or HistoryBudget(1600, 1000, 256))
        return service, model, summarizer

    def test_short_generate_nine_then_add_three_never_summarizes(self):
        from fixtures.compact_tool_results import pose
        generator = Mock()
        generator.generate.return_value = iter([pose(i) for i in range(1, 10)])
        model = ToolCallingModel(responses=[
            tool_call("generate_sequence", request="Generate 9", count=9), AIMessage(content="Готово"),
        ])
        summarizer = ToolCallingModel(responses=[summary_reply()])
        service = assistant(lambda: generator, model=model, summary_model=summarizer)
        list(service.stream("Generate 9"))
        source = service.get_sequence(service.active_sequence_uuid)
        generator.add_elements.return_value = [pose(i) for i in range(10, 13)]
        model.responses = [tool_call("add_sequence_elements", sequence_uuid=str(source.uuid),
                                     count=3, request="Добавь 3"), AIMessage(content="Добавлены")]
        model.i = 0
        list(service.stream("Добавь 3"))
        self.assertEqual(service.get_sequence(service.active_sequence_uuid).elements_count, 12)
        self.assertEqual(summarizer.received_messages, [])
        self.assertNotIn("history_summary", service.state)
        self.assertEqual(service.state["history_turn_starts"], [0, 4])
        self.assertEqual(model.received_messages[2][1:-1], service.state["messages"][:5])

    def test_one_summary_per_request_and_large_current_tool_loop_is_untouched(self):
        service, model, summarizer = self.service(responses=[
            tool_call("get_catalog_tag_names"), tool_call("get_catalog_tag_names"), AIMessage(content="Готово"),
        ])
        service._catalog_tag_service.get_names.return_value = ["Position/Sitting"] * 800
        original = seed_history(service, tool_turns=True, text="gentle seated practice " * 15)
        # Exercise a current turn far above the history budget, while staying
        # inside the separate public-request security limit.
        current = "Текущий запрос " * (MAX_REQUEST_CHARACTERS // len("Текущий запрос "))
        events = list(service.stream_response_events(current))
        self.assertEqual(len(summarizer.received_messages), 1)
        self.assertEqual(len(model.received_messages), 3)
        state = service.state
        counter = HistoryTokenCounter(model)
        retained = compact_history(original[state["history_projection_start"]:])
        self.assertLessEqual(counter.messages(summary_context(state["history_summary"]) + retained), 1000)
        self.assertGreaterEqual(len([m for m in retained if isinstance(m, HumanMessage)]), 2)
        for outgoing in model.received_messages:
            assert_tool_pairs(self, outgoing)
            current_index = next(i for i, m in enumerate(outgoing) if isinstance(m, HumanMessage) and m.content == current)
            self.assertEqual(outgoing[current_index:-1], state["messages"][len(original):len(original) + len(outgoing[current_index:-1])])
        self.assertGreater(counter.messages(model.received_messages[-1]), service._history_budget.max_tokens)
        self.assertEqual(state["messages"][:len(original)], original)
        self.assertEqual(state["history_summary_calls"][0]["input_tokens"], 123)
        self.assertEqual(state["history_summary_calls"][0]["cache_read_tokens"], 32)
        self.assertEqual(state["history_projection_version"], 2)
        self.assertEqual(events[0].type, "request_started")
        self.assertEqual(events[-1].type, "request_finished")
        service._generator_factory.assert_not_called()
        service._catalog_item_service.assert_not_called()
        self.assertEqual(summarizer.tool_names, [])

    def test_later_compaction_merges_previous_summary_only_with_new_turns(self):
        service, model, summarizer = self.service(summaries=[
            summary_reply(), summary_reply(ongoing_preferences=["Standing poses are allowed now."]),
        ])
        original = seed_history(service)
        list(service.stream("Теперь можно стоячие позы. " + "контекст " * 700))
        first = service.state
        covered = first["history_summary_covered_until"]
        first_summary = first["history_summary"]
        self.assertEqual(len(summarizer.received_messages), 1)
        list(service.stream("Что помнишь?"))
        self.assertEqual(len(summarizer.received_messages), 2)
        payload = json.loads(summarizer.received_messages[1][-1].content)
        self.assertEqual(payload["previous_summary"], json.loads(first_summary))
        newly_covered = [message for turn in payload["newly_covered_turns"] for message in turn]
        self.assertEqual(newly_covered[0]["content"], original[covered].content)
        self.assertIn("Теперь можно", newly_covered[-2]["content"])
        second = service.state
        self.assertGreater(second["history_summary_covered_until"], covered)
        self.assertEqual(json.loads(second["history_summary"])["ongoing_preferences"], ["Standing poses are allowed now."])
        self.assertEqual([m for m in model.received_messages[-1] if isinstance(m, HumanMessage)], [second["messages"][-2]])

    def test_summary_is_reused_until_max_and_no_calls_on_read_or_selection(self):
        service, model, summarizer = self.service()
        seed_history(service)
        source = source_sequence()
        service._get_agent().update_state(service._config, {
            "sequences": {str(source.uuid): source.model_dump(mode="json")},
            "active_sequence_uuid": str(source.uuid),
        })
        list(service.stream("Продолжим"))
        saved = service.state["history_summary"]
        for _ in range(3):
            list(service.stream("Да"))
            self.assertEqual(service.state["history_summary"], saved)
        calls = len(model.received_messages)
        service.select_sequence(str(source.uuid))
        self.assertEqual(service.get_sequence(str(source.uuid)), source)
        self.assertEqual(service.get_sequence_variants(str(source.uuid)), [source])
        self.assertEqual(len(model.received_messages), calls)
        self.assertEqual(len(summarizer.received_messages), 1)
        other, other_model, other_summary = self.service()
        list(other.stream("Что выбрано?"))
        self.assertNotIn("history_summary", other.state)
        self.assertIsNone(selected_snapshot(other_model))
        self.assertEqual(other_summary.received_messages, [])

    def test_summary_failure_without_previous_summary_keeps_a_valid_empty_or_recent_suffix(self):
        service, model, summarizer = self.service()
        original = seed_history(service, turns=1, text="очень длинный завершённый запрос " * 1200, tool_turns=True)
        with patch.object(type(summarizer), "invoke", side_effect=TimeoutError("offline")) as call:
            list(service.stream("Продолжим"))
        self.assertEqual(call.call_count, 1)
        self.assertNotIn("history_summary", service.state)
        self.assertEqual(service.state.get("history_summary_covered_until", 0), 0)
        self.assertEqual(service.state["history_projection_start"], len(original))
        self.assertTrue(service.state["history_context_incomplete"])
        self.assertIsNone(service.state["history_summary_calls"][0]["input_tokens"])
        self.assertGreater(service.state["history_summary_calls"][0]["estimated_input_tokens"], 0)
        self.assertEqual(service.state["messages"][:len(original)], original)
        assert_tool_pairs(self, model.received_messages[-1])

    def test_old_replacement_swap_and_addition_results_have_the_009_contract(self):
        from fixtures.compact_tool_results import run_case
        for operation in ("replace", "swap", "add"):
            with self.subTest(operation=operation):
                case = run_case(operation, legacy=True)
                original = deepcopy(case.service.state["messages"])
                projected = compact_history(original)
                result = next(m for m in reversed(projected) if isinstance(m, ToolMessage))
                payload = json.loads(result.content)
                self.assertEqual(payload["operation"], operation)
                self.assertNotIn("elements", payload)
                self.assertNotIn("generation_request", payload)
                self.assertNotIn("image_url", result.content)
                if operation == "replace":
                    self.assertEqual(set(payload["replacement"]["old_catalog_item"]), {"id", "name"})
                if operation == "swap":
                    self.assertTrue(all(entry["name"] for entry in payload["swap"]))
                self.assertEqual(case.service.state["messages"], original)
                assert_tool_pairs(self, projected)

    def test_app_rerun_save_and_variant_navigation_preserve_summarized_session(self):
        from streamlit.testing.v1 import AppTest
        from test_sequence_variants import APP
        from fixtures.compact_tool_results import pose
        generator = Mock()
        generator.generate.return_value = [pose(i) for i in range(1, 10)]
        model = ToolCallingModel(responses=[
            tool_call("generate_sequence", request="Generate 9", count=9), AIMessage(content="Готово"),
        ])
        save = Mock()
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator), patch(
            "langchain_app.dependencies.get_catalog_item_service", return_value=Mock(),
        ), patch("langchain_app.services.assistant_agent.init_chat_model", return_value=model), patch(
            "langchain_app.ui.sequence_import._save_component", return_value=save,
        ):
            app = AppTest.from_file(str(APP)).run()
            app.chat_input[0].set_value("Generate 9").run()
            self.assertFalse(app.exception)
            service = app.session_state.assistant_agent
            source = service.get_sequence(service.active_sequence_uuid)
            old_card = deepcopy(app.session_state.messages[1]["sequence"])
            saved = deepcopy(save.call_args.kwargs["data"]["sequence"])
            summarizer = SummaryModel(responses=[summary_reply()])
            service._summary_model = summarizer
            seed_history(service, turns=50)
            model.responses, model.i = [AIMessage(content="Помню мягкую практику")], 0
            app.chat_input[0].set_value("Какие пожелания ты помнишь?").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(summarizer.received_messages), 1)
            self.assertEqual(app.session_state.messages[1]["sequence"], old_card)
            self.assertEqual(save.call_args.kwargs["data"]["sequence"], saved)
            model.responses, model.i = [tool_call("swap_sequence_elements", sequence_uuid=str(source.uuid),
                first_element_id=str(source.elements[2].element_id), second_element_id=str(source.elements[4].element_id)),
                AIMessage(content="Поменяно")], 0
            app.chat_input[0].set_value("Поменяй третью и пятую").run()
            self.assertFalse(app.exception)
            variant = service.get_sequence(service.active_sequence_uuid)
            calls = len(model.received_messages)
            last_index = len(app.session_state.messages) - 1
            app.button(key=f"assistant-sequence-{last_index}-previous").click().run()
            self.assertEqual(service.active_sequence_uuid, str(source.uuid))
            self.assertEqual(save.call_args.kwargs["data"]["sequence"]["info"]["uuid"], str(source.uuid))
            app.button(key=f"assistant-sequence-{last_index}-next").click().run()
            self.assertEqual(service.active_sequence_uuid, str(variant.uuid))
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(model.received_messages), calls)
            self.assertEqual(len(summarizer.received_messages), 1)
            self.assertEqual(service.get_sequence(str(source.uuid)), source)

    def test_errors_and_invalid_summaries_keep_old_coverage_and_bound_fallback(self):
        bad = [TimeoutError("timeout"), RuntimeError("provider refused"), AIMessage(content=""),
               AIMessage(content="not JSON"), summary_reply(current_goal="huge " * 800),
               summary_reply().model_copy(update={"response_metadata": {"finish_reason": "length"}}),
               tool_call("generate_sequence", request="wrong", count=9)]
        for response in bad:
            with self.subTest(response=type(response).__name__):
                service, model, summarizer = self.service()
                original = seed_history(service, tool_turns=True)
                old_summary = HistorySummary.model_validate_json(summary_reply().text).model_dump_json()
                service._get_agent().update_state(service._config, {
                    "history_summary": old_summary, "history_summary_covered_until": 4,
                })
                with patch.object(type(summarizer), "invoke", side_effect=response if isinstance(response, Exception) else None,
                                  return_value=response) as call:
                    list(service.stream("Объясни выбранную последовательность"))
                self.assertEqual(call.call_count, 1)
                state = service.state
                self.assertEqual(state["history_summary"], old_summary)
                self.assertEqual(state["history_summary_covered_until"], 4)
                self.assertTrue(state["history_context_incomplete"])
                self.assertEqual(state["history_summary_calls"][-1]["status"], "error")
                outgoing = model.received_messages[-1][1:-2]
                self.assertLessEqual(HistoryTokenCounter(model).messages(outgoing), 1600)
                assert_tool_pairs(self, model.received_messages[-1])
                self.assertEqual(state["messages"][:len(original)], original)
                service._generator_factory.assert_not_called()

    def test_failure_with_open_tool_calls_is_repaired_before_projection(self):
        service, model, summarizer = self.service(responses=[tool_call("get_catalog_tag_names")])
        service._catalog_tag_service.get_names.side_effect = RuntimeError("offline")
        with self.assertRaisesRegex(RuntimeError, "offline"):
            list(service.stream("запрос " * 2000))
        service._catalog_tag_service.get_names.side_effect = None
        model.responses, model.i = [AIMessage(content="Продолжаем")], 0
        list(service.stream("Продолжим"))
        self.assertEqual(service.state["history_turn_starts"], [0, 3])
        repaired = service.state["messages"][2]
        self.assertIsInstance(repaired, ToolMessage)
        self.assertEqual(repaired.status, "error")
        assert_tool_pairs(self, model.received_messages[-1])
        summary_input = json.loads(summarizer.received_messages[0][-1].content)
        self.assertEqual(summary_input["newly_covered_turns"][0][-1]["status"], "error")
        self.assertEqual(len(service.state["history_summary_calls"]), 1)

    def test_summary_persists_when_main_model_fails(self):
        service, model, summarizer = self.service()
        seed_history(service)
        service._summary_model = Mock(invoke=Mock(return_value=summary_reply()), model_name="")
        with patch.object(type(model), "_generate", side_effect=RuntimeError("model failed")):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                list(service.stream("Продолжим"))
        self.assertIn("history_summary", service.state)
        self.assertGreater(service.state["history_summary_covered_until"], 0)
        self.assertEqual(service.state["history_summary_calls"][0]["status"], "ok")

    def test_legacy_compaction_precedes_counting_and_keeps_original_and_selected_variant(self):
        source = source_sequence(45)
        calls = tool_call("generate_sequence", request=source.generation_request, count=45)
        payload = legacy_payload(source, {"status": "ok"})
        legacy = ToolMessage(content=json.dumps(payload), name="generate_sequence", tool_call_id=calls.tool_calls[0]["id"])
        messages = [HumanMessage(content="Generate 45"), calls, legacy, AIMessage(content="Создано")]
        original = deepcopy(messages)
        service, model, summarizer = self.service()
        service._get_agent().update_state(service._config, {
            "messages": messages, "history_turn_starts": [0],
            "sequences": {str(source.uuid): source.model_dump(mode="json")},
            "active_sequence_uuid": str(source.uuid),
        })
        list(service.stream("Опиши её"))
        self.assertEqual(summarizer.received_messages, [])
        self.assertNotIn("elements", json.loads(model.received_messages[-1][3].content))
        self.assertEqual(service.state["messages"][2].content, original[2].content)
        self.assertEqual(selected_snapshot(model), source.model_context())
        self.assertEqual(service.get_sequence(str(source.uuid)), source)
        assert_tool_pairs(self, model.received_messages[-1])
        counter = HistoryTokenCounter(model)
        self.assertGreater(counter.messages(messages), service._history_budget.max_tokens)
        self.assertLess(counter.messages(compact_history(messages)), service._history_budget.max_tokens)

    def test_selected_older_variant_save_and_regenerate_remain_authoritative_after_summary(self):
        source = source_sequence()
        newer = source.model_copy(deep=True)
        from uuid import uuid4
        newer.uuid, newer.parent_uuid = uuid4(), source.uuid
        newer.elements.reverse()
        service, model, summarizer = self.service()
        seed_history(service)
        service._get_agent().update_state(service._config, {
            "sequences": {str(s.uuid): s.model_dump(mode="json") for s in (source, newer)},
            "active_sequence_uuid": str(newer.uuid),
        })
        service.select_sequence(str(source.uuid))
        model.responses, model.i = [tool_call("swap_sequence_elements", sequence_uuid=str(source.uuid),
            first_element_id=str(source.elements[2].element_id), second_element_id=str(source.elements[4].element_id)),
            AIMessage(content="Поменяно")], 0
        list(service.stream("Поменяй третью и пятую"))
        self.assertEqual(selected_snapshot(model, 0), source.model_context())
        result = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual(result.parent_uuid, source.uuid)
        self.assertEqual(result.generation_request, source.generation_request)
        from langchain_app.ui.sequence import generated_sequence_view
        self.assertEqual(generated_sequence_view(result)["info"]["uuid"], str(result.uuid))
        self.assertEqual(service.get_sequence(str(source.uuid)), source)
        self.assertEqual(service.get_sequence(str(newer.uuid)), newer)
        service._generator_factory.assert_not_called()
        self.assertEqual(len(summarizer.received_messages), 1)
        model.responses, model.i = [tool_call("generate_sequence", request=source.generation_request, count=9), AIMessage(content="Готово")], 0
        service._generator_factory.return_value.generate.return_value = iter([e.catalog_item for e in source.elements])
        list(service.generate("ignored", sequence_uuid=str(result.uuid)))
        regenerated = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual(regenerated.generation_request, source.generation_request)
        self.assertEqual(regenerated.family_uuid, source.family_uuid)

    def test_invalid_budgets_fail_at_initialization_and_fallback_is_labeled(self):
        for values in ((0, 2, 1), (10, 10, 1), (10, 5, 5), (10, 5, -1), (10, 5, True)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                HistoryBudget(*values)
        with patch.dict("os.environ", {"ASSISTANT_HISTORY_TARGET_TOKENS": "9000"}):
            with self.assertRaises(ValueError):
                assistant(Mock())
        service, model, _ = self.service()
        list(service.stream("Да"))
        self.assertIn("fallback:o200k_base", service.state["history_token_counter"])
        real = HistoryTokenCounter(ChatOpenAI(model="gpt-6-luna", api_key="test", reasoning_effort="none"))
        self.assertTrue(real.fallback)
        self.assertEqual(real.encoding.name, "o200k_base")
        self.assertGreater(real.messages([HumanMessage(content="hello")]), 0)


if __name__ == "__main__":
    unittest.main()
