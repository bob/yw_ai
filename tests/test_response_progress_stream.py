"""Verify public lifecycle events against the real agent/tool loop, offline."""

import unittest
from collections import Counter
from threading import Event
from unittest.mock import Mock, patch
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import TypeAdapter

from langchain_app.schemas import ResponseEvent
from langchain_app.services.assistant_agent import AssistantAgentError
from langchain_app.services.catalog_item import CatalogItemApiError
from langchain_app.services.sequence_generator import SequenceGenerator
from test_assistant_agent import ToolCallingModel, assistant, pose, tool_call, tool_model


class ResponseProgressStreamTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        env.start()
        self.addCleanup(env.stop)

    def gate(self):
        release = Event()
        self.addCleanup(release.set)
        return release

    def wait(self, release):
        if not release.wait(timeout=3):
            raise RuntimeError("Start/progress event was buffered until work completed")

    def assert_lifecycle(self, events, *, outcome="complete"):
        self.assertEqual(events[0].type, "request_started")
        self.assertEqual(events[-1].type, "request_finished")
        self.assertEqual(events[-1].outcome, outcome)
        self.assertEqual(len({event.request_id for event in events}), 1)
        self.assertEqual([event.event_index for event in events], list(range(len(events))))
        times = [event.elapsed_seconds for event in events]
        self.assertEqual(times, sorted(times))
        self.assertTrue(all(time >= 0 for time in times))
        started = [event for event in events if event.type == "step_started"]
        finished = [event for event in events if event.type == "step_finished"]
        self.assertEqual(Counter(event.step_id for event in started), Counter(event.step_id for event in finished))
        self.assertTrue(all(count == 1 for count in Counter(event.step_id for event in finished).values()))
        active = set()
        for event in events:
            if event.type == "step_started":
                if event.parent_step_id:
                    self.assertIn(event.parent_step_id, active)
                active.add(event.step_id)
            elif event.type == "step_finished":
                self.assertIn(event.step_id, active)
                active.remove(event.step_id)
            elif event.type in ("step_progress", "sequence_element", "sequence_committed"):
                self.assertIn(event.step_id, active)
        self.assertFalse(active)
        adapter = TypeAdapter(list[ResponseEvent])
        self.assertEqual(adapter.validate_json(adapter.dump_json(events)), events)

    def test_request_started_precedes_model_initialization_and_closes_on_init_failure(self):
        factory = Mock()
        service = assistant(factory)
        events = []
        with patch("langchain_app.services.assistant_agent.init_chat_model", side_effect=ValueError("secret config")) as init:
            stream = service.stream_response_events("Hello")
            events.append(next(stream))
            init.assert_not_called()
            with self.assertRaisesRegex(ValueError, "secret config"):
                events.extend(stream)
        self.assert_lifecycle(events, outcome="error")
        self.assertEqual([event.type for event in events], ["request_started", "request_finished"])
        self.assertNotIn("secret config", str(events))
        factory.assert_not_called()

    def test_thinking_start_is_visible_while_model_is_blocked(self):
        release = self.gate()
        model = ToolCallingModel(responses=[AIMessage(content="Hello")])
        original = ToolCallingModel._generate

        def blocked(model_instance, messages, **kwargs):
            self.wait(release)
            return original(model_instance, messages, **kwargs)

        events = []
        service = assistant(Mock(), model=model)
        with patch.object(ToolCallingModel, "_generate", new=blocked):
            for event in service.stream_response_events("Hello"):
                events.append(event)
                if event.type == "step_started" and event.code == "thinking":
                    self.assertFalse(model.received_messages)
                    release.set()
        self.assert_lifecycle(events)
        self.assertEqual([event.code for event in events if event.type == "step_started"],
                         ["thinking", "preparing_response"])
        self.assertEqual([event.text for event in events if event.type == "assistant_message"], ["Hello"])

    def test_catalog_tool_start_precedes_blocking_service_and_counts_empty_results(self):
        for name, code in (("search_catalog_items", "searching_catalog_items"),
                           ("get_catalog_tag_names", "loading_catalog_tags")):
            with self.subTest(name=name):
                release = self.gate()
                catalog, tags = Mock(), Mock()
                operation = catalog.search if name == "search_catalog_items" else tags.get_names
                result = [] if name == "search_catalog_items" else ["Standing", "Balance"]

                def blocked(**kwargs):
                    self.wait(release)
                    return result

                operation.side_effect = blocked
                model = ToolCallingModel(responses=[tool_call(name), AIMessage(content="Done")])
                service = assistant(Mock(), model=model, catalog_item_service=catalog, catalog_tag_service=tags)
                events = []
                for event in service.stream_response_events("Find poses"):
                    events.append(event)
                    if event.type == "step_started" and event.code == code:
                        self.assertFalse(any(e.type == "step_finished" and e.step_id == event.step_id for e in events))
                        release.set()
                self.assert_lifecycle(events)
                step = next(event for event in events if event.type == "step_started" and event.code == code)
                finish = next(event for event in events if event.type == "step_finished" and event.step_id == step.step_id)
                self.assertEqual(finish.outcome, "ok")
                self.assertIn("0 catalog items" if name == "search_catalog_items" else "2 catalog tags", finish.summary)
                self.assertIsNotNone(step.tool_call_id)

    def test_generator_search_elements_and_commit_arrive_before_followup_reply(self):
        allow_search, allow_selection, allow_reply = self.gate(), self.gate(), self.gate()
        first, last, hidden = pose(1), pose(2), pose(3)
        first.catalog_tag_names = ["Other/Starting"]
        last.catalog_tag_names = ["Other/Finishing"]
        hidden.image_url = None
        catalog, selector = Mock(), Mock()

        def search(**kwargs):
            self.wait(allow_search)
            return [first, last, hidden]

        def select(*args, **kwargs):
            self.wait(allow_selection)
            return last

        catalog.search.side_effect = search
        selector.choose_next_catalog_item.side_effect = select
        generator = SequenceGenerator(catalog, selector)
        model = ToolCallingModel(responses=[
            tool_call("generate_sequence", request="Two poses", count=2), AIMessage(content="Ready"),
        ])
        original = ToolCallingModel._generate

        def followup(model_instance, messages, **kwargs):
            if isinstance(messages[-1], ToolMessage):
                self.wait(allow_reply)
            return original(model_instance, messages, **kwargs)

        service = assistant(lambda: generator, model=model)
        events = []
        with patch.object(ToolCallingModel, "_generate", new=followup):
            for event in service.stream_response_events("Two poses", require_generation=True):
                events.append(event)
                if event.type == "step_started" and event.code == "searching_catalog_items":
                    self.assertIsNotNone(event.parent_step_id)
                    self.assertEqual(service.state["sequences"], {})
                    allow_search.set()
                elif event.type == "step_progress" and event.completed == 1:
                    self.assertIsNone(service.active_sequence_uuid)
                    self.assertEqual(len([e for e in events if e.type == "sequence_element"]), 1)
                    allow_selection.set()
                elif event.type == "sequence_committed":
                    sequence = service.get_sequence(str(event.sequence_uuid))
                    self.assertIsNotNone(sequence)
                    self.assertEqual(sequence.elements_count, 2)
                    self.assertEqual(service.active_sequence_uuid, str(event.sequence_uuid))
                    self.assertFalse(any(e.type == "assistant_message" and e.purpose == "final" for e in events))
                    allow_reply.set()
        self.assert_lifecycle(events)
        steps = [event for event in events if event.type == "step_started"]
        self.assertEqual([step.code for step in steps],
                         ["thinking", "generating_sequence", "searching_catalog_items", "thinking", "preparing_response"])
        search_step = steps[2]
        finish = next(e for e in events if e.type == "step_finished" and e.step_id == search_step.step_id)
        self.assertEqual(finish.summary, "Found 2 usable catalog candidates.")
        poses = [event for event in events if event.type == "sequence_element"]
        self.assertEqual([event.order for event in poses], [1, 2])
        self.assertEqual([event.catalog_item for event in poses], [first, last])
        self.assertEqual([e.completed for e in events if e.type == "step_progress"], [0, 0, 1, 2])
        self.assertEqual(len([e for e in events if e.type == "sequence_committed"]), 1)

    def test_intermediate_and_final_text_exclude_reasoning_arguments_and_tool_payload(self):
        intermediate = tool_call("search_catalog_items", name="secret arguments")
        intermediate.content = [
            {"type": "reasoning", "reasoning": "secret reasoning"},
            {"type": "text", "text": "I'll search the catalog."},
        ]
        final = AIMessage(content=[
            {"type": "reasoning", "reasoning": "secret final reasoning"},
            {"type": "text", "text": "Nothing found."},
        ])
        catalog = Mock()
        catalog.search.return_value = []
        model = ToolCallingModel(responses=[intermediate, final])
        service = assistant(Mock(), model=model, catalog_item_service=catalog)
        events = list(service.stream_response_events("Search"))
        self.assert_lifecycle(events)
        messages = [event for event in events if event.type == "assistant_message"]
        self.assertEqual([(m.purpose, m.text) for m in messages],
                         [("intermediate", "I'll search the catalog."), ("final", "Nothing found.")])
        self.assertNotIn("secret", str(events))
        self.assertEqual(list(service.stream("Search again")), ["Nothing found."])

    def test_repeated_searches_and_followup_requests_have_distinct_identities(self):
        catalog = Mock()
        catalog.search.return_value = []
        model = ToolCallingModel(responses=[
            tool_call("search_catalog_items", name="Tree"), tool_call("search_catalog_items", name="Mountain"),
            AIMessage(content="Nothing found"),
        ])
        service = assistant(Mock(), model=model, catalog_item_service=catalog)
        requested_id = uuid4()
        first = list(service.stream_response_events("Search", request_id=requested_id))
        second = list(service.stream_response_events("Search again"))
        self.assert_lifecycle(first)
        self.assert_lifecycle(second)
        self.assertEqual(first[0].request_id, requested_id)
        self.assertNotEqual(second[0].request_id, requested_id)
        searches = [e for e in first if e.type == "step_started" and e.code == "searching_catalog_items"]
        self.assertEqual(len(searches), 2)
        self.assertNotEqual(searches[0].step_id, searches[1].step_id)
        self.assertNotEqual(searches[0].tool_call_id, searches[1].tool_call_id)
        self.assertFalse({e.step_id for e in first if e.type == "step_started"} &
                         {e.step_id for e in second if e.type == "step_started"})

    def test_recoverable_catalog_and_validation_errors_finish_only_the_tool(self):
        for invalid in (False, True):
            with self.subTest(invalid=invalid):
                catalog = Mock()
                catalog.search.side_effect = CatalogItemApiError("secret response")
                model = ToolCallingModel(responses=[
                    tool_call("search_catalog_items", limit=0 if invalid else 10), AIMessage(content="Try another search"),
                ])
                events = list(assistant(Mock(), model=model, catalog_item_service=catalog).stream_response_events("Search"))
                self.assert_lifecycle(events)
                search = next(e for e in events if e.type == "step_started" and e.code == "searching_catalog_items")
                finish = next(e for e in events if e.type == "step_finished" and e.step_id == search.step_id)
                self.assertEqual(finish.outcome, "error")
                self.assertNotIn("secret response", str(events))
                if invalid:
                    catalog.search.assert_not_called()

    def test_generator_search_failure_closes_child_parent_and_request(self):
        catalog = Mock()
        catalog.search.side_effect = ValueError("secret service details")
        service = assistant(lambda: SequenceGenerator(catalog, Mock()), model=tool_model("Practice", count=2))
        events = []
        with self.assertRaisesRegex(ValueError, "secret service details"):
            events.extend(service.stream_response_events("Practice", require_generation=True))
        self.assert_lifecycle(events, outcome="error")
        failed = [e for e in events if e.type == "step_finished" and e.outcome == "error"]
        self.assertEqual(len(failed), 2)
        self.assertFalse(any(e.type == "sequence_committed" for e in events))
        self.assertEqual(service.state["sequences"], {})
        self.assertNotIn("secret service details", str(events))

    def test_partial_generation_failure_keeps_previous_variant_and_repairs_tool_history(self):
        generator = Mock()
        generator.generate.return_value = [pose(1)]
        model = tool_model("Original", count=1)
        service = assistant(lambda: generator, model=model)
        list(service.generate("Original"))
        previous = service.state["sequences"]
        source_uuid = service.active_sequence_uuid

        def fail(*args, **kwargs):
            yield pose(2)
            raise ValueError("failed after first pose")

        generator.generate.side_effect = fail
        events = []
        with self.assertRaisesRegex(ValueError, "failed after first pose"):
            events.extend(service.stream_response_events("Ignored", require_generation=True,
                                                         regenerate_sequence_uuid=source_uuid))
        self.assert_lifecycle(events, outcome="error")
        self.assertEqual(str(events[0].source_sequence_uuid), source_uuid)
        self.assertEqual(len([e for e in events if e.type == "sequence_element"]), 1)
        self.assertFalse(any(e.type == "sequence_committed" for e in events))
        self.assertEqual(service.state["sequences"], previous)
        self.assertEqual(service.active_sequence_uuid, source_uuid)
        self.assertEqual(service.state["messages"][-1].status, "error")

    def test_regenerate_commit_survives_failure_of_final_model_call(self):
        generator = Mock()
        generator.generate.side_effect = [[pose(1)], [pose(2)]]
        model = tool_model("Original", count=1)
        service = assistant(lambda: generator, model=model)
        initial = list(service.stream_response_events("Original", require_generation=True))
        original_uuid = service.active_sequence_uuid
        original = ToolCallingModel._generate

        def fail_reply(model_instance, messages, **kwargs):
            if isinstance(messages[-1], ToolMessage):
                raise ValueError("secret model failure")
            return original(model_instance, messages, **kwargs)

        events = []
        with patch.object(ToolCallingModel, "_generate", new=fail_reply):
            with self.assertRaisesRegex(ValueError, "secret model failure"):
                events.extend(service.stream_response_events("Ignored", require_generation=True,
                                                             regenerate_sequence_uuid=original_uuid))
        self.assert_lifecycle(events, outcome="error")
        commits = [event for event in events if event.type == "sequence_committed"]
        self.assertEqual(len(commits), 1)
        variant = service.get_sequence(str(commits[0].sequence_uuid))
        self.assertEqual(str(variant.parent_uuid), original_uuid)
        self.assertEqual(str(variant.uuid), service.active_sequence_uuid)
        self.assertEqual(variant.generation_request, "Original")
        self.assertEqual(variant.operation, "regenerate")
        self.assertNotEqual(initial[0].request_id, events[0].request_id)
        self.assertNotIn("secret model failure", str(events))

    def test_replacement_commits_once_and_no_change_emits_no_commit(self):
        for no_change in (False, True):
            with self.subTest(no_change=no_change):
                first, replacement = pose(1), pose(2)
                first.uuid, replacement.uuid = str(uuid4()), str(uuid4())
                generator = Mock()
                generator.generate.return_value = [first]
                model = tool_model("Original", count=1)
                catalog = Mock()
                service = assistant(lambda: generator, model=model, catalog_item_service=catalog)
                list(service.generate("Original"))
                source = service.get_sequence(service.active_sequence_uuid)
                candidate = first if no_change else replacement
                catalog.search.return_value = [candidate]
                model.responses = [
                    tool_call("search_catalog_items", name=candidate.name),
                    tool_call("replace_sequence_element", sequence_uuid=str(source.uuid),
                              element_id=str(source.elements[0].element_id), replacement_catalog_item_id=candidate.id),
                    AIMessage(content="Done"),
                ]
                events = []
                for event in service.stream_response_events("Replace first"):
                    events.append(event)
                    if event.type == "sequence_committed":
                        self.assertIsNotNone(service.get_sequence(str(event.sequence_uuid)))
                self.assert_lifecycle(events)
                step = next(e for e in events if e.type == "step_started" and e.code == "updating_sequence")
                finish = next(e for e in events if e.type == "step_finished" and e.step_id == step.step_id)
                self.assertEqual(finish.outcome, "no_change" if no_change else "ok")
                self.assertEqual(len([e for e in events if e.type == "sequence_committed"]), 0 if no_change else 1)
                self.assertEqual(len(service.state["sequences"]), 1 if no_change else 2)
                self.assertFalse(any(e.type == "sequence_element" for e in events))

    def test_required_generation_without_commit_finishes_with_error(self):
        factory = Mock()
        service = assistant(factory, model=ToolCallingModel(responses=[AIMessage(content="Please clarify")]))
        events = []
        with self.assertRaises(AssistantAgentError):
            events.extend(service.stream_response_events("Practice", require_generation=True))
        self.assert_lifecycle(events, outcome="error")
        self.assertFalse(any(e.type == "sequence_committed" for e in events))
        factory.assert_not_called()

    def test_invalid_regenerate_source_still_has_a_terminal_error(self):
        for source in (str(uuid4()), "invalid-uuid"):
            with self.subTest(source=source):
                factory = Mock()
                model = ToolCallingModel(responses=[AIMessage(content="Unused")])
                service = assistant(factory, model=model)
                events = []
                with self.assertRaisesRegex(ValueError, "Regenerate source is not stored"):
                    events.extend(service.stream_response_events("Ignored", regenerate_sequence_uuid=source))
                self.assert_lifecycle(events, outcome="error")
                self.assertEqual([e.type for e in events], ["request_started", "request_finished"])
                factory.assert_not_called()
                self.assertEqual(model.received_messages, [])


if __name__ == "__main__":
    unittest.main()
