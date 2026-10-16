"""Verify public lifecycle events against the real agent/tool loop, offline."""

import unittest
from collections import Counter
from threading import Event
from unittest.mock import Mock, patch
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import TypeAdapter

from langchain_app.schemas import CatalogItemSchema, CatalogItemsDisplayedEvent, ResponseEvent
from langchain_app.services.assistant_agent import AssistantAgentError
from langchain_app.services.catalog_item import CatalogItemApiError
from langchain_app.services.sequence_editor import insert_elements, remove_elements, swap_elements
from langchain_app.services.sequence_generator import SequenceGenerator
from langchain_app.services.type_safe import TypeSafeSelectionError
from test_assistant_agent import last_message, ToolCallingModel, assistant, pose, tool_call, tool_model
from test_sequence_addition_tool import addition_call
from test_sequence_insertion import pose as addition_pose, source as addition_source
from test_sequence_removal_tool import removal_call
from test_sequence_swap import sequence, swap_call


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
            elif event.type in ("step_progress", "sequence_element", "sequence_committed", "catalog_items_displayed"):
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
                self.assertFalse(any(e.type == "catalog_items_displayed" for e in events))

    def test_display_publishes_only_chosen_items_inside_its_step_before_followup_reply(self):
        allow_reply = self.gate()
        selected = [pose(2), pose(3), pose(4)]
        selected[0].image_url = None
        selected[1].bilateral = True
        selected[1].catalog_tag_names = ["Seated"]
        catalog = Mock()
        catalog.search.side_effect = [[pose(1)], selected]
        model = ToolCallingModel(responses=[
            tool_call("search_catalog_items", name="First"),
            tool_call("search_catalog_items", name="Second"),
            tool_call("show_catalog_items", catalog_item_ids=[3, 2]), AIMessage(content="Here are the options."),
        ])
        original = ToolCallingModel._generate

        def followup(model_instance, messages, **kwargs):
            if isinstance(last_message(messages), ToolMessage) and last_message(messages).artifact is not None:
                self.wait(allow_reply)
            return original(model_instance, messages, **kwargs)

        factory = Mock()
        service = assistant(factory, model=model, catalog_item_service=catalog)
        events = []
        with patch.object(ToolCallingModel, "_generate", new=followup):
            for event in service.stream_response_events("Show options"):
                events.append(event)
                if isinstance(event, CatalogItemsDisplayedEvent):
                    self.assertFalse(any(e.type == "assistant_message" and e.purpose == "final" for e in events))
                    self.assertFalse(any(e.type == "step_finished" and e.step_id == event.step_id for e in events))
                    allow_reply.set()
        self.assert_lifecycle(events)
        displayed, = [e for e in events if isinstance(e, CatalogItemsDisplayedEvent)]
        self.assertEqual(displayed.items, [selected[1], selected[0]])
        step = next(e for e in events if e.type == "step_started" and e.step_id == displayed.step_id)
        finish = next(e for e in events if e.type == "step_finished" and e.step_id == displayed.step_id)
        self.assertEqual((step.code, step.title), ("showing_catalog_items", "Showing catalog items..."))
        self.assertEqual(displayed.tool_call_id, step.tool_call_id)
        self.assertEqual(displayed.tool_call_id, finish.tool_call_id)
        self.assertLess(step.event_index, displayed.event_index)
        self.assertLess(displayed.event_index, finish.event_index)
        self.assertEqual((finish.outcome, finish.summary), ("ok", "Displayed 2 catalog items."))
        self.assertFalse(any(e.type in ("sequence_element", "sequence_committed") for e in events))
        self.assertFalse(service.state["turn_operation_committed"])
        self.assertIsNone(service.active_sequence_uuid)
        self.assertEqual(service.state["latest_catalog_items"], [item.model_dump(mode="json") for item in selected])
        displayed.items[0].name = "Changed"
        displayed.items[0].catalog_tag_names.append("Changed")
        self.assertEqual(service.state["latest_catalog_items"][1]["name"], "Pose 3")
        result = next(m for m in service.state["messages"] if isinstance(m, ToolMessage) and m.artifact is not None)
        self.assertEqual(result.artifact["items"][0]["catalog_tag_names"], ["Seated"])
        factory.assert_not_called()

    def test_invalid_display_and_repeated_display_emit_no_snapshot_and_finish_with_error(self):
        for ids in ([], [0], [True], ["1"], [1, 1], [1, 99]):
            with self.subTest(ids=ids):
                catalog = Mock()
                catalog.search.return_value = [pose(1)]
                model = ToolCallingModel(responses=[
                    tool_call("search_catalog_items"), tool_call("show_catalog_items", catalog_item_ids=ids),
                    tool_call("show_catalog_items", catalog_item_ids=[1]),
                    tool_call("show_catalog_items", catalog_item_ids=[1]), AIMessage(content="Shown."),
                ])
                events = list(assistant(Mock(), model=model, catalog_item_service=catalog).stream_response_events("Show options"))
                self.assert_lifecycle(events)
                displays = [e for e in events if e.type == "step_started" and e.code == "showing_catalog_items"]
                self.assertEqual(len(displays), 3)
                finishes = [next(e for e in events if e.type == "step_finished" and e.step_id == step.step_id)
                            for step in displays]
                self.assertEqual([e.outcome for e in finishes], ["error", "ok", "error"])
                displayed, = [e for e in events if e.type == "catalog_items_displayed"]
                self.assertEqual(displayed.step_id, displays[1].step_id)
                self.assertEqual([item.id for item in displayed.items], [1])

    def test_empty_or_failed_search_cannot_publish_display(self):
        for failed in (False, True):
            with self.subTest(failed=failed):
                catalog = Mock()
                catalog.search.return_value = []
                if failed:
                    catalog.search.side_effect = CatalogItemApiError("secret API details")
                model = ToolCallingModel(responses=[
                    tool_call("search_catalog_items"), tool_call("show_catalog_items", catalog_item_ids=[1]),
                    AIMessage(content="No options found."),
                ])
                service = assistant(Mock(), model=model, catalog_item_service=catalog)
                events = list(service.stream_response_events("Show options"))
                self.assert_lifecycle(events)
                self.assertFalse(any(e.type == "catalog_items_displayed" for e in events))
                show = next(e for e in events if e.type == "step_started" and e.code == "showing_catalog_items")
                finish = next(e for e in events if e.type == "step_finished" and e.step_id == show.step_id)
                self.assertEqual(finish.outcome, "error")
                self.assertEqual(service.state["latest_catalog_items"], [])
                self.assertNotIn("secret API details", str(events))

    def test_model_failure_before_or_after_display_preserves_only_successful_display(self):
        for show in (False, True):
            with self.subTest(show=show):
                catalog = Mock()
                catalog.search.return_value = [pose(1)]
                calls = [tool_call("search_catalog_items")]
                if show:
                    calls.append(tool_call("show_catalog_items", catalog_item_ids=[1]))
                model = ToolCallingModel(responses=[*calls, AIMessage(content="Unused")])
                original = ToolCallingModel._generate

                def fail_reply(model_instance, messages, **kwargs):
                    if isinstance(last_message(messages), ToolMessage) and (not show or last_message(messages).artifact is not None):
                        raise ValueError("secret model failure")
                    return original(model_instance, messages, **kwargs)

                service = assistant(Mock(), model=model, catalog_item_service=catalog)
                events = []
                with patch.object(ToolCallingModel, "_generate", new=fail_reply):
                    with self.assertRaisesRegex(ValueError, "secret model failure"):
                        events.extend(service.stream_response_events("Show options"))
                self.assert_lifecycle(events, outcome="error")
                displayed = [e for e in events if e.type == "catalog_items_displayed"]
                self.assertEqual(len(displayed), 1 if show else 0)
                if show:
                    self.assertEqual(displayed[0].items, [pose(1)])
                    finish = next(e for e in events if e.type == "step_finished" and e.step_id == displayed[0].step_id)
                    self.assertEqual(finish.outcome, "ok")
                self.assertFalse(any(e.type == "assistant_message" and e.purpose == "final" for e in events))
                self.assertNotIn("secret model failure", str(events))

    def test_later_searches_and_next_request_do_not_update_a_displayed_snapshot(self):
        catalog = Mock()
        catalog.search.side_effect = [[pose(1)], [pose(2)]]
        model = ToolCallingModel(responses=[
            tool_call("search_catalog_items"), tool_call("show_catalog_items", catalog_item_ids=[1]),
            tool_call("search_catalog_items"), tool_call("show_catalog_items", catalog_item_ids=[2]),
            AIMessage(content="Shown."),
        ])
        service = assistant(Mock(), model=model, catalog_item_service=catalog)
        first = list(service.stream_response_events("Show one"))
        self.assert_lifecycle(first)
        first_display, = [e for e in first if e.type == "catalog_items_displayed"]
        model.responses = [tool_call("show_catalog_items", catalog_item_ids=[2]), AIMessage(content="")]
        second = list(service.stream_response_events("Show another"))
        self.assert_lifecycle(second)
        second_display, = [e for e in second if e.type == "catalog_items_displayed"]
        self.assertEqual(first_display.items, [pose(1)])
        self.assertEqual(second_display.items, [pose(2)])
        self.assertNotEqual(first_display.request_id, second_display.request_id)
        self.assertNotEqual(first_display.step_id, second_display.step_id)
        self.assertFalse(any(e.type == "assistant_message" for e in second))

    def test_legacy_adapters_ignore_catalog_display_and_keep_generation_item_types(self):
        catalog = Mock()
        catalog.search.return_value = [pose(1)]
        generator = Mock()
        generator.generate.return_value = [pose(2)]
        model = ToolCallingModel(responses=[
            tool_call("search_catalog_items"), tool_call("show_catalog_items", catalog_item_ids=[1]),
            AIMessage(content="Options."),
        ])
        service = assistant(lambda: generator, model=model, catalog_item_service=catalog)
        self.assertEqual(list(service.stream("Show options")), ["Options."])
        model.responses = [
            tool_call("show_catalog_items", catalog_item_ids=[1]),
            tool_call("generate_sequence", request="Practice", count=1), AIMessage(content="Generated."),
        ]
        chunks = list(service.generate("Practice"))
        self.assertEqual(chunks, [pose(2)])
        self.assertTrue(all(isinstance(chunk, CatalogItemSchema) for chunk in chunks))
        model.responses = [
            tool_call("show_catalog_items", catalog_item_ids=[1]),
            tool_call("generate_sequence", request="Practice", count=1), AIMessage(content="Regenerated."),
        ]
        events = list(service.stream_response_events("Regenerate", regenerate_sequence_uuid=service.active_sequence_uuid))
        self.assert_lifecycle(events)
        self.assertFalse(any(e.type == "catalog_items_displayed" for e in events))
        show = next(e for e in events if e.type == "step_started" and e.code == "showing_catalog_items")
        finish = next(e for e in events if e.type == "step_finished" and e.step_id == show.step_id)
        self.assertEqual(finish.outcome, "error")

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
            if isinstance(last_message(messages), ToolMessage):
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
        self.assertFalse(any(e.type == "catalog_items_displayed" for e in events))

    def addition_service(self, count=5):
        source = addition_source(count)
        catalog, selector = Mock(), Mock()
        catalog.search.return_value = [addition_pose(n) for n in range(100, 150)]
        selector.choose_addition_catalog_item.side_effect = lambda candidates, **kwargs: candidates[0]
        generator = SequenceGenerator(catalog, selector)
        model = ToolCallingModel(responses=[AIMessage(content="Ready")])
        service = assistant(lambda: generator, model=model)
        service._get_agent().update_state(service._config, {
            "sequences": {str(source.uuid): source.model_dump(mode="json")},
            "active_sequence_uuid": str(source.uuid),
        })
        return service, source, model, catalog, selector

    def test_addition_progress_is_live_and_commit_confirms_the_entire_inserted_block(self):
        service, source, model, catalog, selector = self.addition_service()
        search_gate, first_gate, second_gate, commit_gate, reply_gate = [self.gate() for _ in range(5)]
        catalog.search.return_value = [addition_pose(100), addition_pose(101)]
        catalog.search.side_effect = lambda **kwargs: (self.wait(search_gate), catalog.search.return_value)[1]

        def choose(candidates, **kwargs):
            self.wait(second_gate if kwargs["selected_additions"] else first_gate)
            return min(candidates, key=lambda item: item.id)

        selector.choose_addition_catalog_item.side_effect = choose
        model.responses = [addition_call(source, count=2, insert_before_element_id=str(source.elements[2].element_id)),
                           AIMessage(content="Added two poses before the third.")]
        original = ToolCallingModel._generate

        def reply(model_instance, messages, **kwargs):
            if isinstance(last_message(messages), ToolMessage):
                self.wait(reply_gate)
            return original(model_instance, messages, **kwargs)

        def commit(*args, **kwargs):
            self.wait(commit_gate)
            return insert_elements(*args, **kwargs)

        events = []
        with patch.object(ToolCallingModel, "_generate", new=reply), patch(
            "langchain_app.tools.sequence.insert_elements", side_effect=commit,
        ):
            for event in service.stream_response_events("Add two before the third"):
                events.append(event)
                if event.type == "step_started" and event.code == "searching_catalog_items":
                    self.assertIsNotNone(event.parent_step_id)
                    search_gate.set()
                elif event.type == "step_progress":
                    self.assertEqual(event.total, 2)
                    self.assertEqual(service.active_sequence_uuid, str(source.uuid))
                    self.assertEqual(len(service.state["sequences"]), 1)
                    if event.completed == 0:
                        first_gate.set()
                    elif event.completed == 1:
                        second_gate.set()
                    elif event.completed == 2:
                        commit_gate.set()
                elif event.type == "sequence_committed":
                    variant = service.get_sequence(str(event.sequence_uuid))
                    self.assertEqual([e.catalog_item.id for e in variant.elements], [1, 2, 100, 101, 3, 4, 5])
                    self.assertEqual(service.active_sequence_uuid, str(variant.uuid))
                    reply_gate.set()
        self.assert_lifecycle(events)
        update = next(e for e in events if e.type == "step_started" and e.code == "updating_sequence")
        self.assertEqual(update.title, "Adding sequence elements...")
        progress = [e for e in events if e.type == "step_progress"]
        self.assertEqual([e.completed for e in progress], [0, 0, 1, 2])
        self.assertTrue(all(e.step_id == update.step_id and e.tool_call_id == update.tool_call_id for e in progress))
        committed = next(e for e in events if e.type == "sequence_committed")
        self.assertEqual(committed.step_id, update.step_id)
        finish = next(e for e in events if e.type == "step_finished" and e.step_id == update.step_id)
        self.assertEqual((finish.outcome, finish.summary), ("ok", "Added 2 sequence elements; 7 poses in total."))
        self.assertLess(events.index(committed), events.index(finish))
        self.assertEqual(sum(e.type == "sequence_committed" for e in events), 1)
        self.assertFalse(any(e.type in ("sequence_element", "catalog_items_displayed") for e in events))
        self.assertEqual(service.get_sequence(str(source.uuid)), source)

    def test_addition_progress_total_uses_free_space_and_summary_explains_the_limit(self):
        service, source, model, _, _ = self.addition_service(count=43)
        model.responses = [addition_call(source, count=100), AIMessage(content="Added two; limit 45.")]
        events = list(service.stream_response_events("Add 100"))
        self.assert_lifecycle(events)
        self.assertEqual([e.total for e in events if e.type == "step_progress"], [2, 2, 2, 2])
        finish = next(e for e in events if e.type == "step_finished" and "Added 2" in e.summary)
        self.assertIn("45 poses in total", finish.summary)
        self.assertIn("requested 100 additions", finish.summary)
        self.assertEqual(service.get_sequence(service.active_sequence_uuid).elements_count, 45)

    def test_addition_errors_and_no_change_finish_without_commit_or_preview(self):
        for mode in ("full", "catalog", "selection", "invalid"):
            with self.subTest(mode=mode):
                service, source, model, catalog, selector = self.addition_service(count=45 if mode == "full" else 5)
                if mode == "catalog":
                    catalog.search.side_effect = CatalogItemApiError("offline")
                elif mode == "selection":
                    selector.choose_addition_catalog_item.side_effect = [addition_pose(100), TypeSafeSelectionError("offline")]
                model.responses = [addition_call(source, count=2, **(
                    {"insert_before_element_id": str(uuid4())} if mode == "invalid" else {}
                )), AIMessage(content="No addition applied.")]
                events = list(service.stream_response_events("Add two"))
                self.assert_lifecycle(events)
                self.assertFalse(any(e.type in ("sequence_committed", "sequence_element", "catalog_items_displayed") for e in events))
                self.assertEqual(service.active_sequence_uuid, str(source.uuid))
                self.assertEqual(len(service.state["sequences"]), 1)
                update = next(e for e in events if e.type == "step_started" and e.code == "updating_sequence")
                finish = next(e for e in events if e.type == "step_finished" and e.step_id == update.step_id)
                self.assertEqual(finish.outcome, "no_change" if mode == "full" else "error")
                if mode == "full":
                    self.assertIn("45-element limit", finish.summary)
                if mode in ("full", "invalid"):
                    catalog.search.assert_not_called()
                    self.assertFalse(any(e.type == "step_progress" for e in events))
                if mode == "selection":
                    self.assertEqual([e.completed for e in events if e.type == "step_progress"], [0, 0, 1])

    def test_addition_commit_survives_a_failure_of_the_final_model_call(self):
        service, source, model, _, _ = self.addition_service()
        model.responses = [addition_call(source, count=2)]
        original = ToolCallingModel._generate

        def fail_reply(model_instance, messages, **kwargs):
            if isinstance(last_message(messages), ToolMessage):
                raise AssistantAgentError("Model unavailable")
            return original(model_instance, messages, **kwargs)

        events = []
        with patch.object(ToolCallingModel, "_generate", new=fail_reply), self.assertRaises(AssistantAgentError):
            events.extend(service.stream_response_events("Add two"))
        self.assert_lifecycle(events, outcome="error")
        self.assertEqual(sum(e.type == "sequence_committed" for e in events), 1)
        variant = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual((variant.operation, variant.parent_uuid, variant.elements_count), ("add", source.uuid, 7))

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
            if isinstance(last_message(messages), ToolMessage):
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
                self.assertFalse(any(e.type == "catalog_items_displayed" for e in events))

    def test_required_generation_without_commit_finishes_with_error(self):
        factory = Mock()
        service = assistant(factory, model=ToolCallingModel(responses=[AIMessage(content="Please clarify")]))
        events = []
        with self.assertRaises(AssistantAgentError):
            events.extend(service.stream_response_events("Practice", require_generation=True))
        self.assert_lifecycle(events, outcome="error")
        self.assertFalse(any(e.type == "sequence_committed" for e in events))
        factory.assert_not_called()

    def generated_swap_source(self):
        generator = Mock()
        generator.generate.return_value = [element.catalog_item for element in sequence().elements]
        factory, catalog, tags = Mock(return_value=generator), Mock(), Mock()
        model = tool_model("Original practice", count=5)
        service = assistant(factory, model=model, catalog_item_service=catalog, catalog_tag_service=tags)
        list(service.generate("Original practice"))
        factory.reset_mock()
        return service, service.get_sequence(service.active_sequence_uuid), model, factory, catalog, tags

    def test_swap_start_precedes_edit_and_commit_confirms_full_state_before_followup_reply(self):
        service, source, model, factory, catalog, tags = self.generated_swap_source()
        allow_swap, allow_reply = self.gate(), self.gate()
        model.responses = [swap_call(source), AIMessage(content="Swapped third and fifth.")]
        original = ToolCallingModel._generate

        def blocked_swap(*args, **kwargs):
            self.wait(allow_swap)
            return swap_elements(*args, **kwargs)

        def followup(model_instance, messages, **kwargs):
            if isinstance(last_message(messages), ToolMessage) and last_message(messages).name == "swap_sequence_elements":
                self.wait(allow_reply)
            return original(model_instance, messages, **kwargs)

        request_id = uuid4()
        events = []
        with patch("langchain_app.tools.sequence.swap_elements", side_effect=blocked_swap), patch.object(
            ToolCallingModel, "_generate", new=followup,
        ):
            for event in service.stream_response_events("Swap third and fifth", request_id=request_id):
                events.append(event)
                if event.type == "step_started" and event.code == "updating_sequence":
                    self.assertEqual(service.active_sequence_uuid, str(source.uuid))
                    self.assertEqual(len(service.state["sequences"]), 1)
                    allow_swap.set()
                elif event.type == "sequence_committed":
                    variant = service.get_sequence(str(event.sequence_uuid))
                    self.assertEqual(variant.operation, "swap")
                    self.assertEqual([element.catalog_item.id for element in variant.elements], [1, 2, 5, 4, 3])
                    self.assertEqual(service.active_sequence_uuid, str(event.sequence_uuid))
                    self.assertTrue(service.state["turn_operation_committed"])
                    self.assertEqual(service.state["sequences"][str(variant.uuid)], variant.model_dump(mode="json"))
                    self.assertFalse(any(e.type == "step_finished" and e.step_id == event.step_id for e in events))
                    self.assertFalse(any(e.type == "assistant_message" and e.purpose == "final" for e in events))
                    allow_reply.set()
        self.assert_lifecycle(events)
        committed, = [e for e in events if e.type == "sequence_committed"]
        step = next(e for e in events if e.type == "step_started" and e.code == "updating_sequence")
        finish = next(e for e in events if e.type == "step_finished" and e.step_id == step.step_id)
        self.assertEqual(step.title, "Swapping sequence elements...")
        self.assertEqual((finish.outcome, finish.summary), ("ok", "Sequence elements swapped."))
        self.assertEqual(committed.request_id, request_id)
        self.assertEqual(committed.step_id, step.step_id)
        self.assertEqual(committed.tool_call_id, step.tool_call_id)
        self.assertEqual(finish.tool_call_id, step.tool_call_id)
        self.assertEqual(model.received_messages[-1][-2].tool_call_id, step.tool_call_id)
        self.assertLess(step.event_index, committed.event_index)
        self.assertLess(committed.event_index, finish.event_index)
        self.assertFalse(any(e.type in ("sequence_element", "catalog_items_displayed") for e in events))
        self.assertEqual(len(service.state["sequences"]), 2)
        self.assertEqual(service.get_sequence(str(source.uuid)), source)
        factory.assert_not_called()
        catalog.assert_not_called()
        self.assertEqual(catalog.mock_calls, [])
        self.assertEqual(tags.mock_calls, [])

    def test_no_change_and_invalid_swaps_finish_without_commit_or_preview_events(self):
        for case in ("no_change", "unknown_element", "invalid_uuid", "stale_sequence"):
            with self.subTest(case=case):
                service, source, model, factory, catalog, tags = self.generated_swap_source()
                overrides = {
                    "no_change": {"second_element_id": str(source.elements[2].element_id)},
                    "unknown_element": {"second_element_id": str(uuid4())},
                    "invalid_uuid": {"first_element_id": "invalid-uuid"},
                    "stale_sequence": {"sequence_uuid": str(uuid4())},
                }[case]
                model.responses = [swap_call(source, **overrides), AIMessage(content="No swap applied.")]
                before = service.state["sequences"]
                events = list(service.stream_response_events("Swap the pair"))
                self.assert_lifecycle(events)
                step = next(e for e in events if e.type == "step_started" and e.code == "updating_sequence")
                finish = next(e for e in events if e.type == "step_finished" and e.step_id == step.step_id)
                self.assertEqual(step.title, "Swapping sequence elements...")
                self.assertEqual(finish.outcome, "no_change" if case == "no_change" else "error")
                self.assertEqual(finish.tool_call_id, step.tool_call_id)
                if case == "no_change":
                    self.assertEqual(finish.summary, "Both references identify the same sequence element; no change was needed.")
                self.assertFalse(any(e.type in ("sequence_committed", "sequence_element", "catalog_items_displayed") for e in events))
                self.assertEqual(service.state["sequences"], before)
                self.assertEqual(service.active_sequence_uuid, str(source.uuid))
                self.assertFalse(service.state["turn_operation_committed"])
                factory.assert_not_called()
                self.assertEqual(catalog.mock_calls, [])
                self.assertEqual(tags.mock_calls, [])

    def test_removal_start_is_live_and_commit_exposes_full_state_before_empty_final(self):
        service, source, model, factory, catalog, tags = self.generated_swap_source()
        allow_edit, allow_reply = self.gate(), self.gate()
        model.responses = [removal_call(source, element_ids=[
            str(source.elements[i].element_id) for i in (3, 1)
        ]), AIMessage(content="")]
        original = ToolCallingModel._generate

        def edit(*args, **kwargs):
            self.wait(allow_edit)
            return remove_elements(*args, **kwargs)

        def reply(model_instance, messages, **kwargs):
            if isinstance(last_message(messages), ToolMessage):
                self.wait(allow_reply)
            return original(model_instance, messages, **kwargs)

        events, request_id = [], uuid4()
        with patch("langchain_app.tools.sequence.remove_elements", side_effect=edit), patch.object(
            ToolCallingModel, "_generate", new=reply,
        ):
            for event in service.stream_response_events("Remove second and fourth", request_id=request_id):
                events.append(event)
                if event.type == "step_started" and event.code == "updating_sequence":
                    self.assertEqual(service.active_sequence_uuid, str(source.uuid))
                    self.assertEqual(len(service.state["sequences"]), 1)
                    allow_edit.set()
                elif event.type == "sequence_committed":
                    variant = service.get_sequence(str(event.sequence_uuid))
                    self.assertEqual((variant.operation, variant.elements_count), ("remove", 3))
                    self.assertEqual([e.element_id for e in variant.elements],
                                     [source.elements[i].element_id for i in (0, 2, 4)])
                    self.assertEqual(service.active_sequence_uuid, str(variant.uuid))
                    self.assertEqual(service.state["sequences"][str(variant.uuid)], variant.model_dump(mode="json"))
                    self.assertTrue(service.state["turn_operation_committed"])
                    allow_reply.set()
        self.assert_lifecycle(events)
        commit, = [e for e in events if e.type == "sequence_committed"]
        step = next(e for e in events if e.type == "step_started" and e.code == "updating_sequence")
        finish = next(e for e in events if e.type == "step_finished" and e.step_id == step.step_id)
        self.assertEqual(step.title, "Removing sequence elements...")
        self.assertEqual((finish.outcome, finish.summary), ("ok", "Removed 2 sequence elements; 3 poses in total."))
        self.assertEqual(commit.request_id, request_id)
        self.assertEqual((commit.step_id, commit.tool_call_id), (step.step_id, step.tool_call_id))
        self.assertEqual(finish.tool_call_id, step.tool_call_id)
        self.assertEqual(model.received_messages[-1][-2].tool_call_id, step.tool_call_id)
        self.assertLess(step.event_index, commit.event_index)
        self.assertLess(commit.event_index, finish.event_index)
        self.assertEqual([e.code for e in events if e.type == "step_started"],
                         ["thinking", "updating_sequence", "thinking"])
        self.assertFalse(any(e.type in ("sequence_element", "step_progress", "catalog_items_displayed", "assistant_message")
                             for e in events))
        self.assertEqual(service.get_sequence(str(source.uuid)), source)
        factory.assert_not_called()
        self.assertEqual(catalog.mock_calls + tags.mock_calls, [])

    def test_invalid_removals_finish_with_error_without_commit_or_previews(self):
        for case in ("unknown", "all", "duplicate"):
            with self.subTest(case=case):
                service, source, model, factory, catalog, tags = self.generated_swap_source()
                targets = {
                    "unknown": [str(source.elements[0].element_id), str(uuid4())],
                    "all": [str(e.element_id) for e in source.elements],
                    "duplicate": [str(source.elements[0].element_id)] * 2,
                }[case]
                model.responses = [removal_call(source, element_ids=targets), AIMessage(content="Cannot remove these poses.")]
                events = list(service.stream_response_events("Remove poses"))
                self.assert_lifecycle(events)
                step = next(e for e in events if e.type == "step_started" and e.code == "updating_sequence")
                finish = next(e for e in events if e.type == "step_finished" and e.step_id == step.step_id)
                self.assertEqual((step.title, finish.outcome), ("Removing sequence elements...", "error"))
                self.assertEqual(finish.tool_call_id, step.tool_call_id)
                self.assertFalse(any(e.type in ("sequence_committed", "sequence_element", "step_progress", "catalog_items_displayed")
                                     for e in events))
                self.assertEqual(service.active_sequence_uuid, str(source.uuid))
                self.assertEqual(len(service.state["sequences"]), 1)
                self.assertFalse(service.state["turn_operation_committed"])
                factory.assert_not_called()
                self.assertEqual(catalog.mock_calls + tags.mock_calls, [])

    def test_removal_commit_survives_failure_of_final_model_call(self):
        service, source, model, factory, catalog, tags = self.generated_swap_source()
        model.responses = [removal_call(source)]
        original = ToolCallingModel._generate

        def fail_reply(model_instance, messages, **kwargs):
            if isinstance(last_message(messages), ToolMessage):
                raise AssistantAgentError("Model unavailable")
            return original(model_instance, messages, **kwargs)

        events = []
        with patch.object(ToolCallingModel, "_generate", new=fail_reply):
            with self.assertRaisesRegex(AssistantAgentError, "unavailable"):
                events.extend(service.stream_response_events("Remove second"))
        self.assert_lifecycle(events, outcome="error")
        commit, = [e for e in events if e.type == "sequence_committed"]
        finish = next(e for e in events if e.type == "step_finished" and e.step_id == commit.step_id)
        self.assertEqual((finish.outcome, finish.summary), ("ok", "Removed 1 sequence elements; 4 poses in total."))
        self.assertEqual(service.active_sequence_uuid, str(commit.sequence_uuid))
        self.assertEqual(service.get_sequence(str(commit.sequence_uuid)).operation, "remove")
        self.assertEqual(service.get_sequence(str(source.uuid)), source)
        factory.assert_not_called()
        self.assertEqual(catalog.mock_calls + tags.mock_calls, [])

    def test_swap_commit_survives_failure_of_final_model_call(self):
        service, source, model, factory, catalog, tags = self.generated_swap_source()
        model.responses = [swap_call(source)]
        original = ToolCallingModel._generate

        def fail_followup(model_instance, messages, **kwargs):
            if isinstance(last_message(messages), ToolMessage) and last_message(messages).name == "swap_sequence_elements":
                raise RuntimeError("Private model failure")
            return original(model_instance, messages, **kwargs)

        events = []
        with patch.object(ToolCallingModel, "_generate", new=fail_followup):
            with self.assertRaisesRegex(RuntimeError, "Private model failure"):
                events.extend(service.stream_response_events("Swap third and fifth"))
        self.assert_lifecycle(events, outcome="error")
        committed, = [e for e in events if e.type == "sequence_committed"]
        step = next(e for e in events if e.type == "step_started" and e.code == "updating_sequence")
        finish = next(e for e in events if e.type == "step_finished" and e.step_id == step.step_id)
        self.assertEqual((finish.outcome, finish.summary), ("ok", "Sequence elements swapped."))
        self.assertEqual(committed.step_id, step.step_id)
        self.assertEqual(committed.tool_call_id, step.tool_call_id)
        self.assertEqual(service.active_sequence_uuid, str(committed.sequence_uuid))
        self.assertEqual(service.get_sequence(str(committed.sequence_uuid)).operation, "swap")
        self.assertEqual(service.get_sequence(str(source.uuid)), source)
        self.assertEqual(len(service.state["sequences"]), 2)
        self.assertNotIn("Private model failure", str(events))

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
