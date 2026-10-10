"""Exercise the shared assistant and real tool loop without external calls."""

import json
import unittest
from threading import Event
from uuid import uuid4
from unittest.mock import ANY, Mock, patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from openai import APIConnectionError
import httpx
from pydantic import Field

from langchain_app.schemas import CatalogItemSchema
from langchain_app.services.assistant_agent import AssistantAgent, AssistantAgentError
from langchain_app.services.catalog_item import CatalogItemApiError
from langchain_app.services.sequence_generator import SequenceGenerator
from langchain_app.tools import ASSISTANT_TOOLS


class ToolCallingModel(FakeMessagesListChatModel):
    received_requests: list[str] = Field(default_factory=list)
    received_messages: list[list] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.tool_names = [t.name for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received_messages.append(list(messages))
        if isinstance(messages[-1], HumanMessage):
            self.received_requests.append(messages[-1].content)
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        # Every real model response has a new identity, including repeated calls.
        for generation in result.generations:
            generation.message = generation.message.model_copy(deep=True)
            generation.message.id = str(uuid4())
            for tool in generation.message.tool_calls:
                tool["id"] = str(uuid4())
        return result


def tool_model(request, count=9):
    return ToolCallingModel(responses=[AIMessage(content="", tool_calls=[{
        "name": "generate_sequence",
        "args": {"request": request, "count": count},
        "id": "sequence-call",
        "type": "tool_call",
    }]), AIMessage(content="")])


def pose(number):
    return CatalogItemSchema(
        id=number, uuid=None, name=f"Pose {number}", bilateral=False,
        image_url=f"https://example.com/{number}.png", catalog_tag_names=[],
    )


def tool_call(tool_name, **arguments):
    return AIMessage(content="", tool_calls=[{
        "name": tool_name, "args": arguments, "id": f"{tool_name}-call", "type": "tool_call",
    }])


def assistant(factory, **kwargs):
    kwargs.setdefault("catalog_item_service", Mock())
    kwargs.setdefault("catalog_tag_service", Mock())
    return AssistantAgent(factory, **kwargs)


def selected_snapshot(model, call_index=-1):
    system_message = model.received_messages[call_index][0]
    context = system_message.content_blocks[-1]["text"]
    payload = next(line for line in context.splitlines() if line.startswith('{"selected_sequence":'))
    return json.loads(payload)["selected_sequence"]


class AssistantAgentTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_tool_streams_poses_before_generation_finishes(self):
        resume = Event()
        self.addCleanup(resume.set)
        poses = [pose(1), pose(2)]

        def generate(request, **kwargs):
            yield poses[0]
            if not resume.wait(timeout=3):
                raise RuntimeError("First pose was buffered until tool completion")
            yield poses[1]

        generator = Mock()
        generator.generate.side_effect = generate
        factory = Mock(return_value=generator)
        request = "gentle practice with 12 poses"
        model = tool_model(request, count=12)
        service = assistant(factory, model=model)
        sequence = service.generate(request, language="uk")
        factory.assert_not_called()
        self.assertEqual(next(sequence), poses[0])
        resume.set()
        self.assertEqual(list(sequence), poses[1:])
        factory.assert_called_once()
        generator.generate.assert_called_once_with(
            request, language="en", count=12, starting_catalog_item_id=None, progress_observer=ANY,
        )
        self.assertEqual(model.received_requests, [request])

    def test_text_only_model_response_does_not_bypass_tool(self):
        factory = Mock()
        service = assistant(factory, model=ToolCallingModel(
            responses=[AIMessage(content="Here is a yoga sequence")]
        ))
        with self.assertRaisesRegex(AssistantAgentError, "did not call"):
            list(service.generate("practice", language="en"))
        factory.assert_not_called()

    def test_catalog_search_resolves_first_pose_and_generation_preserves_it(self):
        requested = pose(42)
        requested.name = "Tree pose"
        requested.catalog_tag_names = ["Position/Standing"]
        random_start = pose(1)
        random_start.catalog_tag_names = ["Other/Starting", "Position/Sitting"]
        following = pose(43)
        following.catalog_tag_names = ["Position/Standing", "Other/Finishing"]
        catalog = Mock()
        catalog.search.side_effect = [[requested], [random_start, requested, following]]
        selector = Mock()
        selector.choose_next_catalog_item.side_effect = lambda candidates, **kwargs: candidates[0]
        request = "Сгенерируй две асаны, первой поставь позу дерева"
        model = ToolCallingModel(responses=[
            tool_call("search_catalog_items", name="Tree"),
            tool_call("generate_sequence", request=request, count=2, starting_catalog_item_id=42),
            AIMessage(content="Секвенция готова."),
        ])
        generator = SequenceGenerator(catalog, selector)
        service = assistant(lambda: generator, catalog_item_service=catalog, model=model)
        self.assertEqual(list(service.generate(request, language="en")), [requested, following])
        sequence = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual([element.catalog_item.id for element in sequence.elements], [42, 43])
        self.assertEqual(catalog.search.call_count, 2)
        selector.choose_next_catalog_item.assert_called_once()
        self.assertEqual(selector.choose_next_catalog_item.call_args.kwargs["selected_items"][0], requested)

    def test_invalid_starting_item_id_does_not_start_generation(self):
        for item_id in (0, -1, True, "42", 1.5):
            with self.subTest(item_id=item_id):
                factory = Mock()
                service = assistant(factory, model=ToolCallingModel(responses=[
                    tool_call("generate_sequence", request="practice", count=2, starting_catalog_item_id=item_id),
                    AIMessage(content="Invalid first pose ID."),
                ]))
                with self.assertRaises(AssistantAgentError):
                    list(service.generate("practice", language="en"))
                factory.assert_not_called()

    def test_partial_generation_failure_is_propagated(self):
        def generate(*args, **kwargs):
            yield pose(1)
            raise ValueError("generation failed")

        generator = Mock()
        generator.generate.side_effect = generate
        service = assistant(lambda: generator, model=tool_model("practice"))
        sequence = service.generate("practice", language="en")
        self.assertEqual(next(sequence), pose(1))
        with self.assertRaisesRegex(ValueError, "generation failed"):
            list(sequence)

    def test_model_connection_error_is_wrapped(self):
        factory = Mock()
        error = APIConnectionError(request=httpx.Request("POST", "https://example.com"))
        with patch(
            "langchain_app.services.assistant_agent.init_chat_model", side_effect=error
        ):
            with self.assertRaisesRegex(AssistantAgentError, "Language model request failed"):
                list(assistant(factory).generate("practice", language="en"))
        factory.assert_not_called()

    def test_invalid_tool_count_does_not_start_generation(self):
        factory = Mock()
        service = assistant(factory, model=tool_model("practice", count=0))
        with self.assertRaises(AssistantAgentError):
            list(service.generate("practice", language="en"))
        factory.assert_not_called()

    def test_all_tools_are_registered_with_hidden_application_configuration(self):
        model = ToolCallingModel(responses=[AIMessage(content="Какую практику вы хотите?")])
        factory = Mock()
        service = assistant(factory, model=model)
        self.assertEqual(list(service.stream("Помоги выбрать практику")), ["Какую практику вы хотите?"])
        self.assertEqual(set(model.tool_names), {
            "generate_sequence", "search_catalog_items", "get_catalog_tag_names", "replace_sequence_element",
            "show_catalog_items",
        })
        for registered_tool in ASSISTANT_TOOLS:
            self.assertFalse(registered_tool.return_direct)
            self.assertFalse({"language", "base_url", "catalog_folder_uuid", "runtime"}
                             & set(registered_tool.tool_call_schema.model_fields))
        self.assertEqual(list(ASSISTANT_TOOLS[2].tool_call_schema.model_fields), [])
        prompt = model.received_messages[0][0].text
        start_field = ASSISTANT_TOOLS[0].tool_call_schema.model_fields["starting_catalog_item_id"]
        self.assertFalse(start_field.is_required())
        self.assertIsNone(start_field.default)
        self.assertIn("pass starting_catalog_item_id to generate_sequence", prompt)
        self.assertIn("a mirrored (bilateral) pose takes 80 seconds", prompt)
        self.assertIn("Mandatory rule: generate at most 45 elements", prompt)
        self.assertIn("leave duration_seconds null in that case", prompt)
        self.assertIn("Always reply in the user's language", prompt)
        self.assertIn("keep item names and tag names exactly as returned by the API", prompt)
        factory.assert_not_called()

    def test_tags_then_search_then_reply_use_one_agent_and_preserve_api_names(self):
        found = pose(42)
        found.name = "Tree pose"
        found.catalog_tag_names = ["Balance", "Standing"]
        catalog = Mock()
        catalog.search.return_value = [found]
        tags = Mock()
        tags.get_names.return_value = ["Balance", "Standing"]
        reply = "В каталоге есть Tree pose с тегами Balance и Standing."
        model = ToolCallingModel(responses=[
            tool_call("get_catalog_tag_names"),
            tool_call("search_catalog_items", name="Tree", bilateral=False, catalog_tag_names=["Standing"]),
            AIMessage(content=reply),
        ])
        factory = Mock()
        service = assistant(factory, catalog_item_service=catalog, catalog_tag_service=tags, model=model)
        self.assertEqual(list(service.stream("Найди позу дерева стоя")), [reply])
        tags.get_names.assert_called_once_with()
        catalog.search.assert_called_once_with(
            name="Tree", uuid=None, bilateral=False, catalog_tag_names=["Standing"],
            with_tags=True, with_rigman=False, limit=10, rand=False,
        )
        tag_message = model.received_messages[1][-1]
        search_message = model.received_messages[2][-1]
        self.assertIsInstance(tag_message, ToolMessage)
        self.assertEqual(json.loads(tag_message.content)["tag_names"], ["Balance", "Standing"])
        result = json.loads(search_message.content)
        self.assertEqual(result["items"][0]["name"], "Tree pose")
        self.assertEqual(result["returned_count"], 1)
        self.assertNotIn("total_count", result)
        factory.assert_not_called()

    def test_generation_returns_catalog_snapshot_to_model_before_localized_reply(self):
        generator = Mock()
        generated = pose(1)
        generated.name = "Child's pose"
        generator.generate.return_value = [generated]
        reply = "Практика готова: Child's pose."
        model = ToolCallingModel(responses=[
            tool_call("generate_sequence", request="Практика из одной асаны", count=1),
            AIMessage(content=reply),
        ])
        self.assertEqual(list(assistant(lambda: generator, model=model).stream("Практика из одной асаны")),
                         [generated, reply])
        snapshot = json.loads(model.received_messages[1][-1].content)
        self.assertEqual(snapshot["elements"][0]["catalog_item"]["name"], "Child's pose")
        self.assertEqual(snapshot["language"], "en")
        generator.generate.assert_called_once_with(
            "Практика из одной асаны", language="en", count=1, starting_catalog_item_id=None, progress_observer=ANY,
        )

    def test_duration_reaches_generator_and_regeneration_keeps_mirrored_accounting(self):
        poses = [pose(n) for n in range(1, 7)]
        poses[0].catalog_tag_names = ["Other/Starting"]
        poses[0].bilateral = poses[2].bilateral = True
        poses[3].catalog_tag_names = ["Other/Finishing"]
        catalog = Mock()
        catalog.search.return_value = poses
        selector = Mock()
        selector.choose_next_catalog_item.side_effect = lambda candidates, **kwargs: candidates[0]
        generator = SequenceGenerator(catalog, selector)
        request = "Секвенция на четыре минуты"
        model = ToolCallingModel(responses=[
            tool_call("generate_sequence", request=request, count=6, duration_seconds=240),
            AIMessage(content="Готово."),
        ])
        service = assistant(lambda: generator, model=model)
        with patch("langchain_app.services.sequence_generator.sample", side_effect=lambda population, k: population[:k]):
            self.assertEqual(list(service.generate(request)), poses[:4])
            original = service.get_sequence(service.active_sequence_uuid)
            self.assertEqual(list(service.generate("", sequence_uuid=str(original.uuid))), poses[:4])
        sequence = service.get_sequence(service.active_sequence_uuid)
        self.assertEqual(sequence.elements_count, 4)
        self.assertEqual(sequence.parent_uuid, original.uuid)
        self.assertEqual(sequence.generation_request, request)
        self.assertEqual([element.start_mirror for element in sequence.elements], [True, False, True, False])

    def test_generation_count_above_limit_is_rejected_before_factory_and_can_be_corrected(self):
        generator = Mock()
        generator.generate.return_value = [pose(n) for n in range(1, 46)]
        factory = Mock(return_value=generator)
        model = ToolCallingModel(responses=[
            tool_call("generate_sequence", request="46 poses", count=46),
            tool_call("generate_sequence", request="46 poses", count=45),
            AIMessage(content="Максимум — 45 асан."),
        ])
        service = assistant(factory, model=model)
        self.assertEqual(len(list(service.generate("46 poses"))), 45)
        factory.assert_called_once()
        generator.generate.assert_called_once_with(
            "46 poses", language="en", count=45, starting_catalog_item_id=None, progress_observer=ANY,
        )
        self.assertEqual(model.received_messages[1][-1].status, "error")
        self.assertIn("less than or equal to 45", model.received_messages[1][-1].content)

    def test_invalid_duration_does_not_start_generation(self):
        for duration in (0, -1, float("inf"), float("nan")):
            with self.subTest(duration=duration):
                factory = Mock()
                service = assistant(factory, model=ToolCallingModel(responses=[
                    tool_call("generate_sequence", request="practice", count=1, duration_seconds=duration),
                    AIMessage(content="Укажите положительную длительность."),
                ]))
                list(service.stream("practice"))
                factory.assert_not_called()
                self.assertEqual(service._model.received_messages[1][-1].status, "error")

    def test_generator_cannot_commit_or_stream_more_than_45_elements(self):
        generator = Mock()
        generator.generate.return_value = [pose(n) for n in range(1, 47)]
        model = ToolCallingModel(responses=[
            tool_call("generate_sequence", request="45 poses", count=45),
            AIMessage(content="Ошибка генератора."),
        ])
        service = assistant(lambda: generator, model=model)
        output = list(service.stream("45 poses"))
        self.assertEqual(len([value for value in output if isinstance(value, CatalogItemSchema)]), 45)
        self.assertIsNone(service.active_sequence_uuid)
        message = model.received_messages[1][-1]
        self.assertEqual(message.status, "error")
        self.assertEqual(json.loads(message.content)["code"], "sequence_limit_exceeded")

    def test_catalog_api_error_is_returned_to_model_as_error_not_empty_success(self):
        catalog = Mock()
        catalog.search.side_effect = CatalogItemApiError("Invalid response", code="invalid_api_response")
        reply = "Не удалось получить каталог."
        model = ToolCallingModel(responses=[tool_call("search_catalog_items", name="Tree"), AIMessage(content=reply)])
        self.assertEqual(list(assistant(Mock(), model=model, catalog_item_service=catalog).stream("Найди дерево")),
                         [reply])
        message = model.received_messages[1][-1]
        self.assertEqual(message.status, "error")
        self.assertEqual(json.loads(message.content)["code"], "invalid_api_response")

    def test_search_tool_does_not_allow_unlimited_results(self):
        catalog = Mock()
        model = ToolCallingModel(responses=[tool_call("search_catalog_items", limit=0), AIMessage(content="Уточните поиск")])
        self.assertEqual(list(assistant(Mock(), model=model, catalog_item_service=catalog).stream("Покажи все")),
                         ["Уточните поиск"])
        catalog.search.assert_not_called()
        self.assertEqual(model.received_messages[1][-1].status, "error")

    def test_replacement_without_selected_context_returns_explicit_error(self):
        model = ToolCallingModel(responses=[
            tool_call("replace_sequence_element",
                      sequence_uuid="00000000-0000-0000-0000-000000000001",
                      element_id="00000000-0000-0000-0000-000000000002",
                      replacement_catalog_item_id=42),
            AIMessage(content="Сначала нужен выбранный вариант секвенции."),
        ])
        factory = Mock()
        self.assertEqual(list(assistant(factory, model=model).stream("Замени третью асану")),
                         ["Сначала нужен выбранный вариант секвенции."])
        message = model.received_messages[1][-1]
        self.assertEqual(message.status, "error")
        self.assertEqual(json.loads(message.content)["code"], "sequence_not_found")
        factory.assert_not_called()

    def test_generation_is_not_repeated_within_a_request(self):
        generator = Mock()
        generator.generate.return_value = [pose(1)]
        model = ToolCallingModel(responses=[
            tool_call("generate_sequence", request="practice", count=1),
            tool_call("generate_sequence", request="practice", count=1),
            AIMessage(content="Done"),
        ])
        list(assistant(lambda: generator, model=model).generate("practice"))
        generator.generate.assert_called_once()
        message = model.received_messages[2][-1]
        self.assertEqual(json.loads(message.content)["code"], "operation_already_started")

    def test_complete_serializable_snapshot_keeps_occurrences_and_items_without_images(self):
        first = pose(1)
        first.uuid = str(uuid4())
        first.bilateral = True
        first.catalog_tag_names = ["Standing", "Balance"]
        second = pose(2)
        second.image_url = None
        generator = Mock()
        generator.generate.return_value = iter([first, second, first])
        service = assistant(lambda: generator, model=tool_model("Three poses", count=3))
        list(service.generate("Three poses"))
        snapshot = service.state["sequences"][service.active_sequence_uuid]
        self.assertEqual(snapshot["generation_request"], "Three poses")
        self.assertEqual(snapshot["elements_count"], 3)
        self.assertEqual([item["order"] for item in snapshot["elements"]], [1, 2, 3])
        self.assertEqual(len({item["element_id"] for item in snapshot["elements"]}), 3)
        self.assertEqual(snapshot["elements"][0]["catalog_item"], first.model_dump(mode="json"))
        self.assertTrue(snapshot["elements"][0]["start_mirror"])
        self.assertTrue(snapshot["elements"][0]["end_mirror"])
        self.assertIsNone(snapshot["elements"][1]["catalog_item"]["image_url"])
        self.assertEqual(json.loads(json.dumps(snapshot)), snapshot)
        # Neither caller-owned objects nor returned state can mutate stored history.
        first.name = "Changed outside the agent"
        snapshot["elements"][0]["catalog_item"]["name"] = "Changed in the copy"
        self.assertEqual(service.get_sequence(service.active_sequence_uuid).elements[0].catalog_item.name, "Pose 1")

    def test_selected_context_refreshes_after_tools_and_on_the_next_turn(self):
        generated = pose(3)
        generated.uuid = str(uuid4())
        generated.catalog_tag_names = ["Standing", "Balance"]
        generator = Mock()
        generator.generate.return_value = [generated]
        catalog = Mock()
        catalog.search.return_value = [pose(4)]
        model = ToolCallingModel(responses=[
            tool_call("generate_sequence", request="Practice", count=1),
            AIMessage(content="Ready"),
            tool_call("search_catalog_items", name="Tree"),
            AIMessage(content="Alternative found"),
            AIMessage(content="Your first pose is Pose 3"),
        ])
        service = assistant(lambda: generator, model=model, catalog_item_service=catalog)
        list(service.stream("Practice"))
        snapshot = service.state["sequences"][service.active_sequence_uuid]
        self.assertIsNone(selected_snapshot(model, 0))
        list(service.stream("Find an alternative to the first pose"))
        list(service.stream("What is my first pose?"))
        for call_index in range(1, len(model.received_messages)):
            context = selected_snapshot(model, call_index)
            self.assertEqual(context["uuid"], snapshot["uuid"])
            self.assertEqual(context["elements"][0]["catalog_item"]["id"], 3)
            self.assertEqual(context["elements"][0]["catalog_item"]["uuid"], generated.uuid)
            self.assertEqual(context["elements"][0]["catalog_item"]["catalog_tag_names"], generated.catalog_tag_names)
            self.assertNotIn("image_url", context["elements"][0]["catalog_item"])
            self.assertNotIn("rigman", json.dumps(context))
        self.assertEqual(service.state["sequences"], {snapshot["uuid"]: snapshot})
        self.assertFalse(any(isinstance(message, SystemMessage) for message in service.state["messages"]))
        generator.generate.assert_called_once()
        self.assertEqual(len(model.received_messages[-1][0].content_blocks), 2)

    def test_selecting_an_older_variant_overrides_newer_tool_history_without_model_calls(self):
        generator = Mock()
        generator.generate.side_effect = [[pose(1)], [pose(2)]]
        model = ToolCallingModel(responses=[
            tool_call("generate_sequence", request="First practice", count=1), AIMessage(content="First ready"),
            tool_call("generate_sequence", request="Second practice", count=1), AIMessage(content="Second ready"),
            AIMessage(content="Pose 1"),
        ])
        service = assistant(lambda: generator, model=model)
        list(service.generate("First practice"))
        first_uuid = service.active_sequence_uuid
        list(service.generate("Second practice"))
        second_uuid = service.active_sequence_uuid
        calls_before = len(model.received_messages)
        service.select_sequence(first_uuid)
        self.assertEqual(len(model.received_messages), calls_before)
        with self.assertRaises(ValueError):
            service.select_sequence(str(uuid4()))
        self.assertEqual(service.active_sequence_uuid, first_uuid)
        self.assertEqual(list(service.stream("What is selected?")), ["Pose 1"])
        self.assertEqual(selected_snapshot(model)["uuid"], first_uuid)
        self.assertIn(second_uuid, [
            json.loads(message.content).get("uuid")
            for message in model.received_messages[-1] if isinstance(message, ToolMessage)
        ])
        self.assertEqual(len(service.state["sequences"]), 2)
        self.assertEqual(generator.generate.call_count, 2)
        self.assertNotIn("requires a new generated sequence", model.received_messages[-1][0].text)

    def test_regenerate_uses_source_request_and_keeps_family_and_original_snapshot(self):
        generator = Mock()
        generator.generate.side_effect = [[pose(1)], [pose(2)]]
        model = tool_model("Original full request", count=1)
        service = assistant(lambda: generator, model=model)
        list(service.generate("Original full request"))
        original = service.get_sequence(service.active_sequence_uuid)
        list(service.generate("This argument must not replace the source request", sequence_uuid=str(original.uuid)))
        regenerated = service.get_sequence(service.active_sequence_uuid)
        self.assertNotEqual(regenerated.uuid, original.uuid)
        self.assertEqual(regenerated.parent_uuid, original.uuid)
        self.assertEqual(regenerated.family_uuid, original.family_uuid)
        self.assertEqual(regenerated.operation, "regenerate")
        self.assertEqual(regenerated.generation_request, original.generation_request)
        self.assertNotEqual(regenerated.elements[0].element_id, original.elements[0].element_id)
        self.assertEqual(service.get_sequence(str(original.uuid)), original)
        self.assertEqual(model.received_requests, [original.generation_request] * 2)
        self.assertEqual(generator.generate.call_count, 2)

    def test_partial_failure_keeps_previous_snapshot_and_valid_history_for_followup(self):
        def fail(*args, **kwargs):
            yield pose(2)
            raise ValueError("Failed after a streamed pose")

        generator = Mock()
        generator.generate.side_effect = [[pose(1)], fail()]
        model = ToolCallingModel(responses=[
            tool_call("generate_sequence", request="Original", count=1), AIMessage(content="Ready"),
            tool_call("generate_sequence", request="New practice", count=2),
            AIMessage(content="Original is still selected"),
        ])
        service = assistant(lambda: generator, model=model)
        list(service.generate("Original"))
        snapshots_before = service.state["sequences"]
        selected_before = service.active_sequence_uuid
        stream = service.generate("New practice")
        self.assertEqual(next(stream), pose(2))
        self.assertEqual(service.active_sequence_uuid, selected_before)
        with self.assertRaises(ValueError):
            list(stream)
        self.assertEqual(service.state["sequences"], snapshots_before)
        self.assertEqual(service.active_sequence_uuid, selected_before)
        self.assertEqual(service.state["messages"][-1].status, "error")
        list(service.stream("What is selected after the error?"))
        self.assertEqual(selected_snapshot(model)["uuid"], selected_before)

    def test_agent_instances_do_not_share_conversation_or_sequence_state(self):
        generator = Mock()
        generator.generate.return_value = [pose(1)]
        first = assistant(lambda: generator, model=tool_model("Practice", count=1))
        second_model = ToolCallingModel(responses=[AIMessage(content="No selected sequence")])
        second = assistant(Mock(), model=second_model)
        list(first.generate("Practice"))
        list(second.stream("What is selected?"))
        self.assertIsNone(selected_snapshot(second_model))
        self.assertEqual(second.state["sequences"], {})
        self.assertEqual(second_model.received_requests, ["What is selected?"])
        self.assertEqual(len(first.state["sequences"]), 1)


if __name__ == "__main__":
    unittest.main()
