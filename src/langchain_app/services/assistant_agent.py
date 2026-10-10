"""Route requests through one assistant with sequence and catalog tools."""

from collections.abc import Callable, Iterator
from copy import deepcopy
from typing import cast
from uuid import UUID, uuid4

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import InputAgentState, OutputAgentState
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.stream import CustomTransformer
from openai import OpenAIError

from langchain_app.schemas import CatalogItemSchema, GeneratedSequence, Language
from langchain_app.schemas.assistant import AssistantInputState, AssistantState
from langchain_app.schemas.response_progress import (
    AssistantMessageEvent,
    RequestStartedEvent,
    ResponseEvent,
    SequenceCommittedEvent,
    SequenceElementEvent,
)
from langchain_app.services.catalog_item import CatalogItemService
from langchain_app.services.catalog_tag import CatalogTagService
from langchain_app.services.response_progress import ResponseProgress
from langchain_app.services.response_progress_middleware import (
    model_response_progress,
    tool_response_progress,
)
from langchain_app.services.sequence_context import (
    invalidate_catalog_candidates,
    selected_sequence_context,
    sequential_tool_calls,
)
from langchain_app.services.sequence_generator import MAX_SEQUENCE_ELEMENTS, SequenceGenerator
from langchain_app.tools.context import AssistantToolContext

type _AssistantGraph = CompiledStateGraph[
    AgentState, AssistantToolContext, InputAgentState, OutputAgentState,
]

SYSTEM_PROMPT = (
    "You are a yoga assistant with sequence generation, catalog search and display, catalog tags, "
    "and sequence replacement tools. Choose tools according to the user's intent. "
    "For a new sequence, call generate_sequence once with the user's full request, "
    "preserving all preferences and constraints. Infer the pose count: an explicit "
    "count takes precedence over duration; leave duration_seconds null in that case. "
    "For duration alone, pass minutes * 60 as duration_seconds and use minutes * 60 / 40 "
    "as the upper count estimate, round halves up, with a minimum of 1 pose. "
    "An ordinary pose takes 40 seconds; a mirrored (bilateral) pose takes 80 seconds, "
    "the time of two ordinary poses, while remaining one sequence element. "
    "The generator adjusts the actual element count for the selected mirrored poses; "
    "10 minutes allows up to 15 ordinary poses, fewer when poses are mirrored. "
    f"Mandatory rule: generate at most {MAX_SEQUENCE_ELEMENTS} elements in any sequence. "
    f"Always cap count at {MAX_SEQUENCE_ELEMENTS}, even for an explicit higher count "
    "or a longer duration, and explain this limit when it constrains the request. "
    "If neither count nor duration is specified, use 9 poses. "
    "When generating a sequence with a user-specified first pose, resolve the pose "
    "with search_catalog_items or use its verified ID from the selected sequence, "
    "then pass starting_catalog_item_id to generate_sequence. This first pose takes "
    "priority over the random start and need not have the Other/Starting tag. "
    "If the requested pose is ambiguous, ask for clarification before generating. "
    "If it is unavailable or has no thumbnail, explain the problem instead of "
    "substituting a random start. Leave starting_catalog_item_id null when the user "
    "does not request a particular first pose. Random starts use lying 10%, sitting "
    "60%, standing 30%. After a lying start the second pose must be lying on the "
    "back; after a standing start prefer standing and never choose a lying second pose. "
    "Use search_catalog_items for real catalog items; never invent catalog data. "
    "Search results alone are not displayed to the user. For requests to browse or "
    "choose available catalog poses, search as needed, then call show_catalog_items "
    "once with the complete selection of IDs from the latest verified search, in "
    "display order. Browsing includes 'Show seated forward bends', 'What lying "
    "poses are available?', 'Show more options' after a catalog selection, and "
    "'Find replacement candidates, do not change anything yet'. The word 'show' "
    "is not required. A request for candidates without changes authorizes their "
    "display without confirmation, but not generation or replacement. "
    "Do not call show_catalog_items for internal searches during generation, "
    "replacement, or answering a technique question such as 'How do I perform "
    "a seated forward bend?'. Tag requests alone do not authorize display. "
    "Generation or replacement alone keeps its existing sequence output without "
    "a separate catalog gallery. If the user separately asks for catalog display "
    "and a sequence operation, fulfill both separately; display is not a sequence "
    "change. Mandatory Generate and Regenerate commands only generate sequences "
    "and never call show_catalog_items. Do not invent a selection when search "
    "fails or has no results; explain the outcome instead. "
    "Get real tag names with get_catalog_tag_names before using unknown tag filters. "
    "Catalog requests always use language=en. Translate non-English search terms "
    "into English, but keep item names and tag names exactly as returned by the API. "
    "Always reply in the user's language, including after a tool call. "
    "Use replace_sequence_element only for an explicitly requested replacement, "
    "with IDs from the selected sequence and real search results. Before replacing, "
    "check the target element's position in the selected sequence and the candidate's "
    "catalog tags. A replacement with the Other/Starting tag is allowed only at "
    "position 1 (the first element). Never use it at position 2 or later, including "
    "the middle or end of the sequence. If the requested replacement violates "
    "this rule, explain the restriction and offer eligible alternatives instead "
    "of applying the replacement. Ask for clarification "
    "if the target or replacement is ambiguous. A search for alternatives does not "
    "authorize replacement. Use one tool at a time and apply at most one sequence "
    "change per request. In this first release, editing supports replacing one "
    "element per request. If the user asks for multiple replacements, ask which "
    "single element to replace before changing the sequence. Adding, deleting, "
    "moving elements, and changing mirroring are unavailable in this release. "
    "Explain this limitation when asked for these operations; do not simulate "
    "them with generation or replacement tools, or claim they were applied. "
    "A replacement creates a new variant in the same family "
    "and keeps the original in history. Report the old and new item after success. "
    "For no_change, explain that the selected item is already present. "
    "If no editable sequence context is available, explain "
    "that replacement is unavailable; never claim that a sequence has been changed. "
    "Text replies and clarifying questions do not require tool calls. "
    "Do not invent poses or answer a generation request with a sequence in text."
)


class AssistantAgentError(Exception):
    """The assistant could not complete a language-model request."""


class AssistantAgent:
    """Expose generation and catalog tools through a single LangChain agent."""

    def __init__(
        self,
        generator_factory: Callable[[], SequenceGenerator],
        *,
        catalog_item_service: CatalogItemService,
        catalog_tag_service: CatalogTagService,
        model: str | BaseChatModel = "openai:gpt-5.5",
    ) -> None:
        self._generator_factory = generator_factory
        self._catalog_item_service = catalog_item_service
        self._catalog_tag_service = catalog_tag_service
        self._model = model
        self._agent: _AssistantGraph | None = None
        self._checkpointer = InMemorySaver()
        self._config: RunnableConfig = {
            "configurable": {"thread_id": str(uuid4())}, "recursion_limit": 20,
        }

    @property
    def state(self) -> AssistantState:
        """Return a detached view of this session's graph state."""
        values = self._agent.get_state(self._config).values if self._agent else {}
        return cast(AssistantState, deepcopy(values))

    @property
    def active_sequence_uuid(self) -> str | None:
        return self.state.get("active_sequence_uuid")

    def get_sequence(self, sequence_uuid: str) -> GeneratedSequence | None:
        snapshot = self.state.get("sequences", {}).get(sequence_uuid)
        return GeneratedSequence.from_snapshot(snapshot) if snapshot else None

    def get_sequence_variants(self, sequence_uuid: str) -> list[GeneratedSequence]:
        """Return this variant's family in creation order for UI history."""
        snapshots = self.state.get("sequences", {})
        source = snapshots.get(sequence_uuid)
        if source is None:
            return []
        return [
            GeneratedSequence.from_snapshot(snapshot)
            for snapshot in snapshots.values() if snapshot["family_uuid"] == source["family_uuid"]
        ]

    def select_sequence(self, sequence_uuid: str) -> None:
        """Select a stored variant without running the model or any tools."""
        if self.get_sequence(sequence_uuid) is None:
            raise ValueError("The sequence variant is not stored in this assistant session")
        self._get_agent().update_state(self._config, {"active_sequence_uuid": sequence_uuid})

    def _get_agent(self) -> _AssistantGraph:
        if self._agent is None:
            from langchain_app.tools import ASSISTANT_TOOLS

            model = (
                init_chat_model(
                    self._model, timeout=300,
                    model_kwargs={"parallel_tool_calls": False},
                ) if isinstance(self._model, str) else self._model
            )
            self._agent = create_agent(
                model=model, tools=list(ASSISTANT_TOOLS), system_prompt=SYSTEM_PROMPT,
                context_schema=AssistantToolContext, state_schema=AssistantState,
                middleware=[
                    model_response_progress, selected_sequence_context, invalidate_catalog_candidates,
                    tool_response_progress, sequential_tool_calls,
                ],
                checkpointer=self._checkpointer,
            )
        return self._agent

    def _complete_failed_tool_calls(self) -> None:
        """Keep history valid when a tool raises after streaming partial progress."""
        if self._agent is None:
            return
        messages = self.state.get("messages", [])
        answered = set()
        missing = []
        for message in reversed(messages):
            if isinstance(message, ToolMessage):
                answered.add(message.tool_call_id)
            elif isinstance(message, AIMessage):
                missing = [
                    ToolMessage(
                        content='{"status": "error", "code": "tool_execution_failed"}',
                        tool_call_id=call["id"], status="error",
                    )
                    for call in message.tool_calls if call["id"] not in answered
                ]
                break
        if missing:
            self._agent.update_state(self._config, {"messages": missing}, as_node="tools")

    def stream_response_events(
        self, request: str, *, require_generation: bool = False,
        regenerate_sequence_uuid: str | None = None,
        request_id: UUID | None = None,
    ) -> Iterator[ResponseEvent]:
        """Yield public progress, provisional elements and confirmed state commits.

        A fatal failure emits request_finished(error) before propagating the
        exception. Legacy adapters therefore retain their exception behavior.
        """
        progress = ResponseProgress(request_id)
        source_uuid = regenerate_sequence_uuid or self.active_sequence_uuid
        try:
            source_id = UUID(source_uuid) if source_uuid else None
        except ValueError:
            # Invalid Regenerate input still gets a complete request lifecycle.
            source_id = None
        yield progress.event(
            RequestStartedEvent, source_sequence_uuid=source_id,
        )
        parent = None
        active_before = self.active_sequence_uuid
        context = None
        try:
            if regenerate_sequence_uuid is not None:
                parent = self.get_sequence(regenerate_sequence_uuid)
                if parent is None:
                    raise ValueError("The Regenerate source is not stored in this assistant session")
                request = parent.generation_request
            context = AssistantToolContext(
                generator_factory=self._generator_factory,
                catalog_item_service=self._catalog_item_service,
                catalog_tag_service=self._catalog_tag_service,
                user_request=request, require_generation=require_generation,
                regeneration_parent=parent,
            )
            agent = self._get_agent()
            inputs: AssistantInputState = {
                "messages": [{"role": "user", "content": request}],
                "turn_operation_committed": False,
            }
            if parent is not None:
                inputs["active_sequence_uuid"] = str(parent.uuid)
            stream = agent.stream_events(
                inputs,
                context=context, config=self._config,
                # values must be readable through get_state before confirming commit.
                version="v3", transformers=[CustomTransformer], durability="sync",
            )
            for event_type, data in stream.interleave("custom", "values"):
                events = progress.accept(data) if event_type == "custom" else progress.confirm_commits(data)
                for event in events:
                    if isinstance(event, SequenceCommittedEvent):
                        context.generation_completed = progress.generation_completed
                    yield event
            if require_generation and not context.generation_completed:
                raise AssistantAgentError("The language model did not call the generation tool successfully")
            yield from progress.finish()
        except Exception as exc:
            # Preserve a committed variant even if preparing the final reply failed.
            for event in progress.confirm_commits(self.state):
                if isinstance(event, SequenceCommittedEvent) and context is not None:
                    context.generation_completed = progress.generation_completed
                yield event
            self._complete_failed_tool_calls()
            if parent is not None and context is not None and not context.generation_completed and self._agent is not None:
                self._agent.update_state(self._config, {"active_sequence_uuid": active_before})
            yield from progress.finish("Could not complete request. Please try again.")
            if isinstance(exc, (OpenAIError, GraphRecursionError)):
                raise AssistantAgentError(f"Language model request failed: {exc}") from exc
            raise

    def stream(
        self, request: str, *, require_generation: bool = False,
        regenerate_sequence_uuid: str | None = None,
    ) -> Iterator[CatalogItemSchema | str]:
        """Compatibility adapter: provisional items and final public text only."""
        for event in self.stream_response_events(
            request, require_generation=require_generation,
            regenerate_sequence_uuid=regenerate_sequence_uuid,
        ):
            if isinstance(event, SequenceElementEvent):
                yield event.catalog_item
            elif isinstance(event, AssistantMessageEvent) and event.purpose == "final":
                yield event.text

    def generate(
        self, request: str, *, language: Language = "en", sequence_uuid: str | None = None,
    ) -> Iterator[CatalogItemSchema]:
        """Retain the generation interface for Generate/Regenerate controls.

        The legacy language argument does not change the English catalog data.
        """
        for chunk in self.stream(
            request, require_generation=True, regenerate_sequence_uuid=sequence_uuid,
        ):
            if isinstance(chunk, CatalogItemSchema):
                yield chunk
