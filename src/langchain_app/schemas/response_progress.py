"""Public response events and execution history, independent of agent and UI APIs."""

from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, StringConstraints, model_validator

from langchain_app.schemas.base import ApiSchema
from langchain_app.schemas.catalog import CatalogItemSchema

type StepCode = Literal[
    "thinking", "loading_catalog_tags", "searching_catalog_items",
    "generating_sequence", "updating_sequence", "preparing_response",
]
type StepOutcome = Literal["ok", "no_change", "error"]
type RequestOutcome = Literal["complete", "error"]

type _Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
type _Counter = Annotated[int, Field(ge=0, strict=True)]
type _PublicText = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class _EventPosition(ApiSchema):
    event_index: _Counter
    elapsed_seconds: _Seconds


class _ResponseEvent(_EventPosition):
    request_id: UUID


class _StepIdentity(ApiSchema):
    step_id: UUID
    tool_call_id: _PublicText | None = None


class _StepEvent(_ResponseEvent, _StepIdentity):
    pass


class _StepDescription(ApiSchema):
    code: StepCode
    title: _PublicText
    parent_step_id: UUID | None = None


class _ProgressPayload(ApiSchema):
    summary: _PublicText | None = None
    completed: _Counter | None = None
    total: _Counter | None = None

    @model_validator(mode="after")
    def validate_progress(self) -> Self:
        if self.summary is None and self.completed is None and self.total is None:
            raise ValueError("progress must include a summary or a counter")
        if self.completed is not None and self.total is not None and self.completed > self.total:
            raise ValueError("completed must not exceed total")
        return self


class _RequestResult(ApiSchema):
    outcome: RequestOutcome
    duration_seconds: _Seconds
    error_message: _PublicText | None = None

    @model_validator(mode="after")
    def validate_error(self) -> Self:
        if self.outcome == "error" and self.error_message is None:
            raise ValueError("failed requests must include a public error message")
        if self.outcome == "complete" and self.error_message is not None:
            raise ValueError("completed requests must not include a fatal error message")
        return self


class RequestStartedEvent(_ResponseEvent):
    type: Literal["request_started"] = "request_started"
    source_sequence_uuid: UUID | None = None


class StepStartedEvent(_StepEvent, _StepDescription):
    type: Literal["step_started"] = "step_started"


class StepProgressEvent(_StepEvent, _ProgressPayload):
    type: Literal["step_progress"] = "step_progress"


class StepFinishedEvent(_StepEvent):
    type: Literal["step_finished"] = "step_finished"
    outcome: StepOutcome
    duration_seconds: _Seconds
    summary: _PublicText


class AssistantMessageEvent(_ResponseEvent):
    """Whole public text; extracting text blocks belongs to the event producer."""

    type: Literal["assistant_message"] = "assistant_message"
    purpose: Literal["intermediate", "final"]
    text: _PublicText
    message_id: _PublicText | None = None
    step_id: UUID | None = None


class SequenceElementEvent(_StepEvent):
    """A provisional item; this event does not confirm a committed variant."""

    type: Literal["sequence_element"] = "sequence_element"
    order: Annotated[int, Field(ge=1, strict=True)]
    catalog_item: CatalogItemSchema


class SequenceCommittedEvent(_StepEvent):
    """Reference to the full variant already saved in the agent state."""

    type: Literal["sequence_committed"] = "sequence_committed"
    sequence_uuid: UUID


class RequestFinishedEvent(_ResponseEvent, _RequestResult):
    type: Literal["request_finished"] = "request_finished"


type ResponseEvent = Annotated[
    RequestStartedEvent | StepStartedEvent | StepProgressEvent | StepFinishedEvent
    | AssistantMessageEvent | SequenceElementEvent | SequenceCommittedEvent
    | RequestFinishedEvent,
    Field(discriminator="type"),
]


class ExecutionStepProgress(_EventPosition, _ProgressPayload):
    """One public update retained within its execution step."""


class ExecutionStepSnapshot(_EventPosition, _StepIdentity, _StepDescription):
    """A finished step; event position denotes the start of the step."""

    outcome: StepOutcome
    duration_seconds: _Seconds
    summary: _PublicText
    progress: list[ExecutionStepProgress] = Field(default_factory=list)


class ExecutionMessageSnapshot(_EventPosition):
    """Intermediate public text only; final text remains in ChatMessage.content."""

    text: _PublicText
    message_id: _PublicText | None = None
    step_id: UUID | None = None


class ExecutionSnapshot(_RequestResult):
    """Terminal history of one attempt, including attempts without a variant."""

    request_id: UUID
    steps: list[ExecutionStepSnapshot] = Field(default_factory=list)
    intermediate_messages: list[ExecutionMessageSnapshot] = Field(default_factory=list)
    source_sequence_uuid: UUID | None = None
    sequence_uuid: UUID | None = None

    @model_validator(mode="after")
    def validate_step_references(self) -> Self:
        step_ids = {step.step_id for step in self.steps}
        if len(step_ids) != len(self.steps):
            raise ValueError("each execution step must have a unique step_id")
        for step in self.steps:
            if step.parent_step_id is not None and (
                step.parent_step_id not in step_ids or step.parent_step_id == step.step_id
            ):
                raise ValueError("parent_step_id must reference another step in this request")
        for message in self.intermediate_messages:
            if message.step_id is not None and message.step_id not in step_ids:
                raise ValueError("intermediate messages must reference a step in this request")
        return self
