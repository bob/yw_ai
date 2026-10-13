"""Collect one public event stream into session history without UI dependencies."""

from uuid import UUID

from langchain_app.schemas.response_progress import (
    AssistantMessageEvent,
    ExecutionMessageSnapshot,
    ExecutionSnapshot,
    ExecutionStepProgress,
    ExecutionStepSnapshot,
    RequestFinishedEvent,
    RequestStartedEvent,
    ResponseEvent,
    SequenceCommittedEvent,
    StepFinishedEvent,
    StepProgressEvent,
    StepStartedEvent,
)
from langchain_app.services.request_errors import PUBLIC_REQUEST_ERROR


class ResponseExecution:
    """Accept only this attempt's ordered events; retain public data only."""

    def __init__(self, request_id: UUID) -> None:
        self.request_id = request_id
        self.starts: dict[UUID, StepStartedEvent] = {}
        self.progress: dict[UUID, list[ExecutionStepProgress]] = {}
        self.finishes: dict[UUID, StepFinishedEvent] = {}
        self.messages: list[ExecutionMessageSnapshot] = []
        self.final_texts: list[str] = []
        self.source_sequence_uuid: UUID | None = None
        self.sequence_uuid: UUID | None = None
        self.result: RequestFinishedEvent | None = None
        self._last_index = -1
        self._message_ids: set[tuple[str, str]] = set()

    def accept(self, event: ResponseEvent) -> bool:
        if event.request_id != self.request_id or event.event_index <= self._last_index or self.result:
            return False
        self._last_index = event.event_index
        if isinstance(event, RequestStartedEvent):
            self.source_sequence_uuid = event.source_sequence_uuid
        elif isinstance(event, StepStartedEvent):
            self.starts[event.step_id] = event
            self.progress[event.step_id] = []
        elif isinstance(event, StepProgressEvent):
            self.progress[event.step_id].append(ExecutionStepProgress(
                event_index=event.event_index, elapsed_seconds=event.elapsed_seconds,
                summary=event.summary, completed=event.completed, total=event.total,
            ))
        elif isinstance(event, StepFinishedEvent):
            self.finishes[event.step_id] = event
        elif isinstance(event, AssistantMessageEvent):
            if event.message_id is not None:
                identity = (event.purpose, event.message_id)
                if identity in self._message_ids:
                    return False
                self._message_ids.add(identity)
            if event.purpose == "final":
                self.final_texts.append(event.text)
            else:
                self.messages.append(ExecutionMessageSnapshot(
                    event_index=event.event_index, elapsed_seconds=event.elapsed_seconds,
                    text=event.text, message_id=event.message_id, step_id=event.step_id,
                ))
        elif isinstance(event, SequenceCommittedEvent):
            self.sequence_uuid = event.sequence_uuid
        elif isinstance(event, RequestFinishedEvent):
            self.result = event
        return True

    def fail(self, elapsed_seconds: float, *, error_message: str = PUBLIC_REQUEST_ERROR) -> None:
        """Initialization/transport failures also get exactly one terminal result."""
        if self.result is not None:
            return
        for step_id, step in reversed(self.starts.items()):
            if step_id not in self.finishes:
                self.accept(StepFinishedEvent(
                    request_id=self.request_id, event_index=self._last_index + 1,
                    elapsed_seconds=elapsed_seconds, step_id=step_id,
                    tool_call_id=step.tool_call_id, outcome="error",
                    duration_seconds=max(0.0, elapsed_seconds - step.elapsed_seconds),
                    summary="Could not complete this step.",
                ))
        self.accept(RequestFinishedEvent(
            request_id=self.request_id, event_index=self._last_index + 1,
            elapsed_seconds=elapsed_seconds, duration_seconds=elapsed_seconds,
            outcome="error", error_message=error_message,
        ))

    def snapshot(self) -> ExecutionSnapshot:
        if self.result is None:
            raise ValueError("Cannot store an unfinished request")
        return ExecutionSnapshot(
            request_id=self.request_id, outcome=self.result.outcome,
            duration_seconds=self.result.duration_seconds,
            error_message=self.result.error_message,
            source_sequence_uuid=self.source_sequence_uuid, sequence_uuid=self.sequence_uuid,
            intermediate_messages=self.messages,
            steps=[ExecutionStepSnapshot(
                **start.model_dump(exclude={"type", "request_id"}),
                outcome=self.finishes[step_id].outcome,
                duration_seconds=self.finishes[step_id].duration_seconds,
                summary=self.finishes[step_id].summary, progress=self.progress[step_id],
            ) for step_id, start in self.starts.items()],
        )
