"""Normalize request-local progress signals into the public event contract."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, cast
from uuid import UUID, uuid4

from pydantic import BaseModel

from langchain_app.schemas.response_progress import (
    RequestFinishedEvent,
    ResponseEvent,
    SequenceCommittedEvent,
    StepFinishedEvent,
    StepStartedEvent,
)


@dataclass(frozen=True)
class ProgressSignal:
    """Unindexed signal emitted where work actually starts or finishes."""

    event_class: type[BaseModel]
    payload: dict[str, Any]
    emitted_at: float = field(default_factory=monotonic)


@dataclass(frozen=True)
class PendingSequenceCommit:
    """A returned Command is only a candidate until its state is applied."""

    sequence_uuid: UUID
    finished: ProgressSignal


class ResponseProgress:
    """One consumer owns event ordering, active steps and commit confirmation."""

    def __init__(self, request_id: UUID | None = None) -> None:
        self.request_id = request_id or uuid4()
        self.started_at = monotonic()
        self._event_index = 0
        self._elapsed = 0.0
        self._active: dict[UUID, StepStartedEvent] = {}
        self._pending: dict[UUID, PendingSequenceCommit] = {}
        self._committed: set[UUID] = set()
        self.generation_completed = False

    def event[EventT: BaseModel](
        self, event_class: type[EventT], *, emitted_at: float | None = None, **payload: Any,
    ) -> EventT:
        timestamp = monotonic() if emitted_at is None else emitted_at
        self._elapsed = max(self._elapsed, timestamp - self.started_at)
        event = event_class(
            request_id=self.request_id, event_index=self._event_index,
            elapsed_seconds=self._elapsed, **payload,
        )
        self._event_index += 1
        if isinstance(event, StepStartedEvent):
            self._active[event.step_id] = event
        elif isinstance(event, StepFinishedEvent):
            self._active.pop(event.step_id, None)
        return event

    def accept(self, signal: object) -> Iterator[ResponseEvent]:
        if isinstance(signal, PendingSequenceCommit):
            self._pending[signal.sequence_uuid] = signal
        elif isinstance(signal, ProgressSignal):
            yield cast(ResponseEvent, self.event(
                signal.event_class, emitted_at=signal.emitted_at, **signal.payload,
            ))

    def confirm_commits(self, state: Mapping[str, Any]) -> Iterator[ResponseEvent]:
        """Only merged state snapshots can acknowledge a complete variant."""
        sequences = state.get("sequences", {})
        for sequence_uuid, pending in list(self._pending.items()):
            key = str(sequence_uuid)
            if key not in sequences or state.get("active_sequence_uuid") != key:
                continue
            if not state.get("turn_operation_committed"):
                continue
            if sequence_uuid not in self._committed:
                self._committed.add(sequence_uuid)
                step = self._active[pending.finished.payload["step_id"]]
                if step.code == "generating_sequence":
                    self.generation_completed = True
                committed = self.event(
                    SequenceCommittedEvent, sequence_uuid=sequence_uuid,
                    step_id=pending.finished.payload["step_id"],
                    tool_call_id=pending.finished.payload["tool_call_id"],
                )
                yield committed
                # State persistence is part of the operation's duration.
                yield self.event(
                    StepFinishedEvent, **{
                        **pending.finished.payload,
                        "duration_seconds": max(
                            pending.finished.payload["duration_seconds"],
                            committed.elapsed_seconds - step.elapsed_seconds,
                        ),
                    }, emitted_at=self.started_at + committed.elapsed_seconds,
                )
            del self._pending[sequence_uuid]

    def finish(self, error_message: str | None = None) -> Iterator[ResponseEvent]:
        if error_message is not None:
            # Children close before parents, and already finished steps stay closed.
            for step in reversed(list(self._active.values())):
                yield self.event(
                    StepFinishedEvent, step_id=step.step_id, tool_call_id=step.tool_call_id,
                    outcome="error", duration_seconds=max(0.0, monotonic() - self.started_at - step.elapsed_seconds),
                    summary="Could not complete this step.",
                )
        elif self._active or self._pending:
            raise RuntimeError("Request ended before all execution steps were committed or finished")
        duration = monotonic() - self.started_at
        yield self.event(
            RequestFinishedEvent, outcome="error" if error_message else "complete",
            duration_seconds=duration, error_message=error_message,
            emitted_at=self.started_at + duration,
        )
