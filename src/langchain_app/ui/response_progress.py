"""Live status and replay for one assistant request's public execution history."""

from dataclasses import dataclass, field
from time import monotonic
from uuid import UUID, uuid4

import streamlit as st

from langchain_app.schemas.chat import ChatMessage
from langchain_app.schemas.response_progress import (
    AssistantMessageEvent,
    ExecutionSnapshot,
    ExecutionStepSnapshot,
    RequestFinishedEvent,
    ResponseEvent,
    SequenceCommittedEvent,
    StepFinishedEvent,
    StepProgressEvent,
    StepStartedEvent,
)
from langchain_app.services.response_execution import ResponseExecution


@dataclass(frozen=True)
class PendingResponse:
    request: str
    require_generation: bool = False
    sequence_uuid: str | None = None
    message_index: int | None = None
    request_id: UUID = field(default_factory=uuid4)
    accepted_at: float = field(default_factory=monotonic)


def enqueue_response(pending: PendingResponse) -> None:
    """Callbacks run before the script redraws its disabled request controls."""
    if st.session_state.get("pending_response") is not None:
        return
    st.session_state["pending_response"] = pending
    if pending.message_index is None:
        st.session_state.messages.append({"role": "user", "content": pending.request})


def enqueue_regeneration(request: str, sequence_uuid: str | None, message_index: int) -> None:
    enqueue_response(PendingResponse(
        request, require_generation=True, sequence_uuid=sequence_uuid,
        message_index=message_index,
    ))


def retain_execution(message: ChatMessage, snapshot: ExecutionSnapshot) -> None:
    previous = message.get("execution")
    if previous is not None and previous.request_id != snapshot.request_id:
        message.setdefault("execution_history", []).append(previous)
    message["execution"] = snapshot


def _label(snapshot: ExecutionSnapshot) -> str:
    if snapshot.outcome == "error":
        return "Could not complete request"
    return f"Completed in {snapshot.duration_seconds:.0f} s"


def _step_text(title: str, *, summary: str | None = None, outcome: str | None = None,
               duration: float | None = None, completed: int | None = None,
               total: int | None = None) -> str:
    label = f"**{title}**"
    if completed is not None:
        label += f" {completed}/{total} poses" if total is not None else f" {completed} poses"
    if outcome is not None:
        label += f" · {'Failed' if outcome == 'error' else 'No change' if outcome == 'no_change' else 'Done'}"
    if duration is not None:
        label += f" · {duration:.1f} s"
    return label + (f"\n\n{summary}" if summary else "")


def render_execution(snapshot: ExecutionSnapshot) -> None:
    """Replaying history never initializes the model or executes a tool."""
    status = st.status(
        _label(snapshot), state="error" if snapshot.outcome == "error" else "complete",
        expanded=snapshot.outcome == "error",
    )
    entries = sorted([*snapshot.steps, *snapshot.intermediate_messages], key=lambda item: item.event_index)
    for entry in entries:
        if isinstance(entry, ExecutionStepSnapshot):
            status.markdown(_step_text(
                entry.title, summary=entry.summary, outcome=entry.outcome,
                duration=entry.duration_seconds,
            ))
            for update in entry.progress:
                if update.summary:
                    status.caption(update.summary)
                if update.completed is not None:
                    count = f"{update.completed}/{update.total}" if update.total is not None else str(update.completed)
                    status.caption(f"{count} poses")
        else:
            status.markdown(entry.text)
    if snapshot.sequence_uuid is not None:
        status.caption("Sequence change saved in this session.")
    if snapshot.error_message:
        st.error(snapshot.error_message)


class ResponseProgressRenderer:
    """Show feedback before dependencies initialize and keep collapse state intact."""

    def __init__(self, request_id: UUID, accepted_at: float, source_sequence_uuid: str | None = None) -> None:
        self.execution = ResponseExecution(request_id)
        if source_sequence_uuid is not None:
            try:
                self.execution.source_sequence_uuid = UUID(source_sequence_uuid)
            except ValueError:
                # Let the producer report an invalid Regenerate source publicly.
                pass
        self.accepted_at = accepted_at
        self.status = st.status("Thinking...", expanded=True, state="running")
        self._rows = {}
        self._offset: float | None = None

    def accept(self, event: ResponseEvent) -> bool:
        if event.request_id != self.execution.request_id:
            return False
        if self._offset is None:
            self._offset = max(0.0, monotonic() - self.accepted_at - event.elapsed_seconds)
        updates = {"elapsed_seconds": event.elapsed_seconds + self._offset}
        if isinstance(event, RequestFinishedEvent):
            updates["duration_seconds"] = max(event.duration_seconds + self._offset, monotonic() - self.accepted_at)
        event = event.model_copy(update=updates)
        if not self.execution.accept(event):
            return False
        if isinstance(event, StepStartedEvent):
            self._rows[event.step_id] = self.status.empty()
            self._draw_step(event.step_id)
        elif isinstance(event, (StepProgressEvent, StepFinishedEvent)):
            self._draw_step(event.step_id)
        elif isinstance(event, AssistantMessageEvent) and event.purpose == "intermediate":
            self.status.markdown(event.text)
        elif isinstance(event, SequenceCommittedEvent):
            self.status.caption("Sequence change saved in this session.")
        self._update_label()
        return True

    def _draw_step(self, step_id: UUID) -> None:
        start = self.execution.starts[step_id]
        finish = self.execution.finishes.get(step_id)
        progress = self.execution.progress[step_id]
        latest = progress[-1] if progress else None
        counters = [update for update in progress if update.completed is not None]
        counter = counters[-1] if counters else None
        self._rows[step_id].markdown(_step_text(
            start.title, summary=finish.summary if finish else latest.summary if latest else None,
            outcome=finish.outcome if finish else None,
            duration=finish.duration_seconds if finish else None,
            completed=counter.completed if counter else None, total=counter.total if counter else None,
        ))

    def _update_label(self) -> None:
        if self.execution.result is not None:
            snapshot = self.execution.snapshot()
            self.status.update(
                label=_label(snapshot), state="error" if snapshot.outcome == "error" else "complete",
                expanded=snapshot.outcome == "error",
            )
            return
        active = [step for step_id, step in self.execution.starts.items() if step_id not in self.execution.finishes]
        title = active[-1].title if active else "Thinking..."
        if active:
            counters = [update for update in self.execution.progress[active[-1].step_id] if update.completed is not None]
            if counters:
                latest = counters[-1]
                title += (f" {latest.completed}/{latest.total} poses" if latest.total is not None
                          else f" {latest.completed} poses")
        # Omitting expanded preserves the user's live expand/collapse choice.
        self.status.update(label=title, state="running")

    def finish(self) -> ExecutionSnapshot:
        if self.execution.result is None:
            self.execution.fail(max(0.0, monotonic() - self.accepted_at))
        for step_id in self.execution.starts:
            self._draw_step(step_id)
        self._update_label()
        snapshot = self.execution.snapshot()
        if snapshot.error_message:
            st.error(snapshot.error_message)
        return snapshot
