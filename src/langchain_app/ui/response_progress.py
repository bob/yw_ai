"""Live status and replay for one assistant request's public execution history."""

from dataclasses import dataclass, field
from time import monotonic, sleep
from uuid import UUID, uuid4

import streamlit as st

from langchain_app.schemas.chat import ChatMessage
from langchain_app.schemas.response_progress import (
    AssistantMessageEvent,
    ExecutionSnapshot,
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
    with status:
        _render_execution_details(snapshot)
    if snapshot.error_message:
        st.error(snapshot.error_message)


def _render_execution_details(snapshot: ExecutionSnapshot) -> None:
    # Session history can retain instances created before Streamlit reloads a module.
    entries = [
        *((step, True) for step in snapshot.steps),
        *((message, False) for message in snapshot.intermediate_messages),
    ]
    for entry, is_step in sorted(entries, key=lambda item: item[0].event_index):
        if is_step:
            st.markdown(_step_text(
                entry.title, summary=entry.summary, outcome=entry.outcome,
                duration=entry.duration_seconds,
            ))
            for update in entry.progress:
                if update.summary:
                    st.caption(update.summary)
                if update.completed is not None:
                    count = f"{update.completed}/{update.total}" if update.total is not None else str(update.completed)
                    st.caption(f"{count} poses")
        else:
            st.markdown(entry.text)
    if snapshot.sequence_uuid is not None:
        st.caption("Sequence change saved in this session.")


class ResponseProgressRenderer:
    """Rotate public progress in one status line and retain completed history."""

    def __init__(self, request_id: UUID, accepted_at: float, source_sequence_uuid: str | None = None) -> None:
        self.execution = ResponseExecution(request_id)
        if source_sequence_uuid is not None:
            try:
                self.execution.source_sequence_uuid = UUID(source_sequence_uuid)
            except ValueError:
                # Let the producer report an invalid Regenerate source publicly.
                pass
        self.accepted_at = accepted_at
        self.status = st.status("Thinking...", expanded=False, state="running")
        self._current_label = "Thinking..."
        self._label_until = 0.0
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
            self._show_label(event.title)
        elif isinstance(event, StepProgressEvent):
            if event.summary:
                self._show_label(event.summary)
            else:
                title = self.execution.starts[event.step_id].title
                counters = [update for update in self.execution.progress[event.step_id]
                            if update.completed is not None]
                if counters:
                    latest = counters[-1]
                    title += (f" {latest.completed}/{latest.total} poses" if latest.total is not None
                              else f" {latest.completed} poses")
                self._show_label(title, minimum_seconds=0.0)
        elif isinstance(event, StepFinishedEvent):
            self._show_label(event.summary)
        elif isinstance(event, AssistantMessageEvent) and event.purpose == "intermediate":
            self._show_label(event.text)
        elif isinstance(event, SequenceCommittedEvent):
            self._show_label("Sequence change saved in this session.")
        elif isinstance(event, RequestFinishedEvent):
            self._update_terminal_status()
        return True

    def _wait_for_label(self) -> None:
        remaining = self._label_until - monotonic()
        if remaining > 0:
            sleep(remaining)

    def _show_label(self, label: str, *, minimum_seconds: float = 0.6) -> None:
        label = " ".join(label.split())
        if label == self._current_label:
            return
        # Keep fast stages/results readable; consecutive pose counts need no pause.
        self._wait_for_label()
        # Omitting expanded preserves the user's live expand/collapse choice.
        self.status.update(label=label, state="running")
        self._current_label = label
        self._label_until = monotonic() + minimum_seconds

    def _update_terminal_status(self) -> None:
        snapshot = self.execution.snapshot()
        if snapshot.outcome != "error":
            self._wait_for_label()
        self._label_until = 0.0
        self.status.update(
            label=_label(snapshot), state="error" if snapshot.outcome == "error" else "complete",
            expanded=snapshot.outcome == "error",
        )

    def finish(self) -> ExecutionSnapshot:
        if self.execution.result is None:
            self.execution.fail(max(0.0, monotonic() - self.accepted_at))
        snapshot = self.execution.snapshot()
        self._update_terminal_status()
        with self.status:
            _render_execution_details(snapshot)
        if snapshot.error_message:
            st.error(snapshot.error_message)
        return snapshot
