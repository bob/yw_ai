"""Validate the public progress contract without model, catalog or UI calls."""

import json
import unittest
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from langchain_app.schemas import (
    ChatMessage,
    ExecutionSnapshot,
    ExecutionStepSnapshot,
    RequestFinishedEvent,
    ResponseEvent,
    StepFinishedEvent,
    StepProgressEvent,
)


class ResponseProgressSchemaTests(unittest.TestCase):
    def setUp(self):
        self.request_id = str(uuid4())
        self.step_id = str(uuid4())
        self.variant_id = str(uuid4())
        self.events = TypeAdapter(list[ResponseEvent])
        self.event = TypeAdapter(ResponseEvent)

    def payload(self, type, **fields):
        return {
            "request_id": self.request_id, "event_index": 0, "elapsed_seconds": 0,
            "type": type, **fields,
        }

    def step(self, **fields):
        return {
            "event_index": 1, "elapsed_seconds": 0.1, "step_id": self.step_id,
            "tool_call_id": "generate-call", "code": "generating_sequence",
            "title": "Generating sequence...", "outcome": "ok", "duration_seconds": 2,
            "summary": "Sequence created", **fields,
        }

    def test_every_event_survives_json_round_trip_with_its_public_payload(self):
        step_fields = {"step_id": self.step_id, "tool_call_id": "generate-call"}
        payloads = [
            self.payload("request_started"),
            self.payload("step_started", **step_fields,
                         code="generating_sequence", title="Generating sequence..."),
            self.payload("step_progress", **step_fields, completed=1, total=1),
            self.payload("assistant_message", purpose="intermediate", text="Подберу асаны.",
                         message_id="assistant-1", step_id=self.step_id),
            self.payload("sequence_element", **step_fields, order=1, catalog_item={
                "id": 42, "uuid": "tree-pose", "name": "Tree pose", "bilateral": True,
                "image_url": None, "catalog_tag_names": ["Balance"],
            }),
            self.payload("sequence_committed", **step_fields, sequence_uuid=self.variant_id),
            self.payload("step_finished", **step_fields, outcome="ok",
                         duration_seconds=2, summary="Sequence created"),
            self.payload("request_finished", outcome="complete", duration_seconds=2.1),
        ]
        for index, payload in enumerate(payloads):
            payload["event_index"] = index
            payload["elapsed_seconds"] = index / 10
        events = self.events.validate_python(payloads)
        restored = self.events.validate_json(self.events.dump_json(events))
        self.assertEqual(restored, events)
        self.assertEqual([event.type for event in restored], [p["type"] for p in payloads])
        self.assertEqual(restored[4].catalog_item.name, "Tree pose")
        self.assertEqual(str(restored[5].sequence_uuid), self.variant_id)

    def test_final_text_is_distinguished_from_intermediate_text(self):
        for purpose in ("intermediate", "final"):
            with self.subTest(purpose=purpose):
                event = self.event.validate_python(self.payload(
                    "assistant_message", purpose=purpose, text="Готово.",
                ))
                self.assertEqual(event.purpose, purpose)

    def test_unknown_events_raw_payloads_and_missing_fields_are_rejected(self):
        for payload in (
            self.payload("unknown_event"),
            self.payload("request_started", tool_arguments={"request": "raw"}),
            self.payload("assistant_message", purpose="final", text="Ready", reasoning="hidden"),
            self.payload("assistant_message", purpose="reasoning", text="hidden"),
            self.payload("assistant_message", purpose="final", text="  "),
            self.payload("step_started", code="thinking", title="Thinking..."),
            self.payload("sequence_element", step_id=self.step_id, order=1),
            self.payload("sequence_committed", step_id=self.step_id),
            self.payload("request_started", request_id="not-a-uuid"),
        ):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                self.event.validate_python(payload)

    def test_event_positions_and_times_reject_negative_nonfinite_and_coerced_values(self):
        for fields in (
            {"event_index": -1}, {"event_index": True}, {"event_index": 1.5},
            {"event_index": "1"}, {"elapsed_seconds": -0.1},
            {"elapsed_seconds": float("nan")}, {"elapsed_seconds": float("inf")},
            {"elapsed_seconds": True}, {"elapsed_seconds": "1"},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                self.event.validate_python(self.payload("request_started", **fields))

    def test_progress_supports_empty_search_unknown_total_and_known_pose_count(self):
        for fields in (
            {"summary": "No catalog items found", "completed": 0, "total": 0},
            {"completed": 3}, {"total": 12}, {"completed": 3, "total": 12},
        ):
            with self.subTest(fields=fields):
                StepProgressEvent(**self.payload("step_progress", step_id=self.step_id, **fields))
        for fields in (
            {}, {"completed": -1}, {"total": -1}, {"completed": True},
            {"total": 1.5}, {"completed": 13, "total": 12},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                StepProgressEvent(**self.payload("step_progress", step_id=self.step_id, **fields))

    def test_no_change_and_recovered_tool_error_can_belong_to_a_completed_request(self):
        for outcome in ("no_change", "error"):
            with self.subTest(outcome=outcome):
                step = StepFinishedEvent(**self.payload(
                    "step_finished", step_id=self.step_id, outcome=outcome,
                    duration_seconds=1, summary="No matching replacement",
                ))
                snapshot = ExecutionSnapshot(
                    request_id=self.request_id, outcome="complete", duration_seconds=2,
                    steps=[self.step(outcome=step.outcome, summary=step.summary)],
                )
                self.assertEqual(snapshot.outcome, "complete")
                self.assertEqual(snapshot.steps[0].outcome, outcome)

    def test_terminal_errors_require_public_text_and_valid_duration(self):
        for fields in (
            {"outcome": "error"},
            {"outcome": "error", "error_message": " "},
            {"outcome": "complete", "error_message": "Fatal error"},
            {"outcome": "running"},
            {"outcome": "complete", "duration_seconds": -1},
            {"outcome": "complete", "duration_seconds": float("inf")},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                RequestFinishedEvent(**self.payload(
                    "request_finished", **{"duration_seconds": 1, **fields},
                ))

    def test_snapshot_preserves_substeps_progress_messages_and_committed_variant_after_error(self):
        child_id = str(uuid4())
        source_id = str(uuid4())
        snapshot = ExecutionSnapshot(
            request_id=self.request_id, outcome="error", duration_seconds=3,
            error_message="Could not prepare response", source_sequence_uuid=source_id,
            sequence_uuid=self.variant_id, steps=[
                self.step(progress=[{
                    "event_index": 4, "elapsed_seconds": 1.5, "completed": 1, "total": 1,
                }]),
                self.step(step_id=child_id, event_index=2, elapsed_seconds=0.2,
                          parent_step_id=self.step_id, code="searching_catalog_items",
                          title="Searching catalog items...", duration_seconds=1,
                          summary="12 suitable candidates"),
            ], intermediate_messages=[{
                "event_index": 3, "elapsed_seconds": 1.3, "step_id": self.step_id,
                "message_id": "assistant-1", "text": "Нашёл подходящие асаны.",
            }],
        )
        serialized = json.loads(json.dumps(snapshot.model_dump(mode="json")))
        self.assertEqual(ExecutionSnapshot.model_validate(serialized), snapshot)
        self.assertEqual(serialized["source_sequence_uuid"], source_id)
        self.assertEqual(serialized["sequence_uuid"], self.variant_id)
        self.assertEqual(serialized["intermediate_messages"][0]["text"], "Нашёл подходящие асаны.")
        self.assertEqual(serialized["steps"][0]["progress"][0]["completed"], 1)

    def test_snapshot_rejects_duplicate_steps_and_foreign_step_references(self):
        foreign_id = str(uuid4())
        for fields in (
            {"steps": [self.step(), self.step()]},
            {"steps": [self.step(parent_step_id=foreign_id)]},
            {"steps": [self.step(parent_step_id=self.step_id)]},
            {"intermediate_messages": [{
                "event_index": 1, "elapsed_seconds": 1, "text": "Searching",
                "step_id": foreign_id,
            }]},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                ExecutionSnapshot(
                    request_id=self.request_id, outcome="complete", duration_seconds=2, **fields,
                )

    def test_chat_keeps_old_messages_and_separate_failed_regeneration_attempts(self):
        original = ExecutionSnapshot(
            request_id=self.request_id, outcome="complete", duration_seconds=2,
            sequence_uuid=self.variant_id,
        )
        failed = ExecutionSnapshot(
            request_id=uuid4(), outcome="error", duration_seconds=0,
            error_message="Could not initialize catalog", source_sequence_uuid=self.variant_id,
        )
        old_message: ChatMessage = {"role": "assistant", "content": "Ready"}
        message: ChatMessage = {
            "role": "assistant", "content": "Ready", "execution": failed,
            "execution_history": [original],
        }
        self.assertNotIn("execution", old_message)
        self.assertIsNone(message["execution"].sequence_uuid)
        self.assertNotEqual(original.request_id, failed.request_id)
        self.assertEqual(message["execution_history"][0].sequence_uuid, failed.source_sequence_uuid)
        self.assertTrue({"execution", "execution_history"} <= ChatMessage.__optional_keys__)
        another = ExecutionSnapshot(request_id=uuid4(), outcome="complete", duration_seconds=0)
        original.steps.append(ExecutionStepSnapshot(**self.step()))
        self.assertEqual(another.steps, [])


if __name__ == "__main__":
    unittest.main()
