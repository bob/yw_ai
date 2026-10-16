"""Verify atomic occurrence removal, independent variants, and snapshot compatibility."""

import json
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from langchain_app.schemas import CatalogItemSchema, GeneratedSequence
from langchain_app.services.sequence_editor import SequenceEditorError, remove_elements


def sequence(count=5):
    source = GeneratedSequence.from_catalog_items([
        CatalogItemSchema(
            id=number, uuid=str(UUID(int=number)), name=f"Pose {number}",
            bilateral=number % 2 == 0,
            image_url=f"https://example.com/{number}.png",
            catalog_tag_names=["Position/Standing", f"Tag {number}"],
        )
        for number in range(1, count + 1)
    ], request="Original generation request")
    source.name = "Custom practice"
    source.description = "Custom description"
    return source


class SequenceRemovalTests(unittest.TestCase):
    def assert_rejected(self, source, element_ids, code):
        before = source.model_dump(mode="json")
        with patch("langchain_app.services.sequence_editor.uuid4") as new_uuid:
            with self.assertRaises(SequenceEditorError) as error:
                remove_elements(source, element_ids=element_ids, request="Remove poses")
        self.assertEqual(error.exception.code, code)
        new_uuid.assert_not_called()
        self.assertEqual(source.model_dump(mode="json"), before)

    def test_single_first_middle_and_last_removals_preserve_relative_order(self):
        for index in (0, 2, 4):
            with self.subTest(index=index):
                source = sequence()
                before = source.model_dump(mode="json")
                variant = remove_elements(
                    source, element_ids=[source.elements[index].element_id], request="Remove one pose",
                )
                self.assertEqual(
                    [element.catalog_item.id for element in variant.elements],
                    [number for number in range(1, 6) if number != index + 1],
                )
                self.assertEqual([element.order for element in variant.elements], [1, 2, 3, 4])
                self.assertEqual(variant.elements_count, 4)
                self.assertEqual(source.model_dump(mode="json"), before)

    def test_adjacent_and_nonadjacent_targets_use_one_original_snapshot_in_any_order(self):
        for indices, expected in (((1, 3), [1, 3, 5]), ((1, 2, 3), [1, 5]), ((0, 4), [2, 3, 4])):
            source = sequence()
            targets = [source.elements[index].element_id for index in indices]
            results = []
            for element_ids in (targets, targets[::-1]):
                with self.subTest(indices=indices, element_ids=element_ids):
                    with patch("langchain_app.services.sequence_editor.uuid4", wraps=uuid4) as new_uuid:
                        variant = remove_elements(source, element_ids=element_ids, request="Remove several poses")
                    new_uuid.assert_called_once_with()
                    self.assertEqual([element.catalog_item.id for element in variant.elements], expected)
                    self.assertEqual([element.order for element in variant.elements], list(range(1, len(expected) + 1)))
                    results.append(variant.elements)
            self.assertEqual(results[0], results[1])

    def test_survivors_keep_ids_full_catalog_items_and_actual_mirror_flags(self):
        source = sequence()
        flags = [(True, False), (False, True), (True, True), (False, False), (False, True)]
        for element, (start, end) in zip(source.elements, flags):
            element.start_mirror, element.end_mirror = start, end
        variant = remove_elements(
            source, element_ids=[source.elements[1].element_id], request="Remove the second pose",
        )
        for actual, original in zip(variant.elements, [source.elements[index] for index in (0, 2, 3, 4)]):
            expected = original.model_dump(mode="json")
            expected["order"] = actual.order
            self.assertEqual(actual.model_dump(mode="json"), expected)

    def test_removes_only_selected_occurrences_of_repeated_catalog_items(self):
        source = sequence()
        for index in (2, 4):
            source.elements[index].catalog_item = source.elements[0].catalog_item.model_copy(deep=True)
        for indices, expected in (((0,), [2, 1, 4, 1]), ((0, 4), [2, 1, 4]), ((0, 2, 4), [2, 4])):
            with self.subTest(indices=indices):
                variant = remove_elements(
                    source, element_ids=[source.elements[index].element_id for index in indices],
                    request="Remove selected occurrences",
                )
                self.assertEqual([element.catalog_item.id for element in variant.elements], expected)
                self.assertEqual(
                    [element.element_id for element in variant.elements],
                    [element.element_id for index, element in enumerate(source.elements) if index not in indices],
                )

    def test_mirrored_pose_is_removed_as_one_whole_occurrence(self):
        source = sequence(3)
        source.elements[1].start_mirror = source.elements[1].end_mirror = True
        variant = remove_elements(source, element_ids=[source.elements[1].element_id], request="Remove the mirrored pose")
        self.assertEqual([element.catalog_item.id for element in variant.elements], [1, 3])
        self.assertEqual(variant.elements_count, 2)

    def test_variant_preserves_metadata_and_uses_selected_variant_as_parent(self):
        original = sequence()
        selected = GeneratedSequence.from_catalog_items(
            [element.catalog_item for element in original.elements],
            request="Previous regeneration request", parent=original,
        )
        selected.name, selected.description = "Selected variant", "Selected description"
        snapshots = [item.model_dump(mode="json") for item in (original, selected)]
        request = "  Удали вторую и четвёртую асаны\nиз выбранного варианта.  "
        variant = remove_elements(
            selected, element_ids=[selected.elements[index].element_id for index in (1, 3)], request=request,
        )
        self.assertNotIn(variant.uuid, (selected.uuid, original.uuid))
        self.assertEqual(variant.parent_uuid, selected.uuid)
        self.assertEqual(variant.family_uuid, selected.family_uuid)
        self.assertEqual(variant.operation, "remove")
        self.assertEqual(variant.operation_request, request)
        for field in ("generation_request", "name", "description", "language"):
            self.assertEqual(getattr(variant, field), getattr(selected, field))
        self.assertEqual([item.model_dump(mode="json") for item in (original, selected)], snapshots)

    def test_variant_and_history_share_no_mutable_sequence_or_catalog_objects(self):
        source = sequence()
        sibling = GeneratedSequence.from_snapshot(source.model_dump(mode="json"))
        before = source.model_dump(mode="json")
        variant = remove_elements(source, element_ids=[source.elements[1].element_id], request="Remove second")
        self.assertIsNot(variant.elements, source.elements)
        originals = {element.element_id: element for element in source.elements}
        for element in variant.elements:
            original = originals[element.element_id]
            self.assertIsNot(element, original)
            self.assertIsNot(element.catalog_item, original.catalog_item)
            self.assertIsNot(element.catalog_item.catalog_tag_names, original.catalog_item.catalog_tag_names)
            element.catalog_item.name = "Changed variant"
            element.catalog_item.catalog_tag_names.append("Changed tag")
            element.start_mirror = not element.start_mirror
        variant.elements.pop()
        self.assertEqual(source.model_dump(mode="json"), before)
        self.assertEqual(sibling.model_dump(mode="json"), before)
        variant_before = variant.model_dump(mode="json")
        source.elements[0].catalog_item.catalog_tag_names.clear()
        source.elements.clear()
        self.assertEqual(variant.model_dump(mode="json"), variant_before)
        self.assertEqual(sibling.model_dump(mode="json"), before)

    def test_empty_source_is_rejected_before_target_checks(self):
        for targets in ([], [uuid4()], [UUID(int=1), UUID(int=1)]):
            with self.subTest(targets=targets):
                self.assert_rejected(sequence(0), targets, "empty_sequence")

    def test_empty_and_duplicate_targets_are_rejected_before_membership_checks(self):
        source = sequence()
        known, unknown = source.elements[0].element_id, uuid4()
        for targets in ([], [known, known], [unknown, unknown], [known, unknown, known]):
            with self.subTest(targets=targets):
                self.assert_rejected(source, targets, "invalid_arguments")

    def test_unknown_or_mixed_targets_reject_the_entire_removal(self):
        source = sequence()
        known, unknown = source.elements[0].element_id, uuid4()
        for targets in ([unknown], [known, unknown], [unknown, known], [element.element_id for element in source.elements] + [unknown]):
            with self.subTest(targets=targets):
                self.assert_rejected(source, targets, "sequence_element_not_found")

    def test_removing_all_or_the_only_occurrence_is_rejected(self):
        for count in (1, 5):
            with self.subTest(count=count):
                source = sequence(count)
                self.assert_rejected(source, [element.element_id for element in source.elements], "would_empty_sequence")

    def test_removing_all_but_one_is_allowed(self):
        for count, survivor in ((2, 0), (5, 2), (5, 4)):
            with self.subTest(count=count, survivor=survivor):
                source = sequence(count)
                variant = remove_elements(
                    source, element_ids=[element.element_id for index, element in enumerate(source.elements) if index != survivor],
                    request="Keep only the selected pose",
                )
                self.assertEqual(variant.elements_count, 1)
                self.assertEqual(variant.elements[0].element_id, source.elements[survivor].element_id)
                self.assertEqual(variant.elements[0].order, 1)

    def test_already_removed_uuid_is_rejected_in_the_new_variant(self):
        source = sequence()
        removed = source.elements[1].element_id
        variant = remove_elements(source, element_ids=[removed], request="Remove second")
        self.assert_rejected(variant, [removed], "sequence_element_not_found")
        # A failed edit does not prevent a corrected call on the same snapshot.
        corrected = remove_elements(variant, element_ids=[variant.elements[0].element_id], request="Remove first")
        self.assertEqual(corrected.parent_uuid, variant.uuid)
        self.assertEqual(corrected.elements_count, 3)

    def test_imported_long_sequence_is_not_limited_to_45(self):
        source = sequence(50)
        variant = remove_elements(source, element_ids=[source.elements[1].element_id], request="Remove second")
        self.assertEqual(variant.elements_count, 49)
        self.assertEqual([element.catalog_item.id for element in variant.elements], [1, *range(3, 51)])
        self.assertEqual([element.order for element in variant.elements], list(range(1, 50)))
        self.assertEqual(source.elements_count, 50)

    def test_orders_and_count_follow_actual_list_instead_of_imported_metadata(self):
        snapshot = sequence().model_dump(mode="json")
        snapshot["elements_count"] = 1
        source = GeneratedSequence.from_snapshot(snapshot)
        for element, order in zip(source.elements, (50, 10, 30, 20, 40)):
            element.order = order
        variant = remove_elements(
            source, element_ids=[source.elements[index].element_id for index in (1, 3)], request="Remove second and fourth",
        )
        self.assertEqual([element.catalog_item.id for element in variant.elements], [1, 3, 5])
        self.assertEqual([element.order for element in variant.elements], [1, 2, 3])
        self.assertEqual(variant.elements_count, 3)
        self.assertEqual([element.order for element in source.elements], [50, 10, 30, 20, 40])

    def test_removing_boundaries_does_not_apply_generation_or_catalog_rules(self):
        source = sequence()
        for element in source.elements:
            element.catalog_item.image_url = element.catalog_item.uuid = None
            element.catalog_item.catalog_tag_names = ["Other/Starting", "Other/Finish", "Position/Lying on the back"]
        for index in (0, 1, 4):
            with self.subTest(index=index):
                with (
                    patch("langchain_app.services.sequence_editor.starting_pose_allowed", side_effect=AssertionError("Unexpected starting pose check")),
                    patch("langchain_app.services.sequence_editor.second_pose_allowed", side_effect=AssertionError("Unexpected second pose check")),
                    patch("langchain_app.services.sequence_editor.has_catalog_uuid", side_effect=AssertionError("Unexpected catalog check")),
                ):
                    variant = remove_elements(source, element_ids=[source.elements[index].element_id], request="Remove boundary pose")
                self.assertEqual(
                    [element.catalog_item for element in variant.elements],
                    [element.catalog_item for position, element in enumerate(source.elements) if position != index],
                )

    def test_removal_snapshot_round_trip_preserves_all_fields(self):
        source = sequence()
        source.elements[0].start_mirror, source.elements[0].end_mirror = True, False
        variant = remove_elements(source, element_ids=[source.elements[1].element_id], request="Remove second")
        snapshot = json.loads(variant.model_dump_json())
        restored = GeneratedSequence.from_snapshot(snapshot)
        self.assertEqual(restored.operation, "remove")
        self.assertEqual(restored, variant)
        self.assertEqual(restored.model_dump(mode="json"), snapshot)

    def test_existing_operation_snapshots_remain_compatible(self):
        snapshot = sequence().model_dump(mode="json")
        for operation in ("generate", "regenerate", "replace", "swap", "add"):
            with self.subTest(operation=operation):
                restored = GeneratedSequence.from_snapshot({**snapshot, "operation": operation})
                self.assertEqual(restored.operation, operation)
                self.assertEqual(restored.model_dump(mode="json"), {**snapshot, "operation": operation})


if __name__ == "__main__":
    unittest.main()
