"""Verify complete insertion blocks, detached variants, and hard pose constraints."""

import json
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from langchain_app.schemas import CatalogItemSchema, GeneratedSequence
from langchain_app.services.sequence_editor import SequenceEditorError, insert_elements
from langchain_app.services.sequence_generator import MAX_SEQUENCE_ELEMENTS


def pose(number, tags=(), *, bilateral=False):
    return CatalogItemSchema(
        id=number, uuid=str(UUID(int=number)), name=f"Pose {number}",
        bilateral=bilateral, image_url=f"https://example.com/{number}.png",
        catalog_tag_names=list(tags),
    )


def source(count=4):
    sequence = GeneratedSequence.from_catalog_items(
        [pose(number, bilateral=number % 2 == 0) for number in range(1, count + 1)],
        request="Original gentle practice",
    )
    sequence.name = "Custom practice"
    sequence.description = "Custom description"
    return sequence


class SequenceInsertionTests(unittest.TestCase):
    def assert_rejected(self, sequence, items, *, count=None, code="invalid_addition_result", **kwargs):
        before = sequence.model_dump(mode="json")
        originals = None
        if isinstance(items, list):
            originals = [
                item.model_dump(mode="json", warnings=False)
                for item in items if isinstance(item, CatalogItemSchema)
            ]
        with patch("langchain_app.services.sequence_editor.uuid4") as new_uuid:
            with self.assertRaises(SequenceEditorError) as raised:
                insert_elements(
                    sequence, items=items, count=len(items) if count is None else count,
                    request=kwargs.pop("request", "Add poses"), **kwargs,
                )
        self.assertEqual(raised.exception.code, code)
        new_uuid.assert_not_called()
        self.assertEqual(sequence.model_dump(mode="json"), before)
        if originals is not None:
            self.assertEqual(
                [item.model_dump(mode="json", warnings=False) for item in items if isinstance(item, CatalogItemSchema)],
                originals,
            )

    def test_append_one_or_multiple_poses_keeps_original_order(self):
        for count in (1, 3):
            with self.subTest(count=count):
                sequence = source()
                before = sequence.model_dump(mode="json")
                variant = insert_elements(
                    sequence, items=(pose(number) for number in range(10, 10 + count)),
                    count=count, request="Append poses",
                )
                self.assertEqual([element.catalog_item.id for element in variant.elements], [1, 2, 3, 4, *range(10, 10 + count)])
                self.assertEqual([element.order for element in variant.elements], list(range(1, 5 + count)))
                self.assertEqual(variant.elements[:4], sequence.elements)
                self.assertEqual(variant.elements_count, 4 + count)
                self.assertEqual(sequence.model_dump(mode="json"), before)

    def test_prepend_and_internal_insertions_form_one_contiguous_block(self):
        for index, count in ((0, 1), (0, 2), (1, 2), (2, 2), (3, 1)):
            with self.subTest(index=index, count=count):
                sequence = source()
                additions = [
                    pose(10 + offset, ["Other/Starting", "Position/Sitting"] if index == 0 and offset == 0 else [])
                    for offset in range(count)
                ]
                variant = insert_elements(
                    sequence, items=additions, count=count, request="Insert poses",
                    insert_before_element_id=sequence.elements[index].element_id,
                )
                self.assertEqual(
                    [element.catalog_item.id for element in variant.elements],
                    [*range(1, index + 1), *range(10, 10 + count), *range(index + 1, 5)],
                )
                original_ids = {element.element_id for element in sequence.elements}
                self.assertEqual(
                    [element.element_id for element in variant.elements if element.element_id in original_ids],
                    [element.element_id for element in sequence.elements],
                )
                self.assertEqual([element.order for element in variant.elements], list(range(1, 5 + count)))

    def test_preserves_full_original_catalog_fields_ids_and_independent_mirror_flags(self):
        sequence = source()
        for element, flags in zip(sequence.elements, ((True, False), (False, True), (True, True), (False, False))):
            element.start_mirror, element.end_mirror = flags
            element.catalog_item.catalog_tag_names.append(f"Tag {element.order}")
        before = sequence.model_dump(mode="json")
        variant = insert_elements(
            sequence, items=[pose(10)], count=1, request="Insert before third",
            insert_before_element_id=sequence.elements[2].element_id,
        )
        originals = {element.element_id: element for element in sequence.elements}
        for element in variant.elements:
            if element.element_id in originals:
                original = originals[element.element_id]
                self.assertEqual(element.catalog_item.model_dump(mode="json"), original.catalog_item.model_dump(mode="json"))
                self.assertEqual((element.start_mirror, element.end_mirror), (original.start_mirror, original.end_mirror))
        self.assertEqual(sequence.model_dump(mode="json"), before)

    def test_new_occurrences_have_unique_ids_and_both_flags_equal_bilateral(self):
        sequence = source()
        additions = [pose(10), pose(11, bilateral=True)]
        variant = insert_elements(sequence, items=additions, count=2, request="Add two")
        self.assertEqual(len({element.element_id for element in variant.elements}), 6)
        for element, candidate in zip(variant.elements[4:], additions):
            self.assertNotIn(element.element_id, {original.element_id for original in sequence.elements})
            self.assertEqual(element.catalog_item, candidate)
            self.assertEqual((element.start_mirror, element.end_mirror), (candidate.bilateral, candidate.bilateral))

    def test_preserves_metadata_and_branches_from_the_supplied_variant(self):
        original = source()
        selected = GeneratedSequence.from_catalog_items(
            [element.catalog_item for element in original.elements], request="Regenerate", parent=original,
        )
        selected.name, selected.description = "Selected variant", "Selected description"
        before = selected.model_dump(mode="json")
        request = "Добавь две асаны"
        variant = insert_elements(selected, items=[pose(10), pose(11)], count=2, request=request)
        self.assertNotIn(variant.uuid, (selected.uuid, original.uuid))
        self.assertEqual(variant.parent_uuid, selected.uuid)
        self.assertEqual(variant.family_uuid, selected.family_uuid)
        self.assertEqual(variant.operation, "add")
        self.assertEqual(variant.operation_request, request)
        for field in ("generation_request", "name", "description", "language"):
            self.assertEqual(getattr(variant, field), getattr(selected, field))
        self.assertEqual(selected.model_dump(mode="json"), before)

    def test_variant_source_candidates_and_other_snapshots_share_no_mutable_objects(self):
        sequence = source()
        other_snapshot = GeneratedSequence.from_snapshot(sequence.model_dump(mode="json"))
        additions = [pose(10, ["Movement/Twist"]), pose(11)]
        before = sequence.model_dump(mode="json")
        candidates_before = [candidate.model_dump(mode="json") for candidate in additions]
        variant = insert_elements(sequence, items=additions, count=2, request="Add two")
        self.assertIsNot(variant.elements, sequence.elements)
        for actual, original in zip(variant.elements, [*sequence.elements, *additions]):
            catalog_item = original.catalog_item if hasattr(original, "catalog_item") else original
            self.assertIsNot(actual, original)
            self.assertIsNot(actual.catalog_item, catalog_item)
            self.assertIsNot(actual.catalog_item.catalog_tag_names, catalog_item.catalog_tag_names)
            actual.catalog_item.name = "Changed variant"
            actual.catalog_item.catalog_tag_names.append("Changed tag")
            actual.start_mirror = not actual.start_mirror
        variant.elements.pop()
        self.assertEqual(sequence.model_dump(mode="json"), before)
        self.assertEqual(other_snapshot.model_dump(mode="json"), before)
        self.assertEqual([candidate.model_dump(mode="json") for candidate in additions], candidates_before)
        variant_before = variant.model_dump(mode="json")
        sequence.elements[0].catalog_item.catalog_tag_names.append("Parent changed")
        sequence.elements.clear()
        additions[0].catalog_tag_names.clear()
        additions[1].name = "Candidate changed"
        self.assertEqual(variant.model_dump(mode="json"), variant_before)
        self.assertEqual(other_snapshot.model_dump(mode="json"), before)

    def test_new_items_with_a_shared_tag_list_are_copied_individually(self):
        additions = [pose(10), pose(11)]
        additions[1].catalog_tag_names = additions[0].catalog_tag_names
        variant = insert_elements(source(), items=additions, count=2, request="Add two")
        variant.elements[-1].catalog_item.catalog_tag_names.append("Changed")
        self.assertEqual(variant.elements[-2].catalog_item.catalog_tag_names, [])
        self.assertEqual(additions[0].catalog_tag_names, [])

    def test_existing_repetitions_and_unavailable_catalog_data_are_preserved(self):
        sequence = source(2)
        sequence.elements[0].catalog_item.uuid = None
        sequence.elements[0].catalog_item.image_url = None
        sequence.elements[1].catalog_item = sequence.elements[0].catalog_item.model_copy(deep=True)
        before = sequence.model_dump(mode="json")
        variant = insert_elements(sequence, items=[pose(10)], count=1, request="Append")
        self.assertEqual(variant.elements[:2], sequence.elements)
        self.assertEqual([element.catalog_item.id for element in variant.elements], [1, 1, 10])
        self.assertEqual(sequence.model_dump(mode="json"), before)

    def test_duplicate_id_or_case_insensitive_uuid_in_source_or_block_rejects_all(self):
        sequence = source()
        sequence.elements[-1].catalog_item.uuid = "abcdef01-2345-6789-abcd-ef0123456789"
        duplicate_uuid = sequence.elements[-1].catalog_item.uuid.upper()
        first = pose(10).model_copy(update={"uuid": duplicate_uuid})
        for additions in (
            [pose(10), pose(4)],
            [pose(10), pose(11).model_copy(update={"uuid": duplicate_uuid})],
            [pose(10), pose(10).model_copy(update={"uuid": str(uuid4())})],
            [first, pose(11).model_copy(update={"uuid": first.uuid.lower()})],
        ):
            with self.subTest(items=[candidate.id for candidate in additions]):
                self.assert_rejected(sequence, additions, insert_before_element_id=sequence.elements[2].element_id)
        # Also test a UUID alias between two new items without a source collision.
        self.assert_rejected(source(), [first, pose(11).model_copy(update={"uuid": first.uuid.lower()})])

    def test_invalid_or_unusable_candidate_at_end_discards_the_whole_block(self):
        updates = [
            {"uuid": value} for value in (None, "", " \n ", "not-a-uuid")
        ] + [
            {"image_url": None}, {"image_url": "invalid"}, {"id": 0},
            {"bilateral": "false"}, {"catalog_tag_names": [None]},
        ]
        for update in updates:
            with self.subTest(update=update):
                self.assert_rejected(source(), [pose(10), pose(11).model_copy(update=update)])
        for invalid in (None, pose(11).model_dump(), "Pose 11"):
            with self.subTest(item=invalid):
                self.assert_rejected(source(), [pose(10), invalid])
        self.assert_rejected(source(), None, count=1)

    def test_incomplete_empty_or_excessive_blocks_are_rejected(self):
        for additions in ([], [pose(10)], [pose(10), pose(11), pose(12)]):
            with self.subTest(length=len(additions)):
                self.assert_rejected(source(), additions, count=2)

    def test_exact_blocks_fill_the_limit_and_count_mirrored_items_once(self):
        for original_count, count in ((1, 44), (44, 1), (9, 36)):
            with self.subTest(original_count=original_count, count=count):
                sequence = source(original_count)
                variant = insert_elements(
                    sequence, items=[pose(100 + offset, bilateral=True) for offset in range(count)],
                    count=count, request="Fill the sequence",
                )
                self.assertEqual(variant.elements_count, MAX_SEQUENCE_ELEMENTS)
                self.assertEqual([element.order for element in variant.elements], list(range(1, MAX_SEQUENCE_ELEMENTS + 1)))

    def test_overflow_and_imported_oversized_sources_are_never_truncated(self):
        for original_count, count in ((43, 3), (45, 1), (46, 1)):
            with self.subTest(original_count=original_count, count=count):
                self.assert_rejected(
                    source(original_count), [pose(100 + offset) for offset in range(count)],
                )

    def test_empty_source_invalid_counts_and_blank_request_fail_before_new_uuid(self):
        self.assert_rejected(source(0), [pose(10)], code="empty_sequence")
        for count in (0, -1, True, False, 1.5, "1"):
            with self.subTest(count=count):
                self.assert_rejected(source(), [pose(10)], count=count, code="invalid_arguments")
        self.assert_rejected(source(), [pose(10)], request=" \n ", code="invalid_arguments")

    def test_unknown_anchor_is_rejected_even_when_source_is_at_the_limit(self):
        for original_count in (4, 45, 46):
            with self.subTest(original_count=original_count):
                self.assert_rejected(
                    source(original_count), [pose(100)], insert_before_element_id=uuid4(),
                    code="sequence_element_not_found",
                )

    def test_orders_and_count_follow_the_actual_list_after_insertion(self):
        snapshot = source().model_dump(mode="json")
        snapshot["elements_count"] = 1
        sequence = GeneratedSequence.from_snapshot(snapshot)
        for element, order in zip(sequence.elements, (30, 10, 20, 40)):
            element.order = order
        variant = insert_elements(
            sequence, items=[pose(10)], count=1, request="Insert",
            insert_before_element_id=sequence.elements[2].element_id,
        )
        self.assertEqual([element.catalog_item.id for element in variant.elements], [1, 2, 10, 3, 4])
        self.assertEqual([element.order for element in variant.elements], [1, 2, 3, 4, 5])
        self.assertEqual(variant.elements_count, 5)
        self.assertEqual([element.order for element in sequence.elements], [30, 10, 20, 40])

    def test_new_starting_poses_are_allowed_only_in_the_first_position(self):
        for index, additions in (
            (None, [pose(10, ["Other/Starting"])]),
            (2, [pose(10, ["Other/Starting"])]),
            (0, [pose(10)]),
            (0, [pose(10, ["Other/Starting"]), pose(11, ["Other/Starting"])]),
        ):
            with self.subTest(index=index):
                sequence = source()
                self.assert_rejected(
                    sequence, additions,
                    insert_before_element_id=sequence.elements[index].element_id if index is not None else None,
                )

    def test_new_second_pose_obeys_lying_and_standing_start_constraints(self):
        for first_tag, allowed_tag, forbidden_tags in (
            ("Position/Lying on the back", "Position/Lying on the back", ["Position/Standing"]),
            ("Position/Lying on the side", "Position/Lying on the back", ["Position/Lying on the stomach"]),
            ("Position/Lying on the stomach", "Position/Lying on the back", ["Position/Sitting"]),
            ("Position/Standing", "Position/Standing", ["Position/Lying on the back", "Position/Lying on the side", "Position/Lying on the stomach"]),
        ):
            for original_count in (1, 3):
                with self.subTest(first=first_tag, original_count=original_count):
                    sequence = source(original_count)
                    sequence.elements[0].catalog_item.catalog_tag_names = [first_tag]
                    anchor = sequence.elements[1].element_id if original_count > 1 else None
                    for tag in forbidden_tags:
                        self.assert_rejected(sequence, [pose(10, [tag])], insert_before_element_id=anchor)
                    variant = insert_elements(
                        sequence, items=[pose(10, [allowed_tag])], count=1,
                        request="Insert second", insert_before_element_id=anchor,
                    )
                    self.assertEqual(variant.elements[1].catalog_item.id, 10)

    def test_single_prepend_checks_the_new_start_against_the_fixed_second_pose(self):
        for first_tag, fixed_tag, allowed in (
            ("Position/Lying on the stomach", "Position/Standing", False),
            ("Position/Lying on the side", "Position/Lying on the back", True),
            ("Position/Standing", "Position/Lying on the back", False),
            ("Position/Standing", "Position/Sitting", True),
            ("Position/Sitting", "Position/Lying on the stomach", True),
        ):
            with self.subTest(first=first_tag, fixed=fixed_tag):
                sequence = source(1)
                sequence.elements[0].catalog_item.catalog_tag_names = [fixed_tag]
                additions = [pose(10, ["Other/Starting", first_tag])]
                kwargs = {"insert_before_element_id": sequence.elements[0].element_id}
                if allowed:
                    variant = insert_elements(sequence, items=additions, count=1, request="Prepend", **kwargs)
                    self.assertEqual(variant.elements[1].catalog_item, sequence.elements[0].catalog_item)
                else:
                    self.assert_rejected(sequence, additions, **kwargs)

    def test_multiple_prepend_checks_the_pair_of_new_poses(self):
        sequence = source()
        for first_tag, second_tag, allowed in (
            ("Position/Lying on the side", "Position/Standing", False),
            ("Position/Lying on the side", "Position/Lying on the back", True),
            ("Position/Standing", "Position/Lying on the stomach", False),
            ("Position/Standing", "Position/Sitting", True),
        ):
            with self.subTest(first=first_tag, second=second_tag):
                additions = [pose(10, ["Other/Starting", first_tag]), pose(11, [second_tag])]
                kwargs = {"insert_before_element_id": sequence.elements[0].element_id}
                if allowed:
                    variant = insert_elements(sequence, items=additions, count=2, request="Prepend two", **kwargs)
                    self.assertEqual(variant.elements[2].catalog_item, sequence.elements[0].catalog_item)
                else:
                    self.assert_rejected(sequence, additions, **kwargs)

    def test_insertion_after_second_does_not_revalidate_an_old_incompatible_pair(self):
        sequence = source()
        sequence.elements[0].catalog_item.catalog_tag_names = ["Position/Standing"]
        sequence.elements[1].catalog_item.catalog_tag_names = ["Position/Lying on the stomach"]
        for index in (2, None):
            with self.subTest(index=index):
                variant = insert_elements(
                    sequence, items=[pose(10)], count=1, request="Add later",
                    insert_before_element_id=sequence.elements[index].element_id if index is not None else None,
                )
                self.assertEqual(variant.elements[:2], sequence.elements[:2])

    def test_old_start_and_finish_are_preserved_when_new_poses_extend_the_boundaries(self):
        sequence = source()
        sequence.elements[0].catalog_item.catalog_tag_names = ["Other/Starting"]
        sequence.elements[-1].catalog_item.catalog_tag_names = ["Other/Finishing"]
        prepended = insert_elements(
            sequence, items=[pose(10, ["Other/Starting"]), pose(11)], count=2, request="Prepend",
            insert_before_element_id=sequence.elements[0].element_id,
        )
        self.assertEqual(prepended.elements[2].catalog_item, sequence.elements[0].catalog_item)
        appended = insert_elements(sequence, items=[pose(12)], count=1, request="Append")
        self.assertEqual(appended.elements[3].catalog_item, sequence.elements[-1].catalog_item)

    def test_add_snapshot_round_trip_and_all_previous_operations_remain_readable(self):
        sequence = source()
        sequence.elements[1].start_mirror, sequence.elements[1].end_mirror = False, True
        variant = insert_elements(sequence, items=[pose(10, bilateral=True)], count=1, request="Append")
        snapshot = json.loads(variant.model_dump_json())
        self.assertEqual(GeneratedSequence.from_snapshot(snapshot), variant)
        self.assertEqual(GeneratedSequence.from_snapshot(snapshot).model_dump(mode="json"), snapshot)
        for operation in ("generate", "regenerate", "replace", "swap"):
            with self.subTest(operation=operation):
                previous = {**sequence.model_dump(mode="json"), "operation": operation}
                self.assertEqual(GeneratedSequence.from_snapshot(previous).model_dump(mode="json"), previous)


if __name__ == "__main__":
    unittest.main()
