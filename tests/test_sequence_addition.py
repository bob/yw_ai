"""Exercise atomic insertion selection and real TypeSafe payloads without APIs."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

from langchain_typesafe.client import TypeSafeError

from langchain_app.schemas import GeneratedSequence
from langchain_app.schemas.catalog import CatalogItemSearchResult
from langchain_app.services.catalog_item import CatalogItemApiError
from langchain_app.services.sequence_generator import GeneratorAdditionProgress, GeneratorSearchProgress, SequenceGenerator
from langchain_app.services.type_safe import TypeSafeSelectionError, TypeSafeService
from test_sequence_generator import item


def pose(item_id, tags=(), **kwargs):
    return item(item_id, tags, **kwargs).model_copy(update={"uuid": str(UUID(int=item_id))})


def source(*items):
    return GeneratedSequence.from_catalog_items(items, request="original gentle practice")


class SequenceAdditionTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Mock()
        self.classifier = Mock()
        self.classifier.invoke.side_effect = self.classify
        self.generator = SequenceGenerator(self.catalog, TypeSafeService(classifier=self.classifier))
        draw = patch(
            "langchain_app.services.sequence_generator.sample",
            side_effect=lambda population, k: population[:k],
        )
        self.draw = draw.start()
        self.addCleanup(draw.stop)

    @staticmethod
    def answer(choice):
        return SimpleNamespace(choices={"next_yoga_pose": SimpleNamespace(choice=str(choice))})

    def payloads(self):
        return [call.args[0] for call in self.classifier.invoke.call_args_list]

    def classify(self, payload):
        return self.answer(next(iter(payload["questions"]["next_yoga_pose"].criteria)))

    def test_internal_block_context_sees_full_source_and_evolves_after_each_choice(self):
        sequence = source(pose(1), pose(2), pose(3, ["Movement/Twist"]), pose(4, ["Other/Finishing"]))
        sequence.elements[1].start_mirror = True
        sequence.elements[1].end_mirror = False
        original = sequence.model_dump(mode="json")
        self.catalog.search.return_value = [
            pose(n, ["Other/Finishing"] if n == 12 else [], bilateral=True)
            for n in range(10, 14)
        ]
        additions = self.generator.add_elements(
            sequence, "add two challenging poses before the third", language="en", count=2,
            insert_before_element_id=sequence.elements[2].element_id,
        )
        self.assertEqual([entry.id for entry in additions], [10, 11])
        for offset, payload in enumerate(self.payloads()):
            state = payload["state"]
            self.assertEqual(state["mode"], "add")
            self.assertEqual(state["generation_request"], sequence.generation_request)
            self.assertEqual(state["request"], "add two challenging poses before the third")
            self.assertEqual(state["sequence_progress"], {
                "original_elements_count": 4, "added_elements_count": 2, "elements_count": 6,
                "selected_additions_count": offset, "remaining_additions_count": 2 - offset,
                "next_element_position": 3 + offset,
                "is_selecting_last_added_element": offset == 1, "is_selecting_last_element": False,
            })
            self.assertEqual([entry["catalog_item"]["id"] for entry in state["fixed_prefix"]], [1, 2])
            self.assertEqual([entry["catalog_item"]["id"] for entry in state["fixed_suffix"]], [3, 4])
            self.assertEqual([entry["original_order"] for entry in state["fixed_suffix"]], [3, 4])
            self.assertEqual(state["fixed_prefix"][1]["element_id"], str(sequence.elements[1].element_id))
            self.assertTrue(state["fixed_prefix"][1]["start_mirror"])
            self.assertFalse(state["fixed_prefix"][1]["end_mirror"])
            self.assertEqual([entry["id"] for entry in state["selected_additions"]], [10][:offset])
            self.assertEqual(state["fixed_suffix"][0]["catalog_item"]["catalog_tag_names"], ["Movement/Twist"])
            self.assertEqual(state["fixed_prefix"][0]["catalog_item"]["uuid"], sequence.elements[0].catalog_item.uuid)
            serialized = json.dumps(state)
            self.assertNotIn("image_url", serialized)
            self.assertNotIn("rigman", serialized)
            self.assertIn("current request takes priority", " ".join(state["rules"]))
        self.assertEqual(sequence.model_dump(mode="json"), original)
        additions[0].catalog_tag_names.append("changed")
        self.assertEqual(self.catalog.search.return_value[0].catalog_tag_names, [])

    def test_prepend_uses_typesafe_start_and_shared_second_pose_constraints(self):
        sequence = source(pose(1, ["Other/Starting", "Position/Sitting"]), pose(2))
        standing = pose(10, ["Other/Starting", "Position/Standing"])
        other_start = pose(11, ["Other/Starting", "Position/Sitting"])
        lying = pose(12, ["Position/Lying on the back"])
        nonlying = [pose(13, ["Position/Standing"]), pose(14, ["Position/Sitting"])]
        self.catalog.search.return_value = [standing, other_start, lying, *nonlying]
        original = sequence.model_dump(mode="json")
        with patch("langchain_app.services.sequence_generator._random_starting_item") as random_start:
            result = self.generator.add_elements(
                sequence, "prepend two poses", language="en", count=2,
                insert_before_element_id=sequence.elements[0].element_id,
            )
        random_start.assert_not_called()
        self.assertEqual(result, [standing, nonlying[0]])
        payloads = self.payloads()
        self.assertEqual(set(payloads[0]["questions"]["next_yoga_pose"].criteria), {"10", "11"})
        self.assertEqual(set(payloads[1]["questions"]["next_yoga_pose"].criteria), {"13", "14"})
        self.assertEqual(payloads[1]["state"]["fixed_suffix"][0]["catalog_item"]["id"], 1)
        self.assertEqual(sequence.model_dump(mode="json"), original)

    def test_single_prepend_checks_compatibility_with_fixed_second_pose(self):
        starts = [
            pose(10, ["Other/Starting", "Position/Lying on the stomach"]),
            pose(11, ["Other/Starting", "Position/Standing"]),
            pose(12, ["Other/Starting", "Position/Sitting"]),
        ]
        for tag, allowed_ids in (
            ("Position/Standing", {"11", "12"}),
            ("Position/Lying on the back", {"10", "12"}),
            ("Position/Lying on the stomach", None),
        ):
            with self.subTest(tag=tag):
                self.classifier.reset_mock()
                sequence = source(pose(1, [tag]))
                self.catalog.search.return_value = starts if allowed_ids else starts[:2]
                if allowed_ids is None:
                    with self.assertRaises(TypeSafeSelectionError):
                        self.generator.add_elements(
                            sequence, "prepend", language="en", count=1,
                            insert_before_element_id=sequence.elements[0].element_id,
                        )
                    self.assertEqual(self.payloads(), [])
                else:
                    result = self.generator.add_elements(
                        sequence, "prepend", language="en", count=1,
                        insert_before_element_id=sequence.elements[0].element_id,
                    )
                    self.assertEqual(set(self.payloads()[0]["questions"]["next_yoga_pose"].criteria), allowed_ids)
                    self.assertIn(str(result[0].id), allowed_ids)

    def test_inserting_second_pose_filters_before_sampling_and_finishing_preference(self):
        for tag in ("Position/Lying on the back", "Position/Lying on the side", "Position/Lying on the stomach"):
            with self.subTest(tag=tag):
                sequence = source(pose(1, [tag], bilateral=True))
                supine = pose(10, ["Position/Lying on the back"])
                self.catalog.search.return_value = [pose(11, ["Other/Finishing", "Position/Standing"]), supine]
                result = self.generator.add_elements(sequence, "append", language="en", count=1)
                self.assertEqual(result, [supine])
        self.assertEqual(self.payloads(), [])
        self.draw.assert_not_called()

    def test_inserting_after_second_does_not_revalidate_old_incompatible_pair(self):
        sequence = source(pose(1, ["Position/Standing"]), pose(2, ["Position/Lying on the stomach"]), pose(3))
        self.catalog.search.return_value = [pose(10, ["Position/Lying on the side"])]
        result = self.generator.add_elements(
            sequence, "insert", language="en", count=1,
            insert_before_element_id=sequence.elements[2].element_id,
        )
        self.assertEqual([entry.id for entry in result], [10])

    def test_inserting_after_standing_start_excludes_lying_poses_with_fixed_suffix(self):
        sequence = source(pose(1, ["Other/Starting", "Position/Standing"]), pose(2))
        self.catalog.search.return_value = [
            pose(10, ["Position/Lying on the back"]), pose(11, ["Position/Lying on the stomach"]),
            pose(12, ["Position/Standing"]), pose(13, ["Position/Sitting"]),
        ]
        result = self.generator.add_elements(
            sequence, "insert after first", language="en", count=1,
            insert_before_element_id=sequence.elements[1].element_id,
        )
        self.assertEqual([entry.id for entry in result], [12])
        self.assertEqual(set(self.payloads()[0]["questions"]["next_yoga_pose"].criteria), {"12", "13"})
        self.assertFalse(self.payloads()[0]["state"]["sequence_progress"]["is_selecting_last_element"])

    def test_existing_repetitions_and_unavailable_source_items_are_kept_as_context(self):
        sequence = source(pose(1, image=False), pose(1, image=False), pose(2))
        original = sequence.model_dump(mode="json")
        self.catalog.search.return_value = [pose(1), pose(10), pose(11)]
        result = self.generator.add_elements(sequence, "append", language="en", count=1)
        self.assertEqual([entry.id for entry in result], [10])
        self.assertEqual([entry["catalog_item"]["id"] for entry in self.payloads()[0]["state"]["fixed_prefix"]], [1, 1, 2])
        self.assertEqual(sequence.model_dump(mode="json"), original)

    def test_identity_filter_includes_suffix_and_prior_additions_before_batching(self):
        sequence = source(pose(1), pose(2), pose(3))
        suffix_alias = pose(10).model_copy(update={"uuid": sequence.elements[2].catalog_item.uuid.upper()})
        first = pose(11)
        addition_alias = pose(12).model_copy(update={"uuid": first.uuid.upper()})
        valid = [first, addition_alias, *[pose(n) for n in range(20, 32)]]
        self.catalog.search.return_value = [pose(3), suffix_alias, pose(13, ["Other/Starting"]), *valid]
        self.generator.add_elements(
            sequence, "add two", language="en", count=2,
            insert_before_element_id=sequence.elements[2].element_id,
        )
        self.assertEqual(self.draw.call_args_list[0].args, (valid, 10))
        self.assertEqual(self.draw.call_args_list[1].args, (valid[2:], 10))
        self.assertEqual(set(self.payloads()[1]["questions"]["next_yoga_pose"].criteria), {str(n) for n in range(20, 30)})

    def test_append_finishing_pool_only_applies_to_last_pose_of_entire_sequence(self):
        sequence = source(pose(1), pose(2, ["Other/Finishing"]))
        ordinary = pose(10)
        finishing = [pose(n, ["Other/Finishing"]) for n in range(11, 23)]
        self.catalog.search.return_value = [ordinary, *finishing]
        result = self.generator.add_elements(sequence, "add two", language="en", count=2)
        self.assertEqual(result, [ordinary, finishing[0]])
        self.draw.assert_called_once_with([ordinary, *finishing], 10)
        states = [payload["state"]["sequence_progress"] for payload in self.payloads()]
        self.assertFalse(states[0]["is_selecting_last_element"])
        self.assertTrue(states[1]["is_selecting_last_element"])
        self.assertEqual(set(self.payloads()[1]["questions"]["next_yoga_pose"].criteria), {str(n) for n in range(11, 23)})

    def test_exact_addition_counts_fill_limit_without_duration_budget(self):
        for original_count, added_count in ((1, 44), (44, 1), (9, 36)):
            with self.subTest(original_count=original_count):
                sequence = source(*[pose(n, bilateral=True) for n in range(1, original_count + 1)])
                self.catalog.search.return_value = [pose(n, bilateral=True) for n in range(100, 100 + added_count)]
                result = self.generator.add_elements(sequence, "fill", language="en", count=added_count)
                self.assertEqual(len(result), added_count)
                self.assertEqual(len(sequence.elements) + len(result), 45)
                self.assertEqual(len({entry.id for entry in result}), added_count)

    def test_invalid_input_is_rejected_before_catalog_classifier_or_progress(self):
        sequence = source(pose(1))
        observer = Mock()
        cases = [
            (sequence, count, "add", None) for count in (0, -1, True, False, 1.5, "1", 45, 100)
        ] + [
            (source(), 1, "add", None), (sequence, 1, " \n ", None),
            (sequence, 1, "add", uuid4()),
            (source(*[pose(n) for n in range(1, 46)]), 1, "add", None),
        ]
        for selected, count, request, anchor in cases:
            with self.subTest(count=count, request=request, anchor=anchor), self.assertRaises(ValueError):
                self.generator.add_elements(
                    selected, request, language="en", count=count,
                    insert_before_element_id=anchor, progress_observer=observer,
                )
        self.catalog.search.assert_not_called()
        self.classifier.invoke.assert_not_called()
        observer.assert_not_called()

    def test_catalog_loading_uses_existing_settings_and_strips_rigman(self):
        good = pose(10)
        with_rigman = CatalogItemSearchResult(
            **good.model_dump(), rigman_data={"private": "pose"}, second_side_rigman_data={"private": "mirror"},
        )
        self.catalog.search.return_value = [with_rigman, item(11), pose(12, image=False), pose(13).model_copy(update={"uuid": " "}),
                                            pose(14).model_copy(update={"uuid": "invalid"})]
        observed = []
        sequence = source(pose(1), pose(2))
        result = self.generator.add_elements(sequence, "add", language="en", count=1, progress_observer=observed.append)
        self.assertEqual(result, [good])
        self.catalog.search.assert_called_once_with(with_tags=True, with_rigman=False, limit=0, rand=True)
        self.assertEqual(observed, [GeneratorAdditionProgress(0, 1), GeneratorSearchProgress("started"),
                                   GeneratorSearchProgress("finished", candidate_count=1), GeneratorAdditionProgress(1, 1)])
        self.assertEqual(self.payloads(), [])

    def test_search_failure_keeps_error_type_and_reports_catalog_progress(self):
        error = CatalogItemApiError("offline")
        self.catalog.search.side_effect = error
        observed = []
        with self.assertRaises(CatalogItemApiError) as raised:
            self.generator.add_elements(source(pose(1)), "add", language="en", count=1, progress_observer=observed.append)
        self.assertIs(raised.exception, error)
        self.assertEqual(observed, [GeneratorAdditionProgress(0, 1), GeneratorSearchProgress("started"),
                                   GeneratorSearchProgress("finished", outcome="error")])
        self.classifier.invoke.assert_not_called()

    def test_missing_start_second_pose_or_unused_candidates_aborts_whole_block(self):
        for selected, catalog, count, prepend in (
            (source(pose(1)), [], 1, False),
            (source(pose(1)), [pose(10)], 1, True),
            (source(pose(1)), [pose(10)], 2, False),
            (source(pose(1)), [pose(10, ["Other/Starting", "Position/Standing"]),
                               pose(11, ["Position/Lying on the back"])], 2, True),
            (source(pose(1, ["Position/Lying on the stomach"])), [pose(10, ["Position/Standing"])], 1, False),
        ):
            with self.subTest(count=count, prepend=prepend):
                self.catalog.search.return_value = catalog
                original = selected.model_dump(mode="json")
                with self.assertRaises(TypeSafeSelectionError):
                    self.generator.add_elements(
                        selected, "add", language="en", count=count,
                        insert_before_element_id=selected.elements[0].element_id if prepend else None,
                    )
                self.assertEqual(selected.model_dump(mode="json"), original)

    def test_classifier_failure_on_later_step_never_returns_partial_block(self):
        for failure in (TypeSafeError("unavailable"), None, SimpleNamespace(choices=None),
                        SimpleNamespace(choices={}), SimpleNamespace(choices={"next_yoga_pose": SimpleNamespace(choice=[])}),
                        self.answer(999)):
            with self.subTest(failure=failure):
                sequence = source(pose(1), pose(2))
                original = sequence.model_dump(mode="json")
                self.catalog.search.return_value = [pose(n) for n in range(10, 15)]
                self.classifier.invoke.side_effect = [self.answer(10), failure]
                with self.assertRaises(TypeSafeSelectionError):
                    self.generator.add_elements(sequence, "add three", language="en", count=3)
                self.assertEqual(sequence.model_dump(mode="json"), original)


if __name__ == "__main__":
    unittest.main()
