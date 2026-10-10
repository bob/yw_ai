"""Verify sequence selection stages without calling external services."""

import unittest
from collections import Counter
from random import Random
from unittest.mock import Mock, patch

from langchain_app.schemas import CatalogItemSchema
from langchain_app.services.sequence_generator import GeneratorSearchProgress, SequenceGenerator
from langchain_app.services.type_safe import TypeSafeSelectionError


def item(item_id, tags=(), *, image=True, bilateral=False):
    return CatalogItemSchema(
        id=item_id,
        uuid=None,
        name=f"Pose {item_id}",
        bilateral=bilateral,
        image_url="https://example.com/pose.png" if image else None,
        catalog_tag_names=list(tags),
    )


class SequenceGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Mock()
        self.jev = Mock()
        self.generator = SequenceGenerator(self.catalog, self.jev)

    def test_random_start_is_yielded_before_jev_and_passed_in_history(self):
        first = item(1, ["Other/Starting"])
        second = item(2, ["Other/Starting"])
        next_pose = item(3, ["Other/Finishing"])
        self.catalog.search.return_value = [
            first, second, next_pose, item(4, ["Other/Starting"], image=False)
        ]
        self.jev.choose_next_catalog_item.return_value = next_pose
        sequence = self.generator.generate("gentle practice", language="en", count=2)
        with patch(
            "langchain_app.services.sequence_generator.choice", return_value=second
        ) as choose:
            self.assertEqual(next(sequence), second)
            choose.assert_called_once_with([first, second])
        self.jev.choose_next_catalog_item.assert_not_called()
        self.assertEqual(next(sequence), next_pose)
        call = self.jev.choose_next_catalog_item.call_args
        self.assertEqual(call.kwargs["selected_items"][0], second)
        self.assertEqual(call.kwargs["elements_count"], 2)
        self.assertEqual(call.kwargs["request"], "gentle practice")
        self.assertEqual(list(sequence), [])
        self.jev.choose_next_catalog_item.assert_called_once()

    def test_single_pose_does_not_call_jev(self):
        starting = item(1, ["Other/Starting"])
        self.catalog.search.return_value = [starting]
        self.assertEqual(
            list(self.generator.generate("practice", language="en", count=1)),
            [starting],
        )
        self.jev.choose_next_catalog_item.assert_not_called()

    def test_explicit_start_takes_priority_without_requiring_starting_tag(self):
        requested = item(42)
        finishing = item(43, ["Other/Finishing"])
        for starts in ([], [item(1, ["Other/Starting"])]):
            with self.subTest(starts=starts):
                self.catalog.search.return_value = [*starts, requested, finishing]
                self.jev.choose_next_catalog_item.side_effect = lambda candidates, **kwargs: candidates[0]
                with patch("langchain_app.services.sequence_generator.choice") as choose, patch(
                    "langchain_app.services.sequence_generator.choices",
                ) as draw:
                    result = list(self.generator.generate(
                        "Start with Pose 42", language="en", count=2, starting_catalog_item_id=42,
                    ))
                self.assertEqual(result, [requested, finishing])
                choose.assert_not_called()
                draw.assert_not_called()
                self.assertNotIn(requested, self.jev.choose_next_catalog_item.call_args.args[0])

    def test_unavailable_explicit_start_fails_without_random_substitution(self):
        for requested in ([], [item(42, image=False)]):
            with self.subTest(requested=requested):
                self.catalog.search.return_value = [item(1, ["Other/Starting"]), *requested]
                with self.assertRaisesRegex(ValueError, "Starting catalog item 42"):
                    list(self.generator.generate(
                        "Start with Pose 42", language="en", count=1, starting_catalog_item_id=42,
                    ))
        self.jev.choose_next_catalog_item.assert_not_called()

    def test_random_start_position_probabilities_do_not_depend_on_group_size(self):
        lying = [item(n, ["Other/Starting", "Position/Lying on the back"]) for n in range(1, 21)]
        sitting = [item(21, ["Other/Starting", "Position/Sitting"])]
        standing = [item(n, ["Other/Starting", "Position/Standing"]) for n in range(22, 25)]
        self.catalog.search.return_value = [*lying, *sitting, *standing]
        rng = Random(42)
        positions = Counter()
        with patch("langchain_app.services.sequence_generator.choices", side_effect=rng.choices), patch(
            "langchain_app.services.sequence_generator.choice", side_effect=rng.choice,
        ):
            for _ in range(3000):
                start = next(self.generator.generate("practice", language="en", count=1))
                positions[start.catalog_tag_names[-1]] += 1
        for tag, probability in (
            ("Position/Lying on the back", 0.1), ("Position/Sitting", 0.6), ("Position/Standing", 0.3),
        ):
            self.assertAlmostEqual(positions[tag] / 3000, probability, delta=0.03)
        self.jev.choose_next_catalog_item.assert_not_called()

    def test_random_start_uses_available_position_groups(self):
        sitting = item(1, ["Other/Starting", "Position/Sitting on the knees"])
        standing = item(2, ["Other/Starting", "Position/Standing"])
        unclassified = item(3, ["Other/Starting"])
        self.catalog.search.return_value = [sitting, standing, unclassified]
        rng = Random(42)
        selected = Counter()
        with patch("langchain_app.services.sequence_generator.choices", side_effect=rng.choices), patch(
            "langchain_app.services.sequence_generator.choice", side_effect=rng.choice,
        ):
            for _ in range(2000):
                selected[next(self.generator.generate("practice", language="en", count=1)).id] += 1
        self.assertEqual(selected[unclassified.id], 0)
        self.assertAlmostEqual(selected[sitting.id] / 2000, 2 / 3, delta=0.03)
        self.assertAlmostEqual(selected[standing.id] / 2000, 1 / 3, delta=0.03)

    def test_lying_start_requires_supine_second_pose_even_when_it_is_last(self):
        for tag in ("Position/Lying on the back", "Position/Lying on the side", "Position/Lying on the stomach"):
            for explicit in (False, True):
                with self.subTest(tag=tag, explicit=explicit):
                    starting = item(1, ["Other/Starting", tag])
                    supine = item(2, ["Position/Lying on the back"])
                    self.catalog.search.return_value = [
                        starting, item(3, ["Other/Finishing", "Position/Sitting"]), supine,
                    ]
                    self.jev.choose_next_catalog_item.side_effect = lambda candidates, **kwargs: candidates[0]
                    result = list(self.generator.generate(
                        "practice", language="en", count=2,
                        starting_catalog_item_id=1 if explicit else None,
                    ))
                    self.assertEqual(result, [starting, supine])
                    self.assertEqual(self.jev.choose_next_catalog_item.call_args.args[0], [supine])

    def test_standing_start_excludes_all_lying_positions_from_second_pose(self):
        starting = item(1, ["Other/Starting", "Position/Standing"])
        sitting = item(2, ["Position/Sitting"])
        self.catalog.search.return_value = [starting, sitting, *[
            item(n, ["Other/Finishing", tag]) for n, tag in enumerate((
                "Position/Lying on the back", "Position/Lying on the side", "Position/Lying on the stomach",
            ), 3)
        ]]
        self.jev.choose_next_catalog_item.side_effect = lambda candidates, **kwargs: candidates[0]
        self.assertEqual(list(self.generator.generate("practice", language="en", count=2)), [starting, sitting])
        self.assertEqual(self.jev.choose_next_catalog_item.call_args.args[0], [sitting])

    def test_second_pose_restriction_is_applied_before_sampling_and_only_to_second_pose(self):
        starting = item(1, ["Other/Starting", "Position/Lying on the stomach"])
        supine = item(2, ["Position/Lying on the back"])
        other = [item(n, ["Position/Standing"]) for n in range(3, 15)]
        self.catalog.search.return_value = [starting, *other, supine]
        batches = []

        def select(candidates, **kwargs):
            batches.append(list(candidates))
            return candidates[0]

        self.jev.choose_next_catalog_item.side_effect = select
        with patch("langchain_app.services.sequence_generator.sample", side_effect=lambda population, k: population[:k]):
            result = list(self.generator.generate("practice", language="en", count=4))
        self.assertEqual(result[:2], [starting, supine])
        self.assertEqual(batches[0], [supine])
        self.assertEqual(batches[1], other[:10])

    def test_missing_allowed_second_pose_fails_without_relaxing_constraint(self):
        for start_tag, next_tag in (
            ("Position/Lying on the back", "Position/Sitting"),
            ("Position/Standing", "Position/Lying on the back"),
        ):
            with self.subTest(start_tag=start_tag):
                starting = item(1, ["Other/Starting", start_tag])
                self.catalog.search.return_value = [starting, item(2, [next_tag])]
                sequence = self.generator.generate("practice", language="en", count=2)
                self.assertEqual(next(sequence), starting)
                with self.assertRaisesRegex(TypeSafeSelectionError, "second pose"):
                    next(sequence)
        self.jev.choose_next_catalog_item.assert_not_called()

    def test_intermediate_batches_are_sampled_locally_without_repeats(self):
        starting = item(1, ["Other/Starting"])
        remaining = [item(i) for i in range(2, 14)] + [
            item(14, ["Other/Finishing"])
        ]
        self.catalog.search.return_value = [
            starting, *remaining, item(15, image=False)
        ]
        batches = []
        histories = []

        def select(candidates, **kwargs):
            batches.append(list(candidates))
            histories.append(list(kwargs["selected_items"]))
            return candidates[0]

        self.jev.choose_next_catalog_item.side_effect = select
        with patch(
            "langchain_app.services.sequence_generator.sample",
            side_effect=lambda population, k: population[:k],
        ) as draw:
            result = list(self.generator.generate("practice", language="en", count=4))

        self.catalog.search.assert_called_once_with(
            with_tags=True, with_rigman=False, limit=0, rand=True
        )
        self.assertEqual(draw.call_count, 2)
        self.assertEqual(draw.call_args_list[0].args, (remaining, 10))
        self.assertEqual(draw.call_args_list[1].args, (remaining[1:], 10))
        self.assertEqual(batches[:2], [remaining[:10], remaining[1:11]])
        self.assertEqual(histories, [result[:1], result[:2], result[:3]])
        self.assertEqual(batches[-1], [remaining[-1]])

    def test_small_catalog_samples_all_remaining_items(self):
        starting = item(1, ["Other/Starting"])
        remaining = [item(2), item(3, ["Other/Finishing"])]
        self.catalog.search.return_value = [starting, *remaining]
        self.jev.choose_next_catalog_item.side_effect = remaining
        with patch(
            "langchain_app.services.sequence_generator.sample", return_value=remaining
        ) as draw:
            result = list(self.generator.generate("practice", language="en", count=3))
        self.assertEqual(result, [starting, *remaining])
        draw.assert_called_once_with(remaining, 2)
        self.catalog.search.assert_called_once_with(
            with_tags=True, with_rigman=False, limit=0, rand=True
        )

    def test_last_pose_uses_all_eligible_finishing_items(self):
        starting = item(1, ["Other/Starting", "Other/Finishing"])
        finishing = [item(i, ["Other/Finishing"]) for i in range(2, 14)]
        self.catalog.search.return_value = [
            starting, *finishing, item(14),
            item(15, ["Other/Finishing"], image=False),
        ]
        self.jev.choose_next_catalog_item.return_value = finishing[-1]
        sequence = self.generator.generate("practice", language="en", count=2)
        with patch(
            "langchain_app.services.sequence_generator.choice", return_value=starting
        ), patch("langchain_app.services.sequence_generator.sample") as draw:
            self.assertEqual(list(sequence), [starting, finishing[-1]])
        draw.assert_not_called()
        self.assertEqual(self.jev.choose_next_catalog_item.call_args.args, (finishing,))
        self.catalog.search.assert_called_once_with(
            with_tags=True, with_rigman=False, limit=0, rand=True
        )

    def test_missing_unused_finishing_pose_falls_back_to_available_items(self):
        for catalog in (
            [item(1, ["Other/Starting"]), item(2)],
            [item(1, ["Other/Starting", "Other/Finishing"]), item(2)],
            [item(1, ["Other/Starting"]),
             item(2, ["Other/Finishing"], image=False), item(3)],
        ):
            with self.subTest(catalog=catalog):
                self.jev.reset_mock()
                self.catalog.search.return_value = catalog
                alternative = catalog[-1]
                self.jev.choose_next_catalog_item.return_value = alternative
                sequence = self.generator.generate("practice", language="en", count=2)
                self.assertEqual(next(sequence), catalog[0])
                self.assertEqual(list(sequence), [alternative])
                self.jev.choose_next_catalog_item.assert_called_once()
                selection = self.jev.choose_next_catalog_item.call_args
                self.assertEqual(selection.args, ([alternative],))
                self.assertEqual(selection.kwargs["elements_count"], 2)
                self.assertEqual(selection.kwargs["request"], "practice")
                self.assertEqual(selection.kwargs["selected_items"][0], catalog[0])

    def test_missing_eligible_start_fails_before_jev(self):
        for catalog in (
            [],
            [item(1)],
            [item(1, ["Other/Starting"], image=False)],
        ):
            with self.subTest(catalog=catalog):
                self.catalog.search.return_value = catalog
                with self.assertRaisesRegex(ValueError, "Other/Starting"):
                    list(self.generator.generate("practice", language="en", count=2))
        self.jev.choose_next_catalog_item.assert_not_called()

    def test_zero_count_does_not_call_services(self):
        self.assertEqual(
            list(self.generator.generate("practice", language="en", count=0)), []
        )
        self.catalog.search.assert_not_called()
        self.jev.choose_next_catalog_item.assert_not_called()

    def test_negative_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "negative"):
            list(self.generator.generate("practice", language="en", count=-1))
        self.catalog.search.assert_not_called()

    def test_duration_accounts_for_mirrored_poses_including_the_start(self):
        poses = [
            item(1, ["Other/Starting"], bilateral=True), item(2),
            item(3, bilateral=True), item(4, ["Other/Finishing"]), item(5), item(6),
        ]
        self.catalog.search.return_value = poses
        self.jev.choose_next_catalog_item.side_effect = lambda candidates, **kwargs: candidates[0]
        with patch("langchain_app.services.sequence_generator.sample", side_effect=lambda population, k: population[:k]):
            result = list(self.generator.generate(
                "four minutes", language="en", count=6, duration_seconds=240,
            ))
        self.assertEqual(result, poses[:4])
        self.assertEqual(sum(80 if pose.bilateral else 40 for pose in result), 240)
        self.assertEqual(
            [call.kwargs["elements_count"] for call in self.jev.choose_next_catalog_item.call_args_list],
            [5, 5, 4],
        )

    def test_duration_with_only_ordinary_poses_keeps_the_original_count(self):
        poses = [item(1, ["Other/Starting"]), *[item(n) for n in range(2, 15)],
                 item(15, ["Other/Finishing"])]
        self.catalog.search.return_value = poses
        self.jev.choose_next_catalog_item.side_effect = lambda candidates, **kwargs: candidates[0]
        with patch("langchain_app.services.sequence_generator.sample", side_effect=lambda population, k: population[:k]):
            result = list(self.generator.generate("ten minutes", language="en", count=15, duration_seconds=600))
        self.assertEqual(len(result), 15)
        self.assertEqual(result[-1], poses[-1])

    def test_duration_with_only_mirrored_poses_uses_half_as_many_elements(self):
        poses = [item(1, ["Other/Starting"], bilateral=True),
                 *[item(n, bilateral=True) for n in range(2, 16)]]
        self.catalog.search.return_value = poses
        self.jev.choose_next_catalog_item.side_effect = lambda candidates, **kwargs: candidates[0]
        result = list(self.generator.generate("ten minutes", language="en", count=15, duration_seconds=600))
        self.assertEqual(len(result), 8)
        self.assertEqual(sum(80 if pose.bilateral else 40 for pose in result), 640)

    def test_mirrored_finishing_pose_can_fill_the_last_two_time_units(self):
        first = item(1, ["Other/Starting"])
        finishing = item(3, ["Other/Finishing"], bilateral=True)
        self.catalog.search.return_value = [first, item(2), finishing]
        self.jev.choose_next_catalog_item.side_effect = lambda candidates, **kwargs: candidates[0]
        with patch("langchain_app.services.sequence_generator.sample") as draw:
            result = list(self.generator.generate("two minutes", language="en", count=3, duration_seconds=120))
        self.assertEqual(result, [first, finishing])
        self.assertEqual(self.jev.choose_next_catalog_item.call_args.kwargs["elements_count"], 2)
        draw.assert_not_called()

    def test_explicit_count_keeps_mirrored_poses_as_single_elements(self):
        poses = [item(1, ["Other/Starting"], bilateral=True),
                 item(2, bilateral=True), item(3, ["Other/Finishing"], bilateral=True)]
        self.catalog.search.return_value = poses
        self.jev.choose_next_catalog_item.side_effect = lambda candidates, **kwargs: candidates[0]
        result = list(self.generator.generate("three poses", language="en", count=3))
        self.assertEqual(len(result), 3)
        self.assertEqual({pose.id for pose in result}, {1, 2, 3})
        self.assertTrue(all(pose.bilateral for pose in result))

    def test_duration_rounds_halves_up_and_always_yields_at_least_one_pose(self):
        poses = [item(1, ["Other/Starting"]), item(2, ["Other/Finishing"])]
        self.catalog.search.return_value = poses
        self.jev.choose_next_catalog_item.side_effect = lambda candidates, **kwargs: candidates[0]
        for seconds, expected_count in ((1, 1), (59, 1), (60, 2)):
            with self.subTest(seconds=seconds):
                result = list(self.generator.generate("short practice", language="en", count=2, duration_seconds=seconds))
                self.assertEqual(len(result), expected_count)

    def test_short_duration_counts_the_explicit_mirrored_start(self):
        first = item(42, bilateral=True)
        self.catalog.search.return_value = [first, item(43)]
        result = list(self.generator.generate(
            "one minute, start with Pose 42", language="en", count=2,
            starting_catalog_item_id=42, duration_seconds=60,
        ))
        self.assertEqual(result, [first])
        self.jev.choose_next_catalog_item.assert_not_called()

    def test_count_limit_allows_45_and_caps_long_duration_at_45(self):
        poses = [item(1, ["Other/Starting"]), *[item(n) for n in range(2, 46)], item(46, ["Other/Finishing"])]
        self.catalog.search.return_value = poses
        self.jev.choose_next_catalog_item.side_effect = lambda candidates, **kwargs: candidates[0]
        for duration in (None, 7200):
            with self.subTest(duration=duration), patch(
                "langchain_app.services.sequence_generator.sample", side_effect=lambda population, k: population[:k],
            ):
                result = list(self.generator.generate("long practice", language="en", count=45, duration_seconds=duration))
                self.assertEqual(len(result), 45)
                self.assertEqual(result[-1], poses[-1])

    def test_count_above_45_is_rejected_before_search_or_progress(self):
        observer = Mock()
        with self.assertRaisesRegex(ValueError, "must not exceed 45"):
            list(self.generator.generate("46 poses", language="en", count=46, progress_observer=observer))
        self.catalog.search.assert_not_called()
        self.jev.choose_next_catalog_item.assert_not_called()
        observer.assert_not_called()

    def test_invalid_duration_is_rejected_before_search(self):
        for duration in (0, -1, float("inf"), float("nan")):
            with self.subTest(duration=duration), self.assertRaisesRegex(ValueError, "positive and finite"):
                list(self.generator.generate("practice", language="en", count=1, duration_seconds=duration))
        self.catalog.search.assert_not_called()

    def test_optional_observer_brackets_search_and_counts_only_usable_candidates(self):
        observed = []
        first = item(1, ["Other/Starting"])

        def search(**kwargs):
            self.assertEqual(observed, [GeneratorSearchProgress("started")])
            return [first, item(2, image=False)]

        self.catalog.search.side_effect = search
        sequence = self.generator.generate("Practice", language="en", count=1, progress_observer=observed.append)
        self.assertEqual(observed, [])
        self.assertEqual(next(sequence), first)
        self.assertEqual(observed, [
            GeneratorSearchProgress("started"), GeneratorSearchProgress("finished", candidate_count=1),
        ])
        self.assertEqual(list(sequence), [])

    def test_observer_reports_search_failure_before_exception_is_propagated(self):
        observed = []
        self.catalog.search.side_effect = ValueError("Cannot load catalog")
        with self.assertRaisesRegex(ValueError, "Cannot load catalog"):
            list(self.generator.generate("Practice", language="en", count=1, progress_observer=observed.append))
        self.assertEqual(observed, [
            GeneratorSearchProgress("started"), GeneratorSearchProgress("finished", outcome="error"),
        ])

    def test_observer_does_not_invent_search_for_zero_or_invalid_count(self):
        observer = Mock()
        self.assertEqual(list(self.generator.generate("Practice", language="en", count=0,
                                                      progress_observer=observer)), [])
        with self.assertRaises(ValueError):
            list(self.generator.generate("Practice", language="en", count=-1, progress_observer=observer))
        observer.assert_not_called()
        self.catalog.search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
