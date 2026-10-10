"""Verify sequence selection stages without calling external services."""

import unittest
from unittest.mock import Mock, patch

from langchain_app.schemas import CatalogItemSchema
from langchain_app.services.sequence_generator import GeneratorSearchProgress, SequenceGenerator


def item(item_id, tags=(), *, image=True):
    return CatalogItemSchema(
        id=item_id,
        uuid=None,
        name=f"Pose {item_id}",
        bilateral=False,
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
