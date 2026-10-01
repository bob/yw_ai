"""Verify the starting-pose stage without calling external services."""

import unittest
from unittest.mock import Mock, patch

from langchain_app.schemas import CatalogItemSchema
from langchain_app.services.sequence_generator import SequenceGenerator


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
        next_pose = item(3)
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
        remaining = [item(i) for i in range(2, 15)]
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

        self.catalog.search.assert_called_once_with(language="en")
        self.assertEqual(draw.call_count, 2)
        self.assertEqual(draw.call_args_list[0].args, (remaining, 10))
        self.assertEqual(draw.call_args_list[1].args, (remaining[1:], 10))
        self.assertEqual(batches[:2], [remaining[:10], remaining[1:11]])
        self.assertEqual(histories, [result[:1], result[:2], result[:3]])
        self.assertEqual(batches[-1], [starting, *remaining])

    def test_small_catalog_samples_all_remaining_items(self):
        starting = item(1, ["Other/Starting"])
        remaining = [item(2), item(3)]
        self.catalog.search.return_value = [starting, *remaining]
        self.jev.choose_next_catalog_item.side_effect = remaining
        with patch(
            "langchain_app.services.sequence_generator.sample", return_value=remaining
        ) as draw:
            result = list(self.generator.generate("practice", language="en", count=3))
        self.assertEqual(result, [starting, *remaining])
        draw.assert_called_once_with(remaining, 2)
        self.catalog.search.assert_called_once_with(language="en")

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


if __name__ == "__main__":
    unittest.main()
