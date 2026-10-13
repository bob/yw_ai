"""Verify the selection context sent to TypeSafe without external calls."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_app.services.type_safe import TypeSafeSelectionError, TypeSafeService
from test_sequence_generator import item


class TypeSafeServiceTests(unittest.TestCase):
    def test_second_pose_judgment_receives_first_pose_and_transition_rules(self):
        first = item(1, ["Position/Standing"])
        candidates = [item(2, ["Position/Sitting"]), item(3, ["Position/Standing"])]
        classifier = Mock()
        classifier.invoke.return_value = SimpleNamespace(choices={
            "next_yoga_pose": SimpleNamespace(choice="3"),
        })
        service = TypeSafeService(classifier=classifier)
        selected = service.choose_next_catalog_item(
            candidates, selected_items=[first], request="gentle practice", elements_count=3,
        )
        self.assertEqual(selected, candidates[1])
        payload = classifier.invoke.call_args.args[0]
        self.assertEqual(payload["state"]["sequence_progress"]["next_element_position"], 2)
        self.assertEqual(payload["state"]["selected_catalog_items"][0]["catalog_tag_names"], ["Position/Standing"])
        rules = " ".join(payload["state"]["rules"])
        self.assertIn("the second pose must have the Position/Lying on the back tag", rules)
        self.assertIn("prefer a second pose with Position/Standing", rules)
        self.assertIn("the second pose must not have any Position/Lying tag", rules)
        self.assertEqual(set(payload["questions"]["next_yoga_pose"].criteria), {"2", "3"})

    def test_shared_choice_excludes_uuid_alias_and_shortcuts_only_unused_candidate(self):
        first = item(1).model_copy(update={"uuid": "abc"})
        alias = item(2).model_copy(update={"uuid": "ABC"})
        next_pose = item(3)
        classifier = Mock()
        service = TypeSafeService(classifier=classifier)
        selected = service.choose_next_catalog_item(
            [first, alias, next_pose], selected_items=[first], request="practice", elements_count=2,
        )
        self.assertEqual(selected, next_pose)
        classifier.invoke.assert_not_called()

    def test_shared_choice_rejects_empty_pool_missing_answer_and_unknown_id(self):
        classifier = Mock()
        service = TypeSafeService(classifier=classifier)
        with self.assertRaises(TypeSafeSelectionError):
            service.choose_next_catalog_item([], selected_items=[], request="practice", elements_count=1)
        classifier.invoke.assert_not_called()
        for choices in ({}, {"next_yoga_pose": SimpleNamespace(choice="999")}):
            with self.subTest(choices=choices), self.assertRaises(TypeSafeSelectionError):
                classifier.invoke.return_value = SimpleNamespace(choices=choices)
                service.choose_next_catalog_item(
                    [item(1), item(2)], selected_items=[], request="practice", elements_count=1,
                )
