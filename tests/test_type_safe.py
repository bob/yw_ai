"""Verify the selection context sent to TypeSafe without external calls."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_app.services.type_safe import TypeSafeService
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
