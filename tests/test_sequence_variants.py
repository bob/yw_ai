"""Exercise variant history and navigation without external generation calls."""

import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from streamlit.testing.v1 import AppTest

from langchain_app.schemas import CatalogItemSchema

APP = Path(__file__).resolve().parents[1] / "src/langchain_app/generate_sequence.py"


def pose(number):
    return CatalogItemSchema(
        id=number, uuid=None, name=f"Pose {number}", bilateral=False,
        image_url=f"https://example.com/{number}.png", catalog_tag_names=[],
    )


class SequenceVariantTests(unittest.TestCase):
    def test_regenerate_appends_and_navigation_does_not_generate(self):
        generator = Mock()
        generator.generate.side_effect = [[pose(1)], [pose(2)], [pose(3)]]
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator):
            app = AppTest.from_file(str(APP)).run()
            self.assertEqual(len(app.chat_input), 0)
            app.button(key="generate-sequence").click().run()
            self.assertEqual(len(app.chat_input), 1)
            self.assertFalse(app.exception)
            app.button(key="assistant-sequence-0-regenerate").click().run()
            self.assertFalse(app.exception)
            message = app.session_state.messages[0]
            self.assertEqual(len(message["sequence_variants"]), 2)
            self.assertEqual(message["selected_sequence_index"], 1)
            app.button(key="assistant-sequence-0-previous").click().run()
            self.assertEqual(app.session_state.messages[0]["sequence"]["thumbnails"][0]["alt"], "Pose 1")
            app.button(key="assistant-sequence-0-regenerate").click().run()
            self.assertEqual(len(app.session_state.messages[0]["sequence_variants"]), 3)
            app.button(key="assistant-sequence-0-previous").click().run()
            app.button(key="assistant-sequence-0-next").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(generator.generate.call_args_list, [
                call("Generate a yoga sequence.", language="en", count=9)
            ] * 3)

    def test_generation_error_keeps_existing_variant(self):
        generator = Mock()
        generator.generate.side_effect = [[pose(1)], ValueError("generation failed")]
        with patch("langchain_app.dependencies.get_sequence_generator", return_value=generator):
            app = AppTest.from_file(str(APP)).run()
            app.button(key="generate-sequence").click().run()
            app.button(key="assistant-sequence-0-regenerate").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            message = app.session_state.messages[0]
            self.assertEqual(len(message["sequence_variants"]), 1)
            self.assertEqual(message["selected_sequence_index"], 0)
            self.assertEqual(message["sequence"]["thumbnails"][0]["alt"], "Pose 1")
