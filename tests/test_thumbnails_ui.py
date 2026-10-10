"""Verify shared thumbnail rendering and compatibility with sequence UI."""

import unittest
from html.parser import HTMLParser

from streamlit.testing.v1 import AppTest

from langchain_app.ui import AsanaThumbnail as PublicAsanaThumbnail
from langchain_app.ui.sequence import AsanaThumbnail, SequenceRenderer
from langchain_app.ui.thumbnails import ThumbnailRenderer


class MarkupParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


class ThumbnailTests(unittest.TestCase):
    def test_safe_markup_preserves_real_alt_url_and_sequence_mirror_flags(self):
        url = 'https://example.com/pose?a="quoted"&b=<value>'
        name = 'Pose "one" & <script>alert(1)</script>'
        for start, end in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(start=start, end=end):
                thumbnail = {"image_url": url, "alt": name, "start_mirror": start, "end_mirror": end, "uuid": None, "order": 1}
                parser = MarkupParser()
                parser.feed(SequenceRenderer._thumbnail_html(thumbnail))
                image, = [attrs for tag, attrs in parser.tags if tag == "img"]
                self.assertEqual(image, {"src": url, "alt": name})
                self.assertFalse(any(tag == "script" for tag, _ in parser.tags))
                spans = [attrs for tag, attrs in parser.tags if tag == "span"]
                self.assertEqual(len(spans), int(start) + int(end))
                self.assertTrue(all(span["aria-hidden"] == "true" for span in spans))
                self.assertEqual(any("--start" in span["class"] for span in spans), start)
                self.assertEqual(any("--end" in span["class"] for span in spans), end)

    def test_plain_thumbnail_needs_only_url_and_name_and_has_no_brackets(self):
        parser = MarkupParser()
        parser.feed(ThumbnailRenderer._thumbnail_html({"image_url": "https://example.com/pose.png", "alt": "Tree"}))
        self.assertFalse(any(tag == "span" for tag, _ in parser.tags))
        self.assertIs(PublicAsanaThumbnail, AsanaThumbnail)
        self.assertEqual(AsanaThumbnail.__required_keys__, {"uuid", "order", "image_url", "alt", "start_mirror", "end_mirror"})

    def test_sequence_and_plain_thumbnails_render_with_one_shared_style_and_fallback(self):
        def script():
            from langchain_app.ui.sequence import SequenceRenderer
            from langchain_app.ui.thumbnails import ThumbnailRenderer

            sequence = SequenceRenderer(thumbnail_size=80)
            sequence.render_styles()
            sequence.render({
                "info": {"id": 12, "uuid": None, "name": "Practice", "description": "Gentle", "language": "en", "elements_count": 2},
                "thumbnails": [
                    {"uuid": None, "order": 1, "image_url": "https://example.com/pose.png", "alt": "Tree", "start_mirror": True, "end_mirror": False},
                    {"uuid": None, "order": 2, "image_url": "", "alt": "Missing pose", "start_mirror": False, "end_mirror": False},
                ],
            }, key="sequence")
            ThumbnailRenderer().render([
                {"image_url": "https://example.com/pose.png", "alt": "Tree"},
                {"image_url": "", "alt": "Another missing pose"},
            ], key="plain")

        app = AppTest.from_function(script).run()
        self.assertFalse(app.exception)
        html = [node.proto.body for node in app.get("html")]
        styles = [body for body in html if "<style>" in body]
        self.assertEqual(len(styles), 1)
        self.assertIn("--thumbnail-size: 80px", styles[0])
        self.assertIn("object-fit: cover", styles[0])
        self.assertIn("border-radius: 10px", styles[0])
        self.assertEqual([caption.value for caption in app.caption], [
            "Sequence #12 · 2 poses · EN", "Missing pose", "Another missing pose",
        ])
        self.assertEqual([node.value for node in app.markdown], ["**Practice**", "Gentle"])
        self.assertEqual(len(html), 3)
        self.assertEqual(len(app.button), 0)
