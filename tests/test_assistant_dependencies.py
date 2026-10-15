"""Verify migration of the assistant model setting and injected catalog clients."""

import unittest
from unittest.mock import Mock, patch

from langchain_app.dependencies import get_assistant_agent


class AssistantDependenciesTests(unittest.TestCase):
    def test_model_setting_precedence_and_services_are_created_before_tools_run(self):
        for settings, expected in (
            ({"ASSISTANT_AGENT_MODEL": " openai:new ", "SEQUENCE_AGENT_MODEL": "openai:old"}, "openai:new"),
            ({"ASSISTANT_AGENT_MODEL": " ", "SEQUENCE_AGENT_MODEL": " openai:old "}, "openai:old"),
            ({"ASSISTANT_AGENT_MODEL": "", "SEQUENCE_AGENT_MODEL": ""}, "openai:gpt-6-luna"),
        ):
            with self.subTest(settings=settings), patch.dict("os.environ", settings), patch(
                "langchain_app.dependencies.get_sequence_generator", return_value=Mock()
            ) as generator, patch(
                "langchain_app.dependencies.get_catalog_item_service", return_value=Mock()
            ) as items, patch(
                "langchain_app.dependencies.get_catalog_tag_service", return_value=Mock()
            ) as tags, patch("langchain_app.dependencies.AssistantAgent") as agent:
                get_assistant_agent("https://example.com", "folder")
                self.assertEqual(agent.call_args.kwargs["model"], expected)
                self.assertIs(agent.call_args.kwargs["catalog_item_service"], items.return_value)
                self.assertIs(agent.call_args.kwargs["catalog_tag_service"], tags.return_value)
                self.assertIs(agent.call_args.kwargs["generator_factory"](), generator.return_value)
                for dependency in (generator, items, tags):
                    dependency.assert_called_once_with("https://example.com", "folder")
