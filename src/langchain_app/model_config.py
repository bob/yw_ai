"""Shared OpenAI model defaults for the app, examples, and local fixtures."""

from typing import TypedDict

DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_CHAT_MODEL = f"openai:{DEFAULT_OPENAI_MODEL}"


class ChatModelOptions(TypedDict, total=False):
    reasoning_effort: str
    use_responses_api: bool


def chat_model_options(model: str) -> ChatModelOptions:
    """Luna supports Chat Completions tools with reasoning disabled."""
    return {"reasoning_effort": "none", "use_responses_api": False} if model in {
        DEFAULT_OPENAI_MODEL, DEFAULT_CHAT_MODEL,
    } else {}
