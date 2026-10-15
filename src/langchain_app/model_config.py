"""Shared OpenAI model defaults for the app, examples, and local fixtures."""

DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_CHAT_MODEL = f"openai:{DEFAULT_OPENAI_MODEL}"


def chat_model_options(model: str) -> dict[str, str | bool]:
    """Luna supports Chat Completions tools with reasoning disabled."""
    return {"reasoning_effort": "none", "use_responses_api": False} if model in {
        DEFAULT_OPENAI_MODEL, DEFAULT_CHAT_MODEL,
    } else {}
