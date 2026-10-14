"""OpenAI cache routing and provider-reported usage for one assistant session."""

import json
import logging
import os
import re
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from urllib.parse import urlparse
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI

from langchain_app.schemas.assistant import AssistantState
from langchain_app.tools.context import AssistantToolContext

logger = logging.getLogger(__name__)


def prefix_version(prompt: str, tools: Sequence = (), *, context_instructions: str = "") -> str:
    """Hash model-visible instructions and ordered schemas, never session data."""
    payload = {"prompt": prompt, "context_instructions": context_instructions,
               "tools": [convert_to_openai_tool(tool) for tool in tools]}
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode()).hexdigest()[:12]


def openai_model_name(model: str | BaseChatModel) -> str | None:
    if isinstance(model, str):
        provider, _, name = model.partition(":")
        if not name:
            name, provider = provider, "openai" if model.startswith(("gpt-", "o1", "o3", "o4")) else ""
        base_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
        if provider != "openai":
            return None
    elif isinstance(model, ChatOpenAI):
        name = model.model_name
        base_url = model.openai_api_base or getattr(model.root_client, "base_url", None)
    else:
        return None
    if base_url and urlparse(str(base_url)).hostname != "api.openai.com":
        return None
    return name


def retention_values(name: str) -> tuple[str, ...]:
    """Conservative legacy compatibility; do not apply legacy defaults to new APIs."""
    name = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", name)
    if name in {"gpt-5.5", "gpt-5.5-pro"}:
        return ("24h",)
    if name in {
        "gpt-4.1", "gpt-5", "gpt-5-codex", "gpt-5.1", "gpt-5.1-codex",
        "gpt-5.1-codex-max", "gpt-5.1-codex-mini", "gpt-5.1-chat-latest",
        "gpt-5.2", "gpt-5.4",
    }:
        return ("in_memory", "24h")
    if name in {"gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini",
                "gpt-5-mini", "gpt-5-nano", "o1", "o3", "o3-mini", "o4-mini"}:
        return ("in_memory",)
    return ()


@dataclass(frozen=True)
class PromptCacheSettings:
    enabled: bool = True
    retention: str | None = None

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("ASSISTANT_PROMPT_CACHE_ENABLED must be true or false")
        if self.retention not in {None, "in_memory", "24h"}:
            raise ValueError("ASSISTANT_PROMPT_CACHE_RETENTION must be in_memory or 24h")

    @classmethod
    def from_environment(cls):
        enabled = os.getenv("ASSISTANT_PROMPT_CACHE_ENABLED", "true").strip().lower()
        if enabled not in {"true", "false"}:
            raise ValueError("ASSISTANT_PROMPT_CACHE_ENABLED must be true or false")
        return cls(enabled == "true", os.getenv("ASSISTANT_PROMPT_CACHE_RETENTION", "").strip() or None)

    def validate_model(self, model: str | BaseChatModel) -> None:
        name = openai_model_name(model)
        if self.enabled and name and self.retention and self.retention not in retention_values(name):
            raise ValueError(f"Unsupported prompt cache retention {self.retention!r} for {name}")

    def model_settings(self, model: BaseChatModel, key: str) -> dict:
        self.validate_model(model)
        name = openai_model_name(model)
        if not self.enabled or not name:
            return {}
        result = {"prompt_cache_key": key}
        supported = retention_values(name)
        retention = self.retention or (supported[0] if supported else None)
        if retention:
            result["prompt_cache_retention"] = retention
        return result


def reported_usage(message: AIMessage | None) -> dict:
    """Prefer normalized fields; raw Chat/Responses usage only fills missing fields."""
    normalized = (message.usage_metadata or {}) if message is not None else {}
    metadata = message.response_metadata if message is not None else {}
    raw = metadata.get("token_usage") or metadata.get("usage") or {}

    def first(*values):
        return next((value for value in values if value is not None), None)

    def count(value):
        return value if type(value) is int and value >= 0 else None

    result = {
        "input_tokens": count(first(normalized.get("input_tokens"), raw.get("prompt_tokens"), raw.get("input_tokens"))),
        "cached_input_tokens": count(first(
            (normalized.get("input_token_details") or {}).get("cache_read"),
            (raw.get("prompt_tokens_details") or raw.get("input_tokens_details") or {}).get("cached_tokens"),
        )),
        "output_tokens": count(first(normalized.get("output_tokens"), raw.get("completion_tokens"), raw.get("output_tokens"))),
        "reasoning_tokens": count(first(
            (normalized.get("output_token_details") or {}).get("reasoning"),
            (raw.get("completion_tokens_details") or raw.get("output_tokens_details") or {}).get("reasoning_tokens"),
        )),
    }
    if (result["input_tokens"] is not None and result["cached_input_tokens"] is not None
            and result["cached_input_tokens"] > result["input_tokens"]):
        result.update(cached_input_tokens=None, usage_error="cached_input_exceeds_input")
    result["uncached_input_tokens"] = (
        result["input_tokens"] - result["cached_input_tokens"]
        if result["input_tokens"] is not None and result["cached_input_tokens"] is not None else None
    )
    return result


def aggregate_usage(records: Sequence[dict]) -> dict:
    """Unknown usage stays unknown, including when only some calls report it."""
    totals = {"calls": len(records)}
    for field in ("input_tokens", "cached_input_tokens", "uncached_input_tokens", "output_tokens", "reasoning_tokens"):
        values = [record.get(field) for record in records]
        totals[field] = sum(values) if all(value is not None for value in values) else None
    return totals


def usage_cost(records: Sequence[dict], *, input_rate: float, cached_input_rate: float,
               output_rate: float) -> float | None:
    """USD (or another consistent currency), using supplied rates per million tokens."""
    totals = aggregate_usage(records)
    fields = ("uncached_input_tokens", "cached_input_tokens", "output_tokens")
    if any(totals[field] is None for field in fields):
        return None
    return sum(totals[field] * rate for field, rate in zip(
        fields, (input_rate, cached_input_rate, output_rate), strict=True,
    )) / 1_000_000


class PromptCacheSession:
    def __init__(self, settings: PromptCacheSettings):
        self.settings = settings
        self.session_key = uuid4().hex[:16]
        self._records: list[dict] = []

    @property
    def records(self) -> list[dict]:
        return deepcopy(self._records)

    def key(self, model: BaseChatModel, version: str, profile: str) -> str:
        name = getattr(model, "model_name", None) or model._llm_type
        return f"langchain-app:{name}:{version}:{profile}:{self.session_key}"

    def call[ResponseT: AIMessage | ModelResponse](
        self, model: BaseChatModel, version: str, profile: str, request_id: str | None,
        handler: Callable[[dict], ResponseT],
    ) -> ResponseT:
        settings = self.settings.model_settings(model, self.key(model, version, profile))
        record = {
            "request_id": request_id,
            "call_kind": "history_summary" if profile == "history-summary" else "assistant",
            "model": getattr(model, "model_name", None) or model._llm_type,
            "profile": profile, "prefix_version": version, **reported_usage(None),
        }
        try:
            response = handler(settings)
            message = response if isinstance(response, AIMessage) else next(
                (message for message in response.result if isinstance(message, AIMessage)), None,
            )
            record.update(reported_usage(message), status="ok", message_id=message.id if message else None)
            return response
        except Exception as exc:
            record.update(status="error", error_type=type(exc).__name__)
            raise
        finally:
            self._records.append(record)
            logger.info("Assistant model usage", extra={"model_usage": deepcopy(record)})


class PromptCachingMiddleware(AgentMiddleware[AssistantState, AssistantToolContext]):
    def __init__(self, session: PromptCacheSession, version: str):
        self.session, self.version = session, version

    def wrap_model_call(self, request: ModelRequest[AssistantToolContext],
                        handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        return self.session.call(
            request.model, self.version, "assistant", request.runtime.context.request_id,
            lambda settings: handler(request.override(model_settings={**request.model_settings, **settings})),
        )
