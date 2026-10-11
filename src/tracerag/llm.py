"""A minimal OpenAI-compatible chat client.

Implemented on ``urllib`` so the core package has no HTTP dependency. Any
provider that speaks the ``/chat/completions`` shape works: OpenAI, DeepSeek,
Moonshot, Qwen, OpenRouter, vLLM, Ollama and friends. When no API key is
configured the client reports itself as disabled and the pipeline falls back
to its offline deterministic path.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

# USD per one million tokens, matched by substring against the model name.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    "qwen-plus": (0.40, 1.20),
    "moonshot-v1-8k": (0.17, 0.17),
}


class LLMError(RuntimeError):
    """Raised when a provider call fails or returns an unusable payload."""


@dataclass
class LLMConfig:
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    timeout: float = 90.0
    temperature: float = 0.0
    max_tokens: int = 900

    @classmethod
    def from_env(cls, **overrides: Any) -> "LLMConfig":
        config = cls(
            model=os.getenv("TRACERAG_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL,
            api_key=os.getenv("TRACERAG_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
            base_url=os.getenv("TRACERAG_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or DEFAULT_BASE_URL,
        )
        for key, value in overrides.items():
            if value not in (None, ""):
                setattr(config, key, value)
        return config

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    lowered = (model or "").lower()
    for name, (prompt_price, completion_price) in PRICES.items():
        if name in lowered:
            return (
                prompt_tokens * prompt_price + completion_tokens * completion_price
            ) / 1_000_000
    return 0.0


@dataclass
class LLMClient:
    config: LLMConfig = field(default_factory=LLMConfig.from_env)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, dict[str, int]]:
        """Return ``(text, usage)`` for one completion."""

        if not self.enabled:
            raise LLMError("no API key configured")

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": (
                self.config.temperature if temperature is None else temperature
            ),
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        request = urllib.request.Request(
            url=self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise LLMError(f"provider returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"could not reach provider: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError("provider returned a non-JSON response") from exc

        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {str(body)[:200]}") from exc

        usage = body.get("usage") or {}
        return text, {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
        }


def parse_json_block(text: str) -> Any:
    """Best-effort extraction of a JSON object from a model reply."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned[3:]
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise LLMError("model reply did not contain parseable JSON")
