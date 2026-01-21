"""
LLM utilities for evaluator grading.

Provides optional OpenAI/Anthropic client creation and helpers for
OpenAI-compatible evaluation prompts.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional


def _env_flag(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    return None


def should_use_llm() -> bool:
    """
    Decide whether to use LLM evaluation based on env vars.

    If EVAL_USE_LLM is set, it wins. Otherwise enable when any API key is set.
    """
    flag = _env_flag("EVAL_USE_LLM")
    if flag is not None:
        return flag
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))


def get_llm_model() -> str:
    """Return the model to use for evaluation."""
    return os.getenv("EVAL_LLM_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"


def get_llm_temperature() -> float:
    """Return temperature for evaluation prompts."""
    raw = os.getenv("EVAL_LLM_TEMPERATURE")
    if raw:
        try:
            return float(raw)
        except ValueError:
            return 0.0
    return 0.0


def build_llm_client(existing: Any | None = None) -> Optional[Any]:
    """
    Build an LLM client from environment variables.

    Supports OpenAI-compatible and Anthropic clients.
    """
    if existing is not None:
        return existing

    provider = (os.getenv("LLM_PROVIDER") or "openai").lower()
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        try:
            from anthropic import Anthropic
        except Exception:
            return None
        return Anthropic(api_key=api_key)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    try:
        from openai import OpenAI
    except Exception:
        return None
    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm(
    client: Any,
    prompt: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """
    Call an OpenAI- or Anthropic-style client and return text content.
    """
    model = model or get_llm_model()

    if hasattr(client, "chat"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    if hasattr(client, "messages"):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.content:
            return ""
        return response.content[0].text

    raise ValueError("Unsupported LLM client type")


def extract_json(text: str) -> Optional[dict]:
    """
    Extract the first JSON object from a model response.
    """
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None


def coerce_bool(value: Any) -> Optional[bool]:
    """
    Convert common truthy/falsey values into bool.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "y", "1"):
            return True
        if lowered in ("false", "no", "n", "0"):
            return False
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    return None
