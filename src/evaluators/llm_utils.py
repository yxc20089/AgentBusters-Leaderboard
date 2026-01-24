"""
LLM utilities for evaluator grading.

Provides optional OpenAI/Anthropic client creation and helpers for
OpenAI-compatible evaluation prompts.

Supports per-evaluator LLM configuration via EvaluatorLLMConfig.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional


# =============================================================================
# Per-Evaluator LLM Configuration
# =============================================================================


@dataclass
class EvaluatorModelConfig:
    """
    Configuration for a single evaluator's LLM settings.

    NOTE: For reproducibility in LLM-as-judge evaluation, temperature MUST be 0.0.
    This ensures deterministic outputs and consistent benchmark results across runs.
    """

    model: str = "gpt-4o-mini"
    temperature: float = 0.0  # MUST be 0 for reproducible LLM-as-judge evaluation
    max_tokens: int = 512
    provider: Optional[str] = None  # None = auto-detect from model name

    def get_provider(self) -> str:
        """Determine provider from model name or explicit setting."""
        if self.provider:
            return self.provider
        # Auto-detect from model name
        if self.model.startswith("claude"):
            return "anthropic"
        return "openai"


@dataclass
class EvaluatorLLMConfig:
    """
    Per-evaluator LLM configuration.

    Allows different evaluators to use different models for cost optimization
    and quality tuning. Each evaluator can specify its own model, temperature,
    and max_tokens.

    Example usage:
        config = EvaluatorLLMConfig.from_env()
        model = config.get_model("macro")  # Returns model for MacroEvaluator

    Environment variables (all optional, with sensible defaults):
        EVAL_LLM_MACRO_MODEL: Model for MacroEvaluator (default: gpt-4o-mini)
        EVAL_LLM_EXECUTION_MODEL: Model for ExecutionEvaluator (default: gpt-4o-mini)
        EVAL_LLM_GDPVAL_MODEL: Model for GDPValEvaluator (default: gpt-4o)
        EVAL_LLM_BIZFINBENCH_MODEL: Model for BizFinBenchEvaluator (default: gpt-4o-mini)
        EVAL_LLM_DEBATE_MODEL: Model for debate/complex reasoning (default: gpt-4o)

        EVAL_LLM_<NAME>_TEMPERATURE: Temperature for specific evaluator
        EVAL_LLM_<NAME>_MAX_TOKENS: Max tokens for specific evaluator

        EVAL_LLM_DEFAULT_MODEL: Default model for all evaluators
        EVAL_LLM_DEFAULT_TEMPERATURE: Default temperature (default: 0.0)
    """

    # Per-evaluator configurations
    macro: EvaluatorModelConfig = field(default_factory=lambda: EvaluatorModelConfig(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=256,
    ))
    execution: EvaluatorModelConfig = field(default_factory=lambda: EvaluatorModelConfig(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=512,
    ))
    gdpval: EvaluatorModelConfig = field(default_factory=lambda: EvaluatorModelConfig(
        model="gpt-4o",  # More capable for open-ended evaluation
        temperature=0.0,
        max_tokens=2000,
    ))
    bizfinbench: EvaluatorModelConfig = field(default_factory=lambda: EvaluatorModelConfig(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=800,
    ))
    debate: EvaluatorModelConfig = field(default_factory=lambda: EvaluatorModelConfig(
        model="gpt-4o",  # Complex reasoning needs stronger model
        temperature=0.0,  # Must be 0 for reproducibility (LLM-as-judge)
        max_tokens=1500,
    ))
    public_csv: EvaluatorModelConfig = field(default_factory=lambda: EvaluatorModelConfig(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=1000,
    ))

    # Default fallback
    default: EvaluatorModelConfig = field(default_factory=lambda: EvaluatorModelConfig(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=512,
    ))

    def get_config(self, evaluator_name: str) -> EvaluatorModelConfig:
        """Get configuration for a specific evaluator."""
        return getattr(self, evaluator_name.lower(), self.default)

    def get_model(self, evaluator_name: str) -> str:
        """Get model name for a specific evaluator."""
        return self.get_config(evaluator_name).model

    def get_temperature(self, evaluator_name: str) -> float:
        """Get temperature for a specific evaluator."""
        return self.get_config(evaluator_name).temperature

    def get_max_tokens(self, evaluator_name: str) -> int:
        """Get max_tokens for a specific evaluator."""
        return self.get_config(evaluator_name).max_tokens

    @classmethod
    def from_env(cls) -> "EvaluatorLLMConfig":
        """
        Create configuration from environment variables.

        Supports both global defaults and per-evaluator overrides.
        """
        config = cls()

        # Global defaults
        default_model = os.getenv("EVAL_LLM_DEFAULT_MODEL")
        default_temp = os.getenv("EVAL_LLM_DEFAULT_TEMPERATURE")

        if default_model:
            config.default.model = default_model
        if default_temp:
            try:
                config.default.temperature = float(default_temp)
            except ValueError:
                pass

        # Per-evaluator overrides
        evaluator_names = ["macro", "execution", "gdpval", "bizfinbench", "debate", "public_csv"]

        for name in evaluator_names:
            upper_name = name.upper()
            model_config = getattr(config, name)

            # Model override
            model = os.getenv(f"EVAL_LLM_{upper_name}_MODEL")
            if model:
                model_config.model = model
            elif default_model:
                model_config.model = default_model

            # Temperature override
            temp = os.getenv(f"EVAL_LLM_{upper_name}_TEMPERATURE")
            if temp:
                try:
                    model_config.temperature = float(temp)
                except ValueError:
                    pass

            # Max tokens override
            tokens = os.getenv(f"EVAL_LLM_{upper_name}_MAX_TOKENS")
            if tokens:
                try:
                    model_config.max_tokens = int(tokens)
                except ValueError:
                    pass

            # Provider override
            provider = os.getenv(f"EVAL_LLM_{upper_name}_PROVIDER")
            if provider:
                model_config.provider = provider.lower()

        return config


# Global config instance (lazy loaded)
_evaluator_llm_config: Optional[EvaluatorLLMConfig] = None


def get_evaluator_llm_config() -> EvaluatorLLMConfig:
    """Get the global evaluator LLM configuration (cached)."""
    global _evaluator_llm_config
    if _evaluator_llm_config is None:
        _evaluator_llm_config = EvaluatorLLMConfig.from_env()
    return _evaluator_llm_config


def reset_evaluator_llm_config() -> None:
    """Reset the global config (useful for testing)."""
    global _evaluator_llm_config
    _evaluator_llm_config = None


def get_model_for_evaluator(evaluator_name: str) -> str:
    """Get the configured model for a specific evaluator."""
    return get_evaluator_llm_config().get_model(evaluator_name)


def get_temperature_for_evaluator(evaluator_name: str) -> float:
    """Get the configured temperature for a specific evaluator."""
    return get_evaluator_llm_config().get_temperature(evaluator_name)


def get_max_tokens_for_evaluator(evaluator_name: str) -> int:
    """Get the configured max_tokens for a specific evaluator."""
    return get_evaluator_llm_config().get_max_tokens(evaluator_name)


def get_provider_for_evaluator(evaluator_name: str) -> str:
    """Get the configured provider for a specific evaluator."""
    return get_evaluator_llm_config().get_config(evaluator_name).get_provider()


# =============================================================================
# Original utility functions
# =============================================================================


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


def build_llm_client(
    existing: Any | None = None,
    provider: Optional[str] = None,
) -> Optional[Any]:
    """
    Build an LLM client from environment variables.

    Supports OpenAI-compatible and Anthropic clients.

    Args:
        existing: If provided, returns this client directly
        provider: Optional provider override ("openai" or "anthropic").
                  If not specified, uses LLM_PROVIDER env var or defaults to "openai".
    """
    if existing is not None:
        return existing

    effective_provider = (provider or os.getenv("LLM_PROVIDER") or "openai").lower()
    if effective_provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        try:
            from anthropic import Anthropic
        except Exception:
            return None
        return Anthropic(api_key=api_key)

    # Default to OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    try:
        from openai import OpenAI
    except Exception:
        return None
    return OpenAI(api_key=api_key, base_url=base_url)


def build_llm_client_for_evaluator(
    evaluator_name: str,
    existing: Any | None = None,
) -> Optional[Any]:
    """
    Build an LLM client for a specific evaluator using its configured provider.

    Args:
        evaluator_name: Name of the evaluator (e.g., "macro", "gdpval")
        existing: If provided, returns this client directly

    Returns:
        LLM client configured for the evaluator's provider
    """
    if existing is not None:
        return existing

    provider = get_provider_for_evaluator(evaluator_name)
    return build_llm_client(provider=provider)


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
