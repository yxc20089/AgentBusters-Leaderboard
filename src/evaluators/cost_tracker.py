"""
Cost Tracker for efficiency scoring.

Tracks all costs incurred during agent evaluation:
- LLM inference costs (input + output tokens)
- Tool usage costs (tokens retrieved)
"""

import math
from datetime import datetime, timezone
from typing import Optional

import structlog
from pydantic import BaseModel, Field

from cio_agent.models import CostBreakdown, ToolCall


def _utc_now() -> datetime:
    """Get current UTC time (Python 3.12+ compatible)."""
    return datetime.now(timezone.utc)

logger = structlog.get_logger()


class LLMCallRecord(BaseModel):
    """Record of a single LLM API call."""
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: datetime = Field(default_factory=_utc_now)
    purpose: str = ""  # e.g., "analysis", "debate", "evaluation"


# Model pricing (per 1K tokens)
MODEL_PRICING = {
    # OpenAI
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    # Anthropic
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    # Default fallback
    "default": {"input": 0.01, "output": 0.03},
}


class CostTracker:
    """
    Tracks all costs incurred during agent evaluation.

    Features:
    - LLM call cost tracking with model-specific pricing
    - Tool usage cost tracking (token-based)
    - Cost breakdown generation for Alpha Score calculation
    """

    # Tool cost: $0.001 per 1K tokens
    TOOL_COST_PER_1K = 0.001

    def __init__(self):
        self.llm_calls: list[LLMCallRecord] = []
        self.tool_calls: list[ToolCall] = []
        self._llm_cost: float = 0.0
        self._tool_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_tool_tokens: int = 0

    def _get_model_pricing(self, model: str) -> dict[str, float]:
        """Get pricing for a model, with fallback to default."""
        # Normalize model name
        model_lower = model.lower()

        # Try exact match
        if model_lower in MODEL_PRICING:
            return MODEL_PRICING[model_lower]

        # Try prefix match
        for model_name, pricing in MODEL_PRICING.items():
            if model_lower.startswith(model_name):
                return pricing

        return MODEL_PRICING["default"]

    def add_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        purpose: str = "",
    ) -> LLMCallRecord:
        """
        Record an LLM API call and calculate cost.

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            purpose: Purpose of the call (for tracking)

        Returns:
            LLMCallRecord with calculated cost
        """
        pricing = self._get_model_pricing(model)
        cost = (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]

        record = LLMCallRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            purpose=purpose,
        )

        self.llm_calls.append(record)
        self._llm_cost += cost
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

        logger.debug(
            "llm_call_recorded",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            purpose=purpose,
        )

        return record

    def add_tool_calls(self, tool_calls: list[ToolCall]) -> None:
        """
        Record tool calls and calculate costs.

        Args:
            tool_calls: List of tool calls to record
        """
        for call in tool_calls:
            self.tool_calls.append(call)
            self._total_tool_tokens += call.response_tokens
            cost = (call.response_tokens / 1000) * self.TOOL_COST_PER_1K
            self._tool_cost += cost

            logger.debug(
                "tool_call_recorded",
                tool=call.tool_name,
                tokens=call.response_tokens,
                cost_usd=cost,
            )

    def add_tool_call(self, tool_call: ToolCall) -> None:
        """Record a single tool call."""
        self.add_tool_calls([tool_call])

    @property
    def total_cost(self) -> float:
        """Get total cost in USD."""
        return self._llm_cost + self._tool_cost

    def get_breakdown(self) -> CostBreakdown:
        """
        Get detailed cost breakdown.

        Returns:
            CostBreakdown with all cost components
        """
        return CostBreakdown(
            llm_cost_usd=self._llm_cost,
            tool_cost_usd=self._tool_cost,
            total_cost_usd=self.total_cost,
            llm_calls=len(self.llm_calls),
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            tool_calls=len(self.tool_calls),
            tool_tokens=self._total_tool_tokens,
        )

    def calculate_cost_penalty(self) -> float:
        """
        Calculate the logarithmic cost penalty for Alpha Score.

        The formula ln(1 + cost) ensures:
        - Low cost = low penalty
        - Diminishing returns from throwing compute at the problem

        Examples:
        - $1 cost → penalty of 0.69
        - $5 cost → penalty of 1.79
        - $10 cost → penalty of 2.40

        Returns:
            Log cost penalty value
        """
        return math.log(1 + self.total_cost)

    def get_summary(self) -> dict:
        """Get a summary of costs for logging."""
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "llm_cost_usd": round(self._llm_cost, 4),
            "tool_cost_usd": round(self._tool_cost, 4),
            "llm_calls": len(self.llm_calls),
            "tool_calls": len(self.tool_calls),
            "total_tokens": self._total_input_tokens + self._total_output_tokens + self._total_tool_tokens,
        }

    def reset(self) -> None:
        """Reset all tracking."""
        self.llm_calls.clear()
        self.tool_calls.clear()
        self._llm_cost = 0.0
        self._tool_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_tool_tokens = 0
