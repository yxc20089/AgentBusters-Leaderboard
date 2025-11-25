"""
CIO-Agent: FAB++ Dynamic Finance Agent Benchmark
A Green Agent (Evaluator) for the AgentBeats Competition.

This package implements a Chief Investment Officer evaluator that tests
finance agents through dynamic task generation, adversarial debate,
and multi-dimensional scoring.
"""

__version__ = "1.0.0"
__author__ = "AgentBusters Team"

from cio_agent.models import (
    Task,
    AgentResponse,
    EvaluationResult,
    AlphaScore,
    DebateResult,
)

__all__ = [
    "Task",
    "AgentResponse",
    "EvaluationResult",
    "AlphaScore",
    "DebateResult",
]
