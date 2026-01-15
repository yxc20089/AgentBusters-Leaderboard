"""
Options Trading Data Models for Alpha Challenge Benchmark.

This module provides Pydantic models for options trading simulation:
- OptionsContract, OptionsQuote: Contract and pricing data
- Position, Trade, Portfolio: Position and portfolio tracking
- OptionsStrategy, StrategyLeg: Multi-leg strategy definitions
- GreeksSnapshot, RiskMetrics: Risk analysis
- AlphaScore: Evaluation scoring

These models are used by:
- MCP Servers (options_chain, trading_sim, risk_metrics)
- Purple Agent (FinanceAgentExecutor)
- CIO Agent (ComprehensiveEvaluator)
"""

from options_agents.models import (
    # Core Options
    OptionsContract,
    OptionsQuote,
    GreeksSnapshot,
    # Trading
    Trade,
    Position,
    Portfolio,
    # Strategy
    StrategyLeg,
    OptionsStrategy,
    TradeSignal,
    # Tasks
    OptionsTaskCategory,
    OptionsTask,
    # Risk & Scoring
    RiskMetrics,
    StrategyScore,
    ExecutionScore,
    AlphaScore,
)

__all__ = [
    # Core Options
    "OptionsContract",
    "OptionsQuote",
    "GreeksSnapshot",
    # Trading
    "Trade",
    "Position",
    "Portfolio",
    # Strategy
    "StrategyLeg",
    "OptionsStrategy",
    "TradeSignal",
    # Tasks
    "OptionsTaskCategory",
    "OptionsTask",
    # Risk & Scoring
    "RiskMetrics",
    "StrategyScore",
    "ExecutionScore",
    "AlphaScore",
]
