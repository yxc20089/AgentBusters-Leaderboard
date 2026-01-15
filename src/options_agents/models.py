"""
Options Trading Data Models for Alpha Challenge Benchmark.

This module defines Pydantic models for options contracts, positions,
portfolios, strategies, and risk metrics used in the options trading
simulation and evaluation framework.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field


class OptionsTaskCategory(str, Enum):
    """Categories of options trading tasks."""

    OPTIONS_PRICING = "options_pricing"
    GREEKS_ANALYSIS = "greeks_analysis"
    STRATEGY_CONSTRUCTION = "strategy_construction"
    VOLATILITY_TRADING = "volatility_trading"
    PNL_ATTRIBUTION = "pnl_attribution"
    RISK_MANAGEMENT = "risk_management"
    COPY_TRADING = "copy_trading"
    RACE_TO_10M = "race_to_10m"
    STRATEGY_DEFENSE = "strategy_defense"


class OptionsContract(BaseModel):
    """Uniquely identifies an options contract."""

    ticker: str = Field(..., description="Underlying stock ticker symbol")
    expiration: date = Field(..., description="Options expiration date")
    strike: float = Field(..., gt=0, description="Strike price")
    option_type: Literal["call", "put"] = Field(..., description="Call or put option")

    @computed_field
    @property
    def contract_symbol(self) -> str:
        """Generate OCC-style contract symbol.

        Format: TICKER + YYMMDD + C/P + 8-digit strike (strike * 1000, zero-padded)
        Example: AAPL250117C00200000 for AAPL $200 call expiring Jan 17, 2025
        """
        exp_str = self.expiration.strftime("%y%m%d")
        opt_char = "C" if self.option_type == "call" else "P"
        strike_str = f"{int(self.strike * 1000):08d}"
        return f"{self.ticker}{exp_str}{opt_char}{strike_str}"

    @computed_field
    @property
    def days_to_expiry(self) -> int:
        """Calculate days until expiration from today."""
        return (self.expiration - date.today()).days


class GreeksSnapshot(BaseModel):
    """Option Greeks for risk analysis at a point in time."""

    delta: float = Field(..., ge=-1.0, le=1.0, description="Rate of change vs underlying")
    gamma: float = Field(..., ge=0.0, description="Rate of change of delta")
    theta: float = Field(..., description="Time decay (typically negative)")
    vega: float = Field(..., ge=0.0, description="Sensitivity to volatility")
    rho: float = Field(default=0.0, description="Sensitivity to interest rates")
    timestamp: datetime = Field(default_factory=datetime.now)

    @computed_field
    @property
    def is_long_delta(self) -> bool:
        """True if position benefits from underlying price increase."""
        return self.delta > 0

    @computed_field
    @property
    def is_long_gamma(self) -> bool:
        """True if position benefits from large moves in either direction."""
        return self.gamma > 0.01

    @computed_field
    @property
    def is_long_vega(self) -> bool:
        """True if position benefits from volatility increase."""
        return self.vega > 0


class OptionsQuote(BaseModel):
    """Real-time or historical options quote data."""

    contract: OptionsContract
    bid: float = Field(..., ge=0, description="Best bid price")
    ask: float = Field(..., ge=0, description="Best ask price")
    last: float = Field(..., ge=0, description="Last traded price")
    volume: int = Field(default=0, ge=0, description="Trading volume")
    open_interest: int = Field(default=0, ge=0, description="Open interest")
    implied_volatility: float = Field(..., ge=0, description="Implied volatility")
    greeks: GreeksSnapshot
    underlying_price: float = Field(..., gt=0, description="Current underlying price")
    timestamp: datetime = Field(default_factory=datetime.now)

    @computed_field
    @property
    def mid_price(self) -> float:
        """Calculate mid-point between bid and ask."""
        return (self.bid + self.ask) / 2

    @computed_field
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask - self.bid

    @computed_field
    @property
    def spread_pct(self) -> float:
        """Calculate spread as percentage of mid price."""
        mid = self.mid_price
        return (self.spread / mid * 100) if mid > 0 else 0.0


class Trade(BaseModel):
    """Executed options trade record."""

    trade_id: str = Field(default_factory=lambda: str(uuid4()))
    contract: OptionsContract
    action: Literal["buy_to_open", "sell_to_open", "buy_to_close", "sell_to_close"]
    quantity: int = Field(..., gt=0, description="Number of contracts")
    fill_price: float = Field(..., ge=0, description="Execution price per contract")
    commission: float = Field(default=0.65, ge=0, description="Commission per contract")
    slippage: float = Field(default=0.0, ge=0, description="Slippage amount")
    execution_time: datetime = Field(default_factory=datetime.now)
    order_type: Literal["market", "limit"] = "market"

    @computed_field
    @property
    def total_cost(self) -> float:
        """Calculate total trade cost including commissions.

        Each contract = 100 shares, so multiply by 100 for notional.
        """
        notional = self.fill_price * self.quantity * 100
        total_commission = self.commission * self.quantity
        if self.action in ["buy_to_open", "buy_to_close"]:
            return notional + total_commission
        else:
            return -notional + total_commission

    @computed_field
    @property
    def is_opening(self) -> bool:
        """True if this trade opens a new position."""
        return self.action in ["buy_to_open", "sell_to_open"]


class Position(BaseModel):
    """Open or closed options position."""

    position_id: str = Field(default_factory=lambda: str(uuid4()))
    contract: OptionsContract
    quantity: int = Field(..., description="Positive=long, Negative=short")
    entry_price: float = Field(..., ge=0, description="Average entry price")
    entry_date: datetime = Field(default_factory=datetime.now)
    current_price: float = Field(..., ge=0, description="Current market price")
    current_greeks: GreeksSnapshot
    status: Literal["open", "closed", "expired", "assigned", "exercised"] = "open"
    trades: list[Trade] = Field(default_factory=list)

    @computed_field
    @property
    def is_long(self) -> bool:
        """True if this is a long position."""
        return self.quantity > 0

    @computed_field
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L for open positions."""
        if self.status != "open":
            return 0.0
        price_diff = self.current_price - self.entry_price
        return price_diff * self.quantity * 100

    @computed_field
    @property
    def realized_pnl(self) -> float:
        """Calculate realized P&L from closed trades."""
        if self.status == "open":
            return 0.0
        # Sum up all trade P&L
        total = 0.0
        for trade in self.trades:
            total -= trade.total_cost  # Negative for buys, positive for sells
        return total

    @computed_field
    @property
    def market_value(self) -> float:
        """Calculate current market value of position."""
        return self.current_price * abs(self.quantity) * 100

    @computed_field
    @property
    def position_delta(self) -> float:
        """Calculate position delta (scaled by quantity)."""
        return self.current_greeks.delta * self.quantity * 100


class StrategyLeg(BaseModel):
    """Single leg of a multi-leg options strategy."""

    contract: OptionsContract
    action: Literal["buy", "sell"]
    quantity: int = Field(..., gt=0)
    entry_price: float | None = None

    @computed_field
    @property
    def is_long(self) -> bool:
        """True if this leg is a long position."""
        return self.action == "buy"


class OptionsStrategy(BaseModel):
    """Multi-leg options strategy definition."""

    strategy_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Strategy name (e.g., 'Iron Condor')")
    strategy_type: Literal[
        "single_leg",
        "vertical_spread",
        "calendar_spread",
        "diagonal_spread",
        "straddle",
        "strangle",
        "iron_condor",
        "iron_butterfly",
        "covered_call",
        "protective_put",
        "collar",
        "custom",
    ]
    legs: list[StrategyLeg] = Field(..., min_length=1)
    thesis: str = Field(..., description="Investment thesis for the strategy")
    target_profit_pct: float | None = Field(default=None, description="Target profit %")
    stop_loss_pct: float | None = Field(default=None, description="Stop loss %")
    max_days_to_hold: int | None = Field(default=None, description="Max holding period")

    @computed_field
    @property
    def is_credit_strategy(self) -> bool:
        """True if strategy receives net credit on entry."""
        return any(leg.action == "sell" for leg in self.legs)

    @computed_field
    @property
    def num_legs(self) -> int:
        """Number of legs in the strategy."""
        return len(self.legs)


class RiskMetrics(BaseModel):
    """Portfolio risk metrics and performance statistics."""

    sharpe_ratio: float | None = Field(default=None, description="Risk-adjusted return")
    sortino_ratio: float | None = Field(default=None, description="Downside risk-adjusted")
    calmar_ratio: float | None = Field(default=None, description="Return / Max DD")
    max_drawdown_pct: float = Field(default=0.0, description="Maximum drawdown %")
    var_95: float | None = Field(default=None, description="95% Value at Risk")
    var_99: float | None = Field(default=None, description="99% Value at Risk")
    win_rate: float | None = Field(default=None, description="Winning trades %")
    profit_factor: float | None = Field(default=None, description="Gross profit / loss")
    avg_win: float | None = Field(default=None, description="Average winning trade")
    avg_loss: float | None = Field(default=None, description="Average losing trade")
    expectancy: float | None = Field(default=None, description="Expected value per trade")

    @computed_field
    @property
    def risk_rating(self) -> Literal["low", "medium", "high", "extreme"]:
        """Categorize overall portfolio risk level."""
        if self.max_drawdown_pct <= 10:
            return "low"
        elif self.max_drawdown_pct <= 20:
            return "medium"
        elif self.max_drawdown_pct <= 30:
            return "high"
        else:
            return "extreme"


class Portfolio(BaseModel):
    """Complete portfolio state with positions and metrics."""

    portfolio_id: str = Field(default_factory=lambda: str(uuid4()))
    cash: float = Field(default=100000.0, description="Available cash")
    starting_cash: float = Field(default=100000.0, description="Initial capital")
    positions: list[Position] = Field(default_factory=list)
    trades: list[Trade] = Field(default_factory=list)
    portfolio_greeks: GreeksSnapshot | None = None
    risk_metrics: RiskMetrics = Field(default_factory=RiskMetrics)
    daily_values: list[float] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    @computed_field
    @property
    def total_position_value(self) -> float:
        """Sum of all position market values."""
        return sum(p.market_value for p in self.positions if p.status == "open")

    @computed_field
    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        return self.cash + self.total_position_value

    @computed_field
    @property
    def total_return_pct(self) -> float:
        """Total return as percentage of starting capital."""
        return ((self.total_value - self.starting_cash) / self.starting_cash) * 100

    @computed_field
    @property
    def total_pnl(self) -> float:
        """Total P&L in dollars."""
        return self.total_value - self.starting_cash

    @computed_field
    @property
    def open_position_count(self) -> int:
        """Number of open positions."""
        return sum(1 for p in self.positions if p.status == "open")

    @computed_field
    @property
    def net_delta(self) -> float:
        """Net portfolio delta across all positions."""
        return sum(p.position_delta for p in self.positions if p.status == "open")


class OptionsTask(BaseModel):
    """Options trading task for benchmark evaluation."""

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    category: OptionsTaskCategory
    difficulty: Literal["easy", "medium", "hard", "expert"]
    question: str = Field(..., description="Task question or prompt")
    context: dict | None = Field(default=None, description="Additional context data")
    expected_elements: list[str] = Field(
        default_factory=list, description="Expected response elements"
    )
    time_limit_seconds: int = Field(default=300, description="Max time to complete")
    requires_execution: bool = Field(
        default=False, description="Whether task requires trade execution"
    )
    reference_date: date | None = Field(
        default=None, description="Date for historical data queries"
    )

    @computed_field
    @property
    def is_simulation_task(self) -> bool:
        """True if task involves trading simulation."""
        return self.category in [
            OptionsTaskCategory.STRATEGY_CONSTRUCTION,
            OptionsTaskCategory.RISK_MANAGEMENT,
            OptionsTaskCategory.COPY_TRADING,
            OptionsTaskCategory.RACE_TO_10M,
        ]


class TradeSignal(BaseModel):
    """Trade signal from Architect to PM Agent."""

    signal_id: str = Field(default_factory=lambda: str(uuid4()))
    strategy: OptionsStrategy
    urgency: Literal["immediate", "opportunistic", "patient"] = "opportunistic"
    max_entry_price: float | None = Field(default=None, description="Max price to pay")
    min_exit_price: float | None = Field(default=None, description="Min price to receive")
    position_size_pct: float = Field(
        default=5.0, ge=0.1, le=100.0, description="% of portfolio to allocate"
    )
    risk_budget_pct: float = Field(
        default=2.0, ge=0.1, le=20.0, description="Max % to risk on trade"
    )
    rationale: str = Field(..., description="Reasoning for the trade")
    created_at: datetime = Field(default_factory=datetime.now)


class StrategyScore(BaseModel):
    """Scoring breakdown for strategy evaluation."""

    thesis_quality: float = Field(..., ge=0, le=100)
    greeks_awareness: float = Field(..., ge=0, le=100)
    position_sizing: float = Field(..., ge=0, le=100)
    exit_strategy: float = Field(..., ge=0, le=100)

    @computed_field
    @property
    def weighted_score(self) -> float:
        """Calculate weighted strategy score.

        Weights: Thesis 30%, Greeks 25%, Sizing 25%, Exit 20%
        """
        return (
            self.thesis_quality * 0.30
            + self.greeks_awareness * 0.25
            + self.position_sizing * 0.25
            + self.exit_strategy * 0.20
        )


class ExecutionScore(BaseModel):
    """Scoring breakdown for execution evaluation."""

    pnl_accuracy: float = Field(..., ge=0, le=100)
    timing: float = Field(..., ge=0, le=100)
    slippage_realism: float = Field(..., ge=0, le=100)
    cost_efficiency: float = Field(..., ge=0, le=100)

    @computed_field
    @property
    def weighted_score(self) -> float:
        """Calculate weighted execution score.

        Weights: P&L 40%, Timing 30%, Slippage 15%, Cost 15%
        """
        return (
            self.pnl_accuracy * 0.40
            + self.timing * 0.30
            + self.slippage_realism * 0.15
            + self.cost_efficiency * 0.15
        )


class AlphaScore(BaseModel):
    """Complete Alpha Score for options task evaluation."""

    task_id: str
    strategy_score: StrategyScore
    execution_score: ExecutionScore
    debate_multiplier: float = Field(default=1.0, ge=0.5, le=1.2)
    max_drawdown_pct: float = Field(default=0.0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    lookahead_violations: int = Field(default=0, ge=0)

    @computed_field
    @property
    def alpha_score(self) -> float:
        """Calculate final Alpha Score.

        Formula: (Strategy x Execution x Debate) / (Risk x Cost x Temporal)

        Where:
        - Risk Penalty = 1 + (MaxDD / 0.30) x 0.5
        - Cost Penalty = ln(1 + CostUSD)
        - Temporal Penalty = 1 + (Violations x 0.1)
        """
        weighted_score = (
            self.strategy_score.weighted_score / 100
        ) * (self.execution_score.weighted_score / 100)
        numerator = weighted_score * self.debate_multiplier

        risk_penalty = 1 + (min(self.max_drawdown_pct / 30.0, 1.0) * 0.5)
        cost_penalty = math.log(1 + self.cost_usd) if self.cost_usd > 0 else 0.01
        temporal_penalty = 1 + (self.lookahead_violations * 0.1)

        denominator = risk_penalty * cost_penalty * temporal_penalty

        return (numerator / denominator) * 100

    @computed_field
    @property
    def grade(self) -> Literal["S", "A", "B", "C", "D", "F"]:
        """Letter grade based on alpha score."""
        score = self.alpha_score
        if score >= 90:
            return "S"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"
