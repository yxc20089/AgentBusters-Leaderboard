# PM Agent Execution Framework

**Version**: 1.0.0
**Status**: Design Phase
**Generated**: 2026-01-15
**Author**: PM Agent

---

## 1. Executive Summary

This document defines the portfolio management and execution framework for the AgentBusters Alpha Challenge options trading benchmark. It covers trade execution, position management, risk calculations, and scoring rubrics.

## 2. Core Data Models

### 2.1 OptionsContract

```python
class OptionsContract(BaseModel):
    """Uniquely identifies an options contract."""
    ticker: str
    expiration: date
    strike: float
    option_type: Literal["call", "put"]

    @computed_field
    @property
    def contract_symbol(self) -> str:
        """Generate OCC-style contract symbol."""
        exp_str = self.expiration.strftime("%y%m%d")
        opt_char = "C" if self.option_type == "call" else "P"
        strike_str = f"{int(self.strike * 1000):08d}"
        return f"{self.ticker}{exp_str}{opt_char}{strike_str}"
```

### 2.2 Greeks

```python
class Greeks(BaseModel):
    """Option Greeks for risk analysis."""
    delta: float = Field(ge=-1.0, le=1.0)
    gamma: float = Field(ge=0.0)
    theta: float  # Typically negative (time decay)
    vega: float = Field(ge=0.0)
    rho: float
```

### 2.3 Position

```python
class Position(BaseModel):
    """Open options position."""
    position_id: str
    contract: OptionsContract
    quantity: int  # Positive = long, Negative = short
    entry_price: float
    entry_date: datetime
    current_price: float
    current_greeks: Greeks
    status: Literal["open", "closed", "expired", "assigned"]
    unrealized_pnl: float
    realized_pnl: float
```

### 2.4 Portfolio

```python
class Portfolio(BaseModel):
    """Complete portfolio state."""
    cash: float = 100000.0
    starting_cash: float = 100000.0
    positions: list[Position]
    trades: list[Trade]
    portfolio_greeks: Greeks
    max_drawdown: float
    daily_returns: list[float]

    @computed_field
    @property
    def total_return_pct(self) -> float:
        return ((self.total_value - self.starting_cash) / self.starting_cash) * 100
```

## 3. Position Lifecycle State Machine

```
    ┌──────────┐    add_position()    ┌──────────┐
    │  (new)   │─────────────────────►│   OPEN   │
    └──────────┘                      └────┬─────┘
                                           │
                           ┌───────────────┼───────────────┐
                           │               │               │
                           ▼               ▼               ▼
                    partial_close()  full_close()   expiration()
                           │               │               │
                           ▼               │               │
                  ┌────────────────┐       │               │
                  │   PARTIALLY    │       │               │
                  │    CLOSED      │       │               │
                  └───────┬────────┘       │               │
                          │                │               │
                          └────────┬───────┘               │
                                   ▼                       │
                           ┌──────────┐                    │
                           │  CLOSED  │                    │
                           └──────────┘                    │
                                                           │
                    If ITM at expiration:                  │
                    ├── Long: EXERCISED                    │
                    └── Short: ASSIGNED                    │
                                                           │
                    If OTM at expiration:                  │
                    └── EXPIRED (worthless)                │
```

## 4. Trade Execution Engine

### 4.1 Fill Price Calculation

```python
def calculate_fill_price(
    quote: OptionsQuote,
    action: OrderAction,
    quantity: int,
    order_type: OrderType,
) -> tuple[float, float]:
    """
    Returns (fill_price, slippage_amount)

    - Market orders cross the spread (buy at ask, sell at bid)
    - Add slippage based on liquidity and order size
    - Commission: $0.65 per contract
    """
    is_buy = action in ["buy_to_open", "buy_to_close"]
    base_price = quote.ask if is_buy else quote.bid

    # Slippage increases with order size and illiquidity
    slippage = calculate_slippage(quote, quantity)

    fill_price = base_price + slippage if is_buy else base_price - slippage
    return max(0.01, fill_price), slippage
```

### 4.2 Slippage Model

```python
def calculate_slippage(quote: OptionsQuote, quantity: int) -> float:
    """
    Slippage = base_spread_slippage + volume_impact

    - Base: 50% of bid-ask spread
    - Volume impact: +1% per 100 contracts over threshold
    - Liquidity discount for high OI
    """
    spread = quote.ask - quote.bid
    base_slippage = spread * 0.5

    # Volume impact for large orders
    if quantity > 100:
        volume_impact = base_slippage * min(quantity / 100, 1.0)
        base_slippage += volume_impact

    # Liquidity discount
    if quote.open_interest > 1000:
        base_slippage *= 0.8

    return base_slippage
```

## 5. Risk Calculations

### 5.1 Portfolio Greeks Aggregation

```python
def calculate_portfolio_greeks(portfolio: Portfolio) -> Greeks:
    """
    Net Delta = Σ (position.delta * position.quantity * 100)
    Net Gamma = Σ (position.gamma * |position.quantity| * 100)
    Net Theta = Σ (position.theta * position.quantity)
    Net Vega = Σ (position.vega * |position.quantity| * 100)
    """
```

### 5.2 Value at Risk (VaR)

```python
def calculate_var(
    portfolio: Portfolio,
    confidence_level: float = 0.95,
    horizon_days: int = 1,
    method: str = "historical",
) -> dict:
    """
    Historical VaR:
    - Sort historical returns
    - VaR = percentile(returns, (1 - confidence) * 100)
    - Dollar VaR = portfolio_value * VaR_pct

    Parametric VaR:
    - VaR = μ - z * σ * √T
    """
```

### 5.3 Maximum Drawdown

```python
def calculate_max_drawdown(portfolio_values: list[float]) -> dict:
    """
    Peak = running maximum of portfolio value
    Drawdown = (Peak - Current) / Peak
    Max Drawdown = maximum of all drawdowns

    Returns:
        max_drawdown_pct, peak_value, trough_value, recovery_pct
    """
```

## 6. Performance Metrics

### 6.1 Sharpe Ratio

```
Sharpe = (Rp - Rf) / σp

Where:
- Rp = annualized portfolio return
- Rf = risk-free rate (5.25%)
- σp = annualized volatility
```

### 6.2 Sortino Ratio

```
Sortino = (Rp - Rf) / σd

Where:
- σd = downside deviation (only negative returns)
```

### 6.3 P&L Attribution by Greek

```
Delta P&L = Δ × ΔS × quantity × 100
Gamma P&L = 0.5 × Γ × (ΔS)² × quantity × 100
Theta P&L = Θ × Δt × quantity
Vega P&L = ν × Δσ × quantity × 100
Residual = Total P&L - (Delta + Gamma + Theta + Vega)
```

## 7. Scoring Rubrics

### 7.1 Strategy Score (0-100)

| Component | Weight | 100 pts | 75 pts | 50 pts | 25 pts |
|-----------|--------|---------|--------|--------|--------|
| **Thesis Quality** | 30% | Clear catalyst, quantified target | Good thesis, gaps | Basic thesis | Weak/missing |
| **Greeks Awareness** | 25% | Optimal Greeks for strategy | Good management | Basic awareness | Ignored |
| **Position Sizing** | 25% | Kelly-optimal or risk-based | Reasonable | Arbitrary | Over/undersized |
| **Exit Strategy** | 20% | Clear stops + targets | Stops OR targets | Vague | No plan |

### 7.2 Execution Score (0-100)

| Component | Weight | 100 pts | 75 pts | 50 pts | 25 pts |
|-----------|--------|---------|--------|--------|--------|
| **P&L Accuracy** | 40% | Within 1% | Within 5% | Within 10% | >10% error |
| **Timing** | 30% | Optimal entry/exit | Good | Acceptable | Poor |
| **Slippage** | 15% | Realistic | Minor optimism | Ignored | Unrealistic |
| **Cost Efficiency** | 15% | Minimized | Reasonable | High | Excessive |

### 7.3 Risk-Adjusted Return Scoring

| Metric | Excellent (100) | Good (75) | Fair (50) | Poor (25) |
|--------|-----------------|-----------|-----------|-----------|
| **Sharpe Ratio** | > 2.0 | 1.5-2.0 | 1.0-1.5 | < 1.0 |
| **Sortino Ratio** | > 2.5 | 2.0-2.5 | 1.5-2.0 | < 1.5 |
| **Max Drawdown** | < 10% | 10-15% | 15-25% | > 25% |
| **Win Rate** | > 65% | 55-65% | 45-55% | < 45% |
| **Profit Factor** | > 2.0 | 1.5-2.0 | 1.2-1.5 | < 1.2 |

### 7.4 Debate Multiplier

| Rebuttal Quality | Multiplier | Criteria |
|------------------|------------|----------|
| **Exceptional** | 1.2x | New evidence, stress test, quantified defense |
| **Strong** | 1.1x | Good defense with data |
| **Adequate** | 1.0x | Basic defense, repeats prior |
| **Weak** | 0.8x | Poor defense, no evidence |
| **Concession** | 0.5x | Admits flaw or contradicts |

## 8. Alpha Score Formula

```python
def calculate_options_alpha_score(
    strategy_score: float,      # 0-100
    execution_score: float,     # 0-100
    debate_multiplier: float,   # 0.5-1.2
    max_drawdown_pct: float,
    cost_usd: float,
    lookahead_violations: int,
) -> float:
    """
    Alpha Score = (Strategy × Execution × Debate)
                / (Risk × Cost × Temporal)

    Where:
    - Risk Penalty = 1 + (MaxDD / 0.30) × 0.5
    - Cost Penalty = ln(1 + CostUSD)
    - Temporal Penalty = 1 + (Violations × 0.1)
    """
    weighted_score = (strategy_score / 100) * (execution_score / 100)
    numerator = weighted_score * debate_multiplier

    risk_penalty = 1 + (min(max_drawdown_pct / 30.0, 1.0) * 0.5)
    cost_penalty = math.log(1 + cost_usd) if cost_usd > 0 else 0.01
    temporal_penalty = 1 + (lookahead_violations * 0.1)

    denominator = risk_penalty * cost_penalty * temporal_penalty

    return (numerator / denominator) * 100
```

## 9. Race to 10 Million Challenge

```python
class RaceTo10MChallenge:
    starting_capital: float = 100_000
    target_capital: float = 1_000_000
    simulation_period: str = "1Y"
    max_drawdown_limit: float = 0.30  # 30% = disqualification

    scoring = {
        "10x_achiever": 100,    # Reached $1M
        "5x_achiever": 80,      # Reached $500K
        "2x_achiever": 60,      # Reached $200K
        "profitable": 40,       # Any profit
        "capital_preserved": 20, # <5% loss
        "significant_loss": 0    # >5% loss
    }
```
