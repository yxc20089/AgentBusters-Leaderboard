# Options Trading Benchmark Requirements

## Alpha Challenge: Options Trading Simulation Benchmark

**Version**: 1.0.0
**Status**: Design Phase
**Last Updated**: 2026-01-15

---

## 1. Executive Summary

### 1.1 Vision

Extend AgentBusters FAB++ benchmark with **options trading simulation** capabilities that enable:
- Real P&L tracking with historical market data
- Multi-agent collaboration (Architect + PM + Judge)
- Famous trader strategy replication (Copy Trading)
- "Race to 10 Million" portfolio growth challenges
- Adversarial debate on trading thesis and risk management

### 1.2 Goals

| Goal | Success Metric |
|------|----------------|
| **Verifiable Returns** | P&L accuracy within 1% of ground truth |
| **Diverse Agent Behaviors** | Support 10+ distinct trading strategies |
| **Open-Ended Evaluation** | Score strategies on risk-adjusted returns, not just accuracy |
| **Competition Readiness** | Docker image runs end-to-end without manual intervention |
| **Innovation** | Novel benchmark design beyond existing finance benchmarks |

### 1.3 Non-Goals (Phase 1)

- Live trading integration (paper trading only)
- High-frequency trading scenarios (< 1 min timeframes)
- Exotic options (focus on vanilla calls/puts)
- Cryptocurrency options

---

## 2. System Architecture

### 2.1 Multi-Agent Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Alpha Challenge System                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   GREEN SIDE (Evaluator)              PURPLE SIDE (Competitors)             │
│   ┌─────────────────────┐             ┌─────────────────────┐              │
│   │                     │   A2A       │                     │              │
│   │    Judge Agent      │◄───────────►│  Architect Agent    │              │
│   │    (Evaluator)      │  Protocol   │  (Strategy Design)  │              │
│   │                     │             │                     │              │
│   │  - Task Generation  │             │  - Market Analysis  │              │
│   │  - P&L Verification │             │  - Strategy Design  │              │
│   │  - Risk Scoring     │             │  - Greeks Calc      │              │
│   │  - Debate Challenge │             │  - Thesis Defense   │              │
│   │                     │             │                     │              │
│   └─────────────────────┘             └──────────┬──────────┘              │
│                                                  │                          │
│                                                  │ Internal                 │
│                                                  │ Coordination             │
│                                                  ▼                          │
│                                       ┌─────────────────────┐              │
│                                       │                     │              │
│                                       │    PM Agent         │              │
│                                       │  (Execution)        │              │
│                                       │                     │              │
│                                       │  - Order Execution  │              │
│                                       │  - Position Mgmt    │              │
│                                       │  - Risk Monitoring  │              │
│                                       │  - P&L Reporting    │              │
│                                       │                     │              │
│                                       └─────────────────────┘              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                           MCP Server Layer                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │  Options   │  │  Trading   │  │  Risk      │  │  Famous    │           │
│  │  Chain     │  │  Simulator │  │  Metrics   │  │  Traders   │           │
│  │  MCP       │  │  MCP       │  │  MCP       │  │  MCP       │           │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                           │
│  │  SEC       │  │  Yahoo     │  │  Python    │  (Existing)               │
│  │  EDGAR     │  │  Finance   │  │  Sandbox   │                           │
│  └────────────┘  └────────────┘  └────────────┘                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Agent Responsibilities

#### Judge Agent (Green - Evaluator)
| Responsibility | Description |
|----------------|-------------|
| Task Generation | Create dynamic options trading tasks |
| Ground Truth | Calculate expected P&L from historical data |
| Verification | Compare claimed vs actual returns |
| Debate | Challenge thesis with adversarial scenarios |
| Scoring | Calculate Alpha Score with risk adjustments |

#### Architect Agent (Purple - Strategy Designer)
| Responsibility | Description |
|----------------|-------------|
| Market Analysis | Analyze fundamentals, technicals, IV |
| Strategy Design | Create options strategies (spreads, straddles, etc.) |
| Greeks Management | Optimize portfolio Greeks |
| Thesis Creation | Articulate investment thesis |
| Defense | Rebut adversarial challenges |

#### PM Agent (Purple - Portfolio Manager)
| Responsibility | Description |
|----------------|-------------|
| Order Execution | Execute trades via Trading Simulator |
| Position Management | Track open positions, margin |
| Risk Monitoring | Monitor drawdown, exposure limits |
| P&L Reporting | Report realized/unrealized P&L |
| Rebalancing | Adjust positions per strategy |

---

## 3. Functional Requirements

### 3.1 Options Data Access

#### FR-OPT-001: Options Chain Retrieval
- **Description**: Retrieve real-time and historical options chains
- **Input**: Ticker, expiration date, option type (call/put)
- **Output**: Strike prices, bid/ask, volume, open interest, IV
- **Data Source**: Yahoo Finance (`yfinance` library)
- **Priority**: P0 (Critical)

#### FR-OPT-002: Greeks Calculation
- **Description**: Calculate option Greeks
- **Greeks**: Delta, Gamma, Theta, Vega, Rho
- **Method**: Black-Scholes model with dividend adjustment
- **Priority**: P0 (Critical)

#### FR-OPT-003: Historical Options Data
- **Description**: Access historical options prices for backtesting
- **Timeframe**: 2020-present (5 years)
- **Granularity**: Daily OHLCV
- **Priority**: P1 (High)

#### FR-OPT-004: Implied Volatility Surface
- **Description**: Retrieve IV across strikes and expirations
- **Output**: IV smile/skew visualization data
- **Priority**: P2 (Medium)

### 3.2 Trading Simulation

#### FR-SIM-001: Paper Trading Engine
- **Description**: Simulate options trades with realistic execution
- **Features**:
  - Market/Limit order types
  - Bid-ask spread simulation
  - Partial fills
  - Commission calculation ($0.65/contract)
- **Priority**: P0 (Critical)

#### FR-SIM-002: Position Tracking
- **Description**: Track all open positions
- **Data**:
  - Entry price, current price
  - Quantity, direction
  - Unrealized P&L
  - Days to expiration
- **Priority**: P0 (Critical)

#### FR-SIM-003: Historical Replay
- **Description**: Replay historical market conditions
- **Features**:
  - Set simulation date
  - Time-locked data access
  - Forward-only time progression
- **Priority**: P0 (Critical)

#### FR-SIM-004: Margin Calculation
- **Description**: Calculate margin requirements
- **Methods**:
  - Reg-T margin for naked options
  - Spread margin for defined-risk
- **Priority**: P1 (High)

#### FR-SIM-005: Exercise/Assignment
- **Description**: Handle option expiration
- **Features**:
  - Auto-exercise ITM options
  - Early assignment for American options
  - Cash settlement
- **Priority**: P2 (Medium)

### 3.3 Risk Management

#### FR-RISK-001: Portfolio Greeks
- **Description**: Aggregate Greeks across all positions
- **Output**: Net Delta, Gamma, Theta, Vega
- **Priority**: P0 (Critical)

#### FR-RISK-002: Value at Risk (VaR)
- **Description**: Calculate portfolio VaR
- **Methods**: Historical, Parametric, Monte Carlo
- **Confidence Levels**: 95%, 99%
- **Priority**: P1 (High)

#### FR-RISK-003: Drawdown Tracking
- **Description**: Track maximum drawdown
- **Features**:
  - Peak-to-trough calculation
  - Drawdown duration
  - Recovery tracking
- **Priority**: P0 (Critical)

#### FR-RISK-004: Risk Limits
- **Description**: Enforce position limits
- **Limits**:
  - Max position size (% of portfolio)
  - Max sector exposure
  - Max single-name exposure
  - Max drawdown trigger
- **Priority**: P1 (High)

### 3.4 Strategy Evaluation

#### FR-EVAL-001: P&L Verification
- **Description**: Verify agent's claimed P&L
- **Method**: Replay trades against historical prices
- **Tolerance**: ±1% for rounding
- **Priority**: P0 (Critical)

#### FR-EVAL-002: Risk-Adjusted Returns
- **Description**: Calculate risk-adjusted metrics
- **Metrics**:
  - Sharpe Ratio
  - Sortino Ratio
  - Calmar Ratio
  - Information Ratio
- **Priority**: P0 (Critical)

#### FR-EVAL-003: Strategy Classification
- **Description**: Classify strategy type
- **Types**:
  - Directional (bullish/bearish)
  - Volatility (long/short vol)
  - Income (premium collection)
  - Hedging (protective)
- **Priority**: P1 (High)

#### FR-EVAL-004: Benchmark Comparison
- **Description**: Compare to benchmark strategies
- **Benchmarks**:
  - Buy-and-hold underlying
  - Covered call index (BXM)
  - Put-write index (PUT)
- **Priority**: P2 (Medium)

### 3.5 Famous Trader Strategies

#### FR-COPY-001: Strategy Templates
- **Description**: Provide famous trader strategy rules
- **Traders**:
  | Trader | Strategy Type | Key Rules |
  |--------|--------------|-----------|
  | Warren Buffett | Value Covered Calls | Write calls on undervalued stocks |
  | George Soros | Macro Volatility | Event-driven vol trades |
  | Ray Dalio | All-Weather | Balanced risk parity |
  | Keith Gill (DFV) | Deep Value LEAPS | Long-dated calls on overlooked stocks |
  | Carl Icahn | Activist Catalyst | Calls before activism announcements |
- **Priority**: P1 (High)

#### FR-COPY-002: Strategy Adherence Scoring
- **Description**: Score how closely agent follows strategy
- **Dimensions**:
  - Entry criteria match
  - Position sizing
  - Exit criteria match
  - Risk management
- **Priority**: P1 (High)

### 3.6 Race to 10 Million

#### FR-RACE-001: Challenge Mode
- **Description**: Portfolio growth challenge
- **Parameters**:
  - Starting capital: $100,000
  - Target: $1,000,000 (10x)
  - Duration: 1 year simulated
  - Max drawdown: 30% (disqualification)
- **Priority**: P1 (High)

#### FR-RACE-002: Leaderboard
- **Description**: Rank agents by performance
- **Ranking Criteria**:
  1. Final portfolio value
  2. Risk-adjusted return (Sharpe)
  3. Max drawdown
  4. Number of trades (efficiency)
- **Priority**: P2 (Medium)

---

## 4. Non-Functional Requirements

### 4.1 Performance

| Requirement | Target |
|-------------|--------|
| Task Generation | < 5 seconds |
| Trade Execution | < 1 second |
| P&L Calculation | < 2 seconds |
| Full Evaluation | < 5 minutes |

### 4.2 Reliability

| Requirement | Target |
|-------------|--------|
| Data Accuracy | 99.9% match with source |
| Uptime | 99% during evaluation |
| Error Recovery | Automatic retry with backoff |

### 4.3 Scalability

| Requirement | Target |
|-------------|--------|
| Concurrent Evaluations | 10+ parallel |
| Historical Data Range | 5+ years |
| Options Universe | 500+ tickers |

### 4.4 Security

| Requirement | Implementation |
|-------------|----------------|
| API Key Protection | Environment variables |
| Sandbox Isolation | Docker containers |
| No Real Trading | Paper trading only |

---

## 5. Task Categories

### 5.1 New Options-Specific Categories

| Category | Description | Difficulty | Count |
|----------|-------------|------------|-------|
| **Options Pricing** | Calculate theoretical values | Medium | 50 |
| **Greeks Analysis** | Interpret and apply Greeks | Medium | 50 |
| **Strategy Construction** | Build multi-leg strategies | Hard | 40 |
| **Volatility Trading** | IV analysis and strategies | Hard | 30 |
| **P&L Attribution** | Decompose returns | Hard | 30 |
| **Risk Management** | Position sizing, hedging | Expert | 25 |
| **Copy Trading** | Replicate famous strategies | Hard | 25 |
| **Race to 10M** | Portfolio growth challenge | Expert | 10 |
| **Strategy Defense** | Adversarial debate | Expert | 20 |

### 5.2 Example Tasks

#### Options Pricing (Medium)
```
Task: Calculate the theoretical price of a 30-day AAPL $200 call option.

Given:
- Current stock price: $195.50
- Strike price: $200
- Days to expiration: 30
- Risk-free rate: 5.25%
- Implied volatility: 28%

Calculate the Black-Scholes price and explain each component.
```

#### Greeks Analysis (Medium)
```
Task: Your portfolio has the following Greeks:
- Delta: +500
- Gamma: +50
- Theta: -$200/day
- Vega: +$1,000

The market is expected to stay range-bound for the next 2 weeks.
What adjustments would you make and why?
```

#### Strategy Construction (Hard)
```
Task: Construct an iron condor on SPY for the next monthly expiration.

Requirements:
- Max risk: $5,000
- Target credit: $1.50+ per spread
- Wings at approximately ±1.5 standard deviations
- Probability of profit > 60%

Provide exact strikes, quantities, and Greeks analysis.
```

#### Copy Trading (Hard)
```
Task: Implement a Warren Buffett-style covered call strategy on KO (Coca-Cola).

Requirements:
- Analyze KO fundamentals to confirm it's "Buffett-worthy"
- Buy 1000 shares at current price
- Write covered calls at appropriate strike/expiry
- Calculate expected return if stock flat, up 10%, down 10%
- Explain how this aligns with Buffett's principles
```

#### Race to 10M (Expert)
```
Task: You have $100,000 starting capital. Design a 1-year options strategy
to reach $1,000,000 while maintaining max drawdown under 30%.

Requirements:
- Provide specific entry rules and position sizing
- Show monthly return targets
- Explain risk management approach
- Backtest on 2023 data
- Defend against "what if we have a 2008-style crash?" challenge
```

---

## 6. Scoring Rubrics

### 6.1 Alpha Score Formula

```
Alpha Score = (Strategy Score × Execution Score × Debate Multiplier)
            / (Risk Penalty × Cost Penalty × Temporal Penalty)

Where:
- Strategy Score: 0-100 (weighted: Thesis 30%, Greeks 25%, Sizing 25%, Exit 20%)
- Execution Score: 0-100 (weighted: P&L Accuracy 40%, Timing 30%, Slippage 15%, Costs 15%)
- Debate Multiplier: 0.5-1.2 (based on rebuttal quality)
- Risk Penalty: 1 + (DrawdownRatio × 0.5) where DrawdownRatio = MaxDD / 0.30
- Cost Penalty: ln(1 + CostUSD)
- Temporal Penalty: 1 + LookAheadViolations × 0.1
```

### 6.2 Strategy Score Rubric

| Component | Weight | 100 pts | 75 pts | 50 pts | 25 pts |
|-----------|--------|---------|--------|--------|--------|
| **Thesis Quality** | 30% | Clear catalyst, quantified target | Good thesis, some gaps | Basic thesis | Weak/missing thesis |
| **Greeks Awareness** | 25% | Optimal Greeks for strategy | Good Greeks management | Basic awareness | Greeks ignored |
| **Position Sizing** | 25% | Kelly-optimal or risk-based | Reasonable sizing | Arbitrary sizing | Oversized/undersized |
| **Exit Strategy** | 20% | Clear stops + targets | Stops OR targets | Vague exits | No exit plan |

### 6.3 Execution Score Rubric

| Component | Weight | 100 pts | 75 pts | 50 pts | 25 pts |
|-----------|--------|---------|--------|--------|--------|
| **P&L Accuracy** | 40% | Within 1% of actual | Within 5% | Within 10% | >10% error |
| **Trade Timing** | 30% | Optimal entry/exit | Good timing | Acceptable | Poor timing |
| **Slippage Handling** | 15% | Realistic assumptions | Minor optimism | Ignored slippage | Unrealistic |
| **Cost Efficiency** | 15% | Minimized costs | Reasonable costs | High costs | Excessive costs |

### 6.4 Risk-Adjusted Return Scoring

| Metric | Excellent (100) | Good (75) | Fair (50) | Poor (25) |
|--------|-----------------|-----------|-----------|-----------|
| **Sharpe Ratio** | > 2.0 | 1.5-2.0 | 1.0-1.5 | < 1.0 |
| **Sortino Ratio** | > 2.5 | 2.0-2.5 | 1.5-2.0 | < 1.5 |
| **Max Drawdown** | < 10% | 10-15% | 15-25% | > 25% |
| **Win Rate** | > 65% | 55-65% | 45-55% | < 45% |
| **Profit Factor** | > 2.0 | 1.5-2.0 | 1.2-1.5 | < 1.2 |

### 6.5 Debate Multiplier

| Rebuttal Quality | Multiplier | Criteria |
|------------------|------------|----------|
| **Exceptional** | 1.2x | New evidence, quantified defense, stress test |
| **Strong** | 1.1x | Good defense with supporting data |
| **Adequate** | 1.0x | Basic defense, repeats prior points |
| **Weak** | 0.8x | Poor defense, no new evidence |
| **Concession** | 0.5x | Admits flaw or contradicts prior position |

---

## 7. Data Models

### 7.1 Core Entities

```python
# Options Contract
class OptionsContract(BaseModel):
    ticker: str
    expiration: date
    strike: float
    option_type: Literal["call", "put"]

# Options Quote
class OptionsQuote(BaseModel):
    contract: OptionsContract
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

# Position
class Position(BaseModel):
    contract: OptionsContract
    quantity: int  # Positive = long, Negative = short
    entry_price: float
    entry_date: datetime
    current_price: float
    unrealized_pnl: float

# Trade
class Trade(BaseModel):
    trade_id: str
    contract: OptionsContract
    action: Literal["buy_to_open", "sell_to_open", "buy_to_close", "sell_to_close"]
    quantity: int
    price: float
    commission: float
    timestamp: datetime

# Portfolio
class Portfolio(BaseModel):
    cash: float
    positions: List[Position]
    trades: List[Trade]
    realized_pnl: float
    unrealized_pnl: float
    total_value: float

# Strategy
class OptionsStrategy(BaseModel):
    name: str
    type: Literal["directional", "volatility", "income", "hedging"]
    legs: List[Position]
    max_profit: Optional[float]
    max_loss: Optional[float]
    breakeven: List[float]
    probability_of_profit: float
```

### 7.2 Task Models

```python
class OptionsTask(BaseModel):
    task_id: str
    category: OptionsTaskCategory
    difficulty: Literal["easy", "medium", "hard", "expert"]
    question: str
    context: Dict[str, Any]  # Market data, constraints
    ground_truth: Optional[Dict[str, Any]]
    simulation_date: date
    deadline_seconds: int
    scoring_rubric: Dict[str, float]

class OptionsTaskCategory(str, Enum):
    OPTIONS_PRICING = "options_pricing"
    GREEKS_ANALYSIS = "greeks_analysis"
    STRATEGY_CONSTRUCTION = "strategy_construction"
    VOLATILITY_TRADING = "volatility_trading"
    PNL_ATTRIBUTION = "pnl_attribution"
    RISK_MANAGEMENT = "risk_management"
    COPY_TRADING = "copy_trading"
    RACE_TO_10M = "race_to_10m"
    STRATEGY_DEFENSE = "strategy_defense"
```

---

## 8. API Specifications

### 8.1 Options Chain MCP

```python
# Tool: get_options_chain
@mcp.tool()
async def get_options_chain(
    ticker: str,
    expiration: Optional[str] = None,  # YYYY-MM-DD or "nearest"
    option_type: Optional[Literal["call", "put"]] = None
) -> OptionsChainResponse:
    """Get options chain for a ticker"""

# Tool: get_greeks
@mcp.tool()
async def get_greeks(
    ticker: str,
    strike: float,
    expiration: str,
    option_type: Literal["call", "put"]
) -> GreeksResponse:
    """Calculate Greeks for an option"""

# Tool: calculate_option_price
@mcp.tool()
async def calculate_option_price(
    spot_price: float,
    strike_price: float,
    days_to_expiry: int,
    risk_free_rate: float,
    volatility: float,
    option_type: Literal["call", "put"],
    dividend_yield: float = 0.0
) -> BlackScholesResponse:
    """Calculate theoretical option price using Black-Scholes"""
```

### 8.2 Trading Simulator MCP

```python
# Tool: execute_trade
@mcp.tool()
async def execute_trade(
    ticker: str,
    strike: float,
    expiration: str,
    option_type: Literal["call", "put"],
    action: Literal["buy", "sell"],
    quantity: int,
    order_type: Literal["market", "limit"] = "market",
    limit_price: Optional[float] = None
) -> TradeExecutionResponse:
    """Execute an options trade"""

# Tool: get_portfolio
@mcp.tool()
async def get_portfolio() -> PortfolioResponse:
    """Get current portfolio state"""

# Tool: get_position_pnl
@mcp.tool()
async def get_position_pnl(
    position_id: Optional[str] = None
) -> PnLResponse:
    """Get P&L for positions"""

# Tool: close_position
@mcp.tool()
async def close_position(
    position_id: str,
    quantity: Optional[int] = None  # None = close all
) -> TradeExecutionResponse:
    """Close an existing position"""
```

### 8.3 Risk Metrics MCP

```python
# Tool: calculate_portfolio_greeks
@mcp.tool()
async def calculate_portfolio_greeks() -> PortfolioGreeksResponse:
    """Calculate aggregate portfolio Greeks"""

# Tool: calculate_var
@mcp.tool()
async def calculate_var(
    confidence_level: float = 0.95,
    horizon_days: int = 1,
    method: Literal["historical", "parametric", "monte_carlo"] = "historical"
) -> VaRResponse:
    """Calculate Value at Risk"""

# Tool: calculate_sharpe
@mcp.tool()
async def calculate_sharpe(
    returns: List[float],
    risk_free_rate: float = 0.05
) -> float:
    """Calculate Sharpe ratio"""

# Tool: calculate_max_drawdown
@mcp.tool()
async def calculate_max_drawdown(
    portfolio_values: List[float]
) -> DrawdownResponse:
    """Calculate maximum drawdown"""
```

---

## 9. Acceptance Criteria

### 9.1 Phase 1: Core Infrastructure

- [ ] Options Chain MCP returns accurate data for 100 test tickers
- [ ] Trading Simulator executes trades with correct P&L
- [ ] Risk Metrics calculates Sharpe within 1% of reference
- [ ] Historical replay works for 2020-2025 date range
- [ ] Docker compose starts all services without errors

### 9.2 Phase 2: Agent Implementation

- [ ] Architect Agent generates valid options strategies
- [ ] PM Agent executes trades and tracks positions
- [ ] Judge Agent verifies P&L accuracy
- [ ] A2A protocol works between all agents
- [ ] Debate phase challenges and scores rebuttals

### 9.3 Phase 3: Benchmark Tasks

- [ ] 50+ tasks per category (9 categories = 450+ total)
- [ ] Dynamic task generation with ticker substitution
- [ ] Ground truth calculation for all task types
- [ ] Scoring rubrics implemented and tested
- [ ] Baseline purple agent achieves 30%+ accuracy

### 9.4 Phase 4: Competition Submission

- [ ] README with clear setup instructions
- [ ] Demo video (< 3 minutes)
- [ ] Docker image runs end-to-end
- [ ] Baseline agents registered on AgentBeats
- [ ] All tests passing

---

## 10. Dependencies

### 10.1 External APIs

| API | Purpose | Cost | Rate Limit |
|-----|---------|------|------------|
| Yahoo Finance | Options chains, prices | Free | 2000/hour |
| FRED | Risk-free rates | Free | 120/min |
| Alpha Vantage | Historical data (backup) | Free tier | 5/min |

### 10.2 Python Libraries

```toml
[project.dependencies]
# Existing
yfinance = ">=0.2.30"
pandas = ">=2.0.0"
numpy = ">=1.24.0"

# New for options
scipy = ">=1.10.0"          # Black-Scholes
mibian = ">=0.1.3"          # Options pricing
py_vollib = ">=1.0.1"       # Volatility calculations

# Risk metrics
empyrical = ">=0.5.5"       # Performance metrics
pyfolio = ">=0.9.2"         # Portfolio analysis
```

### 10.3 Internal Dependencies

```
Options Chain MCP
    └── Yahoo Finance MCP (existing)

Trading Simulator MCP
    └── Options Chain MCP
    └── Risk Metrics MCP

Architect Agent
    └── Options Chain MCP
    └── SEC EDGAR MCP (existing)
    └── Yahoo Finance MCP (existing)

PM Agent
    └── Trading Simulator MCP
    └── Risk Metrics MCP

Judge Agent
    └── Trading Simulator MCP (for verification)
    └── Risk Metrics MCP
```

---

## 11. Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Yahoo Finance API changes | High | Medium | Abstract data layer, backup APIs |
| Historical options data gaps | Medium | High | Interpolation, synthetic data |
| Complex Greeks calculations | Medium | Medium | Use proven libraries (mibian) |
| Agent coordination failures | High | Low | Robust A2A error handling |
| Evaluation time > 5 min | Medium | Medium | Parallel execution, caching |

---

## 12. Timeline

### Phase 1: Core Infrastructure (Week 1-2)
- Options Chain MCP
- Trading Simulator MCP
- Risk Metrics MCP
- Basic integration tests

### Phase 2: Agent Implementation (Week 3-4)
- Architect Agent (Claude-based)
- PM Agent (Claude-based)
- Judge Agent updates
- A2A protocol integration

### Phase 3: Benchmark Tasks (Week 5-6)
- Task generation for all categories
- Scoring rubric implementation
- Famous Trader strategies
- Race to 10M mode

### Phase 4: Polish (Week 7-8)
- Docker optimization
- Documentation
- Demo video
- Competition submission

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **ATM** | At-the-money: strike ≈ current price |
| **ITM** | In-the-money: call strike < price, put strike > price |
| **OTM** | Out-of-the-money: call strike > price, put strike < price |
| **IV** | Implied Volatility |
| **Greeks** | Delta, Gamma, Theta, Vega, Rho |
| **Iron Condor** | Short strangle + long wings |
| **LEAPS** | Long-term equity anticipation securities (>1 year options) |
| **Covered Call** | Long stock + short call |
| **Protective Put** | Long stock + long put |

## Appendix B: References

1. Finance Agent Benchmark (arXiv:2508.00828)
2. TradingAgents (arXiv:2412.20138)
3. A2A Protocol Specification
4. Black-Scholes-Merton Model
5. CBOE Options Institute
