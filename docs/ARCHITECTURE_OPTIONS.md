# Options Trading Architecture for AgentBusters Alpha Challenge

**Version**: 2.0.0
**Status**: ✅ Implementation Complete
**Updated**: 2026-01-15

---

## 1. Overview

This document describes the architecture for extending AgentBusters FAB++ benchmark with options trading simulation capabilities. The design extends the existing CIO-Agent (Green) and Purple Agent architecture with new MCP servers for options data, trading simulation, and risk metrics.

**Note**: "PM Agent" and "Architect Agent" mentioned in requirements refer to Claude Code subagents used for *designing* this system, not components within it.

## 2. Component Diagram

```
+==========================================================================+
|                    AgentBusters Options Alpha Challenge                   |
+==========================================================================+
|                                                                           |
|   GREEN SIDE (Evaluator)                   PURPLE SIDE (Competitors)      |
|   +-----------------------------+          +----------------------------+ |
|   |   CIO-Agent (Enhanced)      |   A2A    |   FinanceAnalysisAgent     | |
|   |   src/cio_agent/            |<-------->|   src/purple_agent/        | |
|   +-----------------------------+ Protocol +----------------------------+ |
|   |                             |          |                            | |
|   | ComprehensiveEvaluator      |          | FinanceAgentExecutor       | |
|   | ├─ DynamicTaskGenerator     |          | ├─ Task parsing            | |
|   | │  └─ + OptionsTaskGen      |          | ├─ Data gathering          | |
|   | ├─ MacroEvaluator (30%)     |          | ├─ Analysis generation     | |
|   | ├─ FundamentalEvaluator(40%)|          | └─ + Options trading       | |
|   | ├─ ExecutionEvaluator (30%) |          |                            | |
|   | │  └─ + OptionsEvaluator    |          | MCPToolkit                 | |
|   | ├─ AdversarialDebateManager |          | ├─ SEC EDGAR tools         | |
|   | └─ CostTracker              |          | ├─ Yahoo Finance tools     | |
|   |                             |          | ├─ Sandbox tools           | |
|   | A2AOrchestrator             |          | └─ + Options tools         | |
|   | └─ Task dispatch & collect  |          |    └─ Trading Sim          | |
|   +-----------------------------+          +----------------------------+ |
|                                                                           |
+==========================================================================+
|                           MCP Server Layer                                |
+==========================================================================+
|                                                                           |
|  NEW MCP SERVERS (Options Trading)                                        |
|  +-------------------+  +--------------------+  +--------------------+    |
|  | options_chain.py  |  | trading_sim.py     |  | risk_metrics.py    |    |
|  +-------------------+  +--------------------+  +--------------------+    |
|  | get_options_chain |  | create_portfolio   |  | portfolio_greeks   |    |
|  | get_greeks        |  | execute_trade      |  | calculate_var      |    |
|  | calculate_bs      |  | get_positions      |  | calculate_sharpe   |    |
|  | get_iv_surface    |  | close_position     |  | max_drawdown       |    |
|  | get_expirations   |  | advance_time       |  | stress_test        |    |
|  | analyze_strategy  |  | get_pnl_report     |  | pnl_attribution    |    |
|  +-------------------+  +--------------------+  +--------------------+    |
|          │                       │                       │                |
|          └───────────────────────┴───────────────────────┘                |
|                                  │                                        |
|  EXISTING MCP SERVERS            │                                        |
|  +-------------------+  +--------------------+  +--------------------+    |
|  | sec_edgar.py      |  | yahoo_finance.py   |  | sandbox.py         |    |
|  +-------------------+  +--------------------+  +--------------------+    |
|  | get_filing        |  | get_stock_info     |  | execute_python     |    |
|  | search_filings    |  | get_financials     |  | (calculations)     |    |
|  | get_company_info  |  | get_price_history  |  |                    |    |
|  | (temporal lock)   |  | (temporal lock)    |  |                    |    |
|  +-------------------+  +--------------------+  +--------------------+    |
|                                                                           |
+==========================================================================+
```

## 3. Data Flow Architecture

### 3.1 Options Task Execution Flow

```
CIO-Agent (Green/Evaluator)              Purple Agent (Competitor)
      │                                           │
      ├──1. Generate Options Task ────────────────│
      │   (DynamicTaskGenerator + OptionsTaskGen) │
      │                                           │
      ├──2. Send Task via A2A ───────────────────>│
      │                                           │
      │                                           ├──3. Parse Task
      │                                           │   (category, ticker, constraints)
      │                                           │
      │                                           ├──4. Gather Market Data
      │                                           │   ├── get_options_chain()
      │                                           │   ├── get_stock_info()
      │                                           │   └── get_iv_surface()
      │                                           │
      │                                           ├──5. Design Strategy (LLM)
      │                                           │   └── Select strategy type
      │                                           │
      │                                           ├──6. Execute Trades
      │                                           │   ├── execute_trade()
      │                                           │   └── (realistic slippage)
      │                                           │
      │                                           ├──7. Monitor & Close
      │                                           │   ├── advance_time()
      │                                           │   └── close_position()
      │                                           │
      │<─8. Return Response ──────────────────────┤
      │   (strategy, trades, P&L, thesis)         │
      │                                           │
      ├──9. Generate Challenge ───────────────────│
      │   (AdversarialDebateManager)              │
      │                                           │
      ├──10. Send Challenge ─────────────────────>│
      │                                           │
      │<─11. Receive Rebuttal ────────────────────┤
      │                                           │
      ├──12. Evaluate ────────────────────────────│
      │   ├── Verify P&L against market data      │
      │   ├── Check Greeks accuracy               │
      │   ├── Score strategy quality              │
      │   ├── Apply debate multiplier             │
      │   └── Calculate Alpha Score               │
      │                                           │
      └──13. Report Results ──────────────────────┘
```

## 4. New MCP Server Specifications

### 4.1 Options Chain MCP Server (`src/mcp_servers/options_chain.py`)

```python
@mcp.tool
def get_options_chain(
    ticker: str,
    expiration: str | None = None,  # YYYY-MM-DD or "nearest" or "all"
    option_type: Literal["call", "put", "all"] = "all",
    min_strike: float | None = None,
    max_strike: float | None = None,
) -> dict[str, Any]:
    """Get options chain for a ticker with full Greeks."""

@mcp.tool
def calculate_option_price(
    spot_price: float,
    strike_price: float,
    days_to_expiry: int,
    risk_free_rate: float,
    volatility: float,
    option_type: Literal["call", "put"],
    dividend_yield: float = 0.0,
) -> dict[str, Any]:
    """Calculate theoretical option price using Black-Scholes model."""

@mcp.tool
def get_iv_surface(ticker: str, expirations: list[str] | None = None) -> dict[str, Any]:
    """Get implied volatility surface across strikes and expirations."""

@mcp.tool
def get_expirations(ticker: str) -> list[str]:
    """Get all available expiration dates for options."""

@mcp.tool
def analyze_strategy(legs: list[dict]) -> dict[str, Any]:
    """Analyze a multi-leg options strategy for max profit/loss, breakevens."""
```

### 4.2 Trading Simulator MCP Server (`src/mcp_servers/trading_simulator.py`)

```python
@mcp.tool
def execute_trade(
    ticker: str,
    strike: float,
    expiration: str,
    option_type: Literal["call", "put"],
    action: Literal["buy", "sell"],
    quantity: int,
    order_type: Literal["market", "limit"] = "market",
    limit_price: float | None = None,
) -> dict[str, Any]:
    """Execute an options trade with realistic simulation."""

@mcp.tool
def get_portfolio() -> dict[str, Any]:
    """Get current portfolio state with all positions."""

@mcp.tool
def get_position_pnl(position_id: str | None = None) -> dict[str, Any]:
    """Get P&L breakdown for positions."""

@mcp.tool
def close_position(position_id: str, quantity: int | None = None) -> dict[str, Any]:
    """Close an existing position."""

@mcp.tool
def advance_time(days: int) -> dict[str, Any]:
    """Advance simulation time and update positions."""
```

### 4.3 Risk Metrics MCP Server (`src/mcp_servers/risk_metrics.py`)

```python
@mcp.tool
def calculate_portfolio_greeks() -> dict[str, Any]:
    """Calculate aggregate Greeks across all positions."""

@mcp.tool
def calculate_var(
    confidence_level: float = 0.95,
    horizon_days: int = 1,
    method: Literal["historical", "parametric", "monte_carlo"] = "historical",
) -> dict[str, Any]:
    """Calculate Value at Risk for current portfolio."""

@mcp.tool
def calculate_risk_metrics(returns: list[float], risk_free_rate: float = 0.05) -> dict[str, Any]:
    """Calculate Sharpe, Sortino, Calmar ratios."""

@mcp.tool
def calculate_max_drawdown(portfolio_values: list[float]) -> dict[str, Any]:
    """Calculate maximum drawdown with recovery analysis."""

@mcp.tool
def stress_test(scenarios: list[dict]) -> dict[str, Any]:
    """Run stress tests on current portfolio."""
```

## 5. Agent Extensions

### 5.1 Purple Agent Extensions (FinanceAgentExecutor)

The existing `FinanceAgentExecutor` in `src/purple_agent/executor.py` will be extended to handle options trading tasks:

```python
# Extensions to FinanceAgentExecutor

async def _handle_options_task(self, task_info: dict, mcp_toolkit: MCPToolkit) -> dict:
    """Handle options-specific task categories."""

    category = task_info.get("category")

    if category == "strategy_construction":
        return await self._construct_strategy(task_info)
    elif category == "race_to_10m":
        return await self._run_trading_simulation(task_info)
    elif category == "copy_trading":
        return await self._execute_famous_strategy(task_info)
    # ... other categories

async def _construct_strategy(self, task_info: dict) -> dict:
    """Build and execute multi-leg options strategy."""
    # 1. Get options chain via MCP
    # 2. Analyze IV and Greeks
    # 3. Select strategy type
    # 4. Execute trades via trading_sim MCP
    # 5. Return strategy with P&L projection

async def _run_trading_simulation(self, task_info: dict) -> dict:
    """Run Race to 10M simulation over time period."""
    # 1. Create portfolio via trading_sim MCP
    # 2. Loop: analyze -> trade -> advance_time
    # 3. Calculate final P&L and risk metrics
    # 4. Return complete trading history
```

### 5.2 CIO-Agent Extensions (ComprehensiveEvaluator)

The existing `ComprehensiveEvaluator` in `src/cio_agent/evaluator.py` will be extended:

```python
# Extensions to ComprehensiveEvaluator

class OptionsEvaluator:
    """Additional evaluator for options-specific scoring."""

    def verify_pnl(self, reported_pnl: float, trades: list, market_data: dict) -> float:
        """Verify P&L calculations against real market data."""

    def verify_greeks(self, reported_greeks: dict, positions: list) -> float:
        """Verify Greeks calculations are accurate."""

    def score_strategy_quality(self, strategy: dict, market_context: dict) -> float:
        """Score strategy appropriateness for market conditions."""

    def score_risk_management(self, portfolio: dict, limits: dict) -> float:
        """Score risk management discipline."""
```

### 5.3 MCPToolkit Extensions

The existing `MCPToolkit` in `src/purple_agent/mcp_toolkit.py` will add options tools:

```python
# Extensions to MCPToolkit

async def get_options_chain(self, ticker: str, expiration: str = None) -> dict:
    """Get options chain from options_chain MCP server."""

async def execute_options_trade(self, contract: dict, action: str, quantity: int) -> dict:
    """Execute trade via trading_sim MCP server."""

async def get_portfolio_greeks(self) -> dict:
    """Get portfolio Greeks from risk_metrics MCP server."""
```

## 6. Task Categories

| Category | Description | Difficulty | Example |
|----------|-------------|------------|---------|
| **Options Pricing** | Calculate theoretical values | Medium | "Price a 30-day AAPL $200 call" |
| **Greeks Analysis** | Interpret and apply Greeks | Medium | "Portfolio has -500 Delta. How to hedge?" |
| **Strategy Construction** | Build multi-leg strategies | Hard | "Construct iron condor on SPY" |
| **Volatility Trading** | IV analysis and strategies | Hard | "Is TSLA IV cheap vs historical?" |
| **P&L Attribution** | Decompose returns by Greek | Hard | "Decompose 15% return" |
| **Risk Management** | Position sizing, hedging | Expert | "Size strangle within $10K VaR" |
| **Copy Trading** | Replicate famous strategies | Hard | "Execute Buffett-style covered calls" |
| **Race to 10M** | Portfolio growth challenge | Expert | "Grow $100K to $1M in 1 year" |
| **Strategy Defense** | Adversarial debate | Expert | "Defend short vol against crash" |

## 7. Famous Trader Strategy Templates

```python
FAMOUS_TRADER_STRATEGIES = {
    "buffett_covered_calls": {
        "name": "Warren Buffett Style Covered Calls",
        "type": "income",
        "entry_rules": ["P/E below sector median", "Dividend yield > 2%", "Wide moat"],
    },
    "soros_macro_vol": {
        "name": "George Soros Style Macro Volatility",
        "type": "volatility",
        "entry_rules": ["Major macro catalyst", "IV rank below 50"],
    },
    "dfv_deep_value_leaps": {
        "name": "Keith Gill (DFV) Style Deep Value LEAPS",
        "type": "directional",
        "entry_rules": [">50% discount to intrinsic", "Strong catalyst"],
    },
}
```

## 8. Docker Compose Extension

New services to add to existing `docker-compose.yml`:

```yaml
services:
  # ... existing services (cio-agent, purple-agent, etc.)

  # NEW: Options Chain MCP Server
  options-chain-mcp:
    build:
      context: .
      dockerfile: Dockerfile.mcp
    command: python -m mcp_servers.options_chain
    ports:
      - "8104:8000"
    environment:
      - SIMULATION_DATE=${SIMULATION_DATE:-}
    depends_on:
      - yahoo-finance-mcp
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # NEW: Trading Simulator MCP Server
  trading-simulator-mcp:
    build:
      context: .
      dockerfile: Dockerfile.mcp
    command: python -m mcp_servers.trading_sim
    ports:
      - "8105:8000"
    environment:
      - STARTING_CAPITAL=100000
      - COMMISSION_PER_CONTRACT=0.65
      - SIMULATION_DATE=${SIMULATION_DATE:-}
    volumes:
      - trading-data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # NEW: Risk Metrics MCP Server
  risk-metrics-mcp:
    build:
      context: .
      dockerfile: Dockerfile.mcp
    command: python -m mcp_servers.risk_metrics
    ports:
      - "8106:8000"
    depends_on:
      - trading-simulator-mcp

volumes:
  trading-data:
```

## 9. Implementation Status

All phases have been completed. Here's the implementation summary:

### Phase 1: MCP Servers ✅
- ✅ `src/mcp_servers/options_chain.py` - Black-Scholes pricing, Greeks, options chains
- ✅ `src/mcp_servers/trading_sim.py` - Paper trading with slippage, portfolios, P&L tracking
- ✅ `src/mcp_servers/risk_metrics.py` - Portfolio Greeks, VaR, Sharpe/Sortino, stress testing
- ✅ Docker services configured (ports 8104, 8105, 8106)

### Phase 2: Agent Extensions ✅
- ✅ `FinanceAgentExecutor` extended with 7 options task handlers
- ✅ `MCPToolkit` extended with 21 options-related methods
- ✅ `OptionsEvaluator` created in `src/evaluators/options.py`
- ✅ `ComprehensiveEvaluator` integrated with options scoring
- ✅ 9 new `TaskCategory` enums added for options

### Phase 3: Benchmark Tasks ✅
- ✅ 18 options task templates in `DynamicTaskGenerator`
- ✅ Scoring rubrics for P&L, Greeks, Strategy, Risk Management
- ✅ All options categories supported

### Phase 4: Demo & Testing ✅
- ✅ `scripts/run_options_demo.py` - Full demo pipeline
- ✅ Docker build and health checks verified
- ✅ All 3 MCP servers running and healthy

## 10. Running the Options Demo

```bash
# Start all services
docker compose up -d

# Run a single task type
python scripts/run_options_demo.py --task iron_condor --ticker SPY

# Run all task types
python scripts/run_options_demo.py --task all --ticker SPY
```

### Available Task Types
- `iron_condor` - Strategy construction for neutral market
- `volatility` - IV analysis and volatility trading
- `greeks` - Delta hedging and portfolio Greeks
- `risk` - VaR-based position sizing

## 11. Docker Service Ports

| Service | Container | Port |
|---------|-----------|------|
| SEC EDGAR MCP | fab-plus-edgar | 8101 |
| Yahoo Finance MCP | fab-plus-yfinance | 8102 |
| Sandbox MCP | fab-plus-sandbox | 8103 |
| **Options Chain MCP** | fab-plus-options-chain | **8104** |
| **Trading Sim MCP** | fab-plus-trading-sim | **8105** |
| **Risk Metrics MCP** | fab-plus-risk-metrics | **8106** |
| Purple Agent | fab-plus-purple-agent | 8010 |
| CIO-Agent | fab-plus-orchestrator | 8080 |
