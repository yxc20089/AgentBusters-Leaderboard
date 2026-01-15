# Options Trading MCP Servers

This module provides MCP (Model Context Protocol) servers for options trading simulation in the AgentBusters FAB++ benchmark.

## Servers

### 1. Options Chain Server (`options_chain.py`)

Provides options pricing and chain data using Black-Scholes model.

**Port:** 8104

**Tools:**
- `get_options_chain` - Get full options chain for a ticker
- `calculate_option_price` - Black-Scholes pricing with Greeks
- `get_expirations` - List available expiration dates
- `get_iv_surface` - Implied volatility surface
- `analyze_strategy` - Multi-leg strategy analysis

### 2. Trading Simulator Server (`trading_sim.py`)

Paper trading engine with realistic execution simulation.

**Port:** 8105

**Tools:**
- `create_portfolio` - Create new trading portfolio
- `execute_options_trade` - Execute buy/sell orders with slippage
- `get_positions` - View current positions
- `close_position` - Close an existing position
- `advance_time` - Move simulation forward
- `get_pnl_report` - P&L breakdown

### 3. Risk Metrics Server (`risk_metrics.py`)

Portfolio risk analytics and stress testing.

**Port:** 8106

**Tools:**
- `calculate_portfolio_greeks` - Aggregate Greeks across positions
- `calculate_var` - Value at Risk (historical, parametric, Monte Carlo)
- `calculate_risk_metrics` - Sharpe, Sortino, Calmar ratios
- `calculate_max_drawdown` - Maximum drawdown with recovery
- `run_stress_test` - Scenario stress testing
- `calculate_pnl_attribution` - Decompose returns by Greek

## Running

### Via Docker Compose (recommended)

```bash
docker compose up -d options-chain-mcp trading-sim-mcp risk-metrics-mcp
```

### Standalone (development)

```bash
# Options Chain
fastmcp run src/mcp_servers/options_chain.py:create_options_chain_server \
    --transport http --host 0.0.0.0 --port 8104

# Trading Simulator
fastmcp run src/mcp_servers/trading_sim.py:create_trading_sim_server \
    --transport http --host 0.0.0.0 --port 8105

# Risk Metrics
fastmcp run src/mcp_servers/risk_metrics.py:create_risk_metrics_server \
    --transport http --host 0.0.0.0 --port 8106
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SIMULATION_DATE` | (current) | Override simulation date (YYYY-MM-DD) |
| `STARTING_CAPITAL` | 100000 | Initial portfolio capital |
| `COMMISSION_PER_CONTRACT` | 0.65 | Trading commission per contract |

## Black-Scholes Model

The options pricing uses the Black-Scholes-Merton model with dividend yield:

```
d1 = (ln(S/K) + (r - q + sigma^2/2) * T) / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)

Call = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
Put  = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)
```

### Greeks

| Greek | Formula | Description |
|-------|---------|-------------|
| Delta | dV/dS | Price sensitivity to underlying |
| Gamma | d2V/dS2 | Delta's rate of change |
| Theta | dV/dt | Time decay per day |
| Vega | dV/d(sigma) | Volatility sensitivity |
| Rho | dV/dr | Interest rate sensitivity |

## Integration

### Purple Agent

The `MCPToolkit` class in `src/purple_agent/mcp_toolkit.py` provides high-level access:

```python
toolkit = MCPToolkit()

# Get options chain
chain = await toolkit.get_options_chain("SPY", expiration="2025-02-21")

# Execute trade
trade = await toolkit.execute_options_trade(
    ticker="SPY",
    strike=500,
    expiration="2025-02-21",
    option_type="call",
    action="buy",
    quantity=10
)

# Get portfolio Greeks
greeks = await toolkit.get_portfolio_greeks()
```

### Green Agent (Evaluator)

The `OptionsEvaluator` in `src/evaluators/options.py` scores responses:

- P&L Accuracy (25%)
- Greeks Accuracy (25%)
- Strategy Quality (25%)
- Risk Management (25%)

## Demo

Run the options demo script:

```bash
python scripts/run_options_demo.py --task iron_condor --ticker SPY
python scripts/run_options_demo.py --task volatility --ticker TSLA
python scripts/run_options_demo.py --task all --ticker SPY
```
