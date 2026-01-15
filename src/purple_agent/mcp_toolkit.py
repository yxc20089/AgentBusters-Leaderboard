"""
MCP Toolkit for Purple Agent

Uses the actual FastMCP server implementations directly (in-process)
for real financial data access with temporal locking.
"""

import asyncio
import os
from datetime import datetime
from typing import Any
from dataclasses import dataclass, field

import httpx

# Import the MCP server factories (used when no remote MCP URLs provided)
from mcp_servers.sec_edgar import create_edgar_server
from mcp_servers.yahoo_finance import create_yahoo_finance_server
from mcp_servers.sandbox import create_sandbox_server

# Options trading MCP servers
from mcp_servers.options_chain import create_options_chain_server
from mcp_servers.trading_sim import create_trading_sim_server
from mcp_servers.risk_metrics import create_risk_metrics_server


@dataclass
class MCPToolMetrics:
    """Metrics for MCP tool usage."""
    tool_calls: int = 0
    errors: int = 0
    total_time_ms: int = 0
    calls_by_tool: dict = field(default_factory=dict)


@dataclass
class FinancialData:
    """Financial data from MCP sources."""
    ticker: str
    period: str = ""
    revenue: float | None = None
    net_income: float | None = None
    gross_profit: float | None = None
    gross_margin: float | None = None
    operating_income: float | None = None
    operating_margin: float | None = None
    eps: float | None = None
    eps_diluted: float | None = None
    total_assets: float | None = None
    total_liabilities: float | None = None
    total_equity: float | None = None
    cash: float | None = None
    debt: float | None = None
    market_cap: float | None = None
    price: float | None = None
    pe_ratio: float | None = None
    extra: dict = field(default_factory=dict)


class MCPToolkit:
    """
    MCP Toolkit using FastMCP servers directly.

    This toolkit creates in-process MCP servers and calls their tools
    directly for real financial data access.
    """

    def __init__(self, simulation_date: datetime | None = None):
        """
        Initialize the MCP Toolkit.

        Args:
            simulation_date: Optional date for temporal locking
        """
        self.simulation_date = simulation_date

        # Prefer remote MCP endpoints if provided; otherwise fall back to in-process servers.
        self._edgar_url = os.environ.get("MCP_EDGAR_URL")
        self._yfinance_url = os.environ.get("MCP_YFINANCE_URL")
        self._sandbox_url = os.environ.get("MCP_SANDBOX_URL")
        self._http_client = httpx.AsyncClient(timeout=30)

        self._edgar_server = None if self._edgar_url else create_edgar_server(simulation_date=simulation_date)
        self._yfinance_server = None if self._yfinance_url else create_yahoo_finance_server(simulation_date=simulation_date)
        self._sandbox_server = None if self._sandbox_url else create_sandbox_server()

        # Options trading MCP servers (always in-process for now)
        self._options_chain_server = create_options_chain_server(simulation_date=simulation_date)
        self._trading_sim_server = create_trading_sim_server(simulation_date=simulation_date)
        self._risk_metrics_server = create_risk_metrics_server()

        # Metrics
        self._metrics = MCPToolMetrics()
        self._tool_calls: list[dict] = []

    async def _get_tools(self):
        """Get tools from all servers (only used in local mode)."""
        edgar_tools = await self._edgar_server.get_tools() if self._edgar_server else None
        yfinance_tools = await self._yfinance_server.get_tools() if self._yfinance_server else None
        sandbox_tools = await self._sandbox_server.get_tools() if self._sandbox_server else None
        options_chain_tools = await self._options_chain_server.get_tools()
        trading_sim_tools = await self._trading_sim_server.get_tools()
        risk_metrics_tools = await self._risk_metrics_server.get_tools()
        return {
            "edgar": edgar_tools,
            "yfinance": yfinance_tools,
            "sandbox": sandbox_tools,
            "options_chain": options_chain_tools,
            "trading_sim": trading_sim_tools,
            "risk_metrics": risk_metrics_tools,
        }

    def _record_call(self, server: str, tool: str, result: Any, time_ms: int = 0):
        """Record a tool call for metrics."""
        self._metrics.tool_calls += 1
        self._metrics.total_time_ms += time_ms
        key = f"{server}:{tool}"
        self._metrics.calls_by_tool[key] = self._metrics.calls_by_tool.get(key, 0) + 1
        self._tool_calls.append({
            "server": server,
            "tool": tool,
            "timestamp": datetime.now().isoformat(),
            "time_ms": time_ms,
        })

    # =========================================================================
    # Yahoo Finance Tools
    # =========================================================================

    async def get_quote(self, ticker: str) -> dict[str, Any]:
        """
        Get current stock quote from Yahoo Finance MCP.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Stock quote with price, market cap, ratios
        """
        import time
        start = time.time()

        if self._yfinance_url:
            resp = await self._http_client.post(
                f"{self._yfinance_url}/tools/get_quote",
                json={"ticker": ticker},
            )
            resp.raise_for_status()
            result = resp.json()
        else:
            tools = await self._yfinance_server.get_tools()
            get_quote = tools["get_quote"]
            result = get_quote.fn(ticker=ticker)

        elapsed = int((time.time() - start) * 1000)

        self._record_call("yfinance", "get_quote", result, elapsed)
        return result

    async def get_historical_prices(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> list[dict]:
        """
        Get historical price data from Yahoo Finance MCP.

        Args:
            ticker: Stock ticker symbol
            period: Time period (1mo, 3mo, 6mo, 1y, 2y, 5y)
            interval: Data interval (1d, 1wk, 1mo)

        Returns:
            List of historical price data points
        """
        tools = await self._yfinance_server.get_tools()
        get_historical = tools["get_historical_prices"]

        import time
        start = time.time()
        result = get_historical.fn(ticker=ticker, period=period, interval=interval)
        elapsed = int((time.time() - start) * 1000)

        self._record_call("yfinance", "get_historical_prices", result, elapsed)
        return result

    async def get_financials(
        self,
        ticker: str,
        statement_type: str = "income",
        period: str = "quarterly",
    ) -> dict[str, Any]:
        """
        Get financial statements from Yahoo Finance MCP.

        Args:
            ticker: Stock ticker symbol
            statement_type: "income", "balance", or "cashflow"
            period: "quarterly" or "annual"

        Returns:
            Financial statement data
        """
        tools = await self._yfinance_server.get_tools()
        get_financials = tools["get_financials"]

        import time
        start = time.time()
        result = get_financials.fn(ticker=ticker, statement_type=statement_type, period=period)
        elapsed = int((time.time() - start) * 1000)

        self._record_call("yfinance", "get_financials", result, elapsed)
        return result

    async def get_key_statistics(self, ticker: str) -> dict[str, Any]:
        """
        Get key statistics from Yahoo Finance MCP.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Key statistics including P/E, P/B, beta
        """
        import time
        start = time.time()

        if self._yfinance_url:
            resp = await self._http_client.post(
                f"{self._yfinance_url}/tools/get_key_statistics",
                json={"ticker": ticker},
            )
            resp.raise_for_status()
            result = resp.json()
        else:
            tools = await self._yfinance_server.get_tools()
            get_stats = tools["get_key_statistics"]
            result = get_stats.fn(ticker=ticker)

        elapsed = int((time.time() - start) * 1000)

        self._record_call("yfinance", "get_key_statistics", result, elapsed)
        return result

    async def get_analyst_estimates(self, ticker: str) -> dict[str, Any]:
        """
        Get analyst estimates from Yahoo Finance MCP.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Analyst price targets and recommendations
        """
        tools = await self._yfinance_server.get_tools()
        get_estimates = tools["get_analyst_estimates"]

        import time
        start = time.time()
        result = get_estimates.fn(ticker=ticker)
        elapsed = int((time.time() - start) * 1000)

        self._record_call("yfinance", "get_analyst_estimates", result, elapsed)
        return result

    async def get_earnings(self, ticker: str) -> dict[str, Any]:
        """
        Get earnings data from Yahoo Finance MCP.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Historical earnings and upcoming estimates
        """
        tools = await self._yfinance_server.get_tools()
        get_earnings = tools["get_earnings"]

        import time
        start = time.time()
        result = get_earnings.fn(ticker=ticker)
        elapsed = int((time.time() - start) * 1000)

        self._record_call("yfinance", "get_earnings", result, elapsed)
        return result

    # =========================================================================
    # SEC EDGAR Tools
    # =========================================================================

    async def get_company_info(self, ticker: str) -> dict[str, Any]:
        """
        Get company info from SEC EDGAR MCP.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Company info including CIK, name
        """
        import time
        start = time.time()

        if self._edgar_url:
            resp = await self._http_client.post(
                f"{self._edgar_url}/tools/get_company_info",
                json={"ticker": ticker},
            )
            resp.raise_for_status()
            result = resp.json()
        else:
            tools = await self._edgar_server.get_tools()
            get_company = tools["get_company_info"]
            result = get_company.fn(ticker=ticker)

        elapsed = int((time.time() - start) * 1000)

        self._record_call("edgar", "get_company_info", result, elapsed)
        return result

    async def get_filing(
        self,
        ticker: str,
        form_type: str,
        fiscal_year: int | None = None,
    ) -> dict[str, Any]:
        """
        Get SEC filing from EDGAR MCP.

        Args:
            ticker: Stock ticker symbol
            form_type: Filing type (10-K, 10-Q, 8-K)
            fiscal_year: Specific fiscal year

        Returns:
            Filing metadata
        """
        import time
        start = time.time()

        if self._edgar_url:
            payload = {"ticker": ticker, "form_type": form_type}
            if fiscal_year:
                payload["fiscal_year"] = fiscal_year
            resp = await self._http_client.post(
                f"{self._edgar_url}/tools/get_filing",
                json=payload,
            )
            resp.raise_for_status()
            result = resp.json()
        else:
            tools = await self._edgar_server.get_tools()
            get_filing = tools["get_filing"]
            result = get_filing.fn(ticker=ticker, form_type=form_type, fiscal_year=fiscal_year)

        elapsed = int((time.time() - start) * 1000)

        self._record_call("edgar", "get_filing", result, elapsed)
        return result

    async def get_xbrl_financials(
        self,
        ticker: str,
        statement_type: str = "IS",
        fiscal_year: int | None = None,
    ) -> dict[str, Any]:
        """
        Get XBRL financial data from SEC EDGAR MCP.

        Args:
            ticker: Stock ticker symbol
            statement_type: "IS" (Income), "BS" (Balance), "CF" (Cash Flow)
            fiscal_year: Specific fiscal year

        Returns:
            Parsed XBRL financial data
        """
        import time
        start = time.time()

        if self._edgar_url:
            payload = {
                "ticker": ticker,
                "statement_type": statement_type,
                "fiscal_year": fiscal_year,
            }
            resp = await self._http_client.post(
                f"{self._edgar_url}/tools/get_xbrl_financials",
                json=payload,
            )
            resp.raise_for_status()
            result = resp.json()
        else:
            tools = await self._edgar_server.get_tools()
            get_xbrl = tools["get_xbrl_financials"]
            result = get_xbrl.fn(ticker=ticker, statement_type=statement_type, fiscal_year=fiscal_year)

        elapsed = int((time.time() - start) * 1000)

        self._record_call("edgar", "get_xbrl_financials", result, elapsed)
        return result

    # =========================================================================
    # Sandbox Tools
    # =========================================================================

    async def execute_python(
        self,
        code: str,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """
        Execute Python code in the sandbox MCP.

        Pre-loaded libraries: numpy (np), pandas (pd), math, datetime, scipy.stats

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds

        Returns:
            Execution result with stdout, stderr, return value
        """
        import time
        start = time.time()

        if self._sandbox_url:
            resp = await self._http_client.post(
                f"{self._sandbox_url}/tools/execute_python",
                json={"code": code, "timeout": timeout},
            )
            resp.raise_for_status()
            result = resp.json()
        else:
            tools = await self._sandbox_server.get_tools()
            execute = tools["execute_python"]
            result = execute.fn(code=code, timeout=timeout)

        elapsed = int((time.time() - start) * 1000)

        self._record_call("sandbox", "execute_python", result, elapsed)
        return result

    async def calculate_financial_metric(
        self,
        metric: str,
        values: dict[str, float],
    ) -> dict[str, Any]:
        """
        Calculate a financial metric in the sandbox MCP.

        Available metrics: gross_margin, operating_margin, net_margin,
        roe, roa, current_ratio, debt_to_equity, pe_ratio, ev_to_ebitda

        Args:
            metric: Metric name
            values: Input values for calculation

        Returns:
            Calculated metric value and formula
        """
        import time
        start = time.time()

        if self._sandbox_url:
            resp = await self._http_client.post(
                f"{self._sandbox_url}/tools/calculate_financial_metric",
                json={"metric": metric, "values": values},
            )
            resp.raise_for_status()
            result = resp.json()
        else:
            tools = await self._sandbox_server.get_tools()
            calc = tools["calculate_financial_metric"]
            result = calc.fn(metric=metric, values=values)

        elapsed = int((time.time() - start) * 1000)

        self._record_call("sandbox", "calculate_financial_metric", result, elapsed)
        return result

    async def analyze_time_series(
        self,
        data: list[float],
        operations: list[str],
    ) -> dict[str, Any]:
        """
        Analyze time series data in the sandbox MCP.

        Available operations: mean, median, std, var, min, max, range,
        pct_change, cumsum, rolling_mean_5, rolling_mean_10, trend

        Args:
            data: List of numerical values
            operations: List of operations to perform

        Returns:
            Analysis results for each operation
        """
        import time
        start = time.time()

        if self._sandbox_url:
            resp = await self._http_client.post(
                f"{self._sandbox_url}/tools/analyze_time_series",
                json={"data": data, "operations": operations},
            )
            resp.raise_for_status()
            result = resp.json()
        else:
            tools = await self._sandbox_server.get_tools()
            analyze = tools["analyze_time_series"]
            result = analyze.fn(data=data, operations=operations)

        elapsed = int((time.time() - start) * 1000)

        self._record_call("sandbox", "analyze_time_series", result, elapsed)
        return result

    # =========================================================================
    # Options Chain Tools
    # =========================================================================

    async def get_options_chain(
        self,
        ticker: str,
        expiration: str | None = None,
        option_type: str = "all",
        min_strike: float | None = None,
        max_strike: float | None = None,
    ) -> dict[str, Any]:
        """
        Get options chain with Greeks from Options Chain MCP.

        Args:
            ticker: Stock ticker symbol
            expiration: Expiration date (YYYY-MM-DD), "nearest", or None for all
            option_type: "call", "put", or "all"
            min_strike: Minimum strike price filter
            max_strike: Maximum strike price filter

        Returns:
            Options chain with contracts and Greeks
        """
        import time
        start = time.time()

        tools = await self._options_chain_server.get_tools()
        get_chain = tools["get_options_chain"]
        result = get_chain.fn(
            ticker=ticker,
            expiration=expiration,
            option_type=option_type,
            min_strike=min_strike,
            max_strike=max_strike,
        )

        elapsed = int((time.time() - start) * 1000)
        self._record_call("options_chain", "get_options_chain", result, elapsed)
        return result

    async def calculate_option_price(
        self,
        spot_price: float,
        strike_price: float,
        days_to_expiry: int,
        volatility: float,
        risk_free_rate: float = 0.05,
        option_type: str = "call",
        dividend_yield: float = 0.0,
    ) -> dict[str, Any]:
        """
        Calculate theoretical option price using Black-Scholes model.

        Args:
            spot_price: Current stock price
            strike_price: Option strike price
            days_to_expiry: Days until expiration
            volatility: Implied volatility (annualized, e.g., 0.25 for 25%)
            risk_free_rate: Risk-free interest rate
            option_type: "call" or "put"
            dividend_yield: Continuous dividend yield

        Returns:
            Option price and all Greeks (delta, gamma, theta, vega, rho)
        """
        import time
        start = time.time()

        tools = await self._options_chain_server.get_tools()
        calc_price = tools["calculate_option_price"]
        result = calc_price.fn(
            spot_price=spot_price,
            strike_price=strike_price,
            days_to_expiry=days_to_expiry,
            volatility=volatility,
            risk_free_rate=risk_free_rate,
            option_type=option_type,
            dividend_yield=dividend_yield,
        )

        elapsed = int((time.time() - start) * 1000)
        self._record_call("options_chain", "calculate_option_price", result, elapsed)
        return result

    async def calculate_historical_option_price(
        self,
        ticker: str,
        strike_price: float,
        expiration: str,
        historical_date: str,
        option_type: str = "call",
    ) -> dict[str, Any]:
        """
        Calculate what an option would have been worth on a historical date.

        Args:
            ticker: Stock ticker symbol
            strike_price: Option strike price
            expiration: Expiration date (YYYY-MM-DD)
            historical_date: Date to price the option (YYYY-MM-DD)
            option_type: "call" or "put"

        Returns:
            Historical option price with Greeks
        """
        import time
        start = time.time()

        tools = await self._options_chain_server.get_tools()
        calc_hist = tools["calculate_historical_option_price"]
        result = calc_hist.fn(
            ticker=ticker,
            strike_price=strike_price,
            expiration=expiration,
            historical_date=historical_date,
            option_type=option_type,
        )

        elapsed = int((time.time() - start) * 1000)
        self._record_call("options_chain", "calculate_historical_option_price", result, elapsed)
        return result

    async def get_volatility_analysis(
        self,
        ticker: str,
        lookback_days: int = 30,
    ) -> dict[str, Any]:
        """
        Get volatility analysis for a ticker.

        Args:
            ticker: Stock ticker symbol
            lookback_days: Number of days for historical volatility

        Returns:
            Historical volatility, IV rank, IV percentile
        """
        import time
        start = time.time()

        tools = await self._options_chain_server.get_tools()
        get_vol = tools["get_volatility_analysis"]
        result = get_vol.fn(ticker=ticker, lookback_days=lookback_days)

        elapsed = int((time.time() - start) * 1000)
        self._record_call("options_chain", "get_volatility_analysis", result, elapsed)
        return result

    async def get_option_expirations(self, ticker: str) -> dict[str, Any]:
        """
        Get available option expiration dates.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of available expiration dates
        """
        import time
        start = time.time()

        tools = await self._options_chain_server.get_tools()
        get_exp = tools["get_expirations"]
        result = get_exp.fn(ticker=ticker)

        elapsed = int((time.time() - start) * 1000)
        self._record_call("options_chain", "get_expirations", result, elapsed)
        return result

    async def analyze_options_strategy(
        self,
        legs: list[dict],
        spot_price: float,
    ) -> dict[str, Any]:
        """
        Analyze a multi-leg options strategy.

        Args:
            legs: List of strategy legs, each with:
                - option_type: "call" or "put"
                - strike: Strike price
                - action: "buy" or "sell"
                - quantity: Number of contracts
                - premium: Option premium per share
            spot_price: Current stock price

        Returns:
            Strategy analysis with max profit/loss, breakevens, risk/reward
        """
        import time
        start = time.time()

        tools = await self._options_chain_server.get_tools()
        analyze = tools["analyze_strategy"]
        result = analyze.fn(legs=legs, spot_price=spot_price)

        elapsed = int((time.time() - start) * 1000)
        self._record_call("options_chain", "analyze_strategy", result, elapsed)
        return result

    # =========================================================================
    # Trading Simulator Tools
    # =========================================================================

    async def create_portfolio(
        self,
        starting_cash: float = 100000.0,
        portfolio_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a new paper trading portfolio.

        Args:
            starting_cash: Initial cash balance
            portfolio_id: Optional custom ID for the portfolio

        Returns:
            Created portfolio details
        """
        import time
        start = time.time()

        tools = await self._trading_sim_server.get_tools()
        create = tools["create_portfolio"]
        result = create.fn(starting_cash=starting_cash, portfolio_id=portfolio_id)

        elapsed = int((time.time() - start) * 1000)
        self._record_call("trading_sim", "create_portfolio", result, elapsed)
        return result

    async def execute_options_trade(
        self,
        portfolio_id: str,
        ticker: str,
        strike: float,
        expiration: str,
        option_type: str,
        action: str,
        quantity: int,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        """
        Execute an options trade in the simulator.

        Args:
            portfolio_id: Portfolio ID
            ticker: Stock ticker symbol
            strike: Strike price
            expiration: Expiration date (YYYY-MM-DD)
            option_type: "call" or "put"
            action: "buy" or "sell"
            quantity: Number of contracts
            order_type: "market" or "limit"
            limit_price: Required for limit orders

        Returns:
            Trade execution details with fill price
        """
        import time
        start = time.time()

        tools = await self._trading_sim_server.get_tools()
        execute = tools["execute_trade"]
        result = execute.fn(
            portfolio_id=portfolio_id,
            ticker=ticker,
            strike=strike,
            expiration=expiration,
            option_type=option_type,
            action=action,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
        )

        elapsed = int((time.time() - start) * 1000)
        self._record_call("trading_sim", "execute_trade", result, elapsed)
        return result

    async def get_portfolio(self, portfolio_id: str) -> dict[str, Any]:
        """
        Get current portfolio state.

        Args:
            portfolio_id: Portfolio ID

        Returns:
            Portfolio state with positions and P&L
        """
        import time
        start = time.time()

        tools = await self._trading_sim_server.get_tools()
        get_port = tools["get_portfolio"]
        result = get_port.fn(portfolio_id=portfolio_id)

        elapsed = int((time.time() - start) * 1000)
        self._record_call("trading_sim", "get_portfolio", result, elapsed)
        return result

    async def close_position(
        self,
        portfolio_id: str,
        position_id: str,
        quantity: int | None = None,
    ) -> dict[str, Any]:
        """
        Close an existing position.

        Args:
            portfolio_id: Portfolio ID
            position_id: Position ID to close
            quantity: Number of contracts (None for full close)

        Returns:
            Closing trade details
        """
        import time
        start = time.time()

        tools = await self._trading_sim_server.get_tools()
        close = tools["close_position"]
        result = close.fn(
            portfolio_id=portfolio_id,
            position_id=position_id,
            quantity=quantity,
        )

        elapsed = int((time.time() - start) * 1000)
        self._record_call("trading_sim", "close_position", result, elapsed)
        return result

    async def advance_simulation_time(
        self,
        portfolio_id: str,
        days: int,
    ) -> dict[str, Any]:
        """
        Advance simulation time and update positions.

        Args:
            portfolio_id: Portfolio ID
            days: Number of days to advance

        Returns:
            Updated portfolio state with P&L changes
        """
        import time
        start = time.time()

        tools = await self._trading_sim_server.get_tools()
        advance = tools["advance_time"]
        result = advance.fn(portfolio_id=portfolio_id, days=days)

        elapsed = int((time.time() - start) * 1000)
        self._record_call("trading_sim", "advance_time", result, elapsed)
        return result

    async def get_pnl_report(self, portfolio_id: str) -> dict[str, Any]:
        """
        Get detailed P&L report for a portfolio.

        Args:
            portfolio_id: Portfolio ID

        Returns:
            Detailed P&L breakdown by position
        """
        import time
        start = time.time()

        tools = await self._trading_sim_server.get_tools()
        report = tools["get_pnl_report"]
        result = report.fn(portfolio_id=portfolio_id)

        elapsed = int((time.time() - start) * 1000)
        self._record_call("trading_sim", "get_pnl_report", result, elapsed)
        return result

    async def list_portfolios(self) -> dict[str, Any]:
        """
        List all portfolios in the simulator.

        Returns:
            List of portfolio summaries
        """
        import time
        start = time.time()

        tools = await self._trading_sim_server.get_tools()
        list_port = tools["list_portfolios"]
        result = list_port.fn()

        elapsed = int((time.time() - start) * 1000)
        self._record_call("trading_sim", "list_portfolios", result, elapsed)
        return result

    # =========================================================================
    # Risk Metrics Tools
    # =========================================================================

    async def calculate_portfolio_greeks(
        self,
        positions: list[dict],
    ) -> dict[str, Any]:
        """
        Calculate aggregate Greeks across positions.

        Args:
            positions: List of positions, each with:
                - delta, gamma, theta, vega, rho: Individual Greeks
                - quantity: Number of contracts
                - multiplier: Contract multiplier (usually 100)

        Returns:
            Aggregate portfolio Greeks
        """
        import time
        start = time.time()

        tools = await self._risk_metrics_server.get_tools()
        calc_greeks = tools["calculate_portfolio_greeks"]
        result = calc_greeks.fn(positions=positions)

        elapsed = int((time.time() - start) * 1000)
        self._record_call("risk_metrics", "calculate_portfolio_greeks", result, elapsed)
        return result

    async def calculate_var(
        self,
        returns: list[float],
        confidence_level: float = 0.95,
        horizon_days: int = 1,
        portfolio_value: float = 100000.0,
        method: str = "historical",
    ) -> dict[str, Any]:
        """
        Calculate Value at Risk for a portfolio.

        Args:
            returns: List of historical daily returns
            confidence_level: VaR confidence level (e.g., 0.95, 0.99)
            horizon_days: Risk horizon in days
            portfolio_value: Current portfolio value
            method: "historical", "parametric", or "monte_carlo"

        Returns:
            VaR calculation with dollar and percentage values
        """
        import time
        start = time.time()

        tools = await self._risk_metrics_server.get_tools()
        calc_var = tools["calculate_var"]
        result = calc_var.fn(
            returns=returns,
            confidence_level=confidence_level,
            horizon_days=horizon_days,
            portfolio_value=portfolio_value,
            method=method,
        )

        elapsed = int((time.time() - start) * 1000)
        self._record_call("risk_metrics", "calculate_var", result, elapsed)
        return result

    async def calculate_max_drawdown(
        self,
        portfolio_values: list[float],
    ) -> dict[str, Any]:
        """
        Calculate maximum drawdown with recovery analysis.

        Args:
            portfolio_values: List of portfolio values over time

        Returns:
            Max drawdown percentage, peak/trough dates, recovery info
        """
        import time
        start = time.time()

        tools = await self._risk_metrics_server.get_tools()
        calc_dd = tools["calculate_max_drawdown"]
        result = calc_dd.fn(portfolio_values=portfolio_values)

        elapsed = int((time.time() - start) * 1000)
        self._record_call("risk_metrics", "calculate_max_drawdown", result, elapsed)
        return result

    async def calculate_risk_adjusted_returns(
        self,
        returns: list[float],
        risk_free_rate: float = 0.05,
    ) -> dict[str, Any]:
        """
        Calculate Sharpe, Sortino, and Calmar ratios.

        Args:
            returns: List of period returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio, Sortino ratio, Calmar ratio
        """
        import time
        start = time.time()

        tools = await self._risk_metrics_server.get_tools()
        calc_returns = tools["calculate_risk_adjusted_returns"]
        result = calc_returns.fn(returns=returns, risk_free_rate=risk_free_rate)

        elapsed = int((time.time() - start) * 1000)
        self._record_call("risk_metrics", "calculate_risk_adjusted_returns", result, elapsed)
        return result

    async def stress_test_portfolio(
        self,
        positions: list[dict],
        scenarios: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        Run stress tests on portfolio positions.

        Args:
            positions: List of positions with Greeks
            scenarios: Optional custom scenarios with:
                - name: Scenario name
                - spot_change: % change in spot price
                - vol_change: % change in implied volatility

        Returns:
            P&L impact for each stress scenario
        """
        import time
        start = time.time()

        tools = await self._risk_metrics_server.get_tools()
        stress = tools["stress_test"]
        result = stress.fn(positions=positions, scenarios=scenarios)

        elapsed = int((time.time() - start) * 1000)
        self._record_call("risk_metrics", "stress_test", result, elapsed)
        return result

    async def get_pnl_attribution(
        self,
        position: dict,
        spot_change: float,
        vol_change: float,
        time_decay_days: float,
    ) -> dict[str, Any]:
        """
        Decompose P&L by Greek contribution.

        Args:
            position: Position with Greeks and quantity
            spot_change: Change in spot price ($)
            vol_change: Change in implied volatility (absolute, e.g., 0.05)
            time_decay_days: Days of time decay

        Returns:
            P&L breakdown by delta, gamma, theta, vega
        """
        import time
        start = time.time()

        tools = await self._risk_metrics_server.get_tools()
        attr = tools["pnl_attribution"]
        result = attr.fn(
            position=position,
            spot_change=spot_change,
            vol_change=vol_change,
            time_decay_days=time_decay_days,
        )

        elapsed = int((time.time() - start) * 1000)
        self._record_call("risk_metrics", "pnl_attribution", result, elapsed)
        return result

    # =========================================================================
    # Composite Methods
    # =========================================================================

    async def get_comprehensive_analysis(self, ticker: str) -> dict[str, Any]:
        """
        Get comprehensive financial analysis combining all MCP sources.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with comprehensive financial data
        """
        # Fetch data from multiple MCP sources concurrently
        quote_task = self.get_quote(ticker)
        stats_task = self.get_key_statistics(ticker)
        company_task = self.get_company_info(ticker)
        filing_task = self.get_filing(ticker, "10-K")

        quote, stats, company, filing = await asyncio.gather(
            quote_task,
            stats_task,
            company_task,
            filing_task,
            return_exceptions=True,
        )

        # Handle exceptions
        if isinstance(quote, Exception):
            quote = {"error": str(quote)}
        if isinstance(stats, Exception):
            stats = {"error": str(stats)}
        if isinstance(company, Exception):
            company = {"error": str(company)}
        if isinstance(filing, Exception):
            filing = {"error": str(filing)}

        return {
            "ticker": ticker,
            "simulation_date": self.simulation_date.isoformat() if self.simulation_date else None,
            "quote": quote,
            "statistics": stats,
            "company_info": company,
            "recent_filing": filing,
            "mcp_metrics": self.get_metrics(),
        }

    async def analyze_earnings(
        self,
        ticker: str,
        expected_revenue: float | None = None,
        expected_eps: float | None = None,
    ) -> dict[str, Any]:
        """
        Analyze earnings for beat/miss determination.

        Args:
            ticker: Stock ticker symbol
            expected_revenue: Expected revenue for comparison
            expected_eps: Expected EPS for comparison

        Returns:
            Earnings analysis with beat/miss determination
        """
        quote = await self.get_quote(ticker)
        earnings = await self.get_earnings(ticker)
        financials = await self.get_financials(ticker, "income", "quarterly")

        result = {
            "ticker": ticker,
            "quote": quote,
            "earnings_history": earnings,
            "financials": financials,
        }

        # If we have expectations, compute beat/miss
        if expected_revenue and financials.get("data"):
            actual_revenue = financials["data"].get("Total Revenue")
            if actual_revenue:
                result["revenue_vs_expected"] = {
                    "actual": actual_revenue,
                    "expected": expected_revenue,
                    "beat": actual_revenue > expected_revenue,
                    "surprise_pct": (actual_revenue - expected_revenue) / expected_revenue * 100,
                }

        if expected_eps and quote.get("pe_ratio"):
            # Use EPS from earnings if available
            actual_eps = quote.get("current_price", 0) / quote.get("pe_ratio", 1) if quote.get("pe_ratio") else None
            if actual_eps:
                result["eps_vs_expected"] = {
                    "actual": actual_eps,
                    "expected": expected_eps,
                    "beat": actual_eps > expected_eps,
                    "surprise_pct": (actual_eps - expected_eps) / expected_eps * 100,
                }

        return result

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_metrics(self) -> dict[str, Any]:
        """Get usage metrics from the toolkit."""
        return {
            "tool_calls": self._metrics.tool_calls,
            "errors": self._metrics.errors,
            "total_time_ms": self._metrics.total_time_ms,
            "calls_by_tool": self._metrics.calls_by_tool,
        }

    def get_tool_calls(self) -> list[dict]:
        """Get all recorded tool calls."""
        return self._tool_calls

    def reset_metrics(self):
        """Reset metrics."""
        self._metrics = MCPToolMetrics()
        self._tool_calls = []
