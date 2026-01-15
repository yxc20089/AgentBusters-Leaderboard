"""
Risk Metrics MCP Server

Portfolio risk analysis and performance metrics calculation.
Supports:
- Aggregate portfolio Greeks
- Value at Risk (VaR) - Historical, Parametric, Monte Carlo
- Maximum Drawdown analysis
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Stress testing scenarios
- P&L attribution by Greek
"""

import math
from datetime import datetime, date, timedelta
from typing import Any, Literal

import numpy as np
from scipy.stats import norm
from fastmcp import FastMCP
from pydantic import BaseModel, Field
import yfinance as yf

from mcp_servers.options_chain import black_scholes, calculate_historical_volatility


class PortfolioGreeks(BaseModel):
    """Aggregate Greeks for entire portfolio."""
    net_delta: float = Field(..., description="Net delta exposure (shares equivalent)")
    net_gamma: float = Field(..., description="Rate of delta change")
    net_theta: float = Field(..., description="Daily time decay in $")
    net_vega: float = Field(..., description="Exposure to 1% vol change in $")
    net_rho: float = Field(..., description="Exposure to 1% rate change in $")
    delta_dollars: float = Field(..., description="Dollar delta (notional exposure)")
    gamma_dollars: float = Field(..., description="Dollar gamma")


class VaRResult(BaseModel):
    """Value at Risk calculation result."""
    var_95: float = Field(..., description="95% VaR ($ loss)")
    var_99: float = Field(..., description="99% VaR ($ loss)")
    expected_shortfall_95: float = Field(..., description="Expected Shortfall at 95%")
    method: str
    horizon_days: int
    portfolio_value: float


class DrawdownResult(BaseModel):
    """Maximum drawdown analysis."""
    max_drawdown_pct: float
    max_drawdown_dollars: float
    peak_value: float
    trough_value: float
    peak_date: str
    trough_date: str
    current_drawdown_pct: float
    recovery_pct: float


class RiskAdjustedReturns(BaseModel):
    """Risk-adjusted performance metrics."""
    sharpe_ratio: float | None
    sortino_ratio: float | None
    calmar_ratio: float | None
    information_ratio: float | None
    treynor_ratio: float | None
    annualized_return: float
    annualized_volatility: float
    downside_volatility: float
    risk_free_rate: float


class StressTestResult(BaseModel):
    """Result of a stress test scenario."""
    scenario_name: str
    description: str
    portfolio_pnl: float
    portfolio_pnl_pct: float
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float


def create_risk_metrics_server(
    name: str = "risk-metrics-mcp",
) -> FastMCP:
    """
    Create the Risk Metrics MCP server.

    Args:
        name: Server name

    Returns:
        Configured FastMCP server
    """
    mcp = FastMCP(name)
    _risk_free_rate = 0.0525  # 5.25% annual

    @mcp.tool
    def calculate_portfolio_greeks(
        positions: list[dict],
    ) -> dict[str, Any]:
        """
        Calculate aggregate Greeks for a portfolio of options.

        Args:
            positions: List of positions, each with:
                - ticker: Stock symbol
                - strike: Strike price
                - expiration: Expiration date (YYYY-MM-DD)
                - option_type: "call" or "put"
                - quantity: Number of contracts (negative for short)
                - current_price: Optional current option price

        Returns:
            Aggregate portfolio Greeks
        """
        try:
            net_delta = 0.0
            net_gamma = 0.0
            net_theta = 0.0
            net_vega = 0.0
            net_rho = 0.0
            delta_dollars = 0.0
            gamma_dollars = 0.0

            for pos in positions:
                ticker = pos["ticker"]
                strike = pos["strike"]
                expiration = pos["expiration"]
                option_type = pos["option_type"]
                quantity = pos["quantity"]

                # Get underlying data
                stock = yf.Ticker(ticker)
                info = stock.info
                underlying = info.get("regularMarketPrice") or info.get("currentPrice", 0)

                if underlying <= 0:
                    continue

                # Calculate days to expiry
                exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
                days = max((exp_date - date.today()).days, 0)
                T = days / 365.0

                # Get volatility
                hist = stock.history(period="3mo")
                vol = calculate_historical_volatility(hist["Close"].tolist())

                # Calculate Greeks
                result = black_scholes(
                    S=underlying,
                    K=strike,
                    T=T,
                    r=_risk_free_rate,
                    sigma=vol,
                    option_type=option_type,
                )

                # Each contract = 100 shares
                multiplier = quantity * 100

                net_delta += result["delta"] * multiplier
                net_gamma += result["gamma"] * multiplier
                net_theta += result["theta"] * multiplier
                net_vega += result["vega"] * multiplier
                net_rho += result["rho"] * multiplier

                delta_dollars += result["delta"] * multiplier * underlying
                gamma_dollars += result["gamma"] * multiplier * underlying

            return PortfolioGreeks(
                net_delta=round(net_delta, 2),
                net_gamma=round(net_gamma, 4),
                net_theta=round(net_theta, 2),
                net_vega=round(net_vega, 2),
                net_rho=round(net_rho, 2),
                delta_dollars=round(delta_dollars, 2),
                gamma_dollars=round(gamma_dollars, 2),
            ).model_dump()

        except Exception as e:
            return {"error": str(e)}

    @mcp.tool
    def calculate_var(
        portfolio_value: float,
        returns: list[float] | None = None,
        volatility: float | None = None,
        confidence_level: float = 0.95,
        horizon_days: int = 1,
        method: Literal["historical", "parametric", "monte_carlo"] = "parametric",
    ) -> dict[str, Any]:
        """
        Calculate Value at Risk for a portfolio.

        Args:
            portfolio_value: Current portfolio value in $
            returns: Historical daily returns (for historical/monte_carlo methods)
            volatility: Annual volatility (for parametric method)
            confidence_level: VaR confidence level (0.95 or 0.99)
            horizon_days: Time horizon in days
            method: Calculation method

        Returns:
            VaR at 95% and 99% confidence levels
        """
        try:
            if method == "parametric":
                if volatility is None:
                    volatility = 0.20  # Default 20% annual vol

                # Convert to daily and scale by horizon
                daily_vol = volatility / math.sqrt(252)
                horizon_vol = daily_vol * math.sqrt(horizon_days)

                # Z-scores for confidence levels
                z_95 = norm.ppf(0.95)
                z_99 = norm.ppf(0.99)

                var_95 = portfolio_value * horizon_vol * z_95
                var_99 = portfolio_value * horizon_vol * z_99

                # Expected shortfall (CVaR)
                es_95 = portfolio_value * horizon_vol * norm.pdf(z_95) / 0.05

            elif method == "historical":
                if returns is None or len(returns) < 20:
                    return {"error": "Need at least 20 historical returns for historical VaR"}

                returns = np.array(returns)

                # Scale returns to horizon
                if horizon_days > 1:
                    # Rough scaling (assumes i.i.d.)
                    returns = returns * math.sqrt(horizon_days)

                # Percentile-based VaR
                var_95 = portfolio_value * abs(np.percentile(returns, 5))
                var_99 = portfolio_value * abs(np.percentile(returns, 1))

                # Expected shortfall
                tail_returns = returns[returns <= np.percentile(returns, 5)]
                es_95 = portfolio_value * abs(np.mean(tail_returns))

            elif method == "monte_carlo":
                if returns is None or len(returns) < 20:
                    return {"error": "Need historical returns for Monte Carlo VaR"}

                returns = np.array(returns)
                mu = np.mean(returns) * horizon_days
                sigma = np.std(returns) * math.sqrt(horizon_days)

                # Simulate 10,000 scenarios
                np.random.seed(42)  # Reproducibility
                simulated = np.random.normal(mu, sigma, 10000)

                var_95 = portfolio_value * abs(np.percentile(simulated, 5))
                var_99 = portfolio_value * abs(np.percentile(simulated, 1))

                tail = simulated[simulated <= np.percentile(simulated, 5)]
                es_95 = portfolio_value * abs(np.mean(tail))

            else:
                return {"error": f"Unknown method: {method}"}

            return VaRResult(
                var_95=round(var_95, 2),
                var_99=round(var_99, 2),
                expected_shortfall_95=round(es_95, 2),
                method=method,
                horizon_days=horizon_days,
                portfolio_value=portfolio_value,
            ).model_dump()

        except Exception as e:
            return {"error": str(e)}

    @mcp.tool
    def calculate_max_drawdown(
        portfolio_values: list[float],
        dates: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Calculate maximum drawdown from portfolio value history.

        Args:
            portfolio_values: List of portfolio values over time
            dates: Optional list of dates corresponding to values

        Returns:
            Maximum drawdown analysis with peak/trough details
        """
        try:
            if len(portfolio_values) < 2:
                return {"error": "Need at least 2 values for drawdown calculation"}

            values = np.array(portfolio_values)

            # Calculate running maximum
            running_max = np.maximum.accumulate(values)

            # Calculate drawdown at each point
            drawdowns = (running_max - values) / running_max

            # Find maximum drawdown
            max_dd_idx = np.argmax(drawdowns)
            max_dd = drawdowns[max_dd_idx]

            # Find peak (before trough)
            peak_idx = np.argmax(values[:max_dd_idx + 1])
            trough_idx = max_dd_idx

            peak_value = values[peak_idx]
            trough_value = values[trough_idx]
            current_value = values[-1]

            # Current drawdown
            current_peak = running_max[-1]
            current_dd = (current_peak - current_value) / current_peak

            # Recovery from trough
            if trough_idx < len(values) - 1:
                recovery = (current_value - trough_value) / (peak_value - trough_value)
            else:
                recovery = 0.0

            # Get dates if provided
            if dates:
                peak_date = dates[peak_idx]
                trough_date = dates[trough_idx]
            else:
                peak_date = f"index_{peak_idx}"
                trough_date = f"index_{trough_idx}"

            return DrawdownResult(
                max_drawdown_pct=round(max_dd * 100, 2),
                max_drawdown_dollars=round(peak_value - trough_value, 2),
                peak_value=round(peak_value, 2),
                trough_value=round(trough_value, 2),
                peak_date=peak_date,
                trough_date=trough_date,
                current_drawdown_pct=round(current_dd * 100, 2),
                recovery_pct=round(recovery * 100, 2),
            ).model_dump()

        except Exception as e:
            return {"error": str(e)}

    @mcp.tool
    def calculate_risk_adjusted_returns(
        returns: list[float],
        risk_free_rate: float | None = None,
        benchmark_returns: list[float] | None = None,
    ) -> dict[str, Any]:
        """
        Calculate risk-adjusted performance metrics.

        Args:
            returns: List of periodic returns (daily or monthly)
            risk_free_rate: Annual risk-free rate. Uses default if not provided.
            benchmark_returns: Optional benchmark returns for Information Ratio

        Returns:
            Sharpe, Sortino, Calmar and other risk-adjusted metrics
        """
        try:
            if len(returns) < 5:
                return {"error": "Need at least 5 returns for risk metrics"}

            returns = np.array(returns)
            rf = risk_free_rate or _risk_free_rate

            # Assume daily returns, annualize
            periods_per_year = 252

            # Basic statistics
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)

            annualized_return = mean_return * periods_per_year
            annualized_vol = std_return * math.sqrt(periods_per_year)

            # Sharpe Ratio
            rf_daily = rf / periods_per_year
            excess_returns = returns - rf_daily
            sharpe = None
            if std_return > 0:
                sharpe = np.mean(excess_returns) / std_return * math.sqrt(periods_per_year)

            # Sortino Ratio (downside deviation)
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                downside_vol = np.std(negative_returns, ddof=1) * math.sqrt(periods_per_year)
                sortino = (annualized_return - rf) / downside_vol if downside_vol > 0 else None
            else:
                downside_vol = 0.0
                sortino = None

            # Calmar Ratio (return / max drawdown)
            cumulative = np.cumprod(1 + returns)
            max_dd = np.max((np.maximum.accumulate(cumulative) - cumulative) / np.maximum.accumulate(cumulative))
            calmar = annualized_return / max_dd if max_dd > 0 else None

            # Information Ratio (if benchmark provided)
            info_ratio = None
            if benchmark_returns is not None and len(benchmark_returns) == len(returns):
                bench = np.array(benchmark_returns)
                tracking_error = np.std(returns - bench, ddof=1) * math.sqrt(periods_per_year)
                if tracking_error > 0:
                    info_ratio = (annualized_return - np.mean(bench) * periods_per_year) / tracking_error

            return RiskAdjustedReturns(
                sharpe_ratio=round(sharpe, 4) if sharpe is not None else None,
                sortino_ratio=round(sortino, 4) if sortino is not None else None,
                calmar_ratio=round(calmar, 4) if calmar is not None else None,
                information_ratio=round(info_ratio, 4) if info_ratio is not None else None,
                treynor_ratio=None,  # Would need beta calculation
                annualized_return=round(annualized_return * 100, 2),
                annualized_volatility=round(annualized_vol * 100, 2),
                downside_volatility=round(downside_vol * 100, 2),
                risk_free_rate=round(rf * 100, 2),
            ).model_dump()

        except Exception as e:
            return {"error": str(e)}

    @mcp.tool
    def stress_test(
        positions: list[dict],
        scenarios: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        Run stress tests on a portfolio.

        Args:
            positions: List of positions (same format as calculate_portfolio_greeks)
            scenarios: Optional custom scenarios. Each with:
                - name: Scenario name
                - description: Description
                - underlying_change_pct: % change in underlying
                - vol_change_pct: % change in volatility
                - days_elapsed: Days of time decay

        Returns:
            P&L impact under each stress scenario
        """
        try:
            # Default scenarios if none provided
            if scenarios is None:
                scenarios = [
                    {"name": "Market Crash", "description": "-20% drop, +50% vol spike",
                     "underlying_change_pct": -20, "vol_change_pct": 50, "days_elapsed": 1},
                    {"name": "Flash Crash", "description": "-10% drop, +100% vol spike",
                     "underlying_change_pct": -10, "vol_change_pct": 100, "days_elapsed": 1},
                    {"name": "Melt Up", "description": "+10% rally, -20% vol crush",
                     "underlying_change_pct": 10, "vol_change_pct": -20, "days_elapsed": 1},
                    {"name": "Vol Crush", "description": "Flat market, -30% vol",
                     "underlying_change_pct": 0, "vol_change_pct": -30, "days_elapsed": 1},
                    {"name": "1 Week Theta", "description": "Flat market, 7 days pass",
                     "underlying_change_pct": 0, "vol_change_pct": 0, "days_elapsed": 7},
                    {"name": "2008 Style", "description": "-40% crash, +200% vol",
                     "underlying_change_pct": -40, "vol_change_pct": 200, "days_elapsed": 5},
                ]

            results = []

            for scenario in scenarios:
                total_pnl = 0.0
                delta_pnl = 0.0
                gamma_pnl = 0.0
                vega_pnl = 0.0
                theta_pnl = 0.0

                for pos in positions:
                    ticker = pos["ticker"]
                    strike = pos["strike"]
                    expiration = pos["expiration"]
                    option_type = pos["option_type"]
                    quantity = pos["quantity"]

                    # Get current data
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    S = info.get("regularMarketPrice") or info.get("currentPrice", 0)

                    if S <= 0:
                        continue

                    # Get current vol
                    hist = stock.history(period="3mo")
                    sigma = calculate_historical_volatility(hist["Close"].tolist())

                    # Calculate current option value and Greeks
                    exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
                    T = max((exp_date - date.today()).days, 0) / 365.0

                    current = black_scholes(S, strike, T, _risk_free_rate, sigma, option_type)

                    # Apply scenario
                    S_new = S * (1 + scenario.get("underlying_change_pct", 0) / 100)
                    sigma_new = sigma * (1 + scenario.get("vol_change_pct", 0) / 100)
                    days_elapsed = scenario.get("days_elapsed", 0)
                    T_new = max(T - days_elapsed / 365, 0)

                    new_val = black_scholes(S_new, strike, T_new, _risk_free_rate, sigma_new, option_type)

                    # P&L components
                    dS = S_new - S
                    d_sigma = sigma_new - sigma
                    multiplier = quantity * 100

                    # Delta P&L
                    d_pnl = current["delta"] * dS * multiplier
                    delta_pnl += d_pnl

                    # Gamma P&L (second order)
                    g_pnl = 0.5 * current["gamma"] * (dS ** 2) * multiplier
                    gamma_pnl += g_pnl

                    # Vega P&L
                    v_pnl = current["vega"] * (d_sigma * 100) * multiplier  # vega is per 1%
                    vega_pnl += v_pnl

                    # Theta P&L
                    t_pnl = current["theta"] * days_elapsed * multiplier
                    theta_pnl += t_pnl

                    # Total P&L (actual difference)
                    option_pnl = (new_val["price"] - current["price"]) * multiplier
                    total_pnl += option_pnl

                results.append(StressTestResult(
                    scenario_name=scenario["name"],
                    description=scenario.get("description", ""),
                    portfolio_pnl=round(total_pnl, 2),
                    portfolio_pnl_pct=round(total_pnl / 100000 * 100, 2),  # Assume $100k portfolio
                    delta_pnl=round(delta_pnl, 2),
                    gamma_pnl=round(gamma_pnl, 2),
                    vega_pnl=round(vega_pnl, 2),
                    theta_pnl=round(theta_pnl, 2),
                ).model_dump())

            return {
                "scenarios_tested": len(results),
                "results": results,
                "worst_case": min(results, key=lambda x: x["portfolio_pnl"]),
                "best_case": max(results, key=lambda x: x["portfolio_pnl"]),
            }

        except Exception as e:
            return {"error": str(e)}

    @mcp.tool
    def pnl_attribution(
        position: dict,
        start_underlying: float,
        end_underlying: float,
        start_vol: float,
        end_vol: float,
        days_elapsed: int,
    ) -> dict[str, Any]:
        """
        Attribute P&L to individual Greeks for a position.

        Args:
            position: Position with ticker, strike, expiration, option_type, quantity
            start_underlying: Starting underlying price
            end_underlying: Ending underlying price
            start_vol: Starting volatility
            end_vol: Ending volatility
            days_elapsed: Days between start and end

        Returns:
            P&L breakdown by Greek (delta, gamma, theta, vega) plus residual
        """
        try:
            ticker = position["ticker"]
            strike = position["strike"]
            expiration = position["expiration"]
            option_type = position["option_type"]
            quantity = position["quantity"]

            # Calculate starting values
            exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
            T_start = max((exp_date - date.today()).days, 0) / 365.0
            T_end = max(T_start - days_elapsed / 365, 0)

            start = black_scholes(start_underlying, strike, T_start, _risk_free_rate, start_vol, option_type)
            end = black_scholes(end_underlying, strike, T_end, _risk_free_rate, end_vol, option_type)

            multiplier = quantity * 100

            # Total P&L
            total_pnl = (end["price"] - start["price"]) * multiplier

            # Attribution
            dS = end_underlying - start_underlying
            d_sigma = end_vol - start_vol

            delta_pnl = start["delta"] * dS * multiplier
            gamma_pnl = 0.5 * start["gamma"] * (dS ** 2) * multiplier
            theta_pnl = start["theta"] * days_elapsed * multiplier
            vega_pnl = start["vega"] * (d_sigma * 100) * multiplier

            # Residual (unexplained by first-order Greeks)
            explained = delta_pnl + gamma_pnl + theta_pnl + vega_pnl
            residual = total_pnl - explained

            return {
                "total_pnl": round(total_pnl, 2),
                "attribution": {
                    "delta_pnl": round(delta_pnl, 2),
                    "gamma_pnl": round(gamma_pnl, 2),
                    "theta_pnl": round(theta_pnl, 2),
                    "vega_pnl": round(vega_pnl, 2),
                    "residual": round(residual, 2),
                },
                "pct_attribution": {
                    "delta": round(delta_pnl / total_pnl * 100, 1) if total_pnl != 0 else 0,
                    "gamma": round(gamma_pnl / total_pnl * 100, 1) if total_pnl != 0 else 0,
                    "theta": round(theta_pnl / total_pnl * 100, 1) if total_pnl != 0 else 0,
                    "vega": round(vega_pnl / total_pnl * 100, 1) if total_pnl != 0 else 0,
                    "residual": round(residual / total_pnl * 100, 1) if total_pnl != 0 else 0,
                },
                "inputs": {
                    "start_underlying": start_underlying,
                    "end_underlying": end_underlying,
                    "start_vol": round(start_vol, 4),
                    "end_vol": round(end_vol, 4),
                    "days_elapsed": days_elapsed,
                },
            }

        except Exception as e:
            return {"error": str(e)}

    return mcp


# CLI entry point
if __name__ == "__main__":
    server = create_risk_metrics_server()
    server.run()
