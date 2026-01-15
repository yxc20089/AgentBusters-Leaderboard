"""
Options Chain MCP Server

Provides options chain data and theoretical pricing using Black-Scholes.
Uses yfinance for underlying data and current options chains.
Calculates Greeks (delta, gamma, theta, vega, rho) for any option.

Supports:
- Current options chains with live bid/ask
- Theoretical option pricing via Black-Scholes
- Historical option price estimation (from underlying + B-S)
- Greeks calculation
- IV surface and volatility analysis
"""

import math
from datetime import datetime, date, timedelta
from typing import Any, Literal

import numpy as np
from scipy.stats import norm
from fastmcp import FastMCP
from pydantic import BaseModel, Field
import yfinance as yf


class OptionQuote(BaseModel):
    """Single option contract quote."""
    contract_symbol: str
    ticker: str
    strike: float
    expiration: str
    option_type: Literal["call", "put"]
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: float
    in_the_money: bool
    # Calculated Greeks
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None
    theoretical_price: float | None = None


class OptionsChain(BaseModel):
    """Complete options chain for an expiration."""
    ticker: str
    underlying_price: float
    expiration: str
    days_to_expiry: int
    calls: list[OptionQuote]
    puts: list[OptionQuote]
    timestamp: str


class TheoreticalPrice(BaseModel):
    """Black-Scholes theoretical option pricing result."""
    ticker: str
    strike: float
    expiration: str
    option_type: Literal["call", "put"]
    underlying_price: float
    days_to_expiry: int
    risk_free_rate: float
    volatility: float
    # Results
    theoretical_price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    # Probabilities
    prob_itm: float
    prob_otm: float


class VolatilityData(BaseModel):
    """Volatility analysis data."""
    ticker: str
    current_iv: float | None
    historical_volatility_20d: float
    historical_volatility_60d: float
    historical_volatility_252d: float
    iv_percentile: float | None = None
    iv_rank: float | None = None


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
) -> dict[str, float]:
    """
    Calculate Black-Scholes option price and Greeks.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (annual)
        sigma: Volatility (annual)
        option_type: "call" or "put"
        q: Dividend yield (annual)

    Returns:
        Dictionary with price and all Greeks
    """
    if T <= 0:
        # At expiration
        if option_type == "call":
            intrinsic = max(S - K, 0)
            delta = 1.0 if S > K else 0.0
        else:
            intrinsic = max(K - S, 0)
            delta = -1.0 if S < K else 0.0
        return {
            "price": intrinsic,
            "delta": delta,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0,
            "prob_itm": 1.0 if intrinsic > 0 else 0.0,
        }

    # Standard Black-Scholes with dividend adjustment
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # CDF and PDF values
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_neg_d1 = norm.cdf(-d1)
    N_neg_d2 = norm.cdf(-d2)
    n_d1 = norm.pdf(d1)

    # Discount factors
    exp_qT = math.exp(-q * T)
    exp_rT = math.exp(-r * T)

    if option_type == "call":
        price = S * exp_qT * N_d1 - K * exp_rT * N_d2
        delta = exp_qT * N_d1
        rho = K * T * exp_rT * N_d2 / 100  # Per 1% change
        prob_itm = N_d2
    else:
        price = K * exp_rT * N_neg_d2 - S * exp_qT * N_neg_d1
        delta = -exp_qT * N_neg_d1
        rho = -K * T * exp_rT * N_neg_d2 / 100
        prob_itm = N_neg_d2

    # Greeks (same for calls and puts)
    gamma = exp_qT * n_d1 / (S * sigma * sqrt_T)
    theta = (
        -S * exp_qT * n_d1 * sigma / (2 * sqrt_T)
        - r * K * exp_rT * (N_d2 if option_type == "call" else N_neg_d2)
        + q * S * exp_qT * (N_d1 if option_type == "call" else N_neg_d1)
    ) / 365  # Daily theta
    vega = S * exp_qT * n_d1 * sqrt_T / 100  # Per 1% change in vol

    return {
        "price": max(price, 0),
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
        "prob_itm": prob_itm,
    }


def calculate_historical_volatility(prices: list[float], window: int = 20) -> float:
    """Calculate annualized historical volatility from price series."""
    if len(prices) < window + 1:
        return 0.25  # Default 25% if not enough data

    returns = np.diff(np.log(prices[-window-1:]))
    return float(np.std(returns) * np.sqrt(252))


def implied_volatility_newton(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal["call", "put"],
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> float:
    """Calculate implied volatility using Newton-Raphson method."""
    if T <= 0 or market_price <= 0:
        return 0.0

    # Initial guess
    sigma = 0.3

    for _ in range(max_iterations):
        result = black_scholes(S, K, T, r, sigma, option_type)
        price = result["price"]
        vega = result["vega"] * 100  # Convert back from per 1%

        if vega < 1e-10:
            break

        diff = market_price - price
        if abs(diff) < tolerance:
            break

        sigma = sigma + diff / vega
        sigma = max(0.01, min(sigma, 5.0))  # Bound between 1% and 500%

    return sigma


def create_options_chain_server(
    simulation_date: datetime | None = None,
    name: str = "options-chain-mcp",
) -> FastMCP:
    """
    Create the Options Chain MCP server.

    Args:
        simulation_date: Optional date for temporal locking
        name: Server name

    Returns:
        Configured FastMCP server
    """
    mcp = FastMCP(name)
    _simulation_date = simulation_date
    _risk_free_rate = 0.0525  # 5.25% Fed funds rate

    def get_current_date() -> date:
        """Get the effective current date (respecting simulation date)."""
        if _simulation_date:
            return _simulation_date.date()
        return date.today()

    @mcp.tool
    def get_options_chain(
        ticker: str,
        expiration: str | None = None,
        option_type: Literal["call", "put", "all"] = "all",
        min_strike: float | None = None,
        max_strike: float | None = None,
        include_greeks: bool = True,
    ) -> dict[str, Any]:
        """
        Get options chain for a ticker with optional Greeks calculation.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL", "SPY")
            expiration: Expiration date (YYYY-MM-DD) or None for nearest
            option_type: "call", "put", or "all"
            min_strike: Minimum strike price filter
            max_strike: Maximum strike price filter
            include_greeks: Whether to calculate Greeks for each option

        Returns:
            Options chain with calls and/or puts, including Greeks if requested
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            underlying_price = info.get("regularMarketPrice") or info.get("currentPrice", 0)

            # Get available expirations
            expirations = stock.options
            if not expirations:
                return {"error": f"No options available for {ticker}"}

            # Select expiration
            if expiration is None:
                selected_exp = expirations[0]
            elif expiration in expirations:
                selected_exp = expiration
            else:
                # Find nearest expiration
                target = datetime.strptime(expiration, "%Y-%m-%d").date()
                selected_exp = min(expirations, key=lambda x: abs(
                    datetime.strptime(x, "%Y-%m-%d").date() - target
                ))

            # Calculate days to expiry
            exp_date = datetime.strptime(selected_exp, "%Y-%m-%d").date()
            current = get_current_date()
            days_to_expiry = max((exp_date - current).days, 0)
            T = days_to_expiry / 365.0

            # Get chain
            chain = stock.option_chain(selected_exp)

            # Get historical volatility for Greeks calculation
            hist = stock.history(period="3mo", interval="1d")
            hist_vol = calculate_historical_volatility(hist["Close"].tolist())

            def process_options(df, opt_type: Literal["call", "put"]) -> list[dict]:
                options = []
                for _, row in df.iterrows():
                    strike = float(row["strike"])

                    # Apply strike filters
                    if min_strike and strike < min_strike:
                        continue
                    if max_strike and strike > max_strike:
                        continue

                    # Use market IV if available, otherwise historical
                    iv = row.get("impliedVolatility", 0)
                    if iv is None or iv <= 0 or iv > 5:
                        iv = hist_vol

                    quote = {
                        "contract_symbol": row.get("contractSymbol", ""),
                        "ticker": ticker,
                        "strike": strike,
                        "expiration": selected_exp,
                        "option_type": opt_type,
                        "bid": float(row.get("bid", 0) or 0),
                        "ask": float(row.get("ask", 0) or 0),
                        "last_price": float(row.get("lastPrice", 0) or 0),
                        "volume": int(row.get("volume", 0) or 0),
                        "open_interest": int(row.get("openInterest", 0) or 0),
                        "implied_volatility": float(iv),
                        "in_the_money": bool(row.get("inTheMoney", False)),
                    }

                    if include_greeks and underlying_price > 0:
                        greeks = black_scholes(
                            S=underlying_price,
                            K=strike,
                            T=T,
                            r=_risk_free_rate,
                            sigma=iv,
                            option_type=opt_type,
                        )
                        quote.update({
                            "delta": round(greeks["delta"], 4),
                            "gamma": round(greeks["gamma"], 6),
                            "theta": round(greeks["theta"], 4),
                            "vega": round(greeks["vega"], 4),
                            "rho": round(greeks["rho"], 4),
                            "theoretical_price": round(greeks["price"], 2),
                        })

                    options.append(quote)

                return options

            result = {
                "ticker": ticker,
                "underlying_price": underlying_price,
                "expiration": selected_exp,
                "days_to_expiry": days_to_expiry,
                "available_expirations": list(expirations),
                "timestamp": datetime.now().isoformat(),
            }

            if option_type in ["call", "all"]:
                result["calls"] = process_options(chain.calls, "call")
            if option_type in ["put", "all"]:
                result["puts"] = process_options(chain.puts, "put")

            return result

        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    @mcp.tool
    def calculate_option_price(
        ticker: str,
        strike: float,
        expiration: str,
        option_type: Literal["call", "put"],
        volatility: float | None = None,
        underlying_price: float | None = None,
    ) -> dict[str, Any]:
        """
        Calculate theoretical option price using Black-Scholes.

        Args:
            ticker: Stock ticker symbol
            strike: Strike price
            expiration: Expiration date (YYYY-MM-DD)
            option_type: "call" or "put"
            volatility: Annual volatility (e.g., 0.30 for 30%). If None, uses historical.
            underlying_price: Current stock price. If None, fetches from market.

        Returns:
            Theoretical price and all Greeks
        """
        try:
            stock = yf.Ticker(ticker)

            # Get underlying price
            if underlying_price is None:
                info = stock.info
                underlying_price = info.get("regularMarketPrice") or info.get("currentPrice", 0)

            if underlying_price <= 0:
                return {"error": f"Could not get price for {ticker}"}

            # Calculate time to expiry
            exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
            current = get_current_date()
            days_to_expiry = max((exp_date - current).days, 0)
            T = days_to_expiry / 365.0

            # Get volatility
            if volatility is None:
                hist = stock.history(period="3mo", interval="1d")
                volatility = calculate_historical_volatility(hist["Close"].tolist())

            # Calculate
            result = black_scholes(
                S=underlying_price,
                K=strike,
                T=T,
                r=_risk_free_rate,
                sigma=volatility,
                option_type=option_type,
            )

            return TheoreticalPrice(
                ticker=ticker,
                strike=strike,
                expiration=expiration,
                option_type=option_type,
                underlying_price=underlying_price,
                days_to_expiry=days_to_expiry,
                risk_free_rate=_risk_free_rate,
                volatility=volatility,
                theoretical_price=round(result["price"], 4),
                delta=round(result["delta"], 4),
                gamma=round(result["gamma"], 6),
                theta=round(result["theta"], 4),
                vega=round(result["vega"], 4),
                rho=round(result["rho"], 4),
                prob_itm=round(result["prob_itm"], 4),
                prob_otm=round(1 - result["prob_itm"], 4),
            ).model_dump()

        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    @mcp.tool
    def calculate_historical_option_price(
        ticker: str,
        strike: float,
        expiration: str,
        option_type: Literal["call", "put"],
        as_of_date: str,
        volatility: float | None = None,
    ) -> dict[str, Any]:
        """
        Calculate what an option would have been worth on a historical date.

        Uses historical underlying price and Black-Scholes to estimate
        the theoretical option price as of a past date.

        Args:
            ticker: Stock ticker symbol
            strike: Strike price
            expiration: Option expiration date (YYYY-MM-DD)
            option_type: "call" or "put"
            as_of_date: Historical date to price the option (YYYY-MM-DD)
            volatility: Override volatility. If None, uses historical vol as of that date.

        Returns:
            Theoretical price and Greeks as of the historical date
        """
        try:
            stock = yf.Ticker(ticker)

            # Parse dates
            as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
            exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()

            if as_of >= exp_date:
                return {"error": "as_of_date must be before expiration"}

            # Get historical price data
            start_date = as_of - timedelta(days=90)
            hist = stock.history(start=start_date.isoformat(), end=(as_of + timedelta(days=1)).isoformat())

            if hist.empty:
                return {"error": f"No historical data for {ticker} on {as_of_date}"}

            # Get the closing price on or before the as_of_date
            underlying_price = float(hist["Close"].iloc[-1])

            # Calculate time to expiry from that date
            days_to_expiry = (exp_date - as_of).days
            T = days_to_expiry / 365.0

            # Calculate historical volatility as of that date
            if volatility is None:
                prices = hist["Close"].tolist()
                volatility = calculate_historical_volatility(prices, window=20)

            # Calculate theoretical price
            result = black_scholes(
                S=underlying_price,
                K=strike,
                T=T,
                r=_risk_free_rate,
                sigma=volatility,
                option_type=option_type,
            )

            return {
                "ticker": ticker,
                "as_of_date": as_of_date,
                "underlying_price": round(underlying_price, 2),
                "strike": strike,
                "expiration": expiration,
                "option_type": option_type,
                "days_to_expiry": days_to_expiry,
                "volatility_used": round(volatility, 4),
                "theoretical_price": round(result["price"], 4),
                "delta": round(result["delta"], 4),
                "gamma": round(result["gamma"], 6),
                "theta": round(result["theta"], 4),
                "vega": round(result["vega"], 4),
                "rho": round(result["rho"], 4),
                "intrinsic_value": round(max(underlying_price - strike, 0) if option_type == "call" else max(strike - underlying_price, 0), 2),
                "time_value": round(result["price"] - max(underlying_price - strike, 0) if option_type == "call" else result["price"] - max(strike - underlying_price, 0), 2),
            }

        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    @mcp.tool
    def get_volatility_analysis(ticker: str) -> dict[str, Any]:
        """
        Get volatility analysis for a ticker.

        Includes historical volatility at various windows and current IV if available.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Volatility metrics including HV and IV
        """
        try:
            stock = yf.Ticker(ticker)

            # Get historical data
            hist = stock.history(period="1y", interval="1d")
            if hist.empty:
                return {"error": f"No historical data for {ticker}"}

            prices = hist["Close"].tolist()

            # Calculate historical volatility at different windows
            hv_20 = calculate_historical_volatility(prices, 20)
            hv_60 = calculate_historical_volatility(prices, 60)
            hv_252 = calculate_historical_volatility(prices, 252) if len(prices) >= 253 else hv_60

            # Try to get current IV from options chain
            current_iv = None
            iv_percentile = None
            try:
                expirations = stock.options
                if expirations:
                    chain = stock.option_chain(expirations[0])
                    # Get ATM options IV
                    info = stock.info
                    current_price = info.get("regularMarketPrice", 0)
                    if current_price > 0:
                        calls = chain.calls
                        atm = calls.iloc[(calls["strike"] - current_price).abs().argsort()[:3]]
                        ivs = atm["impliedVolatility"].dropna()
                        if not ivs.empty:
                            current_iv = float(ivs.mean())
            except:
                pass

            return VolatilityData(
                ticker=ticker,
                current_iv=round(current_iv, 4) if current_iv else None,
                historical_volatility_20d=round(hv_20, 4),
                historical_volatility_60d=round(hv_60, 4),
                historical_volatility_252d=round(hv_252, 4),
                iv_percentile=None,  # Would need historical IV data
                iv_rank=None,
            ).model_dump()

        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    @mcp.tool
    def get_expirations(ticker: str) -> dict[str, Any]:
        """
        Get all available expiration dates for a ticker's options.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of available expiration dates with days to expiry
        """
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options

            if not expirations:
                return {"error": f"No options available for {ticker}", "ticker": ticker}

            current = get_current_date()
            result = []
            for exp in expirations:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                days = (exp_date - current).days
                result.append({
                    "expiration": exp,
                    "days_to_expiry": days,
                    "is_weekly": days <= 7,
                    "is_monthly": exp_date.day >= 15 and exp_date.day <= 21,
                })

            return {
                "ticker": ticker,
                "expirations": result,
                "total_count": len(result),
            }

        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    @mcp.tool
    def analyze_strategy(
        ticker: str,
        legs: list[dict],
        underlying_price: float | None = None,
    ) -> dict[str, Any]:
        """
        Analyze a multi-leg options strategy.

        Args:
            ticker: Stock ticker symbol
            legs: List of strategy legs, each with:
                - strike: Strike price
                - expiration: Expiration date (YYYY-MM-DD)
                - option_type: "call" or "put"
                - action: "buy" or "sell"
                - quantity: Number of contracts
            underlying_price: Current stock price. If None, fetches from market.

        Returns:
            Strategy analysis including max profit, max loss, breakevens, and net Greeks
        """
        try:
            stock = yf.Ticker(ticker)

            if underlying_price is None:
                info = stock.info
                underlying_price = info.get("regularMarketPrice") or info.get("currentPrice", 0)

            if underlying_price <= 0:
                return {"error": f"Could not get price for {ticker}"}

            # Get historical vol
            hist = stock.history(period="3mo", interval="1d")
            hist_vol = calculate_historical_volatility(hist["Close"].tolist())

            current = get_current_date()

            total_cost = 0.0
            net_delta = 0.0
            net_gamma = 0.0
            net_theta = 0.0
            net_vega = 0.0

            leg_details = []

            for leg in legs:
                strike = leg["strike"]
                expiration = leg["expiration"]
                option_type = leg["option_type"]
                action = leg["action"]
                quantity = leg.get("quantity", 1)

                exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
                days_to_expiry = max((exp_date - current).days, 0)
                T = days_to_expiry / 365.0

                result = black_scholes(
                    S=underlying_price,
                    K=strike,
                    T=T,
                    r=_risk_free_rate,
                    sigma=hist_vol,
                    option_type=option_type,
                )

                price = result["price"]
                multiplier = quantity if action == "buy" else -quantity

                total_cost += price * multiplier * 100  # Each contract = 100 shares
                net_delta += result["delta"] * multiplier * 100
                net_gamma += result["gamma"] * multiplier * 100
                net_theta += result["theta"] * multiplier * 100
                net_vega += result["vega"] * multiplier * 100

                leg_details.append({
                    "strike": strike,
                    "expiration": expiration,
                    "option_type": option_type,
                    "action": action,
                    "quantity": quantity,
                    "price_per_contract": round(price, 2),
                    "total_value": round(price * quantity * 100, 2),
                    "delta": round(result["delta"] * multiplier, 4),
                    "gamma": round(result["gamma"] * multiplier, 6),
                    "theta": round(result["theta"] * multiplier, 4),
                    "vega": round(result["vega"] * multiplier, 4),
                })

            # Determine strategy type
            strategy_type = "custom"
            if len(legs) == 2:
                if all(l["option_type"] == "call" for l in legs):
                    if legs[0]["action"] != legs[1]["action"]:
                        strategy_type = "vertical_call_spread"
                elif all(l["option_type"] == "put" for l in legs):
                    if legs[0]["action"] != legs[1]["action"]:
                        strategy_type = "vertical_put_spread"
                elif legs[0]["strike"] == legs[1]["strike"]:
                    strategy_type = "straddle"
                else:
                    strategy_type = "strangle"
            elif len(legs) == 4:
                call_count = sum(1 for l in legs if l["option_type"] == "call")
                if call_count == 2:
                    strategy_type = "iron_condor"

            return {
                "ticker": ticker,
                "underlying_price": underlying_price,
                "strategy_type": strategy_type,
                "legs": leg_details,
                "net_cost": round(total_cost, 2),
                "is_credit": total_cost < 0,
                "net_greeks": {
                    "delta": round(net_delta, 2),
                    "gamma": round(net_gamma, 4),
                    "theta": round(net_theta, 2),
                    "vega": round(net_vega, 2),
                },
                "volatility_used": round(hist_vol, 4),
            }

        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    return mcp


# CLI entry point
if __name__ == "__main__":
    import sys

    simulation_date = None
    if len(sys.argv) > 1:
        try:
            simulation_date = datetime.fromisoformat(sys.argv[1])
        except ValueError:
            pass

    server = create_options_chain_server(simulation_date=simulation_date)
    server.run()
