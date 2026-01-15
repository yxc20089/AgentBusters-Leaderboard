"""
Trading Simulator MCP Server

Paper trading engine for options with realistic execution simulation.
Supports:
- Portfolio creation and management
- Trade execution with bid/ask spread and slippage
- Position tracking with real-time P&L
- Time advancement for backtesting
- Historical replay of market conditions
"""

import math
import uuid
from datetime import datetime, date, timedelta
from typing import Any, Literal
from dataclasses import dataclass, field

from fastmcp import FastMCP
from pydantic import BaseModel, Field
import yfinance as yf

from mcp_servers.options_chain import black_scholes, calculate_historical_volatility


# Commission per contract (industry standard)
COMMISSION_PER_CONTRACT = 0.65


class TradeRecord(BaseModel):
    """Record of an executed trade."""
    trade_id: str
    timestamp: str
    ticker: str
    strike: float
    expiration: str
    option_type: Literal["call", "put"]
    action: Literal["buy_to_open", "sell_to_open", "buy_to_close", "sell_to_close"]
    quantity: int
    fill_price: float
    commission: float
    slippage: float
    total_cost: float  # Negative = paid, Positive = received


class PositionRecord(BaseModel):
    """Current position in a contract."""
    position_id: str
    ticker: str
    strike: float
    expiration: str
    option_type: Literal["call", "put"]
    quantity: int  # Positive = long, Negative = short
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    opened_at: str
    last_updated: str


class PortfolioState(BaseModel):
    """Complete portfolio state."""
    portfolio_id: str
    cash: float
    starting_cash: float
    positions: list[PositionRecord]
    trades: list[TradeRecord]
    total_value: float
    total_return_pct: float
    unrealized_pnl: float
    realized_pnl: float
    simulation_date: str
    created_at: str


@dataclass
class Position:
    """Internal position tracking."""
    position_id: str
    ticker: str
    strike: float
    expiration: str
    option_type: Literal["call", "put"]
    quantity: int
    avg_entry_price: float
    realized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=datetime.now)

    def get_contract_key(self) -> str:
        return f"{self.ticker}_{self.expiration}_{self.option_type}_{self.strike}"


@dataclass
class Portfolio:
    """Internal portfolio state."""
    portfolio_id: str
    cash: float
    starting_cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[TradeRecord] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


def create_trading_sim_server(
    simulation_date: datetime | None = None,
    starting_capital: float = 100000.0,
    name: str = "trading-sim-mcp",
) -> FastMCP:
    """
    Create the Trading Simulator MCP server.

    Args:
        simulation_date: Optional date for temporal locking
        starting_capital: Initial portfolio capital
        name: Server name

    Returns:
        Configured FastMCP server
    """
    mcp = FastMCP(name)

    # Server state
    _portfolios: dict[str, Portfolio] = {}
    _simulation_date = simulation_date
    _starting_capital = starting_capital
    _risk_free_rate = 0.0525

    def get_current_date() -> date:
        if _simulation_date:
            return _simulation_date.date()
        return date.today()

    def get_option_price(
        ticker: str,
        strike: float,
        expiration: str,
        option_type: Literal["call", "put"],
        as_of_date: date | None = None,
    ) -> tuple[float, float, float]:
        """
        Get option price (theoretical bid, ask, mid).

        Returns: (bid, ask, mid_price)
        """
        try:
            stock = yf.Ticker(ticker)

            # Get underlying price
            if as_of_date and as_of_date < date.today():
                # Historical price
                start = as_of_date - timedelta(days=5)
                hist = stock.history(start=start.isoformat(), end=(as_of_date + timedelta(days=1)).isoformat())
                if hist.empty:
                    return 0, 0, 0
                underlying = float(hist["Close"].iloc[-1])

                # Calculate historical vol
                longer_hist = stock.history(start=(as_of_date - timedelta(days=60)).isoformat(),
                                           end=(as_of_date + timedelta(days=1)).isoformat())
                vol = calculate_historical_volatility(longer_hist["Close"].tolist())
            else:
                # Current price
                info = stock.info
                underlying = info.get("regularMarketPrice") or info.get("currentPrice", 0)
                hist = stock.history(period="3mo")
                vol = calculate_historical_volatility(hist["Close"].tolist())

            if underlying <= 0:
                return 0, 0, 0

            # Calculate days to expiry
            exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
            ref_date = as_of_date or get_current_date()
            days_to_expiry = max((exp_date - ref_date).days, 0)
            T = days_to_expiry / 365.0

            # Get theoretical price
            result = black_scholes(
                S=underlying,
                K=strike,
                T=T,
                r=_risk_free_rate,
                sigma=vol,
                option_type=option_type,
            )

            mid = result["price"]

            # Simulate bid-ask spread based on price and liquidity
            # Wider spreads for cheaper options and further OTM
            if mid < 0.50:
                spread_pct = 0.10  # 10% spread for cheap options
            elif mid < 2.00:
                spread_pct = 0.05  # 5% spread
            else:
                spread_pct = 0.02  # 2% spread for expensive options

            half_spread = mid * spread_pct / 2
            bid = max(0.01, mid - half_spread)
            ask = mid + half_spread

            return round(bid, 2), round(ask, 2), round(mid, 2)

        except Exception:
            return 0, 0, 0

    @mcp.tool
    def create_portfolio(
        portfolio_id: str | None = None,
        starting_cash: float | None = None,
    ) -> dict[str, Any]:
        """
        Create a new trading portfolio.

        Args:
            portfolio_id: Optional custom portfolio ID. Auto-generated if not provided.
            starting_cash: Initial cash. Uses server default if not provided.

        Returns:
            Portfolio state with ID and starting cash
        """
        pid = portfolio_id or f"portfolio_{uuid.uuid4().hex[:8]}"
        cash = starting_cash or _starting_capital

        portfolio = Portfolio(
            portfolio_id=pid,
            cash=cash,
            starting_cash=cash,
        )
        _portfolios[pid] = portfolio

        return {
            "portfolio_id": pid,
            "cash": cash,
            "starting_cash": cash,
            "simulation_date": get_current_date().isoformat(),
            "message": f"Portfolio created with ${cash:,.2f}",
        }

    @mcp.tool
    def execute_trade(
        portfolio_id: str,
        ticker: str,
        strike: float,
        expiration: str,
        option_type: Literal["call", "put"],
        action: Literal["buy_to_open", "sell_to_open", "buy_to_close", "sell_to_close"],
        quantity: int,
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        """
        Execute an options trade.

        Args:
            portfolio_id: Portfolio ID
            ticker: Stock ticker symbol
            strike: Strike price
            expiration: Expiration date (YYYY-MM-DD)
            option_type: "call" or "put"
            action: Trade action (buy_to_open, sell_to_open, buy_to_close, sell_to_close)
            quantity: Number of contracts
            limit_price: Optional limit price. If None, executes at market.

        Returns:
            Trade execution details including fill price and new position
        """
        if portfolio_id not in _portfolios:
            return {"error": f"Portfolio {portfolio_id} not found"}

        portfolio = _portfolios[portfolio_id]

        # Get current option price
        bid, ask, mid = get_option_price(ticker, strike, expiration, option_type)

        if mid <= 0:
            return {"error": f"Could not get price for {ticker} {strike} {option_type}"}

        # Determine fill price based on action
        is_buy = action in ["buy_to_open", "buy_to_close"]

        if is_buy:
            # Buy at ask (or limit if better)
            base_price = ask
            if limit_price and limit_price < ask:
                if limit_price >= bid:
                    base_price = limit_price
                else:
                    return {"error": f"Limit price ${limit_price} below bid ${bid}"}
        else:
            # Sell at bid (or limit if better)
            base_price = bid
            if limit_price and limit_price > bid:
                if limit_price <= ask:
                    base_price = limit_price
                else:
                    return {"error": f"Limit price ${limit_price} above ask ${ask}"}

        # Add slippage (simulates market impact)
        slippage_pct = 0.005  # 0.5% slippage
        slippage = base_price * slippage_pct
        if is_buy:
            fill_price = base_price + slippage
        else:
            fill_price = max(0.01, base_price - slippage)

        fill_price = round(fill_price, 2)

        # Calculate total cost (each contract = 100 shares)
        notional = fill_price * quantity * 100
        commission = COMMISSION_PER_CONTRACT * quantity

        if is_buy:
            total_cost = notional + commission  # Pay for options + commission
            cash_change = -total_cost
        else:
            total_cost = -notional + commission  # Receive premium, pay commission
            cash_change = notional - commission

        # Check if we have enough cash for buys
        if is_buy and portfolio.cash < total_cost:
            return {
                "error": f"Insufficient cash. Need ${total_cost:,.2f}, have ${portfolio.cash:,.2f}"
            }

        # Update cash
        portfolio.cash += cash_change

        # Update positions
        contract_key = f"{ticker}_{expiration}_{option_type}_{strike}"

        if action == "buy_to_open":
            # Open or add to long position
            if contract_key in portfolio.positions:
                pos = portfolio.positions[contract_key]
                # Average up/down
                total_qty = pos.quantity + quantity
                pos.avg_entry_price = (
                    pos.avg_entry_price * pos.quantity + fill_price * quantity
                ) / total_qty
                pos.quantity = total_qty
            else:
                portfolio.positions[contract_key] = Position(
                    position_id=f"pos_{uuid.uuid4().hex[:8]}",
                    ticker=ticker,
                    strike=strike,
                    expiration=expiration,
                    option_type=option_type,
                    quantity=quantity,
                    avg_entry_price=fill_price,
                )

        elif action == "sell_to_open":
            # Open or add to short position
            if contract_key in portfolio.positions:
                pos = portfolio.positions[contract_key]
                total_qty = pos.quantity - quantity
                if total_qty == 0:
                    # Position closed
                    pnl = (pos.avg_entry_price - fill_price) * abs(pos.quantity) * 100
                    pos.realized_pnl += pnl
                    del portfolio.positions[contract_key]
                else:
                    pos.avg_entry_price = (
                        pos.avg_entry_price * abs(pos.quantity) + fill_price * quantity
                    ) / abs(total_qty)
                    pos.quantity = total_qty
            else:
                portfolio.positions[contract_key] = Position(
                    position_id=f"pos_{uuid.uuid4().hex[:8]}",
                    ticker=ticker,
                    strike=strike,
                    expiration=expiration,
                    option_type=option_type,
                    quantity=-quantity,  # Negative for short
                    avg_entry_price=fill_price,
                )

        elif action == "buy_to_close":
            # Close short position
            if contract_key not in portfolio.positions:
                return {"error": f"No short position to close for {contract_key}"}

            pos = portfolio.positions[contract_key]
            if pos.quantity >= 0:
                return {"error": f"Position is long, cannot buy_to_close"}

            if quantity > abs(pos.quantity):
                return {"error": f"Cannot close {quantity} contracts, only {abs(pos.quantity)} short"}

            # Calculate realized P&L (sold high, bought back low = profit)
            pnl = (pos.avg_entry_price - fill_price) * quantity * 100
            pos.realized_pnl += pnl

            pos.quantity += quantity  # Reduce short (add positive)
            if pos.quantity == 0:
                del portfolio.positions[contract_key]

        elif action == "sell_to_close":
            # Close long position
            if contract_key not in portfolio.positions:
                return {"error": f"No long position to close for {contract_key}"}

            pos = portfolio.positions[contract_key]
            if pos.quantity <= 0:
                return {"error": f"Position is short, cannot sell_to_close"}

            if quantity > pos.quantity:
                return {"error": f"Cannot close {quantity} contracts, only {pos.quantity} long"}

            # Calculate realized P&L
            pnl = (fill_price - pos.avg_entry_price) * quantity * 100
            pos.realized_pnl += pnl

            pos.quantity -= quantity
            if pos.quantity == 0:
                del portfolio.positions[contract_key]

        # Record trade
        trade = TradeRecord(
            trade_id=f"trade_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now().isoformat(),
            ticker=ticker,
            strike=strike,
            expiration=expiration,
            option_type=option_type,
            action=action,
            quantity=quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=round(slippage * quantity * 100, 2),
            total_cost=round(cash_change, 2),
        )
        portfolio.trades.append(trade)

        return {
            "success": True,
            "trade": trade.model_dump(),
            "cash_remaining": round(portfolio.cash, 2),
            "position_count": len(portfolio.positions),
            "quote": {"bid": bid, "ask": ask, "mid": mid},
        }

    def _get_portfolio_state(portfolio_id: str) -> dict[str, Any]:
        """Internal helper to get portfolio state (can be called by other functions)."""
        if portfolio_id not in _portfolios:
            return {"error": f"Portfolio {portfolio_id} not found"}

        portfolio = _portfolios[portfolio_id]

        # Calculate current values for all positions
        positions = []
        total_unrealized = 0.0
        total_realized = 0.0

        for key, pos in portfolio.positions.items():
            bid, ask, mid = get_option_price(
                pos.ticker, pos.strike, pos.expiration, pos.option_type
            )

            if pos.quantity > 0:
                # Long position - can sell at bid
                current_price = bid
                unrealized = (current_price - pos.avg_entry_price) * pos.quantity * 100
            else:
                # Short position - would buy back at ask
                current_price = ask
                unrealized = (pos.avg_entry_price - current_price) * abs(pos.quantity) * 100

            total_unrealized += unrealized
            total_realized += pos.realized_pnl

            positions.append(PositionRecord(
                position_id=pos.position_id,
                ticker=pos.ticker,
                strike=pos.strike,
                expiration=pos.expiration,
                option_type=pos.option_type,
                quantity=pos.quantity,
                avg_entry_price=pos.avg_entry_price,
                current_price=current_price,
                unrealized_pnl=round(unrealized, 2),
                realized_pnl=round(pos.realized_pnl, 2),
                opened_at=pos.opened_at.isoformat(),
                last_updated=datetime.now().isoformat(),
            ).model_dump())

        # Calculate position value
        position_value = sum(
            abs(pos.quantity) * (bid if pos.quantity > 0 else ask) * 100
            for key, pos in portfolio.positions.items()
            for bid, ask, _ in [get_option_price(pos.ticker, pos.strike, pos.expiration, pos.option_type)]
        )

        total_value = portfolio.cash + position_value
        total_return_pct = ((total_value - portfolio.starting_cash) / portfolio.starting_cash) * 100

        return PortfolioState(
            portfolio_id=portfolio.portfolio_id,
            cash=round(portfolio.cash, 2),
            starting_cash=portfolio.starting_cash,
            positions=positions,
            trades=[t.model_dump() for t in portfolio.trades[-10:]],  # Last 10 trades
            total_value=round(total_value, 2),
            total_return_pct=round(total_return_pct, 2),
            unrealized_pnl=round(total_unrealized, 2),
            realized_pnl=round(total_realized, 2),
            simulation_date=get_current_date().isoformat(),
            created_at=portfolio.created_at.isoformat(),
        ).model_dump()

    @mcp.tool
    def get_portfolio(portfolio_id: str) -> dict[str, Any]:
        """
        Get current portfolio state with all positions and P&L.

        Args:
            portfolio_id: Portfolio ID

        Returns:
            Complete portfolio state including positions, cash, and returns
        """
        return _get_portfolio_state(portfolio_id)

    @mcp.tool
    def close_position(
        portfolio_id: str,
        position_id: str | None = None,
        contract_key: str | None = None,
        quantity: int | None = None,
    ) -> dict[str, Any]:
        """
        Close an existing position.

        Args:
            portfolio_id: Portfolio ID
            position_id: Position ID to close
            contract_key: Alternative: contract key (ticker_expiration_type_strike)
            quantity: Number of contracts to close. If None, closes entire position.

        Returns:
            Trade execution details
        """
        if portfolio_id not in _portfolios:
            return {"error": f"Portfolio {portfolio_id} not found"}

        portfolio = _portfolios[portfolio_id]

        # Find position
        pos = None
        for key, p in portfolio.positions.items():
            if position_id and p.position_id == position_id:
                pos = p
                break
            if contract_key and key == contract_key:
                pos = p
                break

        if not pos:
            return {"error": "Position not found"}

        qty = quantity or abs(pos.quantity)

        if pos.quantity > 0:
            action = "sell_to_close"
        else:
            action = "buy_to_close"

        return execute_trade(
            portfolio_id=portfolio_id,
            ticker=pos.ticker,
            strike=pos.strike,
            expiration=pos.expiration,
            option_type=pos.option_type,
            action=action,
            quantity=qty,
        )

    @mcp.tool
    def advance_time(
        portfolio_id: str,
        days: int = 1,
    ) -> dict[str, Any]:
        """
        Advance simulation time and update all position values.

        Handles:
        - Time decay (theta)
        - Expiration (options expire worthless or ITM)
        - Updated P&L calculations

        Args:
            portfolio_id: Portfolio ID
            days: Number of days to advance

        Returns:
            Updated portfolio state with any expired positions handled
        """
        nonlocal _simulation_date

        if portfolio_id not in _portfolios:
            return {"error": f"Portfolio {portfolio_id} not found"}

        # Advance the simulation date
        current = get_current_date()
        new_date = current + timedelta(days=days)

        if _simulation_date:
            _simulation_date = datetime.combine(new_date, datetime.min.time())
        else:
            _simulation_date = datetime.combine(new_date, datetime.min.time())

        portfolio = _portfolios[portfolio_id]

        # Check for expirations
        expired_positions = []
        for key, pos in list(portfolio.positions.items()):
            exp_date = datetime.strptime(pos.expiration, "%Y-%m-%d").date()

            if new_date >= exp_date:
                # Option has expired - determine if ITM
                stock = yf.Ticker(pos.ticker)
                info = stock.info
                underlying = info.get("regularMarketPrice") or info.get("currentPrice", 0)

                if pos.option_type == "call":
                    is_itm = underlying > pos.strike
                    settlement_value = max(underlying - pos.strike, 0)
                else:
                    is_itm = underlying < pos.strike
                    settlement_value = max(pos.strike - underlying, 0)

                # Calculate final P&L
                if pos.quantity > 0:
                    # Long position - receive settlement value
                    pnl = (settlement_value - pos.avg_entry_price) * pos.quantity * 100
                    portfolio.cash += settlement_value * pos.quantity * 100
                else:
                    # Short position - pay settlement value
                    pnl = (pos.avg_entry_price - settlement_value) * abs(pos.quantity) * 100
                    portfolio.cash -= settlement_value * abs(pos.quantity) * 100

                pos.realized_pnl += pnl

                expired_positions.append({
                    "position_id": pos.position_id,
                    "contract": key,
                    "quantity": pos.quantity,
                    "settlement_value": settlement_value,
                    "pnl": round(pnl, 2),
                    "is_itm": is_itm,
                })

                del portfolio.positions[key]

        return {
            "success": True,
            "previous_date": current.isoformat(),
            "new_date": new_date.isoformat(),
            "days_advanced": days,
            "expired_positions": expired_positions,
            "portfolio": _get_portfolio_state(portfolio_id),
        }

    @mcp.tool
    def get_pnl_report(
        portfolio_id: str,
        include_trades: bool = True,
    ) -> dict[str, Any]:
        """
        Generate detailed P&L report for a portfolio.

        Args:
            portfolio_id: Portfolio ID
            include_trades: Whether to include full trade history

        Returns:
            Comprehensive P&L breakdown
        """
        if portfolio_id not in _portfolios:
            return {"error": f"Portfolio {portfolio_id} not found"}

        portfolio = _portfolios[portfolio_id]
        state = _get_portfolio_state(portfolio_id)

        # Calculate statistics
        total_trades = len(portfolio.trades)
        winning_trades = sum(1 for t in portfolio.trades if t.total_cost > 0)
        total_commission = sum(t.commission for t in portfolio.trades)
        total_slippage = sum(t.slippage for t in portfolio.trades)

        return {
            "portfolio_id": portfolio_id,
            "summary": {
                "starting_cash": portfolio.starting_cash,
                "current_cash": round(portfolio.cash, 2),
                "total_value": state["total_value"],
                "total_return_pct": state["total_return_pct"],
                "unrealized_pnl": state["unrealized_pnl"],
                "realized_pnl": state["realized_pnl"],
                "total_pnl": round(state["unrealized_pnl"] + state["realized_pnl"], 2),
            },
            "statistics": {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "win_rate": round(winning_trades / total_trades * 100, 1) if total_trades > 0 else 0,
                "total_commission": round(total_commission, 2),
                "total_slippage": round(total_slippage, 2),
                "trading_costs": round(total_commission + total_slippage, 2),
            },
            "positions": state["positions"],
            "trades": [t.model_dump() for t in portfolio.trades] if include_trades else [],
            "simulation_date": get_current_date().isoformat(),
        }

    @mcp.tool
    def list_portfolios() -> dict[str, Any]:
        """
        List all portfolios.

        Returns:
            List of portfolio IDs and basic info
        """
        portfolios = []
        for pid, p in _portfolios.items():
            portfolios.append({
                "portfolio_id": pid,
                "cash": round(p.cash, 2),
                "position_count": len(p.positions),
                "trade_count": len(p.trades),
                "created_at": p.created_at.isoformat(),
            })

        return {
            "portfolios": portfolios,
            "total_count": len(portfolios),
        }

    return mcp


# CLI entry point
if __name__ == "__main__":
    import sys

    simulation_date = None
    starting_capital = 100000.0

    for arg in sys.argv[1:]:
        if arg.startswith("--date="):
            try:
                simulation_date = datetime.fromisoformat(arg.split("=")[1])
            except ValueError:
                pass
        elif arg.startswith("--capital="):
            try:
                starting_capital = float(arg.split("=")[1])
            except ValueError:
                pass

    server = create_trading_sim_server(
        simulation_date=simulation_date,
        starting_capital=starting_capital,
    )
    server.run()
