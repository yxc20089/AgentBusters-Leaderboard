"""
Yahoo Finance MCP Server client with temporal locking (Time Machine).

This client wraps the yahoo-finance-mcp server with:
- Simulation date enforcement
- Look-ahead bias detection
- Cost tracking
"""

from datetime import datetime
from typing import Any, Optional

import structlog
from pydantic import BaseModel, Field

from mcp_servers.base import BaseMCPClient, MCPConfig
from cio_agent.models import TemporalViolation, ViolationSeverity

logger = structlog.get_logger()


class TemporalViolationError(Exception):
    """Raised when an agent attempts to access future data."""
    pass


class PriceData(BaseModel):
    """Historical price data (OHLCV)."""
    ticker: str
    start_date: str
    end_date: str
    data: list[dict[str, Any]] = Field(default_factory=list)
    # Each item: {date, open, high, low, close, volume, adj_close}


class FinancialStatement(BaseModel):
    """Financial statement data from Yahoo Finance."""
    ticker: str
    statement_type: str  # income_statement, balance_sheet, cash_flow
    period: str  # annual, quarterly
    data: dict[str, Any] = Field(default_factory=dict)


class KeyStatistics(BaseModel):
    """Key statistics for a ticker."""
    ticker: str
    as_of_date: str
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    beta: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    fifty_day_average: Optional[float] = None
    two_hundred_day_average: Optional[float] = None
    dividend_yield: Optional[float] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class AnalystEstimates(BaseModel):
    """Analyst estimates and recommendations."""
    ticker: str
    as_of_date: str
    target_mean_price: Optional[float] = None
    target_high_price: Optional[float] = None
    target_low_price: Optional[float] = None
    recommendation_mean: Optional[float] = None
    recommendation_key: Optional[str] = None
    number_of_analyst_opinions: Optional[int] = None
    eps_estimates: dict[str, Any] = Field(default_factory=dict)
    revenue_estimates: dict[str, Any] = Field(default_factory=dict)


class TimeMachineYFinanceClient(BaseMCPClient):
    """
    Yahoo Finance MCP client with Time Machine temporal enforcement.

    Features:
    - Accepts a simulation_date from the Green Agent
    - All data requests are filtered to only return information available as of that date
    - Tracks temporal violations for penalty calculation
    """

    def __init__(
        self,
        config: MCPConfig,
        simulation_date: Optional[datetime] = None,
        temporal_lock_enabled: bool = True
    ):
        super().__init__(config, simulation_date)
        self.temporal_lock_enabled = temporal_lock_enabled
        self.temporal_violations: list[TemporalViolation] = []

    async def health_check(self) -> bool:
        """Check if the Yahoo Finance MCP server is healthy."""
        try:
            await self._request("GET", "/health", "yfinance:health_check")
            return True
        except Exception:
            return False

    def _check_temporal_lock(
        self,
        ticker: str,
        requested_date: str,
        raise_error: bool = True
    ) -> bool:
        """
        Check if the requested date violates temporal constraints.

        Args:
            ticker: Ticker being queried
            requested_date: Date being requested
            raise_error: Whether to raise an error on violation

        Returns:
            True if violation detected, False otherwise
        """
        if not self.temporal_lock_enabled or not self.simulation_date:
            return False

        try:
            req_date = datetime.fromisoformat(requested_date.replace("Z", "+00:00"))
            if req_date.tzinfo:
                req_date = req_date.replace(tzinfo=None)
        except ValueError:
            # Try parsing as date only
            req_date = datetime.strptime(requested_date, "%Y-%m-%d")

        if req_date > self.simulation_date:
            days_ahead = (req_date - self.simulation_date).days
            severity = (
                ViolationSeverity.HIGH if days_ahead > 90
                else ViolationSeverity.MEDIUM if days_ahead > 30
                else ViolationSeverity.LOW
            )

            violation = TemporalViolation(
                ticker=ticker,
                requested_date=requested_date,
                simulation_date=self.simulation_date.isoformat(),
                days_ahead=days_ahead,
                severity=severity,
                tool_name="yahoo-finance-mcp",
                timestamp=datetime.utcnow()
            )
            self.temporal_violations.append(violation)

            logger.warning(
                "temporal_violation_detected",
                ticker=ticker,
                requested=requested_date,
                simulation=self.simulation_date.isoformat(),
                days_ahead=days_ahead,
                severity=severity.value
            )

            if raise_error:
                raise TemporalViolationError(
                    f"403 Future Data Forbidden: Cannot access {requested_date} data "
                    f"when simulation date is {self.simulation_date.isoformat()}"
                )
            return True
        return False

    def _enforce_end_date(self, end_date: Optional[str]) -> str:
        """Enforce that end_date does not exceed simulation_date."""
        if not self.simulation_date:
            return end_date or datetime.utcnow().strftime("%Y-%m-%d")

        sim_date_str = self.simulation_date.strftime("%Y-%m-%d")

        if end_date is None:
            return sim_date_str

        if end_date > sim_date_str:
            logger.info(
                "enforcing_simulation_date",
                original_end=end_date,
                enforced_end=sim_date_str
            )
            return sim_date_str

        return end_date

    async def get_historical_prices(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> PriceData:
        """
        Get historical price data (OHLCV).

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), will be capped at simulation_date
            interval: Data interval (1d, 1wk, 1mo)

        Returns:
            PriceData with OHLCV data
        """
        enforced_end = self._enforce_end_date(end_date)

        # Check for explicit future date requests
        if end_date and end_date != enforced_end:
            self._check_temporal_lock(ticker, end_date, raise_error=False)

        params = {
            "ticker": ticker,
            "interval": interval,
        }
        if start_date:
            params["start"] = start_date
        params["end"] = enforced_end

        result = await self._request(
            "GET",
            f"/price/{ticker}",
            f"yfinance:get_price:{interval}",
            params=params
        )

        return PriceData(
            ticker=ticker,
            start_date=start_date or "",
            end_date=enforced_end,
            data=result.get("data", [])
        )

    async def get_financial_statements(
        self,
        ticker: str,
        statement_type: str = "income_statement",
        period: str = "annual"
    ) -> FinancialStatement:
        """
        Get financial statements.

        Args:
            ticker: Stock ticker symbol
            statement_type: Type of statement (income_statement, balance_sheet, cash_flow)
            period: Period type (annual, quarterly)

        Returns:
            FinancialStatement with financial data
        """
        params = {
            "ticker": ticker,
            "period": period,
        }

        # Add simulation_date to filter historical data
        if self.simulation_date:
            params["as_of_date"] = self.simulation_date.strftime("%Y-%m-%d")

        result = await self._request(
            "GET",
            f"/financials/{ticker}/{statement_type}",
            f"yfinance:get_financials:{statement_type}",
            params=params
        )

        return FinancialStatement(
            ticker=ticker,
            statement_type=statement_type,
            period=period,
            data=result.get("data", {})
        )

    async def get_key_statistics(
        self,
        ticker: str,
        as_of_date: Optional[str] = None
    ) -> KeyStatistics:
        """
        Get key statistics for a ticker.

        Args:
            ticker: Stock ticker symbol
            as_of_date: Date for statistics (defaults to simulation_date)

        Returns:
            KeyStatistics with valuation and technical metrics
        """
        effective_date = as_of_date or (
            self.simulation_date.strftime("%Y-%m-%d") if self.simulation_date
            else datetime.utcnow().strftime("%Y-%m-%d")
        )

        # Check temporal constraint
        if as_of_date:
            self._check_temporal_lock(ticker, as_of_date)

        params = {
            "ticker": ticker,
            "as_of_date": effective_date,
        }

        result = await self._request(
            "GET",
            f"/statistics/{ticker}",
            "yfinance:get_statistics",
            params=params
        )

        return KeyStatistics(
            ticker=ticker,
            as_of_date=effective_date,
            market_cap=result.get("market_cap"),
            pe_ratio=result.get("pe_ratio"),
            forward_pe=result.get("forward_pe"),
            peg_ratio=result.get("peg_ratio"),
            price_to_book=result.get("price_to_book"),
            beta=result.get("beta"),
            fifty_two_week_high=result.get("fifty_two_week_high"),
            fifty_two_week_low=result.get("fifty_two_week_low"),
            fifty_day_average=result.get("fifty_day_average"),
            two_hundred_day_average=result.get("two_hundred_day_average"),
            dividend_yield=result.get("dividend_yield"),
            extra=result.get("extra", {})
        )

    async def get_analyst_estimates(
        self,
        ticker: str,
        as_of_date: Optional[str] = None
    ) -> AnalystEstimates:
        """
        Get analyst estimates and recommendations.

        Args:
            ticker: Stock ticker symbol
            as_of_date: Date for estimates (defaults to simulation_date)

        Returns:
            AnalystEstimates with price targets and recommendations
        """
        effective_date = as_of_date or (
            self.simulation_date.strftime("%Y-%m-%d") if self.simulation_date
            else datetime.utcnow().strftime("%Y-%m-%d")
        )

        # Check temporal constraint
        if as_of_date:
            self._check_temporal_lock(ticker, as_of_date)

        params = {
            "ticker": ticker,
            "as_of_date": effective_date,
        }

        result = await self._request(
            "GET",
            f"/estimates/{ticker}",
            "yfinance:get_estimates",
            params=params
        )

        return AnalystEstimates(
            ticker=ticker,
            as_of_date=effective_date,
            target_mean_price=result.get("target_mean_price"),
            target_high_price=result.get("target_high_price"),
            target_low_price=result.get("target_low_price"),
            recommendation_mean=result.get("recommendation_mean"),
            recommendation_key=result.get("recommendation_key"),
            number_of_analyst_opinions=result.get("number_of_analyst_opinions"),
            eps_estimates=result.get("eps_estimates", {}),
            revenue_estimates=result.get("revenue_estimates", {})
        )

    async def get_company_info(self, ticker: str) -> dict[str, Any]:
        """
        Get basic company information.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with company info (name, sector, industry, etc.)
        """
        result = await self._request(
            "GET",
            f"/info/{ticker}",
            "yfinance:get_info",
            params={"ticker": ticker}
        )
        return result

    def get_temporal_violations(self) -> list[TemporalViolation]:
        """Get all temporal violations logged by this client."""
        return self.temporal_violations

    def calculate_lookahead_penalty(self) -> float:
        """
        Calculate the look-ahead penalty based on temporal violations.

        Returns:
            Penalty multiplier (0.0 to 0.5)
        """
        if not self.temporal_violations:
            return 0.0

        total_days_ahead = sum(v.days_ahead for v in self.temporal_violations)
        return min(0.5, total_days_ahead / 365.0)
