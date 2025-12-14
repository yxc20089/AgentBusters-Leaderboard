"""
AlphaVantage API client for financial data retrieval.

Provides structured access to fundamental data endpoints:
- INCOME_STATEMENT, BALANCE_SHEET, CASH_FLOW
- EARNINGS for beat/miss analysis
- OVERVIEW for company metadata

Rate limiting and caching included for API efficiency.
"""

import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
import structlog
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

logger = structlog.get_logger()

# Rate limiting: AlphaVantage free tier = 5 calls/minute
RATE_LIMIT_CALLS = 5
RATE_LIMIT_PERIOD = 60  # seconds


class IncomeStatementData(BaseModel):
    """Parsed income statement data."""
    fiscal_date_ending: str
    reported_currency: str = "USD"
    gross_profit: Optional[float] = None
    total_revenue: Optional[float] = None
    cost_of_revenue: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    ebitda: Optional[float] = None
    interest_expense: Optional[float] = None
    income_tax_expense: Optional[float] = None
    depreciation_and_amortization: Optional[float] = None
    research_and_development: Optional[float] = None


class BalanceSheetData(BaseModel):
    """Parsed balance sheet data."""
    fiscal_date_ending: str
    reported_currency: str = "USD"
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_shareholder_equity: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    short_term_investments: Optional[float] = None
    total_current_assets: Optional[float] = None
    total_current_liabilities: Optional[float] = None
    long_term_debt: Optional[float] = None
    short_term_debt: Optional[float] = None
    total_debt: Optional[float] = None


class CashFlowData(BaseModel):
    """Parsed cash flow data."""
    fiscal_date_ending: str
    reported_currency: str = "USD"
    operating_cashflow: Optional[float] = None
    capital_expenditures: Optional[float] = None
    free_cash_flow: Optional[float] = None
    dividend_payout: Optional[float] = None


class EarningsData(BaseModel):
    """Parsed earnings data for beat/miss analysis."""
    fiscal_date_ending: str
    reported_eps: Optional[float] = None
    estimated_eps: Optional[float] = None
    surprise: Optional[float] = None
    surprise_percentage: Optional[float] = None


class CompanyOverview(BaseModel):
    """Company metadata and description."""
    symbol: str
    name: str
    description: str = ""
    sector: str = ""
    industry: str = ""
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    eps: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None


class FundamentalData(BaseModel):
    """Complete fundamental data for a ticker."""
    ticker: str
    overview: Optional[CompanyOverview] = None
    annual_income_statements: list[IncomeStatementData] = Field(default_factory=list)
    quarterly_income_statements: list[IncomeStatementData] = Field(default_factory=list)
    annual_balance_sheets: list[BalanceSheetData] = Field(default_factory=list)
    quarterly_balance_sheets: list[BalanceSheetData] = Field(default_factory=list)
    annual_cash_flows: list[CashFlowData] = Field(default_factory=list)
    quarterly_cash_flows: list[CashFlowData] = Field(default_factory=list)
    quarterly_earnings: list[EarningsData] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


def _parse_value(value: Any) -> Optional[float]:
    """Parse AlphaVantage value to float, handling 'None' strings."""
    if value is None or value == "None" or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


class AlphaVantageClient:
    """
    AlphaVantage API client with rate limiting and caching.
    
    Usage:
        client = AlphaVantageClient()
        data = await client.get_fundamental_data("AAPL")
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        cache_ttl_hours: int = 24,
    ):
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY", "")
        if not self.api_key:
            logger.warning("alphavantage_no_api_key", 
                           msg="ALPHAVANTAGE_API_KEY not set. API calls will fail.")
        
        self.cache_dir = cache_dir or Path("data/alphavantage_cache")
        self.cache_ttl_hours = cache_ttl_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting state
        self._call_timestamps: list[float] = []
        self._lock = asyncio.Lock()
    
    async def _rate_limit(self) -> None:
        """Enforce rate limiting before API calls."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            # Remove timestamps older than the rate limit period
            self._call_timestamps = [
                ts for ts in self._call_timestamps 
                if now - ts < RATE_LIMIT_PERIOD
            ]
            
            if len(self._call_timestamps) >= RATE_LIMIT_CALLS:
                # Wait until oldest call expires
                wait_time = RATE_LIMIT_PERIOD - (now - self._call_timestamps[0])
                if wait_time > 0:
                    logger.info("alphavantage_rate_limit_wait", seconds=wait_time)
                    await asyncio.sleep(wait_time)
            
            self._call_timestamps.append(now)
    
    def _get_cache_path(self, ticker: str, endpoint: str) -> Path:
        """Get cache file path for a given ticker and endpoint."""
        return self.cache_dir / f"{ticker}_{endpoint}.json"
    
    def _read_cache(self, ticker: str, endpoint: str) -> Optional[dict]:
        """Read from cache if valid."""
        cache_path = self._get_cache_path(ticker, endpoint)
        if not cache_path.exists():
            return None
        
        try:
            data = json.loads(cache_path.read_text())
            cached_at = datetime.fromisoformat(data.get("_cached_at", ""))
            age_hours = (datetime.utcnow() - cached_at).total_seconds() / 3600
            
            if age_hours < self.cache_ttl_hours:
                logger.debug("alphavantage_cache_hit", ticker=ticker, endpoint=endpoint)
                return data
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "alphavantage_cache_corrupt",
                msg=f"Failed to read cache for {ticker} {endpoint} at {cache_path}: {e}",
                ticker=ticker,
                endpoint=endpoint,
                cache_path=str(cache_path),
                exception=str(e),
            )
        
        return None
    
    def _write_cache(self, ticker: str, endpoint: str, data: dict) -> None:
        """Write data to cache."""
        cache_path = self._get_cache_path(ticker, endpoint)
        data["_cached_at"] = datetime.utcnow().isoformat()
        cache_path.write_text(json.dumps(data, indent=2))
    
    async def _fetch(self, params: dict) -> dict:
        """Make an API request with rate limiting."""
        await self._rate_limit()
        
        params["apikey"] = self.api_key
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
        
        # Check for API errors
        if "Error Message" in data:
            function = params.get("function", "unknown")
            symbol = params.get("symbol", "unknown")
            raise ValueError(
                f"AlphaVantage API error for function '{function}' and symbol '{symbol}': {data['Error Message']}"
            )
        if "Note" in data:
            logger.warning("alphavantage_api_note", note=data["Note"])
        
        return data
    
    async def get_income_statement(self, ticker: str, use_cache: bool = True) -> dict:
        """Fetch income statement data."""
        if use_cache:
            cached = self._read_cache(ticker, "INCOME_STATEMENT")
            if cached:
                return cached
        
        data = await self._fetch({
            "function": "INCOME_STATEMENT",
            "symbol": ticker,
        })
        
        self._write_cache(ticker, "INCOME_STATEMENT", data)
        logger.info("alphavantage_fetched", ticker=ticker, endpoint="INCOME_STATEMENT")
        return data
    
    async def get_balance_sheet(self, ticker: str, use_cache: bool = True) -> dict:
        """Fetch balance sheet data."""
        if use_cache:
            cached = self._read_cache(ticker, "BALANCE_SHEET")
            if cached:
                return cached
        
        data = await self._fetch({
            "function": "BALANCE_SHEET",
            "symbol": ticker,
        })
        
        self._write_cache(ticker, "BALANCE_SHEET", data)
        logger.info("alphavantage_fetched", ticker=ticker, endpoint="BALANCE_SHEET")
        return data
    
    async def get_cash_flow(self, ticker: str, use_cache: bool = True) -> dict:
        """Fetch cash flow data."""
        if use_cache:
            cached = self._read_cache(ticker, "CASH_FLOW")
            if cached:
                return cached
        
        data = await self._fetch({
            "function": "CASH_FLOW",
            "symbol": ticker,
        })
        
        self._write_cache(ticker, "CASH_FLOW", data)
        logger.info("alphavantage_fetched", ticker=ticker, endpoint="CASH_FLOW")
        return data
    
    async def get_earnings(self, ticker: str, use_cache: bool = True) -> dict:
        """Fetch earnings data for beat/miss analysis."""
        if use_cache:
            cached = self._read_cache(ticker, "EARNINGS")
            if cached:
                return cached
        
        data = await self._fetch({
            "function": "EARNINGS",
            "symbol": ticker,
        })
        
        self._write_cache(ticker, "EARNINGS", data)
        logger.info("alphavantage_fetched", ticker=ticker, endpoint="EARNINGS")
        return data
    
    async def get_overview(self, ticker: str, use_cache: bool = True) -> dict:
        """Fetch company overview metadata."""
        if use_cache:
            cached = self._read_cache(ticker, "OVERVIEW")
            if cached:
                return cached
        
        data = await self._fetch({
            "function": "OVERVIEW",
            "symbol": ticker,
        })
        
        self._write_cache(ticker, "OVERVIEW", data)
        logger.info("alphavantage_fetched", ticker=ticker, endpoint="OVERVIEW")
        return data
    
    def _parse_income_statement(self, report: dict) -> IncomeStatementData:
        """Parse raw income statement JSON to structured data."""
        return IncomeStatementData(
            fiscal_date_ending=report.get("fiscalDateEnding", ""),
            reported_currency=report.get("reportedCurrency", "USD"),
            gross_profit=_parse_value(report.get("grossProfit")),
            total_revenue=_parse_value(report.get("totalRevenue")),
            cost_of_revenue=_parse_value(report.get("costOfRevenue")),
            operating_income=_parse_value(report.get("operatingIncome")),
            net_income=_parse_value(report.get("netIncome")),
            ebitda=_parse_value(report.get("ebitda")),
            interest_expense=_parse_value(report.get("interestExpense")),
            income_tax_expense=_parse_value(report.get("incomeTaxExpense")),
            depreciation_and_amortization=_parse_value(report.get("depreciationAndAmortization")),
            research_and_development=_parse_value(report.get("researchAndDevelopment")),
        )
    
    def _parse_balance_sheet(self, report: dict) -> BalanceSheetData:
        """Parse raw balance sheet JSON to structured data."""
        long_term = _parse_value(report.get("longTermDebt"))
        short_term = _parse_value(report.get("shortTermDebt"))
        total_debt = None
        if long_term is not None or short_term is not None:
            total_debt = (long_term or 0) + (short_term or 0)
        
        return BalanceSheetData(
            fiscal_date_ending=report.get("fiscalDateEnding", ""),
            reported_currency=report.get("reportedCurrency", "USD"),
            total_assets=_parse_value(report.get("totalAssets")),
            total_liabilities=_parse_value(report.get("totalLiabilities")),
            total_shareholder_equity=_parse_value(report.get("totalShareholderEquity")),
            cash_and_equivalents=_parse_value(report.get("cashAndCashEquivalentsAtCarryingValue")),
            short_term_investments=_parse_value(report.get("shortTermInvestments")),
            total_current_assets=_parse_value(report.get("totalCurrentAssets")),
            total_current_liabilities=_parse_value(report.get("totalCurrentLiabilities")),
            long_term_debt=long_term,
            short_term_debt=short_term,
            total_debt=total_debt,
        )
    
    def _parse_cash_flow(self, report: dict) -> CashFlowData:
        """Parse raw cash flow JSON to structured data."""
        operating = _parse_value(report.get("operatingCashflow"))
        capex = _parse_value(report.get("capitalExpenditures"))
        fcf = None
        if operating is not None and capex is not None:
            fcf = operating + capex  # CapEx is typically negative (cash outflow)
        
        return CashFlowData(
            fiscal_date_ending=report.get("fiscalDateEnding", ""),
            reported_currency=report.get("reportedCurrency", "USD"),
            operating_cashflow=operating,
            capital_expenditures=capex,
            free_cash_flow=fcf,
            dividend_payout=_parse_value(report.get("dividendPayout")),
        )
    
    def _parse_earnings(self, report: dict) -> EarningsData:
        """Parse raw earnings JSON to structured data."""
        return EarningsData(
            fiscal_date_ending=report.get("fiscalDateEnding", ""),
            reported_eps=_parse_value(report.get("reportedEPS")),
            estimated_eps=_parse_value(report.get("estimatedEPS")),
            surprise=_parse_value(report.get("surprise")),
            surprise_percentage=_parse_value(report.get("surprisePercentage")),
        )
    
    def _parse_overview(self, data: dict) -> CompanyOverview:
        """Parse raw overview JSON to structured data."""
        return CompanyOverview(
            symbol=data.get("Symbol", ""),
            name=data.get("Name", ""),
            description=data.get("Description", ""),
            sector=data.get("Sector", ""),
            industry=data.get("Industry", ""),
            market_cap=_parse_value(data.get("MarketCapitalization")),
            pe_ratio=_parse_value(data.get("PERatio")),
            dividend_yield=_parse_value(data.get("DividendYield")),
            eps=_parse_value(data.get("EPS")),
            fifty_two_week_high=_parse_value(data.get("52WeekHigh")),
            fifty_two_week_low=_parse_value(data.get("52WeekLow")),
        )
    
    async def get_fundamental_data(self, ticker: str, use_cache: bool = True) -> FundamentalData:
        """
        Fetch all fundamental data for a ticker.
        
        Aggregates income statement, balance sheet, cash flow, earnings, and overview
        into a single FundamentalData object.
        """
        logger.info("alphavantage_fetching_all", ticker=ticker)
        
        # Fetch all endpoints (rate limiting handled internally)
        income_raw = await self.get_income_statement(ticker, use_cache)
        balance_raw = await self.get_balance_sheet(ticker, use_cache)
        cash_raw = await self.get_cash_flow(ticker, use_cache)
        earnings_raw = await self.get_earnings(ticker, use_cache)
        overview_raw = await self.get_overview(ticker, use_cache)
        
        # Parse annual reports
        annual_income = [
            self._parse_income_statement(r) 
            for r in income_raw.get("annualReports", [])
        ]
        quarterly_income = [
            self._parse_income_statement(r) 
            for r in income_raw.get("quarterlyReports", [])
        ]
        
        annual_balance = [
            self._parse_balance_sheet(r) 
            for r in balance_raw.get("annualReports", [])
        ]
        quarterly_balance = [
            self._parse_balance_sheet(r) 
            for r in balance_raw.get("quarterlyReports", [])
        ]
        
        annual_cash = [
            self._parse_cash_flow(r) 
            for r in cash_raw.get("annualReports", [])
        ]
        quarterly_cash = [
            self._parse_cash_flow(r) 
            for r in cash_raw.get("quarterlyReports", [])
        ]
        
        quarterly_earnings = [
            self._parse_earnings(r) 
            for r in earnings_raw.get("quarterlyEarnings", [])
        ]
        
        overview = self._parse_overview(overview_raw)
        
        return FundamentalData(
            ticker=ticker,
            overview=overview,
            annual_income_statements=annual_income,
            quarterly_income_statements=quarterly_income,
            annual_balance_sheets=annual_balance,
            quarterly_balance_sheets=quarterly_balance,
            annual_cash_flows=annual_cash,
            quarterly_cash_flows=quarterly_cash,
            quarterly_earnings=quarterly_earnings,
        )
