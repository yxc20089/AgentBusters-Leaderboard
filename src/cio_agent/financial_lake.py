"""
Financial Lake: Local data storage for harvested financial data.

Provides ETL pipeline to populate a local cache of AlphaVantage data
for a universe of tickers, enabling rapid synthetic question generation
without repeated API calls.
"""

import json
from pathlib import Path
from typing import Optional

import structlog

from cio_agent.alphavantage import AlphaVantageClient, FundamentalData

logger = structlog.get_logger()


# Stratified S&P 500 universe by sector (50 tickers)
# This ensures sector diversity and avoids over-indexing on one industry
TICKER_UNIVERSE = {
    "technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "CRM", "ORCL", "ADBE", "INTC"],
    "financials": ["JPM", "BAC", "WFC", "GS", "MS"],
    "healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABBV"],
    "consumer_discretionary": ["AMZN", "TSLA", "HD", "NKE", "MCD"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "industrials": ["CAT", "BA", "HON", "UPS", "RTX"],
    "consumer_staples": ["PG", "KO", "PEP", "WMT", "COST"],
    "utilities": ["NEE", "DUK", "SO", "D", "AEP"],
    "real_estate": ["AMT", "PLD", "CCI", "EQIX", "SPG"],
    "materials": ["LIN", "APD", "SHW", "ECL", "NEM"],
}

# Flatten to list of all tickers
ALL_TICKERS = [ticker for tickers in TICKER_UNIVERSE.values() for ticker in tickers]

# Reverse mapping: ticker -> sector
TICKER_TO_SECTOR = {
    ticker: sector
    for sector, tickers in TICKER_UNIVERSE.items()
    for ticker in tickers
}


class DataQualityReport(dict):
    """Report on data completeness for a ticker."""
    pass


class FinancialLake:
    """
    Local storage layer for AlphaVantage financial data.
    
    Stores pre-fetched fundamental data to enable rapid synthetic
    question generation without hitting API rate limits.
    
    Usage:
        lake = FinancialLake()
        await lake.harvest(["AAPL", "MSFT"])
        data = lake.get("AAPL")
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        alphavantage_client: Optional[AlphaVantageClient] = None,
    ):
        self.data_dir = data_dir or Path("data/financial_lake")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.client = alphavantage_client or AlphaVantageClient()
        self._cache: dict[str, FundamentalData] = {}
    
    def _get_path(self, ticker: str) -> Path:
        """Get storage path for a ticker's data."""
        return self.data_dir / f"{ticker.upper()}.json"
    
    def exists(self, ticker: str) -> bool:
        """Check if data exists for a ticker."""
        return self._get_path(ticker).exists()
    
    def get(self, ticker: str) -> Optional[FundamentalData]:
        """
        Retrieve fundamental data for a ticker from local storage.
        
        Returns None if not found in lake.
        """
        ticker = ticker.upper()
        
        # Check memory cache first
        if ticker in self._cache:
            return self._cache[ticker]
        
        path = self._get_path(ticker)
        if not path.exists():
            return None
        
        try:
            raw = json.loads(path.read_text())
            data = FundamentalData(**raw)
            self._cache[ticker] = data
            return data
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("financial_lake_load_error", ticker=ticker, error=str(e))
            return None
    
    def save(self, data: FundamentalData) -> None:
        """Save fundamental data to local storage."""
        path = self._get_path(data.ticker)
        path.write_text(data.model_dump_json(indent=2))
        self._cache[data.ticker] = data
        logger.info("financial_lake_saved", ticker=data.ticker)
    
    async def harvest_ticker(self, ticker: str, force: bool = False) -> Optional[FundamentalData]:
        """
        Harvest data for a single ticker from AlphaVantage.
        
        Args:
            ticker: Stock ticker symbol
            force: If True, re-fetch even if data exists
            
        Returns:
            FundamentalData or None on error
        """
        ticker = ticker.upper()
        
        if not force and self.exists(ticker):
            logger.info("financial_lake_skip_existing", ticker=ticker)
            return self.get(ticker)
        
        try:
            data = await self.client.get_fundamental_data(ticker)
            self.save(data)
            return data
        except Exception as e:
            logger.error("financial_lake_harvest_error", ticker=ticker, error=str(e))
            return None
    
    async def harvest(
        self,
        tickers: Optional[list[str]] = None,
        force: bool = False,
    ) -> dict[str, bool]:
        """
        Harvest data for multiple tickers.
        
        Args:
            tickers: List of tickers to harvest (defaults to full universe)
            force: If True, re-fetch even if data exists
            
        Returns:
            Dict mapping ticker -> success status
        """
        tickers = tickers or ALL_TICKERS
        results = {}
        
        logger.info("financial_lake_harvest_start", count=len(tickers))
        
        for i, ticker in enumerate(tickers):
            logger.info("financial_lake_harvesting", 
                       ticker=ticker, 
                       progress=f"{i+1}/{len(tickers)}")
            
            data = await self.harvest_ticker(ticker, force=force)
            results[ticker] = data is not None
        
        success_count = sum(results.values())
        logger.info("financial_lake_harvest_complete", 
                   success=success_count, 
                   total=len(tickers))
        
        return results
    
    def validate_data(self, ticker: str, years_required: int = 3) -> DataQualityReport:
        """
        Validate data completeness for a ticker.
        
        Checks:
        - At least N years of annual data
        - No missing quarters in recent history
        - All key financial metrics present
        
        Returns:
            DataQualityReport with validation results
        """
        data = self.get(ticker)
        report = DataQualityReport()
        report["ticker"] = ticker
        report["valid"] = True
        report["issues"] = []
        
        if not data:
            report["valid"] = False
            report["issues"].append("No data found")
            return report
        
        # Check annual data
        annual_count = len(data.annual_income_statements)
        if annual_count < years_required:
            report["issues"].append(
                f"Only {annual_count} years of annual data (need {years_required})"
            )
        
        # Check quarterly data
        quarterly_count = len(data.quarterly_income_statements)
        if quarterly_count < years_required * 4:
            report["issues"].append(
                f"Only {quarterly_count} quarters of data"
            )
        
        # Check earnings data (for beat/miss questions)
        earnings_count = len(data.quarterly_earnings)
        if earnings_count < 4:
            report["issues"].append(f"Only {earnings_count} quarters of earnings data")
        
        # Check overview exists
        if not data.overview or not data.overview.description:
            report["issues"].append("Missing company description")
        
        # Check key metrics in most recent annual data
        if data.annual_income_statements:
            latest = data.annual_income_statements[0]
            if latest.total_revenue is None:
                report["issues"].append("Missing revenue data")
            if latest.net_income is None:
                report["issues"].append("Missing net income data")
        
        report["valid"] = len(report["issues"]) == 0
        report["annual_years"] = annual_count
        report["quarterly_count"] = quarterly_count
        report["has_earnings"] = earnings_count > 0
        
        return report
    
    def get_available_tickers(self) -> list[str]:
        """Get list of tickers with data in the lake."""
        return [
            p.stem for p in self.data_dir.glob("*.json")
        ]
    
    def get_tickers_by_sector(self, sector: str) -> list[str]:
        """Get available tickers for a specific sector."""
        sector_tickers = TICKER_UNIVERSE.get(sector.lower(), [])
        return [t for t in sector_tickers if self.exists(t)]
    
    def get_sector(self, ticker: str) -> Optional[str]:
        """Get sector for a ticker."""
        return TICKER_TO_SECTOR.get(ticker.upper())
    
    def get_peers(self, ticker: str) -> list[str]:
        """Get peer companies (same sector) for a ticker."""
        sector = self.get_sector(ticker)
        if not sector:
            return []
        
        sector_tickers = TICKER_UNIVERSE.get(sector, [])
        return [t for t in sector_tickers if t != ticker.upper() and self.exists(t)]
    
    def get_quality_summary(self) -> dict:
        """Get overall data quality summary for the lake."""
        tickers = self.get_available_tickers()
        reports = [self.validate_data(t) for t in tickers]
        
        valid_count = sum(1 for r in reports if r["valid"])
        
        return {
            "total_tickers": len(tickers),
            "valid_tickers": valid_count,
            "invalid_tickers": len(tickers) - valid_count,
            "sectors": {
                sector: len(self.get_tickers_by_sector(sector))
                for sector in TICKER_UNIVERSE.keys()
            },
            "reports": reports,
        }
