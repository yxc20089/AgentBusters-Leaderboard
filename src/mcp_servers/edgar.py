"""
SEC EDGAR MCP Server client with metering and temporal locking.

This client wraps the sec-edgar-mcp server with additional features:
- Token cost tracking for efficiency scoring
- Noise injection (distractor documents)
- Temporal integrity enforcement
"""

import random
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


class FilingSection(BaseModel):
    """A section from an SEC filing."""
    ticker: str
    form_type: str
    section_name: str
    filing_date: str
    content: str
    token_count: int = 0


class XBRLFinancials(BaseModel):
    """Parsed XBRL financial data."""
    ticker: str
    fiscal_year: int
    fiscal_period: str
    statement_type: str  # BS, IS, CF
    data: dict[str, float]
    filing_date: str


class DistractorDocument(BaseModel):
    """A distractor document for noise injection."""
    doc_type: str  # "press_release", "earnings_call", "analyst_report"
    ticker: str
    date: str
    content: str
    is_distractor: bool = True


class FilingWithNoise(BaseModel):
    """Filing result that may contain distractor documents."""
    official: dict[str, Any]
    distractors: list[DistractorDocument] = Field(default_factory=list)
    test_type: Optional[str] = None


class MeteredEDGARClient(BaseMCPClient):
    """
    SEC EDGAR MCP client with metering, noise injection, and temporal locking.

    Features:
    - Tracks token cost for each request
    - Penalizes inefficient query patterns
    - Injects distractor documents for signal-vs-noise testing
    - Enforces temporal constraints (no future data access)
    """

    def __init__(
        self,
        config: MCPConfig,
        simulation_date: Optional[datetime] = None,
        noise_injection_ratio: float = 0.3,
        temporal_lock_enabled: bool = True
    ):
        super().__init__(config, simulation_date)
        self.noise_injection_ratio = noise_injection_ratio
        self.temporal_lock_enabled = temporal_lock_enabled
        self.temporal_violations: list[TemporalViolation] = []

    async def health_check(self) -> bool:
        """Check if the SEC EDGAR MCP server is healthy."""
        try:
            await self._request("GET", "/health", "edgar:health_check")
            return True
        except Exception:
            return False

    def _check_temporal_lock(self, filing_date: str) -> None:
        """Check if the requested filing date violates temporal constraints."""
        if not self.temporal_lock_enabled or not self.simulation_date:
            return

        requested_date = datetime.fromisoformat(filing_date.replace("Z", "+00:00"))
        if requested_date.tzinfo:
            requested_date = requested_date.replace(tzinfo=None)

        if requested_date > self.simulation_date:
            days_ahead = (requested_date - self.simulation_date).days
            severity = (
                ViolationSeverity.HIGH if days_ahead > 90
                else ViolationSeverity.MEDIUM if days_ahead > 30
                else ViolationSeverity.LOW
            )

            violation = TemporalViolation(
                ticker="",  # Will be filled by caller
                requested_date=filing_date,
                simulation_date=self.simulation_date.isoformat(),
                days_ahead=days_ahead,
                severity=severity,
                tool_name="sec-edgar-mcp",
                timestamp=datetime.utcnow()
            )
            self.temporal_violations.append(violation)

            raise TemporalViolationError(
                f"Look-ahead bias detected: Cannot access {filing_date} data "
                f"when simulation date is {self.simulation_date.isoformat()}. "
                f"Days ahead: {days_ahead}"
            )

    def _generate_distractor(self, ticker: str) -> DistractorDocument:
        """Generate a realistic distractor document."""
        distractor_types = ["press_release", "earnings_call", "analyst_report"]
        doc_type = random.choice(distractor_types)

        templates = {
            "press_release": f"""
                PRESS RELEASE - {ticker}

                {ticker} Announces Strategic Partnership

                FOR IMMEDIATE RELEASE

                [City, Date] - {ticker} today announced a new strategic partnership
                aimed at expanding market presence. The company expects this initiative
                to contribute positively to future growth.

                "We are excited about this opportunity," said the CEO.

                This press release contains forward-looking statements...
                [END]
            """,
            "earnings_call": f"""
                {ticker} Q3 Earnings Call Transcript (UNOFFICIAL)

                CEO: Thank you for joining us today. We're pleased to report...

                Analyst: Can you comment on the margin outlook?

                CFO: We're seeing positive trends in gross margin...

                [This is an unofficial transcript and may contain errors]
            """,
            "analyst_report": f"""
                ANALYST OPINION - {ticker}

                Rating: HOLD
                Price Target: $XXX

                Key Points:
                - Management execution remains solid
                - Competitive pressures in key markets
                - Valuation appears fair at current levels

                [This is not the official 10-K filing]
            """
        }

        return DistractorDocument(
            doc_type=doc_type,
            ticker=ticker,
            date=datetime.utcnow().isoformat(),
            content=templates[doc_type].strip(),
            is_distractor=True
        )

    async def get_filing(
        self,
        ticker: str,
        form_type: str,
        fiscal_year: Optional[int] = None,
        filing_date: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Fetch an SEC filing.

        Args:
            ticker: Company ticker symbol
            form_type: Filing type (10-K, 10-Q, 8-K)
            fiscal_year: Target fiscal year
            filing_date: Specific filing date (YYYY-MM-DD)

        Returns:
            Filing data
        """
        params = {
            "ticker": ticker,
            "form_type": form_type,
        }
        if fiscal_year:
            params["fiscal_year"] = fiscal_year
        if filing_date:
            self._check_temporal_lock(filing_date)
            params["date"] = filing_date

        return await self._request(
            "GET",
            f"/filings/{ticker}/{form_type}",
            f"edgar:get_filing:{form_type}",
            params=params
        )

    async def get_filing_with_noise(
        self,
        ticker: str,
        form_type: str,
        fiscal_year: Optional[int] = None
    ) -> FilingWithNoise:
        """
        Fetch an SEC filing with potential noise injection.

        This method randomly injects distractor documents alongside
        official filings to test if the agent can filter signal from noise.

        Args:
            ticker: Company ticker symbol
            form_type: Filing type
            fiscal_year: Target fiscal year

        Returns:
            FilingWithNoise with official filing and potential distractors
        """
        official_filing = await self.get_filing(ticker, form_type, fiscal_year)

        distractors = []
        test_type = None

        if random.random() < self.noise_injection_ratio:
            # Inject 1-3 distractor documents
            num_distractors = random.randint(1, 3)
            for _ in range(num_distractors):
                distractors.append(self._generate_distractor(ticker))
            test_type = "agent must identify which document is the official filing"

            logger.info(
                "noise_injection",
                ticker=ticker,
                form_type=form_type,
                num_distractors=num_distractors
            )

        return FilingWithNoise(
            official=official_filing,
            distractors=distractors,
            test_type=test_type
        )

    async def get_filing_section(
        self,
        ticker: str,
        form_type: str,
        section_name: str,
        fiscal_year: Optional[int] = None
    ) -> FilingSection:
        """
        Fetch a specific section from an SEC filing.

        Args:
            ticker: Company ticker symbol
            form_type: Filing type
            section_name: Section name (e.g., "Item 1A", "Item 7", "MD&A")
            fiscal_year: Target fiscal year

        Returns:
            FilingSection with the requested section content
        """
        params = {"ticker": ticker}
        if fiscal_year:
            params["fiscal_year"] = fiscal_year

        result = await self._request(
            "GET",
            f"/filings/{ticker}/{form_type}/section/{section_name}",
            f"edgar:get_section:{section_name}",
            params=params
        )

        return FilingSection(
            ticker=ticker,
            form_type=form_type,
            section_name=section_name,
            filing_date=result.get("filing_date", ""),
            content=result.get("content", ""),
            token_count=self._estimate_tokens(result.get("content", ""))
        )

    async def parse_xbrl_financials(
        self,
        ticker: str,
        statement_type: str = "IS",
        fiscal_year: Optional[int] = None,
        filing_date: Optional[str] = None
    ) -> XBRLFinancials:
        """
        Parse XBRL financial data from SEC filings.

        Args:
            ticker: Company ticker symbol
            statement_type: Statement type (IS=Income Statement, BS=Balance Sheet, CF=Cash Flow)
            fiscal_year: Target fiscal year
            filing_date: Specific filing date

        Returns:
            XBRLFinancials with parsed financial data
        """
        if filing_date:
            self._check_temporal_lock(filing_date)

        params = {
            "ticker": ticker,
            "statement_type": statement_type,
        }
        if fiscal_year:
            params["fiscal_year"] = fiscal_year
        if filing_date:
            params["filing_date"] = filing_date

        result = await self._request(
            "GET",
            f"/financials/{ticker}",
            f"edgar:parse_xbrl:{statement_type}",
            params=params
        )

        return XBRLFinancials(
            ticker=ticker,
            fiscal_year=result.get("fiscal_year", fiscal_year or 0),
            fiscal_period=result.get("fiscal_period", "FY"),
            statement_type=statement_type,
            data=result.get("data", {}),
            filing_date=result.get("filing_date", "")
        )

    async def search_filings(
        self,
        ticker: str,
        form_type: str,
        keywords: list[str],
        date_start: Optional[str] = None,
        date_end: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Search filings by keyword.

        Args:
            ticker: Company ticker symbol
            form_type: Filing type
            keywords: Search keywords
            date_start: Start date for search range
            date_end: End date for search range

        Returns:
            List of matching filing excerpts
        """
        if date_end:
            self._check_temporal_lock(date_end)

        json_body = {
            "ticker": ticker,
            "form_type": form_type,
            "keywords": keywords,
        }
        if date_start and date_end:
            json_body["date_range"] = {"start": date_start, "end": date_end}

        result = await self._request(
            "POST",
            "/search",
            "edgar:search",
            json_body=json_body
        )

        return result.get("results", [])

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
