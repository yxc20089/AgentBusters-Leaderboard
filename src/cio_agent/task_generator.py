"""
Dynamic Task Generator for FAB++ Evaluation.

Transforms static FAB questions into dynamic variants by:
- Substituting tickers within the same sector
- Adjusting fiscal years based on simulation date
- Fetching live ground truth from MCP tools
"""

import random
from datetime import datetime
from typing import Any, Optional

import structlog
from pydantic import BaseModel, Field

from cio_agent.models import (
    Task,
    TaskCategory,
    TaskDifficulty,
    TaskRubric,
    GroundTruth,
    FinancialData,
    FABQuestionTemplate,
)
from cio_agent.data_providers.base import DatasetProvider, DatasetExample

logger = structlog.get_logger()


# Sector-based ticker groupings for realistic substitution
SECTOR_TICKERS = {
    "technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "CRM", "ORCL", "ADBE"],
    "financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V"],
    "healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT", "BMY", "AMGN"],
    "consumer_discretionary": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "LOW", "TJX", "CMG", "BKNG"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "KMI"],
    "industrials": ["CAT", "BA", "HON", "UPS", "RTX", "GE", "MMM", "LMT", "UNP", "DE"],
    "consumer_staples": ["PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "CL", "MDLZ", "KHC"],
    "utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "WEC"],
    "real_estate": ["AMT", "PLD", "CCI", "EQIX", "SPG", "PSA", "O", "WELL", "AVB", "EQR"],
    "materials": ["LIN", "APD", "SHW", "ECL", "NEM", "FCX", "NUE", "VMC", "MLM", "DD"],
    "communication": ["GOOG", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "CHTR", "EA", "ATVI"],
}

# Reverse mapping: ticker to sector
TICKER_TO_SECTOR = {
    ticker: sector
    for sector, tickers in SECTOR_TICKERS.items()
    for ticker in tickers
}


class FABDataset(BaseModel):
    """Container for FAB question templates."""
    questions: list[FABQuestionTemplate] = Field(default_factory=list)

    @classmethod
    def load_sample_questions(cls) -> "FABDataset":
        """Load sample FAB questions for testing."""
        questions = [
            # Category 1: Quantitative Retrieval
            FABQuestionTemplate(
                template_id="FAB_001",
                category=TaskCategory.QUANTITATIVE_RETRIEVAL,
                template="What was {ticker}'s total revenue in FY {year}?",
                difficulty=TaskDifficulty.EASY,
                metric="revenue",
                rubric=TaskRubric(
                    criteria=["Extract correct revenue figure", "Use correct fiscal year"],
                    mandatory_elements=["revenue value", "fiscal year reference"],
                ),
            ),
            FABQuestionTemplate(
                template_id="FAB_002",
                category=TaskCategory.QUANTITATIVE_RETRIEVAL,
                template="What was {ticker}'s net income in FY {year}?",
                difficulty=TaskDifficulty.EASY,
                metric="net_income",
                rubric=TaskRubric(
                    criteria=["Extract correct net income figure"],
                    mandatory_elements=["net income value"],
                ),
            ),
            FABQuestionTemplate(
                template_id="FAB_003",
                category=TaskCategory.QUANTITATIVE_RETRIEVAL,
                template="What was {ticker}'s operating cash flow in FY {year}?",
                difficulty=TaskDifficulty.MEDIUM,
                metric="operating_cash_flow",
                rubric=TaskRubric(
                    criteria=["Extract correct operating cash flow"],
                    mandatory_elements=["cash flow value"],
                ),
            ),

            # Category 2: Qualitative Retrieval
            FABQuestionTemplate(
                template_id="FAB_010",
                category=TaskCategory.QUALITATIVE_RETRIEVAL,
                template="What are the primary risk factors disclosed in {ticker}'s FY {year} 10-K?",
                difficulty=TaskDifficulty.MEDIUM,
                metric="risk_factors",
                rubric=TaskRubric(
                    criteria=["Identify major risk factors", "Reference correct filing"],
                    mandatory_elements=["at least 3 risk factors"],
                ),
            ),
            FABQuestionTemplate(
                template_id="FAB_011",
                category=TaskCategory.QUALITATIVE_RETRIEVAL,
                template="Describe {ticker}'s main business segments as reported in FY {year}.",
                difficulty=TaskDifficulty.EASY,
                metric="business_segments",
                rubric=TaskRubric(
                    criteria=["List major business segments", "Describe each segment"],
                ),
            ),

            # Category 3: Numerical Reasoning
            FABQuestionTemplate(
                template_id="FAB_020",
                category=TaskCategory.NUMERICAL_REASONING,
                template="Calculate {ticker}'s gross margin percentage for FY {year}.",
                difficulty=TaskDifficulty.MEDIUM,
                metric="gross_margin",
                rubric=TaskRubric(
                    criteria=["Correct formula applied", "Accurate calculation"],
                    mandatory_elements=["gross margin percentage"],
                ),
                requires_code_execution=True,
            ),
            FABQuestionTemplate(
                template_id="FAB_021",
                category=TaskCategory.NUMERICAL_REASONING,
                template="Calculate the year-over-year revenue growth rate for {ticker} from FY {prev_year} to FY {year}.",
                difficulty=TaskDifficulty.MEDIUM,
                metric="revenue_growth",
                rubric=TaskRubric(
                    criteria=["Fetch both years' revenue", "Apply correct growth formula"],
                    mandatory_elements=["growth rate percentage"],
                ),
                requires_code_execution=True,
            ),
            FABQuestionTemplate(
                template_id="FAB_022",
                category=TaskCategory.NUMERICAL_REASONING,
                template="What was {ticker}'s return on equity (ROE) for FY {year}?",
                difficulty=TaskDifficulty.HARD,
                metric="roe",
                rubric=TaskRubric(
                    criteria=["Correct ROE formula", "Accurate calculation"],
                ),
                requires_code_execution=True,
            ),

            # Category 4: Complex Retrieval
            FABQuestionTemplate(
                template_id="FAB_030",
                category=TaskCategory.COMPLEX_RETRIEVAL,
                template="Compare the operating margins of {ticker} and {comp_ticker} in FY {year}.",
                difficulty=TaskDifficulty.HARD,
                metric="operating_margin_comparison",
                rubric=TaskRubric(
                    criteria=["Retrieve both companies' data", "Calculate operating margins", "Compare and explain"],
                ),
                requires_code_execution=True,
            ),

            # Category 5: Adjustments
            FABQuestionTemplate(
                template_id="FAB_040",
                category=TaskCategory.ADJUSTMENTS,
                template="Calculate {ticker}'s adjusted EBITDA for FY {year}, excluding one-time charges.",
                difficulty=TaskDifficulty.EXPERT,
                metric="adjusted_ebitda",
                rubric=TaskRubric(
                    criteria=["Identify one-time charges", "Apply correct adjustments"],
                    penalty_conditions=["Missing major adjustment items"],
                ),
                requires_code_execution=True,
            ),

            # Category 6: Beat or Miss
            FABQuestionTemplate(
                template_id="FAB_050",
                category=TaskCategory.BEAT_OR_MISS,
                template="Did {ticker} beat or miss analyst EPS estimates in Q{quarter} {year}?",
                difficulty=TaskDifficulty.MEDIUM,
                metric="eps_beat_miss",
                rubric=TaskRubric(
                    criteria=["Retrieve actual EPS", "Retrieve consensus estimate", "Determine beat/miss"],
                ),
            ),

            # Category 7: Trends
            FABQuestionTemplate(
                template_id="FAB_060",
                category=TaskCategory.TRENDS,
                template="What is the 3-year trend in {ticker}'s R&D spending as a percentage of revenue from FY {start_year} to FY {year}?",
                difficulty=TaskDifficulty.HARD,
                metric="rd_trend",
                rubric=TaskRubric(
                    criteria=["Retrieve multiple years of data", "Calculate percentages", "Identify trend direction"],
                ),
                requires_code_execution=True,
            ),

            # Category 8: Financial Modeling
            FABQuestionTemplate(
                template_id="FAB_070",
                category=TaskCategory.FINANCIAL_MODELING,
                template="Build a simple DCF model to estimate {ticker}'s intrinsic value. Use FY {year} financials as the base.",
                difficulty=TaskDifficulty.EXPERT,
                metric="dcf_valuation",
                rubric=TaskRubric(
                    criteria=["Project future cash flows", "Apply appropriate discount rate", "Calculate terminal value"],
                    mandatory_elements=["DCF model code", "intrinsic value estimate"],
                ),
                requires_code_execution=True,
            ),

            # Category 9: Market Analysis
            FABQuestionTemplate(
                template_id="FAB_080",
                category=TaskCategory.MARKET_ANALYSIS,
                template="Given current macroeconomic conditions, provide a Buy/Sell/Hold recommendation for {ticker}. Base your analysis on FY {year} financials.",
                difficulty=TaskDifficulty.EXPERT,
                metric="recommendation",
                rubric=TaskRubric(
                    criteria=["Analyze macro factors", "Assess company fundamentals", "Provide clear recommendation with rationale"],
                ),
            ),

            # =================================================================
            # Options Trading Categories (Alpha Challenge)
            # =================================================================

            # Category 10: Options Pricing
            FABQuestionTemplate(
                template_id="OPT_001",
                category=TaskCategory.OPTIONS_PRICING,
                template="Calculate the theoretical price of a {ticker} {days_to_expiry}-day call option with strike at {strike_pct}% of current price. Assume 25% implied volatility and 5% risk-free rate.",
                difficulty=TaskDifficulty.MEDIUM,
                metric="option_price",
                rubric=TaskRubric(
                    criteria=["Use Black-Scholes model correctly", "Calculate accurate price", "Show all Greeks"],
                    mandatory_elements=["option price", "delta", "gamma", "theta", "vega"],
                ),
                requires_code_execution=True,
            ),
            FABQuestionTemplate(
                template_id="OPT_002",
                category=TaskCategory.OPTIONS_PRICING,
                template="Price a {ticker} put option expiring in 45 days with a strike 5% below current price. Use historical volatility from the last 30 days.",
                difficulty=TaskDifficulty.MEDIUM,
                metric="option_price",
                rubric=TaskRubric(
                    criteria=["Calculate historical volatility", "Apply Black-Scholes", "Accurate pricing"],
                    mandatory_elements=["option price", "implied volatility used"],
                ),
                requires_code_execution=True,
            ),

            # Category 11: Greeks Analysis
            FABQuestionTemplate(
                template_id="OPT_010",
                category=TaskCategory.GREEKS_ANALYSIS,
                template="Your portfolio has -500 delta exposure from {ticker} options. Design a hedge using options or stock to neutralize the delta.",
                difficulty=TaskDifficulty.MEDIUM,
                metric="delta_hedge",
                rubric=TaskRubric(
                    criteria=["Understand delta exposure", "Propose valid hedge", "Calculate hedge size"],
                    mandatory_elements=["hedge instrument", "quantity", "resulting portfolio delta"],
                ),
                requires_code_execution=True,
            ),
            FABQuestionTemplate(
                template_id="OPT_011",
                category=TaskCategory.GREEKS_ANALYSIS,
                template="Analyze the gamma risk of holding 10 at-the-money {ticker} call options expiring in 5 days. What price move would cause the largest P&L impact?",
                difficulty=TaskDifficulty.HARD,
                metric="gamma_analysis",
                rubric=TaskRubric(
                    criteria=["Calculate gamma correctly", "Analyze near-expiry behavior", "Quantify P&L scenarios"],
                ),
                requires_code_execution=True,
            ),

            # Category 12: Strategy Construction
            FABQuestionTemplate(
                template_id="OPT_020",
                category=TaskCategory.STRATEGY_CONSTRUCTION,
                template="Construct an iron condor on {ticker} with 30-day expiration. Select strikes to achieve at least 70% probability of profit.",
                difficulty=TaskDifficulty.HARD,
                metric="strategy_construction",
                rubric=TaskRubric(
                    criteria=["Select appropriate strikes", "Calculate max profit/loss", "Determine probability of profit", "Execute trades correctly"],
                    mandatory_elements=["4 option legs", "max profit", "max loss", "breakeven points"],
                ),
                requires_code_execution=True,
            ),
            FABQuestionTemplate(
                template_id="OPT_021",
                category=TaskCategory.STRATEGY_CONSTRUCTION,
                template="Build a bull call spread on {ticker} targeting 15% upside with limited risk. The position should cost no more than $5,000.",
                difficulty=TaskDifficulty.MEDIUM,
                metric="strategy_construction",
                rubric=TaskRubric(
                    criteria=["Select appropriate strikes", "Stay within budget", "Calculate risk/reward"],
                    mandatory_elements=["long call strike", "short call strike", "total cost", "max profit"],
                ),
                requires_code_execution=True,
            ),
            FABQuestionTemplate(
                template_id="OPT_022",
                category=TaskCategory.STRATEGY_CONSTRUCTION,
                template="Create a protective put strategy for a 100-share position in {ticker}. Choose an expiration and strike that balances cost with downside protection.",
                difficulty=TaskDifficulty.MEDIUM,
                metric="strategy_construction",
                rubric=TaskRubric(
                    criteria=["Explain strike selection rationale", "Calculate protection level", "Analyze cost of insurance"],
                ),
                requires_code_execution=True,
            ),

            # Category 13: Volatility Trading
            FABQuestionTemplate(
                template_id="OPT_030",
                category=TaskCategory.VOLATILITY_TRADING,
                template="Analyze {ticker}'s current implied volatility. Is IV overpriced or underpriced relative to historical volatility? Design a trade to exploit any mispricing.",
                difficulty=TaskDifficulty.HARD,
                metric="volatility_analysis",
                rubric=TaskRubric(
                    criteria=["Calculate IV rank/percentile", "Compare to historical vol", "Design appropriate strategy"],
                    mandatory_elements=["IV analysis", "HV comparison", "trade recommendation"],
                ),
                requires_code_execution=True,
            ),
            FABQuestionTemplate(
                template_id="OPT_031",
                category=TaskCategory.VOLATILITY_TRADING,
                template="{ticker} earnings announcement is in 5 days. Current IV is elevated at 60%. Design a volatility crush strategy.",
                difficulty=TaskDifficulty.EXPERT,
                metric="volatility_strategy",
                rubric=TaskRubric(
                    criteria=["Understand IV crush dynamics", "Select appropriate strategy", "Manage directional risk"],
                    mandatory_elements=["strategy type", "position sizing", "expected IV after earnings"],
                ),
                requires_code_execution=True,
            ),

            # Category 14: P&L Attribution
            FABQuestionTemplate(
                template_id="OPT_040",
                category=TaskCategory.PNL_ATTRIBUTION,
                template="Your {ticker} straddle position made $2,500 today. The stock moved +2%, IV dropped 3 points, and 1 day passed. Decompose the P&L by Greek contribution.",
                difficulty=TaskDifficulty.HARD,
                metric="pnl_attribution",
                rubric=TaskRubric(
                    criteria=["Calculate delta P&L", "Calculate gamma P&L", "Calculate theta P&L", "Calculate vega P&L"],
                    mandatory_elements=["delta contribution", "gamma contribution", "theta contribution", "vega contribution"],
                ),
                requires_code_execution=True,
            ),

            # Category 15: Risk Management
            FABQuestionTemplate(
                template_id="OPT_050",
                category=TaskCategory.RISK_MANAGEMENT,
                template="Size a {ticker} strangle position to stay within a $10,000 VaR limit at 95% confidence over 1 day.",
                difficulty=TaskDifficulty.EXPERT,
                metric="position_sizing",
                rubric=TaskRubric(
                    criteria=["Calculate position VaR", "Size appropriately", "Consider tail risk"],
                    mandatory_elements=["position size", "VaR calculation", "risk parameters"],
                ),
                requires_code_execution=True,
            ),
            FABQuestionTemplate(
                template_id="OPT_051",
                category=TaskCategory.RISK_MANAGEMENT,
                template="Stress test your {ticker} options portfolio against: 1) 20% market crash, 2) 50% IV spike, 3) Flash crash scenario. Report expected P&L for each.",
                difficulty=TaskDifficulty.EXPERT,
                metric="stress_test",
                rubric=TaskRubric(
                    criteria=["Model each scenario correctly", "Calculate P&L impact", "Identify worst case"],
                    mandatory_elements=["P&L for each scenario", "worst case scenario", "hedging recommendations"],
                ),
                requires_code_execution=True,
            ),

            # Category 16: Copy Trading (Famous Trader Strategies)
            FABQuestionTemplate(
                template_id="OPT_060",
                category=TaskCategory.COPY_TRADING,
                template="Execute a Warren Buffett-style covered call strategy on {ticker}. Select strikes that generate income while maintaining upside participation.",
                difficulty=TaskDifficulty.HARD,
                metric="copy_trading",
                rubric=TaskRubric(
                    criteria=["Understand Buffett's value approach", "Select appropriate strike", "Balance income vs upside"],
                    mandatory_elements=["stock position", "call strike selection", "premium collected", "max profit cap"],
                ),
                requires_code_execution=True,
            ),
            FABQuestionTemplate(
                template_id="OPT_061",
                category=TaskCategory.COPY_TRADING,
                template="Replicate a Keith Gill (DFV) style deep value LEAPS position on {ticker}. Use fundamental analysis to justify the position.",
                difficulty=TaskDifficulty.EXPERT,
                metric="copy_trading",
                rubric=TaskRubric(
                    criteria=["Identify deep value characteristics", "Select far-dated options", "Build conviction thesis"],
                    mandatory_elements=["fundamental analysis", "LEAPS selection", "position sizing", "thesis"],
                ),
                requires_code_execution=True,
            ),

            # Category 17: Race to 10M
            FABQuestionTemplate(
                template_id="OPT_070",
                category=TaskCategory.RACE_TO_10M,
                template="Starting with $100,000, design and execute an options trading strategy to maximize portfolio growth over 30 days. Target aggressive but not reckless growth.",
                difficulty=TaskDifficulty.EXPERT,
                metric="portfolio_growth",
                rubric=TaskRubric(
                    criteria=["Design coherent strategy", "Execute trades", "Manage risk", "Track P&L"],
                    mandatory_elements=["strategy description", "trades executed", "daily P&L", "final portfolio value"],
                ),
                requires_code_execution=True,
            ),
            FABQuestionTemplate(
                template_id="OPT_071",
                category=TaskCategory.RACE_TO_10M,
                template="You have $50,000 and 60 days. Using options on {ticker} and two other stocks of your choice, try to double your money while limiting max drawdown to 30%.",
                difficulty=TaskDifficulty.EXPERT,
                metric="portfolio_growth",
                rubric=TaskRubric(
                    criteria=["Stock selection rationale", "Strategy selection", "Risk management", "Execution quality"],
                    mandatory_elements=["portfolio composition", "P&L history", "max drawdown", "final return"],
                ),
                requires_code_execution=True,
            ),

            # Category 18: Strategy Defense (Adversarial Debate)
            FABQuestionTemplate(
                template_id="OPT_080",
                category=TaskCategory.STRATEGY_DEFENSE,
                template="You've built a short volatility position on {ticker} collecting $5,000 in premium. Defend this strategy against the counter-argument that a 10% gap down would cause devastating losses.",
                difficulty=TaskDifficulty.EXPERT,
                metric="strategy_defense",
                rubric=TaskRubric(
                    criteria=["Acknowledge risk honestly", "Present mitigating factors", "Describe contingency plans"],
                    mandatory_elements=["risk acknowledgment", "hedge plan", "probability analysis"],
                ),
            ),
            FABQuestionTemplate(
                template_id="OPT_081",
                category=TaskCategory.STRATEGY_DEFENSE,
                template="Your iron condor on {ticker} is being challenged due to upcoming earnings. The challenger argues that earnings volatility makes your position extremely risky. Defend or adjust your strategy.",
                difficulty=TaskDifficulty.EXPERT,
                metric="strategy_defense",
                rubric=TaskRubric(
                    criteria=["Address earnings risk", "Present defense or adjustment", "Show quantitative analysis"],
                ),
            ),
        ]

        return cls(questions=questions)


class DynamicTaskGenerator:
    """
    Generates dynamic FAB task variants for evaluation.

    Takes FAB questions as templates and dynamically generates variants by:
    1. Substituting tickers (within same sector for fairness)
    2. Adjusting fiscal years based on simulation date
    3. Fetching live ground truth via MCP tools
    """

    def __init__(
        self,
        fab_dataset: Optional[FABDataset] = None,
        edgar_client: Optional[Any] = None,  # MeteredEDGARClient
        yfinance_client: Optional[Any] = None,  # TimeMachineYFinanceClient
        dataset_provider: Optional[DatasetProvider] = None,
    ):
        self.dataset_provider = dataset_provider

        if dataset_provider:
            examples: list[DatasetExample] = dataset_provider.load()
            self.dataset_examples_by_id = {ex.example_id: ex for ex in examples}
            self.fab_dataset = FABDataset(questions=dataset_provider.to_templates())
        else:
            self.dataset_examples_by_id = {}
            self.fab_dataset = fab_dataset or FABDataset.load_sample_questions()
        self.edgar_client = edgar_client
        self.yfinance_client = yfinance_client

    def get_question_by_id(self, template_id: str) -> Optional[FABQuestionTemplate]:
        """Get a question template by ID."""
        for q in self.fab_dataset.questions:
            if q.template_id == template_id:
                return q
        return None

    def get_questions_by_category(self, category: TaskCategory) -> list[FABQuestionTemplate]:
        """Get all questions in a specific category."""
        return [q for q in self.fab_dataset.questions if q.category == category]

    def sample_similar_company(self, original_ticker: str) -> str:
        """
        Sample a different company from the same sector.

        Args:
            original_ticker: Original ticker to find similar company for

        Returns:
            A different ticker from the same sector
        """
        sector = TICKER_TO_SECTOR.get(original_ticker.upper())

        if not sector:
            # Unknown sector, pick random from technology
            sector = "technology"

        candidates = [t for t in SECTOR_TICKERS[sector] if t != original_ticker.upper()]

        if not candidates:
            return original_ticker

        return random.choice(candidates)

    def get_available_fiscal_years(
        self,
        ticker: str,
        simulation_date: datetime
    ) -> list[int]:
        """
        Get available fiscal years for a ticker given the simulation date.

        Args:
            ticker: Company ticker
            simulation_date: Simulated current date

        Returns:
            List of available fiscal years (up to 5 years back)
        """
        current_year = simulation_date.year

        # Most 10-Ks are filed 60-90 days after fiscal year end
        # So if simulation date is in Q1, the prior year's 10-K may not be available
        if simulation_date.month < 4:
            latest_available = current_year - 2
        else:
            latest_available = current_year - 1

        # Return last 5 available years
        return list(range(latest_available, latest_available - 5, -1))

    async def fetch_ground_truth(
        self,
        ticker: str,
        fiscal_year: int,
        metric: str
    ) -> GroundTruth:
        """
        Fetch ground truth data from MCP tools.

        Args:
            ticker: Company ticker
            fiscal_year: Target fiscal year
            metric: Primary metric to fetch

        Returns:
            GroundTruth with verified financial data
        """
        financials = FinancialData()
        macro_thesis = ""
        key_themes = []

        if self.edgar_client:
            try:
                # Fetch XBRL financials
                xbrl_is = await self.edgar_client.parse_xbrl_financials(
                    ticker=ticker,
                    statement_type="IS",
                    fiscal_year=fiscal_year
                )
                xbrl_bs = await self.edgar_client.parse_xbrl_financials(
                    ticker=ticker,
                    statement_type="BS",
                    fiscal_year=fiscal_year
                )
                xbrl_cf = await self.edgar_client.parse_xbrl_financials(
                    ticker=ticker,
                    statement_type="CF",
                    fiscal_year=fiscal_year
                )

                # Map XBRL data to FinancialData
                is_data = xbrl_is.data
                bs_data = xbrl_bs.data
                cf_data = xbrl_cf.data

                financials = FinancialData(
                    revenue=is_data.get("Revenues") or is_data.get("RevenueFromContractWithCustomerExcludingAssessedTax"),
                    gross_profit=is_data.get("GrossProfit"),
                    operating_income=is_data.get("OperatingIncomeLoss"),
                    net_income=is_data.get("NetIncomeLoss"),
                    total_assets=bs_data.get("Assets"),
                    total_liabilities=bs_data.get("Liabilities"),
                    shareholders_equity=bs_data.get("StockholdersEquity"),
                    operating_cash_flow=cf_data.get("NetCashProvidedByUsedInOperatingActivities"),
                )

                # Calculate derived metrics
                if financials.revenue and financials.gross_profit:
                    financials.gross_margin = financials.gross_profit / financials.revenue
                if financials.revenue and financials.operating_income:
                    financials.operating_margin = financials.operating_income / financials.revenue
                if financials.revenue and financials.net_income:
                    financials.net_margin = financials.net_income / financials.revenue

            except Exception as e:
                logger.warning("failed_to_fetch_ground_truth", ticker=ticker, error=str(e))

        # Generate macro thesis based on sector
        sector = TICKER_TO_SECTOR.get(ticker.upper(), "technology")
        sector_themes = {
            "technology": ["AI adoption", "cloud growth", "digital transformation", "chip demand"],
            "financials": ["interest rate environment", "credit quality", "capital markets activity"],
            "healthcare": ["drug pipeline", "regulatory approvals", "aging population trends"],
            "energy": ["oil prices", "energy transition", "production volumes"],
            "consumer_discretionary": ["consumer spending", "e-commerce trends", "brand strength"],
        }
        key_themes = sector_themes.get(sector, ["industry dynamics", "competitive position"])
        macro_thesis = f"Analysis should consider {', '.join(key_themes[:3])} as primary factors."

        return GroundTruth(
            macro_thesis=macro_thesis,
            financials=financials,
            key_themes=key_themes,
        )

    async def generate_task(
        self,
        template_id: str,
        simulation_date: datetime,
        substitute_ticker: bool = True,
        substitute_year: bool = True,
    ) -> Optional[Task]:
        """
        Generate a dynamic task variant from a FAB template.

        Args:
            template_id: ID of the FAB question template
            simulation_date: Simulated current date for temporal constraints
            substitute_ticker: Whether to substitute the ticker
            substitute_year: Whether to substitute the fiscal year

        Returns:
            Dynamic Task variant or None if template not found
        """
        template = self.get_question_by_id(template_id)
        if not template:
            logger.error("template_not_found", template_id=template_id)
            return None

        # Default ticker based on category/sector
        default_tickers = {
            # FAB categories
            TaskCategory.QUANTITATIVE_RETRIEVAL: "AAPL",
            TaskCategory.QUALITATIVE_RETRIEVAL: "MSFT",
            TaskCategory.NUMERICAL_REASONING: "GOOGL",
            TaskCategory.COMPLEX_RETRIEVAL: "NVDA",
            TaskCategory.ADJUSTMENTS: "META",
            TaskCategory.BEAT_OR_MISS: "AMZN",
            TaskCategory.TRENDS: "TSLA",
            TaskCategory.FINANCIAL_MODELING: "JPM",
            TaskCategory.MARKET_ANALYSIS: "V",
            # Options trading categories
            TaskCategory.OPTIONS_PRICING: "SPY",
            TaskCategory.GREEKS_ANALYSIS: "AAPL",
            TaskCategory.STRATEGY_CONSTRUCTION: "SPY",
            TaskCategory.VOLATILITY_TRADING: "TSLA",
            TaskCategory.PNL_ATTRIBUTION: "QQQ",
            TaskCategory.RISK_MANAGEMENT: "SPY",
            TaskCategory.COPY_TRADING: "AAPL",
            TaskCategory.RACE_TO_10M: "SPY",
            TaskCategory.STRATEGY_DEFENSE: "TSLA",
        }

        original_ticker = default_tickers.get(template.category, "AAPL")

        # Substitute ticker if requested
        if substitute_ticker:
            new_ticker = self.sample_similar_company(original_ticker)
        else:
            new_ticker = original_ticker

        # Get available years and substitute
        available_years = self.get_available_fiscal_years(new_ticker, simulation_date)
        if substitute_year and available_years:
            new_year = random.choice(available_years)
        else:
            new_year = available_years[0] if available_years else simulation_date.year - 1

        # For comparison tasks, get a competitor ticker
        comp_ticker = self.sample_similar_company(new_ticker)

        # Check if template contains placeholders for dynamic substitution
        has_placeholders = "{ticker}" in template.template or "{year}" in template.template

        if has_placeholders:
            # Format the question with substituted values
            # Include options trading placeholders for Options categories
            question_text = template.template.format(
                ticker=new_ticker,
                year=new_year,
                prev_year=new_year - 1,
                start_year=new_year - 2,
                quarter=random.randint(1, 4),
                comp_ticker=comp_ticker,
                # Options trading placeholders
                days_to_expiry=random.choice([7, 14, 30, 45, 60, 90]),
                strike_pct=random.choice([95, 100, 105, 110]),
            )
            task_id = f"{template_id}_variant_{new_ticker}_{new_year}"
        else:
            # Literal question from CSV - use as-is, extract year from text if present
            question_text = template.template
            import re
            year_match = re.search(r'\b(20\d{2})\b', question_text)
            if year_match:
                new_year = int(year_match.group(1))
            # Use "LITERAL" to indicate this is a literal question, not a generated variant
            new_ticker = "LITERAL"
            task_id = f"{template_id}_literal_{new_year}"

        # Fetch ground truth (prefer dataset-provided)
        dataset_example = self.dataset_examples_by_id.get(template.template_id)
        if dataset_example and dataset_example.ground_truth:
            ground_truth = dataset_example.ground_truth
        else:
            ground_truth = await self.fetch_ground_truth(new_ticker, new_year, template.metric)

        task = Task(
            question_id=task_id,
            category=template.category,
            question=question_text,
            ticker=new_ticker,
            fiscal_year=new_year,
            simulation_date=simulation_date,
            ground_truth=ground_truth,
            difficulty=template.difficulty,
            rubric=template.rubric,
            requires_code_execution=template.requires_code_execution,
        )

        logger.info(
            "task_generated",
            task_id=task_id,
            category=template.category.value,
            ticker=new_ticker,
            year=new_year,
        )

        return task

    async def generate_task_batch(
        self,
        count: int,
        simulation_date: datetime,
        categories: Optional[list[TaskCategory]] = None,
        difficulties: Optional[list[TaskDifficulty]] = None,
    ) -> list[Task]:
        """
        Generate a batch of dynamic tasks.

        Args:
            count: Number of tasks to generate
            simulation_date: Simulated current date
            categories: Optional filter by categories
            difficulties: Optional filter by difficulties

        Returns:
            List of generated tasks
        """
        # Filter templates
        templates = self.fab_dataset.questions

        if categories:
            templates = [t for t in templates if t.category in categories]

        if difficulties:
            templates = [t for t in templates if t.difficulty in difficulties]

        if not templates:
            logger.warning("no_matching_templates")
            return []

        tasks = []
        for _ in range(count):
            template = random.choice(templates)
            task = await self.generate_task(template.template_id, simulation_date)
            if task:
                tasks.append(task)

        return tasks

    def get_category_distribution(self) -> dict[str, int]:
        """Get the distribution of questions across categories."""
        distribution = {}
        for q in self.fab_dataset.questions:
            cat = q.category.value
            distribution[cat] = distribution.get(cat, 0) + 1
        return distribution
