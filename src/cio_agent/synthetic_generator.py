"""
Synthetic Task Generator: Generator-Verifier-Refiner Architecture.

Generates synthetic FAB-style questions using the "reverse-engineering" approach:
1. Start with Python-calculated ground truth from Financial Lake
2. Have LLM generate naturalistic question that leads to that answer
3. Verify solvability with separate agent
4. Refine ambiguous questions

Covers all 9 FAB categories with appropriate prompting strategies.
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
)
from cio_agent.financial_lake import FinancialLake, TICKER_UNIVERSE
from cio_agent.alphavantage import FundamentalData

logger = structlog.get_logger()


class RubricComponent(BaseModel):
    """Single component of a weighted rubric."""
    name: str
    description: str
    expected_value: Optional[str] = None
    weight: float = 0.25
    
    
class EnhancedRubric(BaseModel):
    """Enhanced rubric with weighted components for LLM-as-a-Judge."""
    components: list[RubricComponent] = Field(default_factory=list)
    max_score: int = 100
    
    def to_task_rubric(self) -> TaskRubric:
        """Convert to simple TaskRubric for compatibility."""
        return TaskRubric(
            criteria=[c.description for c in self.components],
            max_score=self.max_score,
        )


class SyntheticQuestion(BaseModel):
    """A synthetically generated question with ground truth."""
    question_id: str
    category: TaskCategory
    difficulty: TaskDifficulty
    question: str
    ground_truth_value: Any
    ground_truth_formatted: str
    ticker: str
    fiscal_year: int
    calculation_steps: list[str] = Field(default_factory=list)
    rubric: EnhancedRubric
    requires_code_execution: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class CalculationResult(BaseModel):
    """Result of a ground truth calculation."""
    value: float
    formatted: str
    steps: list[str]
    components: dict[str, Any] = Field(default_factory=dict)


def _format_currency(value: float, billions: bool = True) -> str:
    """Format currency value for display."""
    if billions:
        return f"${value / 1e9:.2f}B"
    return f"${value / 1e6:.2f}M"


def _format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format percentage for display."""
    return f"{value * 100:.{decimal_places}f}%"


class SyntheticTaskGenerator:
    """
    Generates synthetic FAB-style tasks using ground truth from Financial Lake.
    
    Uses the "reverse-engineering" approach: calculate the answer first,
    then generate a question that leads to that answer.
    
    Usage:
        generator = SyntheticTaskGenerator()
        questions = await generator.generate_batch(count=10)
    """
    
    # Category weights matching FAB benchmark distribution
    CATEGORY_WEIGHTS = {
        TaskCategory.QUANTITATIVE_RETRIEVAL: 0.19,
        TaskCategory.QUALITATIVE_RETRIEVAL: 0.18,
        TaskCategory.NUMERICAL_REASONING: 0.15,
        TaskCategory.BEAT_OR_MISS: 0.13,
        TaskCategory.COMPLEX_RETRIEVAL: 0.10,
        TaskCategory.ADJUSTMENTS: 0.08,
        TaskCategory.TRENDS: 0.07,
        TaskCategory.FINANCIAL_MODELING: 0.09,
        TaskCategory.MARKET_ANALYSIS: 0.06,
    }
    
    def __init__(
        self,
        financial_lake: Optional[FinancialLake] = None,
        llm_client: Optional[Any] = None,  # For question generation
    ):
        self.lake = financial_lake or FinancialLake()
        self.llm_client = llm_client
        self._question_counter = 0
    
    def _get_available_data(self, ticker: str) -> Optional[FundamentalData]:
        """Get financial data for a ticker from the lake."""
        return self.lake.get(ticker)
    
    def _select_random_ticker(self, sector: Optional[str] = None) -> Optional[str]:
        """Select a random ticker with available data."""
        if sector:
            available = self.lake.get_tickers_by_sector(sector)
        else:
            available = self.lake.get_available_tickers()
        
        if not available:
            return None
        return random.choice(available)
    
    def _generate_question_id(self, category: TaskCategory) -> str:
        """Generate unique question ID."""
        self._question_counter += 1
        cat_code = category.value.replace(" ", "_").upper()[:4]
        return f"SYN_{cat_code}_{self._question_counter:04d}"
    
    # =========================================================================
    # CATEGORY GENERATORS
    # =========================================================================
    
    def generate_quantitative_retrieval(
        self,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
    ) -> Optional[SyntheticQuestion]:
        """
        Generate a Quantitative Retrieval question (19% of benchmark).
        
        Tests extraction of specific numerical values without modification.
        """
        ticker = ticker or self._select_random_ticker()
        if not ticker:
            return None
        
        data = self._get_available_data(ticker)
        if not data or not data.annual_income_statements:
            return None
        
        # Select fiscal year
        available_statements = data.annual_income_statements
        if fiscal_year:
            statement = next(
                (s for s in available_statements 
                 if s.fiscal_date_ending.startswith(str(fiscal_year))),
                None
            )
        else:
            statement = random.choice(available_statements[:3])  # Recent 3 years
        
        if not statement:
            return None
        
        year = int(statement.fiscal_date_ending[:4])
        
        # Select a metric to query
        metrics = [
            ("total revenue", statement.total_revenue, "revenue"),
            ("net income", statement.net_income, "net_income"),
            ("gross profit", statement.gross_profit, "gross_profit"),
            ("operating income", statement.operating_income, "operating_income"),
            ("EBITDA", statement.ebitda, "ebitda"),
        ]
        
        # Filter to available metrics
        available_metrics = [(name, val, key) for name, val, key in metrics if val is not None]
        if not available_metrics:
            return None
        
        metric_name, metric_value, metric_key = random.choice(available_metrics)
        
        # Generate question
        question_templates = [
            f"What was {ticker}'s {metric_name} in fiscal year {year}?",
            f"Report the {metric_name} for {ticker} in FY{year}.",
            f"Find {ticker}'s {metric_name} from their FY{year} annual report.",
        ]
        question = random.choice(question_templates)
        
        # Build rubric
        rubric = EnhancedRubric(components=[
            RubricComponent(
                name="retrieval_accuracy",
                description=f"Correctly extracted {metric_name} value",
                expected_value=_format_currency(metric_value),
                weight=0.50,
            ),
            RubricComponent(
                name="fiscal_year",
                description="Referenced correct fiscal year",
                expected_value=str(year),
                weight=0.30,
            ),
            RubricComponent(
                name="units",
                description="Correctly stated units (millions/billions USD)",
                weight=0.20,
            ),
        ])
        
        return SyntheticQuestion(
            question_id=self._generate_question_id(TaskCategory.QUANTITATIVE_RETRIEVAL),
            category=TaskCategory.QUANTITATIVE_RETRIEVAL,
            difficulty=TaskDifficulty.EASY,
            question=question,
            ground_truth_value=metric_value,
            ground_truth_formatted=_format_currency(metric_value),
            ticker=ticker,
            fiscal_year=year,
            calculation_steps=[f"Retrieved {metric_name} from FY{year} income statement"],
            rubric=rubric,
        )
    
    def generate_qualitative_retrieval(
        self,
        ticker: Optional[str] = None,
    ) -> Optional[SyntheticQuestion]:
        """
        Generate a Qualitative Retrieval question (18% of benchmark).
        
        Tests extraction of non-numerical text (descriptions, risk factors).
        """
        ticker = ticker or self._select_random_ticker()
        if not ticker:
            return None
        
        data = self._get_available_data(ticker)
        if not data or not data.overview:
            return None
        
        overview = data.overview
        if not overview.description:
            return None
        
        # Question types
        question_types = [
            (
                f"Describe {ticker}'s main business and products.",
                overview.description,
                "business_description",
            ),
            (
                f"What industry does {ticker} operate in? Describe their primary offerings.",
                f"{overview.industry}: {overview.description[:200]}...",
                "industry_description",
            ),
            (
                f"Provide an overview of {ticker}'s business model.",
                overview.description,
                "business_model",
            ),
        ]
        
        question, expected_answer, question_type = random.choice(question_types)
        
        # Extract key themes from description
        key_themes = []
        if overview.sector:
            key_themes.append(overview.sector)
        if overview.industry:
            key_themes.append(overview.industry)
        
        rubric = EnhancedRubric(components=[
            RubricComponent(
                name="content_accuracy",
                description="Answer captures key business description elements",
                weight=0.40,
            ),
            RubricComponent(
                name="sector_industry",
                description=f"Correctly identifies sector ({overview.sector}) or industry ({overview.industry})",
                expected_value=overview.sector,
                weight=0.30,
            ),
            RubricComponent(
                name="completeness",
                description="Provides sufficient detail without irrelevant information",
                weight=0.30,
            ),
        ])
        
        return SyntheticQuestion(
            question_id=self._generate_question_id(TaskCategory.QUALITATIVE_RETRIEVAL),
            category=TaskCategory.QUALITATIVE_RETRIEVAL,
            difficulty=TaskDifficulty.EASY,
            question=question,
            ground_truth_value=expected_answer,
            ground_truth_formatted=expected_answer[:500],
            ticker=ticker,
            fiscal_year=datetime.now().year,
            calculation_steps=["Retrieved company overview from filings"],
            rubric=rubric,
            metadata={"question_type": question_type, "key_themes": key_themes},
        )
    
    def generate_numerical_reasoning(
        self,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
    ) -> Optional[SyntheticQuestion]:
        """
        Generate a Numerical Reasoning question (15% of benchmark).
        
        Tests arithmetic manipulation: margins, growth rates, CAGR.
        """
        ticker = ticker or self._select_random_ticker()
        if not ticker:
            return None
        
        data = self._get_available_data(ticker)
        if not data or len(data.annual_income_statements) < 2:
            return None
        
        # Choose calculation type
        calc_types = ["gross_margin", "operating_margin", "yoy_growth", "net_margin"]
        calc_type = random.choice(calc_types)
        
        statements = data.annual_income_statements
        
        if calc_type == "gross_margin":
            statement = statements[0]
            year = int(statement.fiscal_date_ending[:4])
            
            if not statement.gross_profit or not statement.total_revenue:
                return None
            
            margin = statement.gross_profit / statement.total_revenue
            
            question = f"Calculate {ticker}'s gross margin percentage for fiscal year {year}."
            ground_truth = margin
            formatted = _format_percentage(margin)
            steps = [
                f"Gross Profit = {_format_currency(statement.gross_profit)}",
                f"Total Revenue = {_format_currency(statement.total_revenue)}",
                f"Gross Margin = Gross Profit / Revenue = {formatted}",
            ]
            
        elif calc_type == "operating_margin":
            statement = statements[0]
            year = int(statement.fiscal_date_ending[:4])
            
            if not statement.operating_income or not statement.total_revenue:
                return None
            
            margin = statement.operating_income / statement.total_revenue
            
            question = f"What was {ticker}'s operating margin in FY{year}?"
            ground_truth = margin
            formatted = _format_percentage(margin)
            steps = [
                f"Operating Income = {_format_currency(statement.operating_income)}",
                f"Total Revenue = {_format_currency(statement.total_revenue)}",
                f"Operating Margin = Operating Income / Revenue = {formatted}",
            ]
            
        elif calc_type == "yoy_growth":
            if len(statements) < 2:
                return None
            
            current = statements[0]
            prior = statements[1]
            year = int(current.fiscal_date_ending[:4])
            prior_year = int(prior.fiscal_date_ending[:4])
            
            if not current.total_revenue or not prior.total_revenue:
                return None
            
            growth = (current.total_revenue - prior.total_revenue) / prior.total_revenue
            
            question = f"Calculate the year-over-year revenue growth rate for {ticker} from FY{prior_year} to FY{year}."
            ground_truth = growth
            formatted = _format_percentage(growth)
            steps = [
                f"FY{year} Revenue = {_format_currency(current.total_revenue)}",
                f"FY{prior_year} Revenue = {_format_currency(prior.total_revenue)}",
                f"YoY Growth = (Current - Prior) / Prior = {formatted}",
            ]
            
        else:  # net_margin
            statement = statements[0]
            year = int(statement.fiscal_date_ending[:4])
            
            if not statement.net_income or not statement.total_revenue:
                return None
            
            margin = statement.net_income / statement.total_revenue
            
            question = f"Calculate {ticker}'s net profit margin for FY{year}."
            ground_truth = margin
            formatted = _format_percentage(margin)
            steps = [
                f"Net Income = {_format_currency(statement.net_income)}",
                f"Total Revenue = {_format_currency(statement.total_revenue)}",
                f"Net Margin = Net Income / Revenue = {formatted}",
            ]
        
        rubric = EnhancedRubric(components=[
            RubricComponent(
                name="formula_application",
                description="Applied correct formula for calculation",
                weight=0.30,
            ),
            RubricComponent(
                name="data_retrieval",
                description="Retrieved correct input values",
                weight=0.30,
            ),
            RubricComponent(
                name="calculation",
                description=f"Calculation result within 1% of {formatted}",
                expected_value=formatted,
                weight=0.30,
            ),
            RubricComponent(
                name="units",
                description="Correctly expressed as percentage",
                weight=0.10,
            ),
        ])
        
        return SyntheticQuestion(
            question_id=self._generate_question_id(TaskCategory.NUMERICAL_REASONING),
            category=TaskCategory.NUMERICAL_REASONING,
            difficulty=TaskDifficulty.MEDIUM,
            question=question,
            ground_truth_value=ground_truth,
            ground_truth_formatted=formatted,
            ticker=ticker,
            fiscal_year=year,
            calculation_steps=steps,
            rubric=rubric,
            requires_code_execution=True,
            metadata={"calculation_type": calc_type},
        )
    
    def generate_beat_or_miss(
        self,
        ticker: Optional[str] = None,
    ) -> Optional[SyntheticQuestion]:
        """
        Generate a Beat or Miss question (13% of benchmark).
        
        Tests comparison of actual EPS vs analyst estimates.
        """
        ticker = ticker or self._select_random_ticker()
        if not ticker:
            return None
        
        data = self._get_available_data(ticker)
        if not data or not data.quarterly_earnings:
            return None
        
        # Find a quarter with both reported and estimated EPS
        valid_earnings = [
            e for e in data.quarterly_earnings
            if e.reported_eps is not None and e.estimated_eps is not None
        ]
        
        if not valid_earnings:
            return None
        
        earnings = random.choice(valid_earnings[:4])  # Recent 4 quarters
        
        # Determine beat/miss
        beat = earnings.reported_eps > earnings.estimated_eps
        result = "beat" if beat else "missed"
        
        # Calculate surprise
        surprise_pct = earnings.surprise_percentage or (
            (earnings.reported_eps - earnings.estimated_eps) / abs(earnings.estimated_eps) * 100
            if earnings.estimated_eps != 0 else 0
        )
        
        # Parse quarter from date
        date = earnings.fiscal_date_ending
        year = int(date[:4])
        month = int(date[5:7])
        quarter = (month - 1) // 3 + 1
        
        question = f"Did {ticker} beat or miss analyst EPS expectations in Q{quarter} {year}? By how much?"
        
        formatted = f"{result.capitalize()} by {abs(surprise_pct):.1f}% (Reported: ${earnings.reported_eps:.2f}, Estimated: ${earnings.estimated_eps:.2f})"
        
        rubric = EnhancedRubric(components=[
            RubricComponent(
                name="beat_miss_determination",
                description=f"Correctly determined that {ticker} {result} expectations",
                expected_value=result,
                weight=0.40,
            ),
            RubricComponent(
                name="reported_eps",
                description="Retrieved correct reported EPS",
                expected_value=f"${earnings.reported_eps:.2f}",
                weight=0.20,
            ),
            RubricComponent(
                name="estimated_eps",
                description="Retrieved correct consensus estimate",
                expected_value=f"${earnings.estimated_eps:.2f}",
                weight=0.20,
            ),
            RubricComponent(
                name="surprise_magnitude",
                description="Calculated correct surprise percentage",
                expected_value=f"{abs(surprise_pct):.1f}%",
                weight=0.20,
            ),
        ])
        
        beat_miss_str = "Beat" if beat else "Missed"
        
        return SyntheticQuestion(
            question_id=self._generate_question_id(TaskCategory.BEAT_OR_MISS),
            category=TaskCategory.BEAT_OR_MISS,
            difficulty=TaskDifficulty.MEDIUM,
            question=question,
            ground_truth_value={"beat": beat, "surprise_pct": surprise_pct},
            ground_truth_formatted=formatted,
            ticker=ticker,
            fiscal_year=year,
            calculation_steps=[
                f"Retrieved Q{quarter} {year} earnings",
                f"Reported EPS = ${earnings.reported_eps:.2f}",
                f"Estimated EPS = ${earnings.estimated_eps:.2f}",
                f"{beat_miss_str} by {abs(surprise_pct):.1f}%",
            ],
            rubric=rubric,
            metadata={"quarter": quarter},
        )
    
    def generate_trends(
        self,
        ticker: Optional[str] = None,
    ) -> Optional[SyntheticQuestion]:
        """
        Generate a Trends question (7% of benchmark).
        
        Tests longitudinal analysis over multiple periods.
        """
        ticker = ticker or self._select_random_ticker()
        if not ticker:
            return None
        
        data = self._get_available_data(ticker)
        if not data or len(data.annual_income_statements) < 3:
            return None
        
        statements = data.annual_income_statements[:3]  # Last 3 years
        
        # Calculate revenue trend
        revenues = []
        years = []
        for s in statements:
            if s.total_revenue:
                revenues.append(s.total_revenue)
                years.append(int(s.fiscal_date_ending[:4]))
        
        if len(revenues) < 3:
            return None
        
        # Reverse to chronological order
        revenues = revenues[::-1]
        years = years[::-1]
        
        # Calculate CAGR
        start_val = revenues[0]
        end_val = revenues[-1]
        n_years = len(revenues) - 1
        cagr = (end_val / start_val) ** (1 / n_years) - 1
        
        # Determine trend direction
        if cagr > CAGR_STRONG_GROWTH_THRESHOLD:
            trend = "strong growth"
        elif cagr > CAGR_MODERATE_GROWTH_THRESHOLD:
            trend = "moderate growth"
        elif cagr > CAGR_STABLE_THRESHOLD:
            trend = "stable"
        elif cagr > CAGR_MODERATE_DECLINE_THRESHOLD:
            trend = "moderate decline"
        else:
            trend = "significant decline"
        
        question = f"Analyze the revenue trend for {ticker} from FY{years[0]} to FY{years[-1]}. What is the CAGR?"
        
        formatted = f"{_format_percentage(cagr)} CAGR ({trend})"
        
        steps = [f"FY{year} Revenue = {_format_currency(rev)}" for year, rev in zip(years, revenues)]
        steps.append(f"CAGR = (End/Start)^(1/n) - 1 = {_format_percentage(cagr)}")
        
        rubric = EnhancedRubric(components=[
            RubricComponent(
                name="data_retrieval",
                description="Retrieved revenue for all required years",
                weight=0.30,
            ),
            RubricComponent(
                name="cagr_calculation",
                description=f"Calculated CAGR within 1% of {_format_percentage(cagr)}",
                expected_value=_format_percentage(cagr),
                weight=0.40,
            ),
            RubricComponent(
                name="trend_identification",
                description=f"Correctly identified trend as {trend}",
                expected_value=trend,
                weight=0.30,
            ),
        ])
        
        return SyntheticQuestion(
            question_id=self._generate_question_id(TaskCategory.TRENDS),
            category=TaskCategory.TRENDS,
            difficulty=TaskDifficulty.HARD,
            question=question,
            ground_truth_value={"cagr": cagr, "trend": trend},
            ground_truth_formatted=formatted,
            ticker=ticker,
            fiscal_year=years[-1],
            calculation_steps=steps,
            rubric=rubric,
            requires_code_execution=True,
            metadata={"years": years, "revenues": revenues},
        )
    
    def generate_adjustments(
        self,
        ticker: Optional[str] = None,
    ) -> Optional[SyntheticQuestion]:
        """
        Generate an Adjustments question (8% of benchmark).
        
        Tests EBITDA calculation from components.
        """
        ticker = ticker or self._select_random_ticker()
        if not ticker:
            return None
        
        data = self._get_available_data(ticker)
        if not data or not data.annual_income_statements:
            return None
        
        statement = data.annual_income_statements[0]
        year = int(statement.fiscal_date_ending[:4])
        
        # Need components to calculate EBITDA
        net_income = statement.net_income
        interest = statement.interest_expense
        tax = statement.income_tax_expense
        da = statement.depreciation_and_amortization
        
        if not all([net_income, interest, tax, da]):
            return None
        
        # Calculate EBITDA
        ebitda = net_income + interest + tax + da
        
        question = f"Calculate {ticker}'s EBITDA for FY{year} starting from Net Income. Show your work."
        
        formatted = _format_currency(ebitda)
        
        steps = [
            f"Net Income = {_format_currency(net_income)}",
            f"+ Interest Expense = {_format_currency(interest)}",
            f"+ Income Tax Expense = {_format_currency(tax)}",
            f"+ Depreciation & Amortization = {_format_currency(da)}",
            f"= EBITDA = {formatted}",
        ]
        
        rubric = EnhancedRubric(components=[
            RubricComponent(
                name="formula_knowledge",
                description="Correctly identified EBITDA = Net Income + Interest + Tax + D&A",
                weight=0.30,
            ),
            RubricComponent(
                name="component_retrieval",
                description="Retrieved all four components accurately",
                weight=0.30,
            ),
            RubricComponent(
                name="calculation",
                description=f"Final EBITDA within 2% of {formatted}",
                expected_value=formatted,
                weight=0.30,
            ),
            RubricComponent(
                name="work_shown",
                description="Showed calculation steps clearly",
                weight=0.10,
            ),
        ])
        
        return SyntheticQuestion(
            question_id=self._generate_question_id(TaskCategory.ADJUSTMENTS),
            category=TaskCategory.ADJUSTMENTS,
            difficulty=TaskDifficulty.MEDIUM,
            question=question,
            ground_truth_value=ebitda,
            ground_truth_formatted=formatted,
            ticker=ticker,
            fiscal_year=year,
            calculation_steps=steps,
            rubric=rubric,
            requires_code_execution=True,
            metadata={
                "net_income": net_income,
                "interest": interest,
                "tax": tax,
                "da": da,
            },
        )
    
    def generate_financial_modeling(
        self,
        ticker: Optional[str] = None,
    ) -> Optional[SyntheticQuestion]:
        """
        Generate a Financial Modeling question (9% of benchmark).
        
        Tests M&A firepower calculation or similar modeling.
        """
        ticker = ticker or self._select_random_ticker()
        if not ticker:
            return None
        
        data = self._get_available_data(ticker)
        if not data:
            return None
        
        if not data.annual_balance_sheets or not data.annual_income_statements:
            return None
        
        balance = data.annual_balance_sheets[0]
        income = data.annual_income_statements[0]
        year = int(balance.fiscal_date_ending[:4])
        
        # Get components for M&A firepower
        cash = balance.cash_and_equivalents or 0
        st_investments = balance.short_term_investments or 0
        total_debt = balance.total_debt or 0
        ebitda = income.ebitda
        
        if not ebitda or ebitda <= 0:
            return None
        
        # M&A Firepower = (Cash + ST Investments) + (3x EBITDA) - Total Debt
        max_leverage = 3.0
        debt_capacity = max_leverage * ebitda
        firepower = (cash + st_investments) + debt_capacity - total_debt
        
        question = f"Calculate {ticker}'s M&A firepower as of FY{year}, assuming a maximum leverage of {max_leverage}x EBITDA."
        
        formatted = _format_currency(firepower)
        
        steps = [
            f"Cash & Equivalents = {_format_currency(cash)}",
            f"Short-term Investments = {_format_currency(st_investments)}",
            f"EBITDA = {_format_currency(ebitda)}",
            f"Debt Capacity (3x EBITDA) = {_format_currency(debt_capacity)}",
            f"Total Debt = {_format_currency(total_debt)}",
            f"M&A Firepower = (Cash + ST Invest) + Debt Capacity - Debt = {formatted}",
        ]
        
        rubric = EnhancedRubric(components=[
            RubricComponent(
                name="concept_understanding",
                description="Understood M&A firepower concept and formula",
                weight=0.25,
            ),
            RubricComponent(
                name="data_retrieval",
                description="Retrieved all required balance sheet and income statement items",
                weight=0.25,
            ),
            RubricComponent(
                name="leverage_application",
                description="Correctly applied 3x leverage multiplier to EBITDA",
                weight=0.25,
            ),
            RubricComponent(
                name="calculation",
                description=f"Final result within 5% of {formatted}",
                expected_value=formatted,
                weight=0.25,
            ),
        ])
        
        return SyntheticQuestion(
            question_id=self._generate_question_id(TaskCategory.FINANCIAL_MODELING),
            category=TaskCategory.FINANCIAL_MODELING,
            difficulty=TaskDifficulty.EXPERT,
            question=question,
            ground_truth_value=firepower,
            ground_truth_formatted=formatted,
            ticker=ticker,
            fiscal_year=year,
            calculation_steps=steps,
            rubric=rubric,
            requires_code_execution=True,
            metadata={
                "cash": cash,
                "st_investments": st_investments,
                "ebitda": ebitda,
                "total_debt": total_debt,
                "max_leverage": max_leverage,
            },
        )
    
    def generate_market_analysis(
        self,
        sector: Optional[str] = None,
    ) -> Optional[SyntheticQuestion]:
        """
        Generate a Market Analysis question (6% of benchmark).
        
        Tests cross-company comparison within a sector.
        """
        sector = sector or random.choice(list(TICKER_UNIVERSE.keys()))
        available = self.lake.get_tickers_by_sector(sector)
        
        if len(available) < 3:
            return None
        
        # Select 3 companies for comparison
        tickers = random.sample(available, 3)
        
        # Gather data
        companies_data = []
        for t in tickers:
            data = self._get_available_data(t)
            if data and data.annual_income_statements and data.overview:
                statement = data.annual_income_statements[0]
                if statement.total_revenue and statement.net_income:
                    companies_data.append({
                        "ticker": t,
                        "name": data.overview.name,
                        "revenue": statement.total_revenue,
                        "net_income": statement.net_income,
                        "net_margin": statement.net_income / statement.total_revenue,
                        "year": int(statement.fiscal_date_ending[:4]),
                    })
        
        if len(companies_data) < 3:
            return None
        
        # Sort by net margin for ranking
        companies_data.sort(key=lambda x: x["net_margin"], reverse=True)
        
        year = companies_data[0]["year"]
        ticker_str = ", ".join([c["ticker"] for c in companies_data])
        
        question = f"Compare the net profit margins of {ticker_str} for FY{year}. Rank them from highest to lowest."
        
        formatted = ", ".join([
            f"{c['ticker']}: {_format_percentage(c['net_margin'])}" 
            for c in companies_data
        ])
        
        steps = []
        for c in companies_data:
            steps.append(f"{c['ticker']}: Net Income / Revenue = {_format_percentage(c['net_margin'])}")
        steps.append(f"Ranking: {' > '.join([c['ticker'] for c in companies_data])}")
        
        rubric = EnhancedRubric(components=[
            RubricComponent(
                name="data_retrieval",
                description="Retrieved financial data for all three companies",
                weight=0.30,
            ),
            RubricComponent(
                name="margin_calculation",
                description="Correctly calculated net margins for each company",
                weight=0.30,
            ),
            RubricComponent(
                name="ranking",
                description=f"Correctly ranked as {' > '.join([c['ticker'] for c in companies_data])}",
                expected_value=" > ".join([c['ticker'] for c in companies_data]),
                weight=0.30,
            ),
            RubricComponent(
                name="analysis_quality",
                description="Provided insight into why margins differ",
                weight=0.10,
            ),
        ])
        
        return SyntheticQuestion(
            question_id=self._generate_question_id(TaskCategory.MARKET_ANALYSIS),
            category=TaskCategory.MARKET_ANALYSIS,
            difficulty=TaskDifficulty.HARD,
            question=question,
            ground_truth_value={"ranking": [c["ticker"] for c in companies_data]},
            ground_truth_formatted=formatted,
            ticker=tickers[0],  # Primary ticker
            fiscal_year=year,
            calculation_steps=steps,
            rubric=rubric,
            requires_code_execution=True,
            metadata={"sector": sector, "companies": companies_data},
        )
    
    def generate_complex_retrieval(
        self,
        ticker: Optional[str] = None,
    ) -> Optional[SyntheticQuestion]:
        """
        Generate a Complex Retrieval question (10% of benchmark).
        
        Tests synthesis from multiple data points.
        """
        ticker = ticker or self._select_random_ticker()
        if not ticker:
            return None
        
        data = self._get_available_data(ticker)
        if not data:
            return None
        
        if not data.annual_balance_sheets or not data.annual_income_statements:
            return None
        
        balance = data.annual_balance_sheets[0]
        year = int(balance.fiscal_date_ending[:4])
        
        # Calculate debt-to-equity ratio
        total_debt = balance.total_debt
        equity = balance.total_shareholder_equity
        
        if not total_debt or not equity or equity <= 0:
            return None
        
        debt_to_equity = total_debt / equity
        
        question = f"Calculate {ticker}'s debt-to-equity ratio as of FY{year}. Also provide the total debt and shareholder equity figures."
        
        formatted = f"D/E Ratio: {debt_to_equity:.2f}x (Debt: {_format_currency(total_debt)}, Equity: {_format_currency(equity)})"
        
        steps = [
            f"Total Debt = {_format_currency(total_debt)}",
            f"Shareholder Equity = {_format_currency(equity)}",
            f"Debt-to-Equity = Total Debt / Equity = {debt_to_equity:.2f}x",
        ]
        
        rubric = EnhancedRubric(components=[
            RubricComponent(
                name="debt_retrieval",
                description=f"Retrieved total debt ({_format_currency(total_debt)})",
                expected_value=_format_currency(total_debt),
                weight=0.25,
            ),
            RubricComponent(
                name="equity_retrieval",
                description=f"Retrieved shareholder equity ({_format_currency(equity)})",
                expected_value=_format_currency(equity),
                weight=0.25,
            ),
            RubricComponent(
                name="ratio_calculation",
                description=f"Calculated D/E ratio correctly ({debt_to_equity:.2f}x)",
                expected_value=f"{debt_to_equity:.2f}x",
                weight=0.40,
            ),
            RubricComponent(
                name="interpretation",
                description="Provided context on leverage level",
                weight=0.10,
            ),
        ])
        
        return SyntheticQuestion(
            question_id=self._generate_question_id(TaskCategory.COMPLEX_RETRIEVAL),
            category=TaskCategory.COMPLEX_RETRIEVAL,
            difficulty=TaskDifficulty.HARD,
            question=question,
            ground_truth_value={"ratio": debt_to_equity, "debt": total_debt, "equity": equity},
            ground_truth_formatted=formatted,
            ticker=ticker,
            fiscal_year=year,
            calculation_steps=steps,
            rubric=rubric,
            requires_code_execution=True,
        )
    
    # =========================================================================
    # BATCH GENERATION
    # =========================================================================
    
    def generate_by_category(
        self,
        category: TaskCategory,
        ticker: Optional[str] = None,
    ) -> Optional[SyntheticQuestion]:
        """Generate a question for a specific category."""
        generators = {
            TaskCategory.QUANTITATIVE_RETRIEVAL: self.generate_quantitative_retrieval,
            TaskCategory.QUALITATIVE_RETRIEVAL: self.generate_qualitative_retrieval,
            TaskCategory.NUMERICAL_REASONING: self.generate_numerical_reasoning,
            TaskCategory.BEAT_OR_MISS: self.generate_beat_or_miss,
            TaskCategory.COMPLEX_RETRIEVAL: self.generate_complex_retrieval,
            TaskCategory.ADJUSTMENTS: self.generate_adjustments,
            TaskCategory.TRENDS: self.generate_trends,
            TaskCategory.FINANCIAL_MODELING: self.generate_financial_modeling,
            TaskCategory.MARKET_ANALYSIS: self.generate_market_analysis,
        }
        
        generator = generators.get(category)
        if not generator:
            return None
        
        if category == TaskCategory.MARKET_ANALYSIS:
            return generator()  # Different signature
        
        return generator(ticker=ticker)
    
    def generate_batch(
        self,
        count: int = 100,
        categories: Optional[list[TaskCategory]] = None,
        respect_weights: bool = True,
    ) -> list[SyntheticQuestion]:
        """
        Generate a batch of synthetic questions.
        
        Args:
            count: Number of questions to generate
            categories: Optional filter to specific categories
            respect_weights: If True, use FAB benchmark category distribution
            
        Returns:
            List of generated questions
        """
        questions = []
        available_categories = categories or list(TaskCategory)
        
        if respect_weights and not categories:
            # Generate according to benchmark weights
            for category, weight in self.CATEGORY_WEIGHTS.items():
                cat_count = max(1, int(count * weight))
                for _ in range(cat_count):
                    q = self.generate_by_category(category)
                    if q:
                        questions.append(q)
        else:
            # Generate evenly or for specific categories
            per_category = count // len(available_categories)
            for category in available_categories:
                for _ in range(per_category):
                    q = self.generate_by_category(category)
                    if q:
                        questions.append(q)
        
        logger.info("synthetic_batch_generated", count=len(questions))
        return questions
    
    def to_task(self, question: SyntheticQuestion, simulation_date: Optional[datetime] = None) -> Task:
        """Convert a SyntheticQuestion to a Task for evaluation."""
        simulation_date = simulation_date or datetime.now()
        
        ground_truth = GroundTruth(
            macro_thesis=question.ground_truth_formatted,
            numerical_answer=question.ground_truth_value if isinstance(question.ground_truth_value, (int, float)) else None,
            key_themes=question.calculation_steps,
        )
        
        return Task(
            question_id=question.question_id,
            category=question.category,
            question=question.question,
            ticker=question.ticker,
            fiscal_year=question.fiscal_year,
            simulation_date=simulation_date,
            ground_truth=ground_truth,
            difficulty=question.difficulty,
            rubric=question.rubric.to_task_rubric(),
            requires_code_execution=question.requires_code_execution,
        )
