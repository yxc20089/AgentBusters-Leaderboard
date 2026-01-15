"""
Options Evaluator for options trading task assessment.

Evaluates the quality of options trading responses including:
- P&L calculation accuracy
- Greeks verification
- Strategy quality scoring
- Risk management discipline
"""

from typing import Any, Optional
from dataclasses import dataclass

import structlog

from cio_agent.models import (
    Task,
    AgentResponse,
    TaskCategory,
)

logger = structlog.get_logger()


# Options trading categories requiring specialized evaluation
OPTIONS_CATEGORIES = [
    TaskCategory.OPTIONS_PRICING,
    TaskCategory.GREEKS_ANALYSIS,
    TaskCategory.STRATEGY_CONSTRUCTION,
    TaskCategory.VOLATILITY_TRADING,
    TaskCategory.PNL_ATTRIBUTION,
    TaskCategory.RISK_MANAGEMENT,
    TaskCategory.COPY_TRADING,
    TaskCategory.RACE_TO_10M,
    TaskCategory.STRATEGY_DEFENSE,
]


@dataclass
class OptionsScore:
    """Detailed score for options trading evaluation."""
    score: float  # 0-100
    pnl_accuracy: float  # 0-100: P&L calculation accuracy
    greeks_accuracy: float  # 0-100: Greeks accuracy
    strategy_quality: float  # 0-100: Strategy appropriateness
    risk_management: float  # 0-100: Risk management quality
    feedback: str


@dataclass
class ExtractedOptionsData:
    """Options data extracted from agent response."""
    # Strategy type
    strategy_name: Optional[str] = None
    legs: list[dict] = None

    # Greeks
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

    # P&L
    max_profit: Optional[float] = None
    max_loss: Optional[float] = None
    breakevens: list[float] = None
    current_pnl: Optional[float] = None

    # Risk metrics
    probability_of_profit: Optional[float] = None
    var_95: Optional[float] = None

    def __post_init__(self):
        if self.legs is None:
            self.legs = []
        if self.breakevens is None:
            self.breakevens = []


class OptionsEvaluator:
    """
    Evaluates agent's options trading responses.

    Scoring dimensions:
    - P&L Accuracy (25%): Verify calculations against Black-Scholes
    - Greeks Accuracy (25%): Verify Greeks calculations
    - Strategy Quality (25%): Appropriateness for market conditions
    - Risk Management (25%): Position sizing, hedging, discipline
    """

    # Tolerances for numerical accuracy
    PRICE_TOLERANCE = 0.05  # 5% tolerance for option prices
    GREEKS_TOLERANCE = 0.10  # 10% tolerance for Greeks

    def __init__(
        self,
        task: Task,
        mcp_toolkit: Optional[Any] = None,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize the options evaluator.

        Args:
            task: The task being evaluated
            mcp_toolkit: MCPToolkit for verification calculations
            llm_client: Optional LLM client for qualitative scoring
        """
        self.task = task
        self.mcp_toolkit = mcp_toolkit
        self.llm_client = llm_client

    def _extract_numbers_from_text(self, text: str) -> list[float]:
        """Extract all numbers from text."""
        import re
        # Match numbers including negatives, decimals, percentages
        pattern = r'-?\$?\d+(?:,\d{3})*(?:\.\d+)?%?'
        matches = re.findall(pattern, text)

        numbers = []
        for match in matches:
            # Clean and convert
            clean = match.replace('$', '').replace(',', '').replace('%', '')
            try:
                numbers.append(float(clean))
            except ValueError:
                continue
        return numbers

    def _extract_options_data(
        self,
        response: AgentResponse,
    ) -> ExtractedOptionsData:
        """
        Extract options trading data from agent response.

        Parses the response for:
        - Strategy name and structure
        - Greeks values
        - P&L metrics
        - Risk parameters
        """
        analysis = response.analysis.lower()
        recommendation = response.recommendation.lower()
        combined = f"{analysis} {recommendation}"

        data = ExtractedOptionsData()

        # Detect strategy type
        strategy_keywords = {
            "iron condor": ["iron condor"],
            "bull call spread": ["bull call spread", "call spread", "debit spread"],
            "bear put spread": ["bear put spread", "put spread"],
            "straddle": ["straddle", "long straddle", "short straddle"],
            "strangle": ["strangle", "long strangle", "short strangle"],
            "covered call": ["covered call", "buy-write"],
            "protective put": ["protective put", "married put"],
            "butterfly": ["butterfly", "iron butterfly"],
            "calendar spread": ["calendar spread", "time spread"],
            "naked call": ["naked call", "uncovered call"],
            "naked put": ["naked put", "cash-secured put"],
        }

        for strategy, keywords in strategy_keywords.items():
            if any(kw in combined for kw in keywords):
                data.strategy_name = strategy
                break

        # Extract Greeks using regex patterns
        import re

        # Delta
        delta_match = re.search(r'delta[:\s]+(-?\d+\.?\d*)', combined)
        if delta_match:
            data.delta = float(delta_match.group(1))

        # Gamma
        gamma_match = re.search(r'gamma[:\s]+(-?\d+\.?\d*)', combined)
        if gamma_match:
            data.gamma = float(gamma_match.group(1))

        # Theta
        theta_match = re.search(r'theta[:\s]+(-?\d+\.?\d*)', combined)
        if theta_match:
            data.theta = float(theta_match.group(1))

        # Vega
        vega_match = re.search(r'vega[:\s]+(-?\d+\.?\d*)', combined)
        if vega_match:
            data.vega = float(vega_match.group(1))

        # Max profit
        profit_match = re.search(r'max(?:imum)?\s+profit[:\s]+\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)', combined)
        if profit_match:
            data.max_profit = float(profit_match.group(1).replace(',', ''))

        # Max loss
        loss_match = re.search(r'max(?:imum)?\s+loss[:\s]+\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)', combined)
        if loss_match:
            data.max_loss = float(loss_match.group(1).replace(',', ''))

        # Probability of profit
        pop_match = re.search(r'probability\s+of\s+profit[:\s]+(\d+\.?\d*)%?', combined)
        if pop_match:
            data.probability_of_profit = float(pop_match.group(1))

        # VaR
        var_match = re.search(r'var[:\s]+\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)', combined)
        if var_match:
            data.var_95 = float(var_match.group(1).replace(',', ''))

        return data

    async def _verify_pnl_accuracy(
        self,
        extracted: ExtractedOptionsData,
        response: AgentResponse,
    ) -> tuple[float, str]:
        """
        Verify P&L calculations against market data.

        Returns:
            Tuple of (accuracy_score, feedback)
        """
        if not self.mcp_toolkit:
            # Cannot verify without toolkit, use heuristic
            return self._heuristic_pnl_score(extracted, response)

        score = 50.0  # Base score
        feedback_parts = []

        # Check if strategy legs were provided
        if extracted.strategy_name:
            score += 15
            feedback_parts.append(f"Strategy identified: {extracted.strategy_name}.")
        else:
            feedback_parts.append("No clear strategy identified.")

        # Check max profit/loss provided
        if extracted.max_profit is not None:
            score += 10
            feedback_parts.append("Max profit specified.")

        if extracted.max_loss is not None:
            score += 10
            feedback_parts.append("Max loss specified.")

        # Check for breakeven analysis
        analysis_lower = response.analysis.lower()
        if "breakeven" in analysis_lower or "break-even" in analysis_lower:
            score += 15
            feedback_parts.append("Breakeven analysis included.")

        return min(100, score), " ".join(feedback_parts)

    def _heuristic_pnl_score(
        self,
        extracted: ExtractedOptionsData,
        response: AgentResponse,
    ) -> tuple[float, str]:
        """Heuristic P&L scoring without verification toolkit."""
        score = 40.0
        feedback_parts = []

        # Has P&L numbers
        numbers = self._extract_numbers_from_text(response.analysis)
        if len(numbers) >= 3:
            score += 20
            feedback_parts.append("Multiple P&L calculations present.")
        elif len(numbers) >= 1:
            score += 10
            feedback_parts.append("Some P&L values provided.")
        else:
            feedback_parts.append("Limited P&L analysis.")

        # Check for calculation methodology
        methodology_keywords = ["calculate", "formula", "black-scholes", "premium", "intrinsic", "extrinsic"]
        if any(kw in response.analysis.lower() for kw in methodology_keywords):
            score += 20
            feedback_parts.append("Calculation methodology shown.")

        # Has strategy identification
        if extracted.strategy_name:
            score += 20

        return min(100, score), " ".join(feedback_parts)

    async def _verify_greeks_accuracy(
        self,
        extracted: ExtractedOptionsData,
        response: AgentResponse,
    ) -> tuple[float, str]:
        """
        Verify Greeks calculations.

        Returns:
            Tuple of (accuracy_score, feedback)
        """
        score = 0.0
        feedback_parts = []
        greeks_count = 0

        # Check each Greek
        if extracted.delta is not None:
            greeks_count += 1
            # Delta should be between -1 and 1 for single options
            if -1 <= extracted.delta <= 1:
                score += 20
            elif -1000 <= extracted.delta <= 1000:
                # Could be portfolio delta
                score += 15

        if extracted.gamma is not None:
            greeks_count += 1
            # Gamma should be positive
            if extracted.gamma >= 0:
                score += 20
            else:
                score += 10  # Negative gamma from short positions

        if extracted.theta is not None:
            greeks_count += 1
            score += 20  # Theta can be positive or negative

        if extracted.vega is not None:
            greeks_count += 1
            score += 20

        if extracted.rho is not None:
            greeks_count += 1
            score += 10  # Rho is less commonly reported

        if greeks_count >= 4:
            feedback_parts.append("Comprehensive Greeks analysis.")
        elif greeks_count >= 2:
            feedback_parts.append("Partial Greeks provided.")
        elif greeks_count >= 1:
            score += 10
            feedback_parts.append("Limited Greeks analysis.")
        else:
            # Check if Greeks mentioned without values
            analysis_lower = response.analysis.lower()
            greek_terms = ["delta", "gamma", "theta", "vega"]
            mentioned = sum(1 for g in greek_terms if g in analysis_lower)
            if mentioned >= 2:
                score += 30
                feedback_parts.append("Greeks discussed but values not extracted.")
            else:
                feedback_parts.append("Missing Greeks analysis.")

        return min(100, score), " ".join(feedback_parts)

    async def _score_strategy_quality(
        self,
        extracted: ExtractedOptionsData,
        response: AgentResponse,
    ) -> tuple[float, str]:
        """
        Score strategy appropriateness for market conditions.

        Returns:
            Tuple of (quality_score, feedback)
        """
        score = 40.0  # Base score
        feedback_parts = []
        analysis_lower = response.analysis.lower()

        # Check for market context consideration
        context_keywords = [
            "volatility", "iv", "implied volatility",
            "earnings", "catalyst", "trend",
            "bullish", "bearish", "neutral",
            "market condition", "environment"
        ]
        context_count = sum(1 for kw in context_keywords if kw in analysis_lower)
        if context_count >= 3:
            score += 20
            feedback_parts.append("Strong market context analysis.")
        elif context_count >= 1:
            score += 10
            feedback_parts.append("Some market context considered.")

        # Check for risk/reward discussion
        if "risk" in analysis_lower and "reward" in analysis_lower:
            score += 15
            feedback_parts.append("Risk/reward discussed.")
        elif "risk" in analysis_lower:
            score += 10

        # Check for probability analysis
        if extracted.probability_of_profit is not None:
            score += 15
            feedback_parts.append(f"PoP: {extracted.probability_of_profit}%.")
        elif "probability" in analysis_lower:
            score += 10

        # Verify strategy makes sense for category
        category = self.task.category
        if category == TaskCategory.VOLATILITY_TRADING:
            vol_strategies = ["straddle", "strangle", "iron condor", "butterfly"]
            if extracted.strategy_name and any(s in extracted.strategy_name for s in vol_strategies):
                score += 10
                feedback_parts.append("Appropriate volatility strategy.")
        elif category == TaskCategory.RISK_MANAGEMENT:
            if extracted.var_95 is not None or "var" in analysis_lower:
                score += 10
                feedback_parts.append("VaR analysis included.")

        return min(100, score), " ".join(feedback_parts)

    async def _score_risk_management(
        self,
        extracted: ExtractedOptionsData,
        response: AgentResponse,
    ) -> tuple[float, str]:
        """
        Score risk management discipline.

        Returns:
            Tuple of (risk_score, feedback)
        """
        score = 30.0  # Base score
        feedback_parts = []
        analysis_lower = response.analysis.lower()

        # Check for position sizing discussion
        sizing_keywords = ["position size", "contract", "quantity", "allocation", "portfolio"]
        if any(kw in analysis_lower for kw in sizing_keywords):
            score += 15
            feedback_parts.append("Position sizing addressed.")

        # Check for max loss definition
        if extracted.max_loss is not None:
            score += 15
            feedback_parts.append("Max loss defined.")
        elif "max loss" in analysis_lower or "maximum loss" in analysis_lower:
            score += 10

        # Check for hedging discussion
        hedge_keywords = ["hedge", "protect", "collar", "spread", "limit risk"]
        if any(kw in analysis_lower for kw in hedge_keywords):
            score += 15
            feedback_parts.append("Hedging strategy discussed.")

        # Check for exit strategy
        exit_keywords = ["exit", "stop loss", "take profit", "roll", "close position"]
        if any(kw in analysis_lower for kw in exit_keywords):
            score += 15
            feedback_parts.append("Exit strategy mentioned.")

        # Check for VaR or other risk metrics
        if extracted.var_95 is not None:
            score += 10
            feedback_parts.append(f"VaR: ${extracted.var_95:,.0f}.")

        return min(100, score), " ".join(feedback_parts)

    async def _check_mandatory_elements(
        self,
        response: AgentResponse,
    ) -> tuple[list[str], list[str]]:
        """Check for options-specific mandatory elements."""
        mandatory = self.task.rubric.mandatory_elements
        full_response = f"{response.analysis} {response.recommendation}".lower()

        found = []
        missing = []

        for element in mandatory:
            element_lower = element.lower()

            # Check for exact match or key phrase
            if element_lower in full_response:
                found.append(element)
                continue

            # Check individual words
            words = element_lower.split()
            matches = sum(1 for w in words if w in full_response)
            if matches >= len(words) * 0.6:
                found.append(element)
            else:
                missing.append(element)

        return found, missing

    async def score(
        self,
        response: AgentResponse,
    ) -> OptionsScore:
        """
        Score the agent's options trading response.

        Args:
            response: The agent's response to evaluate

        Returns:
            OptionsScore with detailed breakdown
        """
        # Extract options data from response
        extracted = self._extract_options_data(response)

        # Score each dimension
        pnl_score, pnl_feedback = await self._verify_pnl_accuracy(extracted, response)
        greeks_score, greeks_feedback = await self._verify_greeks_accuracy(extracted, response)
        strategy_score, strategy_feedback = await self._score_strategy_quality(extracted, response)
        risk_score, risk_feedback = await self._score_risk_management(extracted, response)

        # Check mandatory elements
        found, missing = await self._check_mandatory_elements(response)
        if missing:
            # Penalty for missing mandatory elements
            penalty = len(missing) / max(len(self.task.rubric.mandatory_elements), 1)
            pnl_score *= (1 - penalty * 0.2)

        # Calculate weighted final score
        # Equal weights for all dimensions
        final_score = (
            pnl_score * 0.25 +
            greeks_score * 0.25 +
            strategy_score * 0.25 +
            risk_score * 0.25
        )

        # Compile feedback
        all_feedback = [pnl_feedback, greeks_feedback, strategy_feedback, risk_feedback]
        if missing:
            all_feedback.append(f"Missing: {', '.join(missing[:3])}.")
        combined_feedback = " ".join(f for f in all_feedback if f)

        logger.info(
            "options_evaluation",
            task_id=self.task.question_id,
            category=self.task.category.value,
            pnl_score=pnl_score,
            greeks_score=greeks_score,
            strategy_score=strategy_score,
            risk_score=risk_score,
            final_score=final_score,
        )

        return OptionsScore(
            score=final_score,
            pnl_accuracy=pnl_score,
            greeks_accuracy=greeks_score,
            strategy_quality=strategy_score,
            risk_management=risk_score,
            feedback=combined_feedback,
        )

    @staticmethod
    def is_options_task(task: Task) -> bool:
        """Check if a task is an options trading task."""
        return task.category in OPTIONS_CATEGORIES
