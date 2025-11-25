"""
Adversarial Debate Manager for CIO-Agent.

Conducts adversarial debate to test agent robustness:
1. Generates counter-arguments to the agent's thesis
2. Evaluates the quality of the agent's rebuttal
3. Assigns debate multiplier (0.5x - 1.2x)
"""

from typing import Any, Optional

import structlog
from pydantic import BaseModel, Field

from cio_agent.models import (
    DebateResult,
    DebateRebuttal,
    ConvictionLevel,
    Task,
    AgentResponse,
    FinancialData,
)

logger = structlog.get_logger()


class CounterArgumentConfig(BaseModel):
    """Configuration for counter-argument generation."""
    risk_focus: list[str] = Field(
        default_factory=lambda: [
            "margin_compression",
            "competitive_threats",
            "regulatory_risks",
            "valuation_concerns",
            "cyclical_risks",
            "execution_risks",
        ]
    )
    tone: str = "skeptical_professional"
    max_length: int = 500


class AdversarialDebateManager:
    """
    Manages the adversarial debate phase of evaluation.

    The debate tests agent conviction by:
    1. Generating plausible counter-arguments based on financial data
    2. Sending challenges to the agent
    3. Evaluating rebuttal quality
    4. Assigning debate multiplier

    Debate Multipliers:
    - 1.2x: Agent provides NEW evidence and successfully defends thesis
    - 1.0x: Agent repeats previous evidence without new insights
    - 0.5x: Agent hallucinates, contradicts itself, or immediately concedes
    """

    # Debate multiplier values
    MULTIPLIER_STRONG = 1.2
    MULTIPLIER_NEUTRAL = 1.0
    MULTIPLIER_WEAK = 0.5

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        config: Optional[CounterArgumentConfig] = None,
    ):
        """
        Initialize the debate manager.

        Args:
            llm_client: LLM client for generating counter-arguments
            config: Configuration for counter-argument generation
        """
        self.llm_client = llm_client
        self.config = config or CounterArgumentConfig()

    def _generate_heuristic_counter(
        self,
        agent_thesis: str,
        financial_data: FinancialData,
        task: Task,
    ) -> str:
        """
        Generate a counter-argument using heuristics (fallback when no LLM).
        """
        counter_parts = []

        # Analyze thesis for keywords to challenge
        thesis_lower = agent_thesis.lower()

        # Challenge bullish thesis
        if any(word in thesis_lower for word in ["buy", "bullish", "strong", "growth", "outperform"]):
            counter_parts.append(
                "The bullish outlook may be overly optimistic. Consider the following risks:"
            )

            if financial_data.gross_margin and financial_data.gross_margin < 0.4:
                counter_parts.append(
                    f"- Gross margin of {financial_data.gross_margin*100:.1f}% is below industry average, "
                    "indicating pricing pressure or cost challenges."
                )

            if financial_data.pe_ratio and financial_data.pe_ratio > 25:
                counter_parts.append(
                    f"- P/E ratio of {financial_data.pe_ratio:.1f}x implies high growth expectations "
                    "that may not materialize in a slowing economy."
                )

            counter_parts.append(
                "- Competitive dynamics in the sector are intensifying, with new entrants "
                "and existing players expanding capacity."
            )

        # Challenge bearish thesis
        elif any(word in thesis_lower for word in ["sell", "bearish", "weak", "decline", "underperform"]):
            counter_parts.append(
                "The bearish thesis may overlook key positives:"
            )

            if financial_data.operating_cash_flow and financial_data.operating_cash_flow > 0:
                counter_parts.append(
                    "- Positive operating cash flow provides financial flexibility "
                    "and runway for strategic initiatives."
                )

            counter_parts.append(
                "- Management has shown ability to adapt to challenging conditions historically."
            )

        # Generic challenge
        else:
            counter_parts.append(
                "Several factors warrant closer scrutiny of this analysis:"
            )
            counter_parts.append(
                "- How sustainable is the current performance trajectory?"
            )
            counter_parts.append(
                "- What are the key risks that could derail the thesis?"
            )

        counter_parts.append(
            "\nDefend your thesis with additional evidence or revise your recommendation."
        )

        return "\n".join(counter_parts)

    async def generate_counter_argument(
        self,
        agent_thesis: str,
        financial_data: FinancialData,
        task: Task,
    ) -> str:
        """
        Generate a plausible counter-thesis using LLM or heuristics.

        Args:
            agent_thesis: The agent's initial thesis/recommendation
            financial_data: Available financial data for the task
            task: The evaluation task

        Returns:
            Counter-argument challenging the agent's thesis
        """
        if not self.llm_client:
            return self._generate_heuristic_counter(agent_thesis, financial_data, task)

        # Format financial data for prompt
        fin_summary = []
        if financial_data.revenue:
            fin_summary.append(f"Revenue: ${financial_data.revenue:,.0f}")
        if financial_data.net_income:
            fin_summary.append(f"Net Income: ${financial_data.net_income:,.0f}")
        if financial_data.gross_margin:
            fin_summary.append(f"Gross Margin: {financial_data.gross_margin*100:.1f}%")
        if financial_data.operating_margin:
            fin_summary.append(f"Operating Margin: {financial_data.operating_margin*100:.1f}%")
        if financial_data.pe_ratio:
            fin_summary.append(f"P/E Ratio: {financial_data.pe_ratio:.1f}x")

        fin_data_str = "\n".join(fin_summary) if fin_summary else "Limited financial data available"

        prompt = f"""
You are a skeptical Risk Manager reviewing an investment recommendation.
Your role is to generate a challenging counter-argument to test the analyst's conviction.

TASK: {task.question}
TICKER: {task.ticker}
FISCAL YEAR: {task.fiscal_year}

AGENT'S THESIS:
{agent_thesis}

AVAILABLE FINANCIAL DATA:
{fin_data_str}

Generate a critical counter-argument that challenges the thesis using data-driven reasoning.
Focus on risks the analyst may have overlooked, such as:
- Margin compression or profitability concerns
- Competitive threats and market dynamics
- Regulatory or geopolitical risks
- Valuation concerns relative to growth
- Cyclical or macro headwinds

Your counter-argument should be:
- Specific and data-driven where possible
- Professional in tone
- Challenging but fair
- Between 100-300 words

End with: "Defend your thesis with additional evidence or revise your recommendation."
"""

        try:
            counter = await self.llm_client.generate(prompt)
            logger.info("counter_argument_generated", length=len(counter))
            return counter
        except Exception as e:
            logger.warning("llm_counter_failed", error=str(e))
            return self._generate_heuristic_counter(agent_thesis, financial_data, task)

    def _detect_hallucination(
        self,
        rebuttal: str,
        financial_data: FinancialData,
    ) -> bool:
        """
        Detect potential hallucinations in the rebuttal.

        Checks for:
        - Specific numbers that don't match available data
        - References to data that wasn't provided
        """
        import re

        # Extract numbers from rebuttal
        numbers = re.findall(r"\$?([\d,]+(?:\.\d+)?)[%BMK]?", rebuttal)

        # If rebuttal claims very specific large numbers, check against data
        for num_str in numbers:
            try:
                num = float(num_str.replace(",", ""))
                # Skip small numbers (percentages, ratios, etc.)
                if num < 100:
                    continue

                # Check if number roughly matches any known financial figure
                known_values = [
                    financial_data.revenue,
                    financial_data.net_income,
                    financial_data.total_assets,
                    financial_data.market_cap,
                    financial_data.operating_cash_flow,
                ]

                # If the number is very specific but doesn't match anything
                # it might be hallucinated
                if not any(v and abs(num - v) / v < 0.2 for v in known_values if v):
                    # Large specific number with no match
                    if num > 1000000:  # More than $1M
                        logger.warning("potential_hallucination", number=num)
                        # Don't immediately flag, just note it
            except ValueError:
                continue

        return False  # Be conservative about flagging hallucinations

    def _detect_contradiction(
        self,
        original_thesis: str,
        rebuttal: str,
    ) -> bool:
        """
        Detect if the rebuttal contradicts the original thesis.
        """
        thesis_lower = original_thesis.lower()
        rebuttal_lower = rebuttal.lower()

        # Check for explicit stance reversal
        bullish_words = ["buy", "bullish", "strong", "outperform", "positive"]
        bearish_words = ["sell", "bearish", "weak", "underperform", "negative"]

        thesis_bullish = any(w in thesis_lower for w in bullish_words)
        thesis_bearish = any(w in thesis_lower for w in bearish_words)

        rebuttal_bullish = any(w in rebuttal_lower for w in bullish_words)
        rebuttal_bearish = any(w in rebuttal_lower for w in bearish_words)

        # Direct contradiction
        if thesis_bullish and rebuttal_bearish and not rebuttal_bullish:
            return True
        if thesis_bearish and rebuttal_bullish and not rebuttal_bearish:
            return True

        return False

    def _detect_immediate_concession(self, rebuttal: str) -> bool:
        """
        Detect if the agent immediately concedes without defending.
        """
        rebuttal_lower = rebuttal.lower()

        concession_phrases = [
            "you're right",
            "you are right",
            "i agree",
            "valid point",
            "i concede",
            "perhaps you're correct",
            "fair point, i'll revise",
            "let me reconsider",
            "maybe i was wrong",
        ]

        # Check for concession without substantial defense
        has_concession = any(phrase in rebuttal_lower for phrase in concession_phrases)

        # Short response with concession = immediate concession
        if has_concession and len(rebuttal) < 200:
            return True

        return False

    def _detect_new_evidence(
        self,
        original_thesis: str,
        rebuttal: str,
    ) -> bool:
        """
        Detect if the rebuttal provides new evidence not in the original thesis.
        """
        import re

        # Extract key data points from each
        original_numbers = set(re.findall(r"\d+\.?\d*%?", original_thesis))
        rebuttal_numbers = set(re.findall(r"\d+\.?\d*%?", rebuttal))

        new_numbers = rebuttal_numbers - original_numbers

        # Extract key phrases/terms
        original_terms = set(original_thesis.lower().split())
        rebuttal_terms = set(rebuttal.lower().split())

        # Financial/analytical terms that indicate new evidence
        evidence_terms = {
            "guidance", "management", "quarter", "yoy", "sequential",
            "segment", "margin", "growth", "benchmark", "competitor",
            "market share", "filing", "10-k", "10-q", "call",
            "estimate", "consensus", "outlook", "forecast",
        }

        new_evidence_terms = rebuttal_terms.intersection(evidence_terms) - original_terms.intersection(evidence_terms)

        # New numbers or new evidence terms suggest new evidence
        return len(new_numbers) >= 2 or len(new_evidence_terms) >= 2

    async def score_rebuttal(
        self,
        counter_argument: str,
        rebuttal: str,
        original_thesis: str,
        financial_data: FinancialData,
    ) -> tuple[float, ConvictionLevel, dict[str, bool]]:
        """
        Score the quality of the agent's rebuttal.

        Args:
            counter_argument: The challenge sent to the agent
            rebuttal: The agent's defense
            original_thesis: The original thesis being defended
            financial_data: Available financial data

        Returns:
            Tuple of (multiplier, conviction_level, analysis_flags)
        """
        # Detect various quality indicators
        hallucination = self._detect_hallucination(rebuttal, financial_data)
        contradiction = self._detect_contradiction(original_thesis, rebuttal)
        immediate_concession = self._detect_immediate_concession(rebuttal)
        new_evidence = self._detect_new_evidence(original_thesis, rebuttal)

        analysis_flags = {
            "hallucination_detected": hallucination,
            "contradiction_detected": contradiction,
            "immediate_concession": immediate_concession,
            "new_evidence_provided": new_evidence,
        }

        # Determine multiplier
        if hallucination or contradiction or immediate_concession:
            multiplier = self.MULTIPLIER_WEAK
            conviction = ConvictionLevel.LOW
            logger.info(
                "weak_rebuttal",
                hallucination=hallucination,
                contradiction=contradiction,
                concession=immediate_concession,
            )
        elif new_evidence:
            multiplier = self.MULTIPLIER_STRONG
            conviction = ConvictionLevel.HIGH
            logger.info("strong_rebuttal", new_evidence=True)
        else:
            multiplier = self.MULTIPLIER_NEUTRAL
            conviction = ConvictionLevel.MEDIUM
            logger.info("neutral_rebuttal")

        # If LLM is available, get more nuanced scoring
        if self.llm_client and not (hallucination or contradiction or immediate_concession):
            try:
                llm_multiplier, llm_conviction = await self._llm_score_rebuttal(
                    counter_argument, rebuttal, financial_data
                )
                # Use LLM score if it differs significantly
                if abs(llm_multiplier - multiplier) >= 0.2:
                    multiplier = llm_multiplier
                    conviction = llm_conviction
            except Exception as e:
                logger.warning("llm_rebuttal_scoring_failed", error=str(e))

        return multiplier, conviction, analysis_flags

    async def _llm_score_rebuttal(
        self,
        counter_argument: str,
        rebuttal: str,
        financial_data: FinancialData,
    ) -> tuple[float, ConvictionLevel]:
        """
        Use LLM to score rebuttal quality.
        """
        prompt = f"""
Evaluate the quality of this investment rebuttal.

CHALLENGE:
{counter_argument}

REBUTTAL:
{rebuttal}

Score the rebuttal quality:
- 1.2: Agent provides NEW evidence from data and successfully defends thesis
- 1.0: Agent repeats previous evidence without new insights
- 0.5: Agent hallucinates (cites non-existent data), contradicts itself, or immediately concedes

Respond with ONLY a single number: 0.5, 1.0, or 1.2
"""

        result = await self.llm_client.generate(prompt)
        result = result.strip()

        try:
            multiplier = float(result)
            if multiplier not in [0.5, 1.0, 1.2]:
                multiplier = 1.0
        except ValueError:
            multiplier = 1.0

        conviction = (
            ConvictionLevel.HIGH if multiplier == 1.2
            else ConvictionLevel.LOW if multiplier == 0.5
            else ConvictionLevel.MEDIUM
        )

        return multiplier, conviction

    async def conduct_debate(
        self,
        task: Task,
        agent_response: AgentResponse,
        agent_rebuttal: DebateRebuttal,
    ) -> DebateResult:
        """
        Conduct the full debate and return results.

        Args:
            task: The evaluation task
            agent_response: The agent's initial response
            agent_rebuttal: The agent's rebuttal to the challenge

        Returns:
            DebateResult with multiplier and analysis
        """
        # Generate counter-argument
        counter_argument = await self.generate_counter_argument(
            agent_thesis=agent_response.recommendation,
            financial_data=agent_response.extracted_financials,
            task=task,
        )

        # Score the rebuttal
        multiplier, conviction, flags = await self.score_rebuttal(
            counter_argument=counter_argument,
            rebuttal=agent_rebuttal.defense,
            original_thesis=agent_response.recommendation,
            financial_data=agent_response.extracted_financials,
        )

        # Generate feedback
        feedback_parts = []
        if conviction == ConvictionLevel.HIGH:
            feedback_parts.append("Strong defense with new evidence.")
        elif conviction == ConvictionLevel.LOW:
            if flags["hallucination_detected"]:
                feedback_parts.append("Rebuttal contains potentially hallucinated data.")
            if flags["contradiction_detected"]:
                feedback_parts.append("Rebuttal contradicts original thesis.")
            if flags["immediate_concession"]:
                feedback_parts.append("Agent conceded without substantial defense.")
        else:
            feedback_parts.append("Adequate defense but no new evidence provided.")

        logger.info(
            "debate_completed",
            multiplier=multiplier,
            conviction=conviction.value,
            flags=flags,
        )

        return DebateResult(
            counter_argument=counter_argument,
            agent_rebuttal=agent_rebuttal.defense,
            debate_multiplier=multiplier,
            conviction_level=conviction,
            new_evidence_provided=flags["new_evidence_provided"],
            hallucination_detected=flags["hallucination_detected"],
            immediate_concession=flags["immediate_concession"],
            feedback=" ".join(feedback_parts),
        )
