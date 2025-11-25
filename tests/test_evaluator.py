"""
Tests for the CIO-Agent evaluation system.
"""

import pytest
from datetime import datetime

from cio_agent.models import (
    Task,
    TaskCategory,
    TaskDifficulty,
    GroundTruth,
    FinancialData,
    AgentResponse,
    AlphaScore,
    TaskRubric,
)
from cio_agent.task_generator import DynamicTaskGenerator, FABDataset
from evaluators.macro import MacroEvaluator
from evaluators.fundamental import FundamentalEvaluator
from evaluators.cost_tracker import CostTracker


class TestAlphaScore:
    """Tests for Alpha Score calculation."""

    def test_calculate_basic(self):
        """Test basic Alpha Score calculation."""
        score = AlphaScore.calculate(
            role_score=85.0,
            debate_multiplier=1.2,
            cost_usd=2.50,
            lookahead_penalty=0.0,
        )

        assert score.score > 0
        assert score.role_score == 85.0
        assert score.debate_multiplier == 1.2
        assert score.cost_usd == 2.50
        assert score.lookahead_penalty == 0.0

    def test_higher_score_lower_cost(self):
        """Test that lower cost leads to higher Alpha Score."""
        score_low_cost = AlphaScore.calculate(
            role_score=80.0,
            debate_multiplier=1.0,
            cost_usd=1.0,
            lookahead_penalty=0.0,
        )

        score_high_cost = AlphaScore.calculate(
            role_score=80.0,
            debate_multiplier=1.0,
            cost_usd=10.0,
            lookahead_penalty=0.0,
        )

        assert score_low_cost.score > score_high_cost.score

    def test_debate_multiplier_impact(self):
        """Test debate multiplier impact on score."""
        score_strong = AlphaScore.calculate(
            role_score=70.0,
            debate_multiplier=1.2,
            cost_usd=2.0,
            lookahead_penalty=0.0,
        )

        score_weak = AlphaScore.calculate(
            role_score=70.0,
            debate_multiplier=0.5,
            cost_usd=2.0,
            lookahead_penalty=0.0,
        )

        assert score_strong.score > score_weak.score

    def test_penalty_impact(self):
        """Test look-ahead penalty impact."""
        score_clean = AlphaScore.calculate(
            role_score=80.0,
            debate_multiplier=1.0,
            cost_usd=2.0,
            lookahead_penalty=0.0,
        )

        score_penalized = AlphaScore.calculate(
            role_score=80.0,
            debate_multiplier=1.0,
            cost_usd=2.0,
            lookahead_penalty=0.3,
        )

        assert score_clean.score > score_penalized.score


class TestMacroEvaluator:
    """Tests for Macro Evaluator."""

    def test_score_with_matching_themes(self):
        """Test scoring with matching themes."""
        ground_truth = GroundTruth(
            macro_thesis="AI chip demand is driving growth",
            key_themes=["AI adoption", "cloud growth", "chip demand"],
        )

        evaluator = MacroEvaluator(ground_truth)

        analysis = """
        The company is benefiting from strong AI adoption trends.
        Cloud growth continues to drive revenue expansion.
        Chip demand remains robust across data center customers.
        """

        result = evaluator.score(analysis)

        assert result.score > 50  # Should be positive
        assert result.theme_coverage > 0.5
        assert len(result.themes_identified) > 0

    def test_score_with_no_matching_themes(self):
        """Test scoring with no matching themes."""
        ground_truth = GroundTruth(
            macro_thesis="AI chip demand is driving growth",
            key_themes=["AI adoption", "cloud growth"],
        )

        evaluator = MacroEvaluator(ground_truth)

        analysis = "The company sells products to customers."

        result = evaluator.score(analysis)

        assert result.theme_coverage < 0.5
        assert len(result.themes_missed) > 0


class TestFundamentalEvaluator:
    """Tests for Fundamental Evaluator."""

    def test_exact_match(self):
        """Test scoring with exact financial matches."""
        ground_truth = GroundTruth(
            macro_thesis="",
            financials=FinancialData(
                revenue=100_000_000_000,
                net_income=20_000_000_000,
                gross_margin=0.40,
            ),
        )

        evaluator = FundamentalEvaluator(ground_truth)

        agent_financials = FinancialData(
            revenue=100_000_000_000,
            net_income=20_000_000_000,
            gross_margin=0.40,
        )

        result = evaluator.score(agent_financials)

        assert result.score == 100.0
        assert result.correct_fields == result.total_fields

    def test_within_tolerance(self):
        """Test scoring within tolerance."""
        ground_truth = GroundTruth(
            macro_thesis="",
            financials=FinancialData(revenue=100_000_000_000),
        )

        evaluator = FundamentalEvaluator(ground_truth, tolerance=0.01)

        agent_financials = FinancialData(
            revenue=100_500_000_000,  # 0.5% off
        )

        result = evaluator.score(agent_financials)

        assert result.score == 100.0  # Within 1% tolerance


class TestCostTracker:
    """Tests for Cost Tracker."""

    def test_llm_cost_tracking(self):
        """Test LLM cost tracking."""
        tracker = CostTracker()

        tracker.add_llm_call(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            purpose="analysis",
        )

        breakdown = tracker.get_breakdown()

        assert breakdown.llm_cost_usd > 0
        assert breakdown.llm_calls == 1
        assert breakdown.total_input_tokens == 1000
        assert breakdown.total_output_tokens == 500

    def test_total_cost_calculation(self):
        """Test total cost calculation."""
        tracker = CostTracker()

        tracker.add_llm_call("gpt-4o", 1000, 500, "test")

        from cio_agent.models import ToolCall
        from datetime import datetime

        tracker.add_tool_call(ToolCall(
            tool_name="test",
            params={},
            timestamp=datetime.utcnow(),
            response_tokens=5000,
        ))

        breakdown = tracker.get_breakdown()

        assert breakdown.total_cost_usd == breakdown.llm_cost_usd + breakdown.tool_cost_usd


class TestDynamicTaskGenerator:
    """Tests for Dynamic Task Generator."""

    def test_load_sample_questions(self):
        """Test loading sample questions."""
        dataset = FABDataset.load_sample_questions()

        assert len(dataset.questions) > 0
        assert any(q.category == TaskCategory.QUANTITATIVE_RETRIEVAL for q in dataset.questions)

    @pytest.mark.asyncio
    async def test_generate_task(self):
        """Test generating a dynamic task."""
        generator = DynamicTaskGenerator()

        task = await generator.generate_task(
            template_id="FAB_001",
            simulation_date=datetime(2023, 1, 1),
        )

        assert task is not None
        assert task.question_id.startswith("FAB_001_variant")
        assert task.simulation_date == datetime(2023, 1, 1)

    def test_sample_similar_company(self):
        """Test company sampling within sector."""
        generator = DynamicTaskGenerator()

        # AAPL is in technology sector
        similar = generator.sample_similar_company("AAPL")

        assert similar != "AAPL"
        # Should be another tech company
        tech_tickers = ["MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "CRM", "ORCL", "ADBE"]
        assert similar in tech_tickers


class TestTaskModel:
    """Tests for Task model."""

    def test_is_numerical_task(self):
        """Test numerical task detection."""
        numerical_task = Task(
            question_id="test",
            category=TaskCategory.NUMERICAL_REASONING,
            question="Calculate X",
            ticker="AAPL",
            fiscal_year=2023,
            simulation_date=datetime.now(),
            ground_truth=GroundTruth(macro_thesis=""),
        )

        non_numerical_task = Task(
            question_id="test",
            category=TaskCategory.QUALITATIVE_RETRIEVAL,
            question="Describe X",
            ticker="AAPL",
            fiscal_year=2023,
            simulation_date=datetime.now(),
            ground_truth=GroundTruth(macro_thesis=""),
        )

        assert numerical_task.is_numerical_task is True
        assert non_numerical_task.is_numerical_task is False
