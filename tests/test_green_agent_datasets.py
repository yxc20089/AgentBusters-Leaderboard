"""
Unit tests for Green Agent dataset integration.

Tests the integration of BizFinBench and public.csv datasets into the Green Agent.
Requires Python 3.10+ for match statement syntax.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List


class TestGreenAgentDatasetConfig:
    """Test Green Agent dataset configuration."""

    def test_default_synthetic_mode(self):
        """Test default initialization uses synthetic mode."""
        from cio_agent.green_agent import GreenAgent
        
        agent = GreenAgent(dataset_type="synthetic")
        
        assert agent.dataset_type == "synthetic"
        assert agent._examples is None
        assert agent.dataset_provider is None
        assert agent.dataset_evaluator is None

    def test_bizfinbench_mode_initialization(self):
        """Test BizFinBench mode initializes provider and evaluator."""
        from cio_agent.green_agent import GreenAgent
        from cio_agent.local_datasets import BizFinBenchProvider
        from evaluators import BizFinBenchEvaluator
        
        agent = GreenAgent(
            dataset_type="bizfinbench",
            dataset_path="data/BizFinBench.v2",
            task_type="event_logic_reasoning",
            language="en",
            limit=3,
        )
        
        assert agent.dataset_type == "bizfinbench"
        assert agent.dataset_path == "data/BizFinBench.v2"
        assert agent.task_type == "event_logic_reasoning"
        assert agent.language == "en"
        assert agent.limit == 3
        assert isinstance(agent.dataset_provider, BizFinBenchProvider)
        assert isinstance(agent.dataset_evaluator, BizFinBenchEvaluator)
        assert agent._examples is not None
        assert len(agent._examples) == 3

    def test_public_csv_mode_initialization(self):
        """Test public_csv mode initializes provider and evaluator."""
        from cio_agent.green_agent import GreenAgent
        from cio_agent.local_datasets import CsvFinanceDatasetProvider
        from evaluators import PublicCsvEvaluator
        
        agent = GreenAgent(
            dataset_type="public_csv",
            dataset_path="finance-agent/data/public.csv",
            limit=3,
        )
        
        assert agent.dataset_type == "public_csv"
        assert agent.dataset_path == "finance-agent/data/public.csv"
        assert isinstance(agent.dataset_provider, CsvFinanceDatasetProvider)
        assert isinstance(agent.dataset_evaluator, PublicCsvEvaluator)
        assert agent._examples is not None
        assert len(agent._examples) == 3

    def test_synthetic_with_questions(self):
        """Test synthetic mode with provided questions."""
        from cio_agent.green_agent import GreenAgent
        
        questions = [
            {"question_id": "q1", "question": "Test?", "category": "Beat or Miss"},
        ]
        
        agent = GreenAgent(
            synthetic_questions=questions,
            dataset_type="synthetic",
        )
        
        assert len(agent.synthetic_questions) == 1
        assert agent.dataset_type == "synthetic"


class TestGreenAgentExecutorConfig:
    """Test Green Agent Executor configuration."""

    def test_executor_stores_dataset_config(self):
        """Test executor stores all dataset configuration."""
        from cio_agent.green_executor import GreenAgentExecutor
        
        executor = GreenAgentExecutor(
            synthetic_questions=None,
            dataset_type="bizfinbench",
            dataset_path="data/BizFinBench.v2",
            task_type="event_logic_reasoning",
            language="en",
            limit=5,
        )
        
        assert executor.dataset_type == "bizfinbench"
        assert executor.dataset_path == "data/BizFinBench.v2"
        assert executor.task_type == "event_logic_reasoning"
        assert executor.language == "en"
        assert executor.limit == 5

    def test_executor_default_values(self):
        """Test executor default values."""
        from cio_agent.green_executor import GreenAgentExecutor
        
        executor = GreenAgentExecutor()
        
        assert executor.dataset_type == "synthetic"
        assert executor.dataset_path is None
        assert executor.task_type is None
        assert executor.language == "en"
        assert executor.limit is None
        assert executor.synthetic_questions is None

    def test_executor_public_csv_config(self):
        """Test executor with public_csv configuration."""
        from cio_agent.green_executor import GreenAgentExecutor
        
        executor = GreenAgentExecutor(
            dataset_type="public_csv",
            dataset_path="finance-agent/data/public.csv",
            limit=10,
        )
        
        assert executor.dataset_type == "public_csv"
        assert executor.dataset_path == "finance-agent/data/public.csv"
        assert executor.limit == 10


class TestGreenAgentMethods:
    """Test Green Agent methods."""

    def test_validate_request_missing_roles(self):
        """Test validate_request rejects missing roles."""
        from cio_agent.green_agent import GreenAgent, EvalRequest
        
        agent = GreenAgent(dataset_type="synthetic")
        
        request = EvalRequest(
            participants={},
            config={}
        )
        
        ok, msg = agent.validate_request(request)
        assert ok is False
        assert "Missing roles" in msg

    def test_validate_request_success(self):
        """Test validate_request accepts valid request."""
        from cio_agent.green_agent import GreenAgent, EvalRequest
        
        agent = GreenAgent(dataset_type="synthetic")
        
        request = EvalRequest(
            participants={"purple_agent": "http://localhost:9110"},
            config={"num_tasks": 1}
        )
        
        ok, msg = agent.validate_request(request)
        assert ok is True
        assert msg == "ok"

    def test_extract_recommendation(self):
        """Test _extract_recommendation method."""
        from cio_agent.green_agent import GreenAgent
        
        agent = GreenAgent(dataset_type="synthetic")
        
        assert agent._extract_recommendation("The company beat expectations") == "Beat"
        assert agent._extract_recommendation("They missed the target") == "Miss"
        assert agent._extract_recommendation("I recommend to buy") == "Buy"
        assert agent._extract_recommendation("Sell the stock") == "Sell"
        assert agent._extract_recommendation("Hold your position") == "Hold"
        assert agent._extract_recommendation("No clear recommendation") == "Unknown"


class TestDatasetEvaluatorIntegration:
    """Test dataset evaluator integration with GreenAgent."""

    def test_bizfinbench_evaluator_with_agent(self):
        """Test BizFinBench evaluator integration with GreenAgent."""
        from cio_agent.green_agent import GreenAgent
        
        agent = GreenAgent(
            dataset_type="bizfinbench",
            dataset_path="data/BizFinBench.v2",
            task_type="event_logic_reasoning",
            language="en",
            limit=1,
        )
        
        example = agent._examples[0]
        result = agent.dataset_evaluator.evaluate(
            predicted=example.answer,
            expected=example.answer,
            task_type="event_logic_reasoning",
        )
        
        assert result.score == 1.0
        assert result.is_correct

    def test_public_csv_evaluator_with_agent(self):
        """Test public.csv evaluator integration with GreenAgent."""
        from cio_agent.green_agent import GreenAgent
        
        agent = GreenAgent(
            dataset_type="public_csv",
            dataset_path="finance-agent/data/public.csv",
            limit=1,
        )
        
        example = agent._examples[0]
        result = agent.dataset_evaluator.evaluate(
            predicted=example.answer,
            expected=example.answer,
            rubric=None,
        )
        
        assert result.score >= 0.0


class TestDatasetTypeValidation:
    """Test dataset type validation."""

    def test_synthetic_without_path(self):
        """Test synthetic mode works without dataset_path."""
        from cio_agent.green_agent import GreenAgent
        
        agent = GreenAgent(dataset_type="synthetic")
        
        assert agent.dataset_provider is None
        assert agent.dataset_evaluator is None

    def test_bizfinbench_without_path_no_provider(self):
        """Test BizFinBench without path doesn't initialize provider."""
        from cio_agent.green_agent import GreenAgent
        
        agent = GreenAgent(dataset_type="bizfinbench")
        
        assert agent.dataset_provider is None

    def test_all_supported_dataset_types(self):
        """Test all supported dataset types are handled."""
        from cio_agent.green_agent import GreenAgent
        
        supported_types = ["synthetic", "bizfinbench", "public_csv"]
        
        for dtype in supported_types:
            agent = GreenAgent(dataset_type=dtype)
            assert agent.dataset_type == dtype


class TestEvaluationResultFormat:
    """Test evaluation result format from dataset evaluators."""

    def test_bizfinbench_result_has_required_fields(self):
        """Test BizFinBench evaluation result contains required fields."""
        from cio_agent.green_agent import GreenAgent
        from evaluators import EvalResult
        
        agent = GreenAgent(
            dataset_type="bizfinbench",
            dataset_path="data/BizFinBench.v2",
            task_type="event_logic_reasoning",
            language="en",
            limit=1,
        )
        
        result = agent.dataset_evaluator.evaluate(
            predicted="2,1,4,3",
            expected="2,1,4,3",
            task_type="event_logic_reasoning",
        )
        
        assert isinstance(result, EvalResult)
        assert hasattr(result, 'score')
        assert hasattr(result, 'is_correct')
        assert hasattr(result, 'feedback')
        assert hasattr(result, 'details')

    def test_public_csv_result_has_required_fields(self):
        """Test public.csv evaluation result contains required fields."""
        from cio_agent.green_agent import GreenAgent
        from evaluators import EvalResult
        
        agent = GreenAgent(
            dataset_type="public_csv",
            dataset_path="finance-agent/data/public.csv",
            limit=1,
        )
        
        rubric = [{"operator": "correctness", "criteria": "test"}]
        result = agent.dataset_evaluator.evaluate(
            predicted="test is here",
            rubric=rubric,
        )
        
        assert isinstance(result, EvalResult)
        assert hasattr(result, 'score')
        assert hasattr(result, 'correct_count')
        assert hasattr(result, 'total_count')


class TestBatchEvaluation:
    """Test batch evaluation with datasets."""

    def test_bizfinbench_batch_with_limit(self):
        """Test BizFinBench loads correct number of examples with limit."""
        from cio_agent.green_agent import GreenAgent
        
        agent = GreenAgent(
            dataset_type="bizfinbench",
            dataset_path="data/BizFinBench.v2",
            task_type="user_sentiment_analysis",
            language="en",
            limit=5,
        )
        
        assert len(agent._examples) == 5
        
        results = []
        for example in agent._examples:
            result = agent.dataset_evaluator.evaluate(
                predicted=example.answer,
                expected=example.answer,
                task_type="user_sentiment_analysis",
            )
            results.append(result)
        
        assert len(results) == 5
        assert all(r.score == 1.0 for r in results)

    def test_public_csv_batch_with_limit(self):
        """Test public.csv loads correct number of examples with limit."""
        from cio_agent.green_agent import GreenAgent
        
        agent = GreenAgent(
            dataset_type="public_csv",
            dataset_path="finance-agent/data/public.csv",
            limit=5,
        )
        
        assert len(agent._examples) == 5
        
        for example in agent._examples:
            assert hasattr(example, 'question')
            assert hasattr(example, 'answer')
            assert hasattr(example, 'category')


class TestConvertSyntheticToTasks:
    """Test _convert_synthetic_to_tasks method."""

    def test_convert_synthetic_to_tasks(self):
        """Test converting synthetic questions to FABTask objects."""
        from cio_agent.green_agent import GreenAgent
        from cio_agent.models import Task as FABTask
        
        questions = [
            {
                "question_id": "q1",
                "question": "What is the revenue?",
                "category": "Quantitative Retrieval",
                "difficulty": "medium",
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "ground_truth_formatted": "1.5 billion",
            },
            {
                "question_id": "q2",
                "question": "Did company beat?",
                "category": "Beat or Miss",
                "difficulty": "hard",
                "ticker": "NVDA",
                "fiscal_year": 2025,
                "ground_truth_formatted": "Beat",
            },
        ]
        
        agent = GreenAgent(
            synthetic_questions=questions,
            dataset_type="synthetic",
        )
        
        tasks = agent._convert_synthetic_to_tasks(num_tasks=2)
        
        assert len(tasks) == 2
        assert all(isinstance(t, FABTask) for t in tasks)
        assert tasks[0].question_id == "q1"
        assert tasks[1].question_id == "q2"

    def test_convert_synthetic_respects_limit(self):
        """Test that num_tasks limit is respected."""
        from cio_agent.green_agent import GreenAgent
        
        questions = [{"question_id": f"q{i}", "question": f"Q{i}?"} for i in range(10)]
        
        agent = GreenAgent(
            synthetic_questions=questions,
            dataset_type="synthetic",
        )
        
        tasks = agent._convert_synthetic_to_tasks(num_tasks=3)
        assert len(tasks) == 3
