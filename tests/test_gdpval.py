"""
Unit tests for GDPVal dataset integration.

Tests cover:
- GDPValDatasetConfig validation
- Dataset loading from HuggingFace
- GDPValEvaluator scoring
- Integration with GreenAgent
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestGDPValDatasetConfig:
    """Test GDPValDatasetConfig validation."""

    def test_default_config(self):
        """Test default GDPVal config values."""
        from cio_agent.eval_config import GDPValDatasetConfig

        config = GDPValDatasetConfig()

        assert config.type == "gdpval"
        assert config.hf_dataset == "openai/gdpval"
        assert config.sectors is None  # All sectors
        assert config.occupations is None  # All occupations
        assert config.limit is None
        assert config.shuffle is True
        assert config.include_reference_files is True

    def test_config_with_filters(self):
        """Test GDPVal config with sector/occupation filters."""
        from cio_agent.eval_config import GDPValDatasetConfig

        config = GDPValDatasetConfig(
            sectors=["Government", "Manufacturing"],
            occupations=["Accountants and Auditors"],
            limit=10,
        )

        assert config.sectors == ["Government", "Manufacturing"]
        assert config.occupations == ["Accountants and Auditors"]
        assert config.limit == 10

    def test_config_in_evaluation_config(self):
        """Test GDPValDatasetConfig in EvaluationConfig."""
        from cio_agent.eval_config import (
            EvaluationConfig,
            GDPValDatasetConfig,
            SamplingConfig,
        )

        config = EvaluationConfig(
            name="GDPVal Test",
            datasets=[
                GDPValDatasetConfig(
                    limit=5,
                )
            ],
            sampling=SamplingConfig(strategy="sequential", total_limit=5),
        )

        assert len(config.datasets) == 1
        assert config.datasets[0].type == "gdpval"


class TestGDPValDatasetLoader:
    """Test loading GDPVal dataset from HuggingFace."""

    @pytest.fixture
    def mock_hf_dataset(self):
        """Create mock HuggingFace dataset."""
        mock_data = [
            {
                "task_id": "task-001",
                "sector": "Government",
                "occupation": "Budget Analysts",
                "prompt": "Prepare a budget variance report...",
                "reference_files": ["budget.xlsx"],
                "reference_file_urls": ["https://example.com/budget.xlsx"],
                "reference_file_hf_uris": ["hf://datasets/openai/gdpval/budget.xlsx"],
            },
            {
                "task_id": "task-002",
                "sector": "Professional, Scientific, and Technical Services",
                "occupation": "Accountants and Auditors",
                "prompt": "Review the audit findings and prepare a summary...",
                "reference_files": [],
                "reference_file_urls": [],
                "reference_file_hf_uris": [],
            },
        ]

        mock_train = MagicMock()
        mock_train.__iter__ = lambda self: iter(mock_data)

        mock_dataset = {"train": mock_train}
        return mock_dataset

    def test_load_gdpval_examples(self, mock_hf_dataset):
        """Test loading GDPVal examples through ConfigurableDatasetLoader."""
        from cio_agent.eval_config import (
            EvaluationConfig,
            GDPValDatasetConfig,
            SamplingConfig,
            ConfigurableDatasetLoader,
        )

        with patch("datasets.load_dataset", return_value=mock_hf_dataset):
            config = EvaluationConfig(
                datasets=[
                    GDPValDatasetConfig(
                        limit=2,
                        shuffle=False,  # Disable shuffle for deterministic test
                    )
                ],
                sampling=SamplingConfig(strategy="sequential", total_limit=2),
            )

            loader = ConfigurableDatasetLoader(config)
            examples = loader.load()

            assert len(examples) == 2
            assert all(ex.dataset_type == "gdpval" for ex in examples)
            # Check that we have both tasks (order may vary)
            example_ids = {ex.example_id for ex in examples}
            assert "task-001" in example_ids
            assert "task-002" in example_ids

    def test_load_with_sector_filter(self, mock_hf_dataset):
        """Test loading with sector filter."""
        from cio_agent.eval_config import (
            EvaluationConfig,
            GDPValDatasetConfig,
            SamplingConfig,
            ConfigurableDatasetLoader,
        )

        with patch("datasets.load_dataset", return_value=mock_hf_dataset):
            config = EvaluationConfig(
                datasets=[
                    GDPValDatasetConfig(
                        sectors=["Government"],
                    )
                ],
                sampling=SamplingConfig(strategy="sequential", total_limit=10),
            )

            loader = ConfigurableDatasetLoader(config)
            examples = loader.load()

            assert len(examples) == 1
            assert examples[0].category == "Government"

    def test_reference_files_in_metadata(self, mock_hf_dataset):
        """Test that reference files are included in metadata."""
        from cio_agent.eval_config import (
            EvaluationConfig,
            GDPValDatasetConfig,
            SamplingConfig,
            ConfigurableDatasetLoader,
        )

        with patch("datasets.load_dataset", return_value=mock_hf_dataset):
            config = EvaluationConfig(
                datasets=[
                    GDPValDatasetConfig(
                        include_reference_files=True,
                        shuffle=False,  # Disable shuffle for deterministic test
                    )
                ],
                sampling=SamplingConfig(strategy="sequential", total_limit=2),
            )

            loader = ConfigurableDatasetLoader(config)
            examples = loader.load()

            # Find example with and without reference files
            example_with_refs = next(
                (ex for ex in examples if ex.metadata.get("has_reference_files")), None
            )
            example_without_refs = next(
                (ex for ex in examples if not ex.metadata.get("has_reference_files")), None
            )

            # One example has reference files, one doesn't
            assert example_with_refs is not None
            assert len(example_with_refs.metadata.get("reference_files", [])) == 1
            assert example_without_refs is not None


class TestGDPValEvaluator:
    """Test GDPValEvaluator scoring."""

    def test_evaluator_initialization(self):
        """Test evaluator initializes correctly."""
        from evaluators.gdpval_evaluator import GDPValEvaluator

        evaluator = GDPValEvaluator(use_llm=False)

        assert evaluator.name == "gdpval"
        assert evaluator.use_llm is False

    def test_heuristic_evaluation(self):
        """Test heuristic evaluation when LLM unavailable."""
        from evaluators.gdpval_evaluator import GDPValEvaluator

        evaluator = GDPValEvaluator(use_llm=False)

        # Short response
        result = evaluator.evaluate(
            predicted="OK",
            task_prompt="Prepare a detailed budget analysis...",
        )
        assert result.score < 0.3

        # Longer structured response
        result = evaluator.evaluate(
            predicted="""
            Budget Analysis Report

            1. Executive Summary:
            The Q4 budget shows a variance of 5% from projections.

            2. Key Findings:
            - Revenue exceeded targets by 3%
            - Operating expenses were under budget by 2%
            - Capital expenditure aligned with projections

            3. Recommendations:
            - Continue current cost control measures
            - Review revenue forecasting methodology
            """,
            task_prompt="Prepare a detailed budget analysis...",
        )
        assert result.score >= 0.5  # Should score higher due to structure

    def test_missing_task_prompt(self):
        """Test evaluation fails gracefully without task prompt."""
        from evaluators.gdpval_evaluator import GDPValEvaluator

        evaluator = GDPValEvaluator(use_llm=False)

        result = evaluator.evaluate(
            predicted="Some response",
            task_prompt="",
        )

        assert result.score == 0.0
        assert "missing_task_prompt" in str(result.details)

    def test_empty_response(self):
        """Test evaluation handles empty response."""
        from evaluators.gdpval_evaluator import GDPValEvaluator

        evaluator = GDPValEvaluator(use_llm=False)

        result = evaluator.evaluate(
            predicted="",
            task_prompt="Prepare a budget report",
        )

        assert result.score == 0.0

    @patch("evaluators.gdpval_evaluator.build_llm_client")
    @patch("evaluators.gdpval_evaluator.call_llm")
    def test_llm_evaluation(self, mock_call_llm, mock_build_client):
        """Test LLM-based evaluation."""
        from evaluators.gdpval_evaluator import GDPValEvaluator

        mock_client = MagicMock()
        mock_build_client.return_value = mock_client
        mock_call_llm.return_value = '{"completion": 20, "accuracy": 18, "format": 22, "professionalism": 20, "feedback": "Good work"}'

        evaluator = GDPValEvaluator(use_llm=True, llm_client=mock_client)

        result = evaluator.evaluate(
            predicted="Detailed budget analysis with all required sections...",
            task_prompt="Prepare a budget variance report",
            occupation="Budget Analysts",
            sector="Government",
        )

        assert result.score == 0.80  # (20+18+22+20)/100
        assert result.details.get("llm_used") is True
        assert result.details.get("completion") == 20
        assert result.details.get("accuracy") == 18


class TestGDPValYamlConfig:
    """Test loading GDPVal config from YAML."""

    def test_load_gdpval_yaml(self, tmp_path):
        """Test loading GDPVal config from YAML file."""
        from cio_agent.eval_config import EvaluationConfig

        yaml_content = """
name: "GDPVal Test"
datasets:
  - type: gdpval
    hf_dataset: "openai/gdpval"
    sectors:
      - Government
    limit: 5
sampling:
  strategy: random
  total_limit: 5
  seed: 42
"""
        config_file = tmp_path / "test_gdpval.yaml"
        config_file.write_text(yaml_content)

        config = EvaluationConfig.from_yaml(config_file)

        assert config.name == "GDPVal Test"
        assert len(config.datasets) == 1
        assert config.datasets[0].type == "gdpval"
        assert config.datasets[0].sectors == ["Government"]
        assert config.datasets[0].limit == 5


class TestGDPValIntegration:
    """Integration tests for GDPVal with GreenAgent."""

    def test_green_agent_has_gdpval_evaluator(self):
        """Test GreenAgent initializes GDPVal evaluator."""
        from cio_agent.green_agent import GreenAgent
        from cio_agent.eval_config import (
            EvaluationConfig,
            GDPValDatasetConfig,
            SamplingConfig,
        )

        # Mock the dataset loading
        with patch("datasets.load_dataset") as mock_load:
            mock_train = MagicMock()
            mock_train.__iter__ = lambda self: iter([])
            mock_load.return_value = {"train": mock_train}

            config = EvaluationConfig(
                datasets=[
                    GDPValDatasetConfig(limit=1),
                ],
                sampling=SamplingConfig(strategy="sequential", total_limit=1),
            )

            agent = GreenAgent(eval_config=config)

            assert "gdpval" in agent._evaluators
            assert agent._evaluators["gdpval"] is not None
