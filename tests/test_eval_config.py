"""
Unit tests for evaluation config system.

Tests the EvaluationConfig, ConfigurableDatasetLoader, and sampling strategies.
"""

import pytest
import tempfile
import yaml
from pathlib import Path


class TestEvaluationConfig:
    """Test EvaluationConfig loading and validation."""

    def test_default_config_creation(self):
        """Test creating default evaluation config."""
        from cio_agent.eval_config import create_default_config
        
        config = create_default_config()
        
        assert config.name == "FAB++ Default Evaluation"
        assert len(config.datasets) == 3
        assert config.sampling.strategy == "stratified"
        assert config.sampling.total_limit == 50

    def test_config_from_dict(self):
        """Test loading config from dictionary."""
        from cio_agent.eval_config import EvaluationConfig
        
        data = {
            "name": "Test Config",
            "datasets": [
                {"type": "bizfinbench", "path": "data/BizFinBench.v2", "task_types": ["event_logic_reasoning"]},
            ],
            "sampling": {"strategy": "random", "total_limit": 10}
        }
        
        config = EvaluationConfig.from_dict(data)
        
        assert config.name == "Test Config"
        assert len(config.datasets) == 1
        assert config.datasets[0].type == "bizfinbench"
        assert config.sampling.total_limit == 10

    def test_config_from_yaml(self):
        """Test loading config from YAML file."""
        from cio_agent.eval_config import EvaluationConfig
        
        yaml_content = """
name: YAML Test Config
datasets:
  - type: public_csv
    path: finance-agent/data/public.csv
    limit: 5
sampling:
  strategy: sequential
  total_limit: 5
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            config = EvaluationConfig.from_yaml(f.name)
            
        assert config.name == "YAML Test Config"
        assert len(config.datasets) == 1
        assert config.datasets[0].type == "public_csv"
        assert config.sampling.strategy == "sequential"

    def test_bizfinbench_all_task_types_expansion(self):
        """Test that 'all' expands to all task types."""
        from cio_agent.eval_config import BizFinBenchDatasetConfig
        
        config = BizFinBenchDatasetConfig(
            path="data/BizFinBench.v2",
            task_types=["all"]
        )
        
        assert len(config.task_types) >= 5
        assert "event_logic_reasoning" in config.task_types
        assert "user_sentiment_analysis" in config.task_types


class TestConfigurableDatasetLoader:
    """Test ConfigurableDatasetLoader with real datasets."""

    def test_load_bizfinbench_examples(self):
        """Test loading examples from BizFinBench."""
        from cio_agent.eval_config import (
            EvaluationConfig, BizFinBenchDatasetConfig,
            SamplingConfig, ConfigurableDatasetLoader
        )
        
        config = EvaluationConfig(
            datasets=[
                BizFinBenchDatasetConfig(
                    path="data/BizFinBench.v2",
                    task_types=["event_logic_reasoning"],
                    languages=["en"],
                    limit_per_task=3,
                )
            ],
            sampling=SamplingConfig(strategy="sequential", total_limit=3),
        )
        
        loader = ConfigurableDatasetLoader(config)
        examples = loader.load()
        
        assert len(examples) == 3
        assert all(ex.dataset_type == "bizfinbench" for ex in examples)
        assert all(ex.task_type == "event_logic_reasoning" for ex in examples)

    def test_load_public_csv_examples(self):
        """Test loading examples from public.csv."""
        from cio_agent.eval_config import (
            EvaluationConfig, PublicCsvDatasetConfig,
            SamplingConfig, ConfigurableDatasetLoader
        )
        
        config = EvaluationConfig(
            datasets=[
                PublicCsvDatasetConfig(
                    path="finance-agent/data/public.csv",
                    limit=5,
                )
            ],
            sampling=SamplingConfig(strategy="sequential", total_limit=5),
        )
        
        loader = ConfigurableDatasetLoader(config)
        examples = loader.load()
        
        assert len(examples) == 5
        assert all(ex.dataset_type == "public_csv" for ex in examples)

    def test_load_multiple_datasets(self):
        """Test loading from multiple datasets."""
        from cio_agent.eval_config import (
            EvaluationConfig, BizFinBenchDatasetConfig, PublicCsvDatasetConfig,
            SamplingConfig, ConfigurableDatasetLoader
        )
        
        config = EvaluationConfig(
            datasets=[
                BizFinBenchDatasetConfig(
                    path="data/BizFinBench.v2",
                    task_types=["event_logic_reasoning"],
                    languages=["en"],
                    limit_per_task=2,
                ),
                PublicCsvDatasetConfig(
                    path="finance-agent/data/public.csv",
                    limit=3,
                ),
            ],
            sampling=SamplingConfig(strategy="sequential", total_limit=5),
        )
        
        loader = ConfigurableDatasetLoader(config)
        examples = loader.load()
        
        assert len(examples) == 5
        assert sum(1 for ex in examples if ex.dataset_type == "bizfinbench") >= 1
        assert sum(1 for ex in examples if ex.dataset_type == "public_csv") >= 1


class TestSamplingStrategies:
    """Test different sampling strategies."""

    def test_sequential_sampling(self):
        """Test sequential (no shuffle) sampling."""
        from cio_agent.eval_config import (
            EvaluationConfig, BizFinBenchDatasetConfig,
            SamplingConfig, ConfigurableDatasetLoader
        )
        
        config = EvaluationConfig(
            datasets=[
                BizFinBenchDatasetConfig(
                    path="data/BizFinBench.v2",
                    task_types=["event_logic_reasoning"],
                    languages=["en"],
                    limit_per_task=5,
                    shuffle=False,  # No shuffle within dataset
                )
            ],
            sampling=SamplingConfig(strategy="sequential", total_limit=3, seed=42),
        )
        
        loader = ConfigurableDatasetLoader(config)
        examples1 = loader.load()
        
        # Create new loader with same config
        loader2 = ConfigurableDatasetLoader(config)
        examples2 = loader2.load()
        
        # Sequential should be deterministic
        assert [ex.example_id for ex in examples1] == [ex.example_id for ex in examples2]

    def test_random_sampling_with_seed(self):
        """Test random sampling with seed for reproducibility."""
        from cio_agent.eval_config import (
            EvaluationConfig, BizFinBenchDatasetConfig,
            SamplingConfig, ConfigurableDatasetLoader
        )
        
        config = EvaluationConfig(
            datasets=[
                BizFinBenchDatasetConfig(
                    path="data/BizFinBench.v2",
                    task_types=["event_logic_reasoning", "user_sentiment_analysis"],
                    languages=["en"],
                    limit_per_task=3,
                )
            ],
            sampling=SamplingConfig(strategy="random", total_limit=5, seed=42),
        )
        
        loader1 = ConfigurableDatasetLoader(config)
        examples1 = loader1.load()
        
        # Same seed should give same results
        config2 = EvaluationConfig(
            datasets=[
                BizFinBenchDatasetConfig(
                    path="data/BizFinBench.v2",
                    task_types=["event_logic_reasoning", "user_sentiment_analysis"],
                    languages=["en"],
                    limit_per_task=3,
                )
            ],
            sampling=SamplingConfig(strategy="random", total_limit=5, seed=42),
        )
        loader2 = ConfigurableDatasetLoader(config2)
        examples2 = loader2.load()
        
        assert [ex.example_id for ex in examples1] == [ex.example_id for ex in examples2]

    def test_stratified_sampling(self):
        """Test stratified (equal per dataset) sampling."""
        from cio_agent.eval_config import (
            EvaluationConfig, BizFinBenchDatasetConfig, PublicCsvDatasetConfig,
            SamplingConfig, ConfigurableDatasetLoader
        )
        
        config = EvaluationConfig(
            datasets=[
                BizFinBenchDatasetConfig(
                    path="data/BizFinBench.v2",
                    task_types=["event_logic_reasoning"],
                    languages=["en"],
                    limit_per_task=10,
                ),
                PublicCsvDatasetConfig(
                    path="finance-agent/data/public.csv",
                    limit=10,
                ),
            ],
            sampling=SamplingConfig(strategy="stratified", total_limit=10, seed=42),
        )
        
        loader = ConfigurableDatasetLoader(config)
        examples = loader.load()
        
        # Should have balanced representation
        bizfin_count = sum(1 for ex in examples if ex.dataset_type == "bizfinbench")
        public_count = sum(1 for ex in examples if ex.dataset_type == "public_csv")
        
        # With stratified, should be roughly equal
        assert abs(bizfin_count - public_count) <= 2


class TestDatasetLoaderSummary:
    """Test loader summary statistics."""

    def test_summary_statistics(self):
        """Test summary returns correct statistics."""
        from cio_agent.eval_config import (
            EvaluationConfig, BizFinBenchDatasetConfig, PublicCsvDatasetConfig,
            SamplingConfig, ConfigurableDatasetLoader
        )
        
        config = EvaluationConfig(
            datasets=[
                BizFinBenchDatasetConfig(
                    path="data/BizFinBench.v2",
                    task_types=["event_logic_reasoning"],
                    languages=["en"],
                    limit_per_task=3,
                ),
                PublicCsvDatasetConfig(
                    path="finance-agent/data/public.csv",
                    limit=2,
                ),
            ],
            sampling=SamplingConfig(strategy="sequential", total_limit=5),
        )
        
        loader = ConfigurableDatasetLoader(config)
        loader.load()
        summary = loader.summary()
        
        assert "total" in summary
        assert "by_dataset" in summary
        assert summary["total"] == 5
        assert "bizfinbench" in summary["by_dataset"]
        assert "public_csv" in summary["by_dataset"]


class TestGreenAgentWithConfig:
    """Test GreenAgent with EvaluationConfig."""

    def test_green_agent_with_config(self):
        """Test GreenAgent initialization with config."""
        from cio_agent.green_agent import GreenAgent
        from cio_agent.eval_config import (
            EvaluationConfig, BizFinBenchDatasetConfig, PublicCsvDatasetConfig,
            SamplingConfig
        )
        
        config = EvaluationConfig(
            name="Test Evaluation",
            datasets=[
                BizFinBenchDatasetConfig(
                    path="data/BizFinBench.v2",
                    task_types=["event_logic_reasoning"],
                    languages=["en"],
                    limit_per_task=2,
                ),
                PublicCsvDatasetConfig(
                    path="finance-agent/data/public.csv",
                    limit=3,
                ),
            ],
            sampling=SamplingConfig(strategy="random", total_limit=5, seed=42),
        )
        
        agent = GreenAgent(eval_config=config)
        
        assert agent.eval_config is not None
        assert agent.dataset_loader is not None
        assert agent._loaded_examples is not None
        assert len(agent._loaded_examples) == 5
        assert "bizfinbench" in agent._evaluators
        assert "public_csv" in agent._evaluators

    def test_green_agent_with_config_path(self):
        """Test GreenAgent with config file path."""
        from cio_agent.green_agent import GreenAgent
        
        # Use the quick test config
        agent = GreenAgent(eval_config="config/eval_quick.yaml")
        
        assert agent.eval_config is not None
        assert agent._loaded_examples is not None
        assert len(agent._loaded_examples) <= 20  # Quick config has limit of 10


class TestSyntheticDatasetSupport:
    """Test synthetic dataset integration."""

    def test_synthetic_config_type(self):
        """Test SyntheticDatasetConfig type."""
        from cio_agent.eval_config import SyntheticDatasetConfig
        
        config = SyntheticDatasetConfig(
            path="data/synthetic_questions/questions.json",
            limit=5,
            shuffle=True,
            weight=0.5,
        )
        
        assert config.type == "synthetic"
        assert config.path == "data/synthetic_questions/questions.json"
        assert config.limit == 5
        assert config.weight == 0.5

    def test_load_synthetic_dataset(self):
        """Test loading synthetic dataset through loader."""
        from cio_agent.eval_config import (
            EvaluationConfig, SyntheticDatasetConfig,
            SamplingConfig, ConfigurableDatasetLoader
        )
        
        config = EvaluationConfig(
            datasets=[
                SyntheticDatasetConfig(
                    path="data/synthetic_questions/questions.json",
                    limit=3,
                )
            ],
            sampling=SamplingConfig(strategy="sequential", total_limit=3),
        )
        
        loader = ConfigurableDatasetLoader(config)
        examples = loader.load()
        
        assert len(examples) == 3
        assert all(ex.dataset_type == "synthetic" for ex in examples)
        assert all(hasattr(ex, 'question') for ex in examples)
        assert all(hasattr(ex, 'answer') for ex in examples)

    def test_synthetic_in_multi_dataset_config(self):
        """Test synthetic dataset alongside other datasets."""
        from cio_agent.eval_config import (
            EvaluationConfig, SyntheticDatasetConfig, PublicCsvDatasetConfig,
            SamplingConfig, ConfigurableDatasetLoader
        )
        
        config = EvaluationConfig(
            datasets=[
                SyntheticDatasetConfig(
                    path="data/synthetic_questions/questions.json",
                    limit=2,
                ),
                PublicCsvDatasetConfig(
                    path="finance-agent/data/public.csv",
                    limit=3,
                ),
            ],
            sampling=SamplingConfig(strategy="sequential", total_limit=5),
        )
        
        loader = ConfigurableDatasetLoader(config)
        examples = loader.load()
        
        synthetic_count = sum(1 for ex in examples if ex.dataset_type == "synthetic")
        public_count = sum(1 for ex in examples if ex.dataset_type == "public_csv")
        
        assert synthetic_count >= 1
        assert public_count >= 1
        assert len(examples) == 5

    def test_green_agent_with_synthetic(self):
        """Test GreenAgent recognizes synthetic evaluator."""
        from cio_agent.green_agent import GreenAgent
        from cio_agent.eval_config import (
            EvaluationConfig, SyntheticDatasetConfig, SamplingConfig
        )
        
        config = EvaluationConfig(
            datasets=[
                SyntheticDatasetConfig(
                    path="data/synthetic_questions/questions.json",
                    limit=2,
                )
            ],
            sampling=SamplingConfig(strategy="sequential", total_limit=2),
        )
        
        agent = GreenAgent(eval_config=config)
        
        assert "synthetic" in agent._evaluators
        assert agent._loaded_examples is not None
        assert len(agent._loaded_examples) == 2
        assert all(ex.dataset_type == "synthetic" for ex in agent._loaded_examples)

    def test_extract_recommendation_for_synthetic(self):
        """Test _extract_recommendation works for synthetic."""
        from cio_agent.green_agent import GreenAgent
        
        agent = GreenAgent(dataset_type="synthetic")
        
        # Test various response patterns
        assert agent._extract_recommendation("The company beat expectations") == "Beat"
        assert agent._extract_recommendation("They missed the target") == "Miss"
        assert agent._extract_recommendation("I recommend to buy NVDA") == "Buy"
        assert agent._extract_recommendation("Sell the position") == "Sell"
        assert agent._extract_recommendation("Hold your current position") == "Hold"


class TestCryptoDatasetSupport:
    """Test crypto dataset integration."""

    def test_load_crypto_dataset(self):
        """Test loading crypto dataset through loader."""
        from cio_agent.eval_config import (
            EvaluationConfig, CryptoDatasetConfig,
            SamplingConfig, ConfigurableDatasetLoader
        )

        config = EvaluationConfig(
            datasets=[
                CryptoDatasetConfig(
                    path="data/crypto/scenarios",
                    scenarios=["sample_btc_window"],
                    limit=1,
                    shuffle=False,
                )
            ],
            sampling=SamplingConfig(strategy="sequential", total_limit=1),
        )

        loader = ConfigurableDatasetLoader(config)
        examples = loader.load()

        assert len(examples) == 1
        assert examples[0].dataset_type == "crypto"
        assert examples[0].metadata.get("data_path")

