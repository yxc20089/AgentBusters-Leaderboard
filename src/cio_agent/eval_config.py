"""
Evaluation configuration for Green Agent.

Supports multi-dataset evaluation with configurable:
- Multiple datasets (BizFinBench, public.csv, options, crypto scenarios)
- Task type filtering
- Language filtering
- Shuffle and sampling strategies
"""

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml

logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field, field_validator

from cio_agent.crypto_benchmark import (
    CryptoEvaluationConfig,
    discover_crypto_scenarios,
    prepare_crypto_scenarios,
)
from cio_agent.crypto_benchmark import (
    CryptoEvaluationConfig,
    discover_crypto_scenarios,
    prepare_crypto_scenarios,
)

class BizFinBenchDatasetConfig(BaseModel):
    """Configuration for BizFinBench dataset (fetched from HuggingFace)."""
    type: Literal["bizfinbench"] = "bizfinbench"
    # path is deprecated - data is fetched from HuggingFace dynamically
    path: Optional[str] = Field(
        default=None,
        description="Deprecated: Data is now fetched from HuggingFace API"
    )
    task_types: List[str] = Field(
        default=["Financial_Time_Reasoning"],
        description="Task types to include. Use HuggingFace subset names or legacy names."
    )
    languages: List[str] = Field(
        default=["en"],
        description="Languages to include: 'en', 'cn', or both"
    )
    limit_per_task: Optional[int] = Field(
        default=None,
        description="Limit per task type (before total limit)"
    )
    shuffle: bool = Field(
        default=False,
        description="Shuffle examples within this dataset (default False for reproducibility)"
    )
    weight: float = Field(
        default=1.0,
        description="Sampling weight relative to other datasets"
    )

    @field_validator("task_types", mode="before")
    @classmethod
    def expand_all_task_types(cls, v):
        if v == "all" or v == ["all"]:
            # Valid task types from HiThink-Research/BizFinBench.v2
            # Note: financial_report_analysis is only available in Chinese (cn)
            return [
                "anomaly_information_tracing",
                "event_logic_reasoning",
                "financial_data_description",
                "financial_quantitative_computation",
                "user_sentiment_analysis",
                "stock_price_predict",
                "financial_multi_turn_perception",
                "financial_report_analysis",  # cn only
            ]
        return v


class PublicCsvDatasetConfig(BaseModel):
    """Configuration for public.csv dataset."""
    type: Literal["public_csv"] = "public_csv"
    path: str = "finance-agent/data/public.csv"
    categories: Optional[List[str]] = Field(
        default=None,
        description="Categories to include. None means all."
    )
    limit: Optional[int] = Field(
        default=None,
        description="Limit number of examples"
    )
    shuffle: bool = Field(
        default=False,
        description="Shuffle examples (default False for reproducibility)"
    )
    weight: float = Field(
        default=1.0,
        description="Sampling weight relative to other datasets"
    )


class SyntheticDatasetConfig(BaseModel):
    """Configuration for synthetic questions."""
    type: Literal["synthetic"] = "synthetic"
    path: str = Field(
        description="Path to synthetic questions JSON file"
    )
    limit: Optional[int] = None
    shuffle: bool = False  # Default False for reproducibility
    weight: float = 1.0


class OptionsDatasetConfig(BaseModel):
    """Configuration for Options Alpha Challenge dataset."""
    type: Literal["options"] = "options"
    path: str = Field(
        default="data/options/questions.json",
        description="Path to options questions JSON file"
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="Categories to include. None means all. Options: Options Pricing, Greeks Analysis, Strategy Construction, Volatility Trading, P&L Attribution, Risk Management, Copy Trading, Race to 10M, Strategy Defense"
    )
    limit: Optional[int] = Field(
        default=None,
        description="Limit number of examples"
    )
    shuffle: bool = Field(
        default=False,
        description="Shuffle examples (default False for reproducibility)"
    )
    weight: float = Field(
        default=1.0,
        description="Sampling weight relative to other datasets"
    )


# Crypto trading benchmark dataset
class CryptoDatasetConfig(BaseModel):
    """Configuration for crypto trading benchmark scenarios."""
    type: Literal["crypto"] = "crypto"
    path: str = Field(
        default="data/crypto/scenarios",
        description="Path to crypto scenarios directory or JSON file"
    )
    remote_manifest: Optional[str] = Field(
        default=None,
        description="Optional URL or file path to a remote manifest JSON"
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Cache directory for remote scenarios (defaults to path)"
    )
    cache_ttl_hours: int = Field(
        default=24,
        description="Cache TTL in hours for remote scenarios (0 disables expiration)"
    )
    download_on_missing: bool = Field(
        default=True,
        description="Download missing scenarios when using remote manifest"
    )
    scenarios: Optional[List[str]] = Field(
        default=None,
        description="Optional list of scenario names to include"
    )
    limit: Optional[int] = Field(
        default=None,
        description="Limit number of scenarios"
    )
    shuffle: bool = Field(
        default=False,
        description="Shuffle scenarios (default False for reproducibility)"
    )
    weight: float = Field(
        default=1.0,
        description="Sampling weight relative to other datasets"
    )
    max_steps: Optional[int] = Field(
        default=None,
        description="Max market states per scenario"
    )
    stride: int = Field(
        default=1,
        description="Downsample market states by taking every Nth bar"
    )
    evaluation: CryptoEvaluationConfig = Field(
        default_factory=CryptoEvaluationConfig,
        description="Evaluation settings for crypto benchmark"
    )

    # PostgreSQL direct query mode
    pg_enabled: bool = Field(
        default=False,
        description="Enable PostgreSQL direct query mode instead of JSON files"
    )
    pg_dsn: Optional[str] = Field(
        default=None,
        description="PostgreSQL DSN connection string (overrides individual params)"
    )
    pg_host: str = Field(
        default="localhost",
        description="PostgreSQL host"
    )
    pg_port: int = Field(
        default=5432,
        description="PostgreSQL port"
    )
    pg_dbname: str = Field(
        default="market_data",
        description="PostgreSQL database name"
    )
    pg_user: str = Field(
        default="postgres",
        description="PostgreSQL username"
    )
    pg_password: Optional[str] = Field(
        default=None,
        description="PostgreSQL password"
    )
    pg_ohlcv_table: str = Field(
        default="market_data.candles_1m",
        description="PostgreSQL table for OHLCV data"
    )
    pg_funding_table: Optional[str] = Field(
        default="market_data.funding_rates",
        description="PostgreSQL table for funding rates (optional)"
    )

    # Hidden window anti-overfitting configuration
    hidden_seed_config: Optional[str] = Field(
        default=None,
        description="Name of hidden seed config in ~/.agentbusters/hidden_seeds.yaml"
    )
    window_count: int = Field(
        default=12,
        description="Number of evaluation windows to generate"
    )
    window_min_bars: int = Field(
        default=1440,
        description="Minimum bars per window (1440 = 1 day at 1m timeframe)"
    )
    window_max_bars: int = Field(
        default=10080,
        description="Maximum bars per window (10080 = 7 days at 1m timeframe)"
    )
    symbols: List[str] = Field(
        default_factory=lambda: ["BTCUSDT"],
        description="Symbols to include in evaluation"
    )
    date_range_start: str = Field(
        default="2020-01-01",
        description="Start of date range for window selection"
    )
    date_range_end: str = Field(
        default="2025-12-31",
        description="End of date range for window selection"
    )

    @field_validator("stride")
    @classmethod
    def validate_stride(cls, value: int) -> int:
        if value < 1:
            raise ValueError("stride must be >= 1")
        return value

    @field_validator("cache_ttl_hours")
    @classmethod
    def validate_cache_ttl_hours(cls, value: int) -> int:
        if value < 0:
            raise ValueError("cache_ttl_hours must be >= 0")
        return value


class GDPValDatasetConfig(BaseModel):
    """Configuration for OpenAI GDPVal dataset (fetched from HuggingFace)."""
    type: Literal["gdpval"] = "gdpval"
    hf_dataset: str = Field(
        default="openai/gdpval",
        description="HuggingFace dataset identifier"
    )
    sectors: Optional[List[str]] = Field(
        default=None,
        description="Filter by sectors. None means all. Options: Professional/Scientific/Technical Services, Government, Information, Manufacturing, etc."
    )
    occupations: Optional[List[str]] = Field(
        default=None,
        description="Filter by occupations. None means all. 44 occupations available."
    )
    limit: Optional[int] = Field(
        default=None,
        description="Limit number of tasks"
    )
    shuffle: bool = Field(
        default=False,
        description="Shuffle tasks (default False for reproducibility)"
    )
    weight: float = Field(
        default=1.0,
        description="Sampling weight relative to other datasets"
    )
    include_reference_files: bool = Field(
        default=True,
        description="Include reference file URLs in metadata for multi-modal evaluation"
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Cache directory for HuggingFace datasets"
    )


# Union type for all dataset configs
DatasetConfig = Union[
    BizFinBenchDatasetConfig,
    PublicCsvDatasetConfig,
    SyntheticDatasetConfig,
    OptionsDatasetConfig,
    CryptoDatasetConfig,
    GDPValDatasetConfig,
]


class SamplingConfig(BaseModel):
    """Configuration for sampling strategy."""
    strategy: Literal["sequential", "random", "stratified", "weighted"] = Field(
        default="stratified",
        description=(
            "sequential: no shuffle, take in order. "
            "random: global shuffle. "
            "stratified: equal samples per dataset/category (default). "
            "weighted: sample by dataset weights."
        )
    )
    total_limit: Optional[int] = Field(
        default=None,
        description="Total number of examples across all datasets"
    )
    seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class LLMEvaluationConfig(BaseModel):
    """Configuration for LLM-as-judge dataset evaluation."""
    enabled: Optional[bool] = Field(
        default=None,
        description="Enable LLM grading for dataset evaluators (bizfinbench/public_csv)."
    )
    model: Optional[str] = Field(
        default=None,
        description="Model override for LLM grading (defaults to LLM_MODEL/EVAL_LLM_MODEL)."
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature override for LLM grading."
    )


class EvaluationConfig(BaseModel):
    """Main evaluation configuration."""
    name: str = Field(
        default="FAB++ Evaluation",
        description="Name of this evaluation run"
    )
    version: str = "1.0"
    datasets: List[DatasetConfig] = Field(
        default_factory=list,
        description="List of datasets to include"
    )
    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig,
        description="Sampling configuration"
    )
    debate: bool = Field(
        default=False,
        description="Whether to conduct adversarial debate"
    )
    timeout_seconds: int = Field(
        default=300,
        description="Timeout per question in seconds"
    )
    llm_eval: LLMEvaluationConfig = Field(
        default_factory=LLMEvaluationConfig,
        description="LLM-as-judge configuration for dataset evaluators."
    )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "EvaluationConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationConfig":
        """Load configuration from dictionary."""
        return cls.model_validate(data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


@dataclass
class LoadedExample:
    """A loaded example from any dataset."""
    example_id: str
    question: str
    answer: str
    dataset_type: str
    category: Optional[str] = None
    task_type: Optional[str] = None
    language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfigurableDatasetLoader:
    """
    Loads and samples examples from multiple datasets based on configuration.
    
    Supports:
    - Multiple dataset types
    - Per-dataset limits and filtering
    - Global sampling strategies
    - Reproducible shuffling
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self._examples: List[LoadedExample] = []
        self._by_dataset: Dict[str, List[LoadedExample]] = {}
        self._loaded = False

    def load(self) -> List[LoadedExample]:
        """Load all examples from configured datasets."""
        if self._loaded:
            return self._examples

        # Set random seed for reproducibility (env overrides config)
        seed = self.config.sampling.seed
        env_seed = os.environ.get("EVAL_SCENARIO_SEED")
        if env_seed is not None:
            try:
                seed = int(env_seed)
            except ValueError:
                pass
        if seed is not None:
            random.seed(seed)

        all_examples = []

        for dataset_config in self.config.datasets:
            examples = self._load_dataset(dataset_config)
            
            # Per-dataset shuffle
            if dataset_config.shuffle:
                random.shuffle(examples)
            
            self._by_dataset[dataset_config.type] = examples
            all_examples.extend(examples)

        # Re-seed before sampling to ensure deterministic results
        # (data loading may have consumed random state unpredictably)
        if seed is not None:
            random.seed(seed)

        # Apply sampling strategy
        self._examples = self._apply_sampling(all_examples)
        self._loaded = True
        
        return self._examples

    def _load_dataset(self, config: DatasetConfig) -> List[LoadedExample]:
        """Load examples from a single dataset configuration."""
        if config.type == "bizfinbench":
            return self._load_bizfinbench(config)
        elif config.type == "public_csv":
            return self._load_public_csv(config)
        elif config.type == "synthetic":
            return self._load_synthetic(config)
        elif config.type == "options":
            return self._load_options(config)
        elif config.type == "crypto":
            return self._load_crypto(config)
        elif config.type == "gdpval":
            return self._load_gdpval(config)
        else:
            raise ValueError(f"Unknown dataset type: {config.type}")

    def _load_bizfinbench(self, config: BizFinBenchDatasetConfig) -> List[LoadedExample]:
        """Load BizFinBench examples from HuggingFace API."""
        from cio_agent.data_providers import BizFinBenchProvider

        examples = []

        for task_type in config.task_types:
            for language in config.languages:
                try:
                    # New HuggingFace-based provider (path is ignored)
                    provider = BizFinBenchProvider(
                        task_type=task_type,
                        language=language,
                        limit=config.limit_per_task,
                    )
                    raw_examples = provider.load()

                    for ex in raw_examples:
                        examples.append(LoadedExample(
                            example_id=ex.example_id,
                            question=ex.question,
                            answer=ex.answer,
                            dataset_type="bizfinbench",
                            task_type=task_type,
                            language=language,
                            metadata={"source": f"bizfinbench_{task_type}_{language}"},
                        ))
                except (ValueError, RuntimeError) as e:
                    # Skip if task type doesn't exist or HuggingFace fetch fails
                    import logging
                    logging.warning(f"Failed to load BizFinBench {task_type}/{language}: {e}")
                    continue

        return examples

    def _load_public_csv(self, config: PublicCsvDatasetConfig) -> List[LoadedExample]:
        """Load public.csv examples."""
        from cio_agent.data_providers import CsvFinanceDatasetProvider
        
        provider = CsvFinanceDatasetProvider(path=config.path)
        raw_examples = provider.load()
        
        examples = []
        for ex in raw_examples:
            category = ex.category.value if hasattr(ex.category, 'value') else str(ex.category)
            
            # Filter by category if specified
            if config.categories and category not in config.categories:
                continue
            
            examples.append(LoadedExample(
                example_id=ex.example_id,
                question=ex.question,
                answer=ex.answer,
                dataset_type="public_csv",
                category=category,
                metadata={
                    "source": "public_csv",
                    "rubric": getattr(ex, 'rubric', None),
                },
            ))
        
        # Apply per-dataset limit
        if config.limit:
            examples = examples[:config.limit]
        
        return examples

    def _load_synthetic(self, config: SyntheticDatasetConfig) -> List[LoadedExample]:
        """Load synthetic questions."""
        import json
        
        with open(config.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle wrapped format {"questions": [...]} or direct list
        if isinstance(data, dict) and "questions" in data:
            questions = data["questions"]
        elif isinstance(data, list):
            questions = data
        else:
            raise ValueError(f"Invalid synthetic questions format: {config.path}")
        
        examples = []
        for q in questions:
            examples.append(LoadedExample(
                example_id=q.get("question_id", f"synthetic_{len(examples)}"),
                question=q.get("question", ""),
                answer=q.get("ground_truth_formatted", q.get("ground_truth_value", "")),
                dataset_type="synthetic",
                category=q.get("category"),
                metadata=q,
            ))
        
        if config.limit:
            examples = examples[:config.limit]

        return examples

    def _load_options(self, config: "OptionsDatasetConfig") -> List[LoadedExample]:
        """Load Options Alpha Challenge questions."""
        from cio_agent.data_providers import OptionsDatasetProvider

        provider = OptionsDatasetProvider(
            path=config.path,
            categories=config.categories,
            limit=config.limit,
            shuffle=config.shuffle,
        )
        raw_examples = provider.load()

        examples = []
        for ex in raw_examples:
            category = ex.category.value if hasattr(ex.category, 'value') else str(ex.category)

            examples.append(LoadedExample(
                example_id=ex.example_id,
                question=ex.question,
                answer=ex.answer,
                dataset_type="options",
                category=category,
                metadata={
                    "source": "options",
                    "rubric": getattr(ex, 'rubric', None),
                    **ex.metadata,
                },
            ))

        return examples

    def _load_gdpval(self, config: "GDPValDatasetConfig") -> List[LoadedExample]:
        """Load GDPVal tasks from HuggingFace."""
        try:
            from datasets import load_dataset
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "The 'datasets' library is required for GDPVal. "
                "Install it with: pip install datasets"
            )

        import logging
        logger = logging.getLogger(__name__)

        # Load dataset from HuggingFace
        logger.info(f"Loading GDPVal dataset from {config.hf_dataset}")
        dataset = load_dataset(
            config.hf_dataset,
            cache_dir=config.cache_dir,
            trust_remote_code=True,
        )

        # GDPVal has a single 'train' split with 220 tasks
        data = dataset["train"]

        examples = []
        for row in data:
            task_id = row["task_id"]
            sector = row.get("sector", "")
            occupation = row.get("occupation", "")
            prompt = row.get("prompt", "")

            # Apply sector filter
            if config.sectors and sector not in config.sectors:
                continue

            # Apply occupation filter
            if config.occupations and occupation not in config.occupations:
                continue

            # Build metadata
            metadata = {
                "source": "gdpval",
                "sector": sector,
                "occupation": occupation,
            }

            # Include reference files if enabled
            if config.include_reference_files:
                reference_files = row.get("reference_files", [])
                reference_urls = row.get("reference_file_urls", [])
                reference_hf_uris = row.get("reference_file_hf_uris", [])

                metadata["reference_files"] = reference_files
                metadata["reference_file_urls"] = reference_urls
                metadata["reference_file_hf_uris"] = reference_hf_uris
                metadata["has_reference_files"] = len(reference_files) > 0

            examples.append(LoadedExample(
                example_id=task_id,
                question=prompt,
                answer="",  # GDPVal is open-ended, no ground truth answer
                dataset_type="gdpval",
                category=sector,
                task_type=occupation,
                metadata=metadata,
            ))

        # Apply limit
        if config.limit:
            examples = examples[:config.limit]

        logger.info(f"Loaded {len(examples)} GDPVal tasks")
        return examples

    def _load_crypto(self, config: "CryptoDatasetConfig") -> List[LoadedExample]:
        """Load crypto trading scenarios."""
        import hashlib
        import os

        # Check if using PostgreSQL with hidden windows
        if config.pg_enabled and config.hidden_seed_config:
            return self._load_crypto_from_postgres(config)

        # Auto-construct remote_manifest from EVAL_DATA_REPO if not explicitly set
        remote_manifest = config.remote_manifest
        if not remote_manifest:
            eval_data_repo = os.environ.get("EVAL_DATA_REPO")
            if eval_data_repo:
                # Build GitHub raw URL for the crypto manifest
                # Expected structure: {repo}/crypto/eval_hidden/manifest.json
                remote_manifest = (
                    f"https://raw.githubusercontent.com/{eval_data_repo}/main/crypto/eval_hidden/manifest.json"
                )
                logger.info(f"Using EVAL_DATA_REPO manifest: {remote_manifest}")

        # Fall back to JSON-based loading
        scenario_root = prepare_crypto_scenarios(
            path=Path(config.path),
            remote_manifest=remote_manifest,
            scenarios=config.scenarios,
            cache_dir=Path(config.cache_dir) if config.cache_dir else None,
            cache_ttl_hours=config.cache_ttl_hours,
            download_on_missing=config.download_on_missing,
        )

        scenario_indices = discover_crypto_scenarios(
            path=scenario_root,
            scenarios=config.scenarios,
            limit=config.limit,
            shuffle=config.shuffle,
        )

        examples = []
        for idx, scenario in enumerate(scenario_indices):
            # Generate anonymous scenario ID to avoid leaking time window info
            anon_id = self._anonymize_scenario_id(scenario.scenario_id, idx)

            # Generic question without revealing specific time periods
            question = f"Crypto trading scenario {idx + 1}"

            examples.append(LoadedExample(
                example_id=anon_id,
                question=question,
                answer="",
                dataset_type="crypto",
                metadata={
                    "scenario_id": anon_id,
                    "name": f"scenario_{idx + 1}",
                    "description": "",  # Omit description to avoid leaking info
                    "data_path": str(scenario.data_path),
                    "metadata": {
                        k: v for k, v in scenario.metadata.items()
                        if k not in ("start", "end", "period", "name", "description")
                    },
                    "max_steps": config.max_steps,
                    "stride": config.stride,
                    "evaluation": config.evaluation.model_dump(),
                },
            ))

        return examples

    def _anonymize_scenario_id(self, original_id: str, index: int) -> str:
        """Generate anonymous scenario ID from original."""
        import hashlib
        if original_id.startswith("scenario_"):
            return original_id
        # Use seed from config if available for consistency
        seed = self.config.sampling.seed or 42
        text = f"{seed}|{original_id}|{index}"
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"scenario_{digest}"

    def _load_crypto_from_postgres(self, config: "CryptoDatasetConfig") -> List[LoadedExample]:
        """Load crypto scenarios directly from PostgreSQL with hidden windows."""
        from cio_agent.crypto_benchmark import PostgresMarketDataLoader
        from cio_agent.hidden_windows import load_hidden_seed, select_evaluation_windows

        master_seed = load_hidden_seed(config.hidden_seed_config)
        if master_seed is None:
            import logging
            logging.warning(
                f"Could not load hidden seed '{config.hidden_seed_config}'. "
                "Falling back to JSON scenarios."
            )
            # Temporarily disable pg_enabled and recurse
            config_copy = config.model_copy()
            config_copy.pg_enabled = False
            return self._load_crypto(config_copy)

        # Create loader
        loader = PostgresMarketDataLoader(
            dsn=config.pg_dsn,
            host=config.pg_host,
            port=config.pg_port,
            dbname=config.pg_dbname,
            user=config.pg_user,
            password=config.pg_password,
            ohlcv_table=config.pg_ohlcv_table,
            funding_table=config.pg_funding_table,
        )

        examples = []
        try:
            conn = loader._connect()

            # Select windows using hidden seed
            windows = select_evaluation_windows(
                master_seed=master_seed,
                window_count=config.window_count,
                symbols=config.symbols,
                date_range=(config.date_range_start, config.date_range_end),
                min_bars=config.window_min_bars,
                max_bars=config.window_max_bars,
                conn=conn,
                ohlcv_table=config.pg_ohlcv_table,
            )

            if config.limit:
                windows = windows[:config.limit]

            for idx, window in enumerate(windows):
                from datetime import datetime

                start_dt = datetime.fromisoformat(window["start"].replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(window["end"].replace("Z", "+00:00"))

                # Load market data
                states = loader.load_window(
                    symbol=window["symbol"],
                    start=start_dt,
                    end=end_dt,
                    timeframe="1m",
                )

                if not states:
                    continue

                # Use anonymous scenario ID
                anon_id = window["scenario_id"]

                examples.append(LoadedExample(
                    example_id=anon_id,
                    question=f"Crypto trading scenario {idx + 1}",
                    answer="",
                    dataset_type="crypto",
                    metadata={
                        "scenario_id": anon_id,
                        "name": f"scenario_{idx + 1}",
                        "description": "",
                        "symbol": window["symbol"],
                        "market_states": states,  # Include states directly
                        "max_steps": config.max_steps,
                        "stride": config.stride,
                        "evaluation": config.evaluation.model_dump(),
                    },
                ))

        finally:
            loader.close()

        return examples

    def _apply_sampling(self, examples: List[LoadedExample]) -> List[LoadedExample]:
        """Apply sampling strategy to collected examples."""
        strategy = self.config.sampling.strategy
        total_limit = self.config.sampling.total_limit

        if strategy == "sequential":
            # No shuffle, just take in order
            result = examples
        
        elif strategy == "random":
            # Global shuffle
            result = examples.copy()
            random.shuffle(result)
        
        elif strategy == "stratified":
            # Equal samples per dataset type
            result = self._stratified_sample(examples)
        
        elif strategy == "weighted":
            # Sample by dataset weights
            result = self._weighted_sample(examples)
        
        else:
            result = examples

        # Apply total limit
        if total_limit and len(result) > total_limit:
            result = result[:total_limit]

        return result

    def _stratified_sample(self, examples: List[LoadedExample]) -> List[LoadedExample]:
        """Sample equal number from each dataset type."""
        by_type: Dict[str, List[LoadedExample]] = {}
        for ex in examples:
            key = f"{ex.dataset_type}_{ex.task_type or 'default'}"
            if key not in by_type:
                by_type[key] = []
            by_type[key].append(ex)
        
        # Shuffle within each group
        for group in by_type.values():
            random.shuffle(group)
        
        # Interleave: take one from each group in round-robin
        result = []
        max_len = max(len(g) for g in by_type.values()) if by_type else 0
        
        for i in range(max_len):
            for group in by_type.values():
                if i < len(group):
                    result.append(group[i])
        
        return result

    def _weighted_sample(self, examples: List[LoadedExample]) -> List[LoadedExample]:
        """Sample according to dataset weights."""
        # Get weights
        weight_map = {}
        for config in self.config.datasets:
            weight_map[config.type] = config.weight
        
        # Group by dataset type
        by_type: Dict[str, List[LoadedExample]] = {}
        for ex in examples:
            if ex.dataset_type not in by_type:
                by_type[ex.dataset_type] = []
            by_type[ex.dataset_type].append(ex)
        
        # Shuffle within groups
        for group in by_type.values():
            random.shuffle(group)
        
        # Calculate sample counts based on weights
        total_limit = self.config.sampling.total_limit or len(examples)
        total_weight = sum(weight_map.get(t, 1.0) for t in by_type.keys())
        
        result = []
        for dtype, group in by_type.items():
            weight = weight_map.get(dtype, 1.0)
            count = int((weight / total_weight) * total_limit)
            result.extend(group[:count])
        
        random.shuffle(result)
        return result

    def get_by_dataset(self, dataset_type: str) -> List[LoadedExample]:
        """Get examples for a specific dataset type."""
        if not self._loaded:
            self.load()
        return self._by_dataset.get(dataset_type, [])

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of loaded examples."""
        if not self._loaded:
            self.load()
        
        by_dataset = {}
        by_task = {}
        by_category = {}
        
        for ex in self._examples:
            # By dataset
            if ex.dataset_type not in by_dataset:
                by_dataset[ex.dataset_type] = 0
            by_dataset[ex.dataset_type] += 1
            
            # By task type
            if ex.task_type:
                if ex.task_type not in by_task:
                    by_task[ex.task_type] = 0
                by_task[ex.task_type] += 1
            
            # By category
            if ex.category:
                if ex.category not in by_category:
                    by_category[ex.category] = 0
                by_category[ex.category] += 1
        
        return {
            "total": len(self._examples),
            "by_dataset": by_dataset,
            "by_task_type": by_task,
            "by_category": by_category,
        }


# Default config for quick testing
def create_default_config() -> EvaluationConfig:
    """Create a default evaluation configuration."""
    return EvaluationConfig(
        name="FAB++ Default Evaluation",
        datasets=[
            BizFinBenchDatasetConfig(
                # Data fetched from HuggingFace BizFinBench.v2 API
                task_types=["event_logic_reasoning", "user_sentiment_analysis"],
                languages=["en"],
                limit_per_task=10,
            ),
            PublicCsvDatasetConfig(
                path="finance-agent/data/public.csv",
                limit=20,
            ),
            OptionsDatasetConfig(
                path="data/options/questions.json",
                limit=10,
            ),
        ],
        sampling=SamplingConfig(
            strategy="stratified",
            total_limit=50,
            seed=42,
        ),
    )
