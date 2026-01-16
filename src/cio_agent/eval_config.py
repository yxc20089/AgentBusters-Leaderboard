"""
Evaluation configuration for Green Agent.

Supports multi-dataset evaluation with configurable:
- Multiple datasets (BizFinBench, public.csv, etc.)
- Task type filtering
- Language filtering
- Shuffle and sampling strategies
"""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator


class BizFinBenchDatasetConfig(BaseModel):
    """Configuration for BizFinBench dataset."""
    type: Literal["bizfinbench"] = "bizfinbench"
    path: str = "data/BizFinBench.v2"
    task_types: List[str] = Field(
        default=["event_logic_reasoning"],
        description="Task types to include. Use 'all' for all available."
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
        default=True,
        description="Shuffle examples within this dataset"
    )
    weight: float = Field(
        default=1.0,
        description="Sampling weight relative to other datasets"
    )

    @field_validator("task_types", mode="before")
    @classmethod
    def expand_all_task_types(cls, v):
        if v == "all" or v == ["all"]:
            return [
                "event_logic_reasoning",
                "user_sentiment_analysis",
                "financial_quantitative_computation",
                "industry_chain_knowledge",
                "financial_terminology_quiz",
                "conterfactual",  # Note: spelling matches dataset
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
        default=True,
        description="Shuffle examples"
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
    shuffle: bool = True
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
        default=True,
        description="Shuffle examples"
    )
    weight: float = Field(
        default=1.0,
        description="Sampling weight relative to other datasets"
    )


# Union type for all dataset configs
DatasetConfig = Union[BizFinBenchDatasetConfig, PublicCsvDatasetConfig, SyntheticDatasetConfig, OptionsDatasetConfig]


class SamplingConfig(BaseModel):
    """Configuration for sampling strategy."""
    strategy: Literal["sequential", "random", "stratified", "weighted"] = Field(
        default="random",
        description=(
            "sequential: no shuffle, take in order. "
            "random: global shuffle. "
            "stratified: equal samples per dataset/category. "
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

        # Set random seed for reproducibility
        if self.config.sampling.seed is not None:
            random.seed(self.config.sampling.seed)

        all_examples = []

        for dataset_config in self.config.datasets:
            examples = self._load_dataset(dataset_config)
            
            # Per-dataset shuffle
            if dataset_config.shuffle:
                random.shuffle(examples)
            
            self._by_dataset[dataset_config.type] = examples
            all_examples.extend(examples)

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
        else:
            raise ValueError(f"Unknown dataset type: {config.type}")

    def _load_bizfinbench(self, config: BizFinBenchDatasetConfig) -> List[LoadedExample]:
        """Load BizFinBench examples."""
        from cio_agent.datasets import BizFinBenchProvider
        
        examples = []
        
        for task_type in config.task_types:
            for language in config.languages:
                try:
                    provider = BizFinBenchProvider(
                        base_path=config.path,
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
                except FileNotFoundError:
                    # Skip if task type / language combination doesn't exist
                    continue
        
        return examples

    def _load_public_csv(self, config: PublicCsvDatasetConfig) -> List[LoadedExample]:
        """Load public.csv examples."""
        from cio_agent.datasets import CsvFinanceDatasetProvider
        
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
        
        with open(config.path, "r") as f:
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
        from cio_agent.datasets import OptionsDatasetProvider

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
                path="data/BizFinBench.v2",
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
