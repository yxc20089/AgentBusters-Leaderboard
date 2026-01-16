"""
BizFinBench dataset provider using HuggingFace API.

Dynamically fetches data from HiThink-Research/BizFinBench on HuggingFace Hub.
No local data files required - questions are fetched on-demand and cached.

Usage:
    provider = BizFinBenchProvider(
        task_type="financial_quantitative_computation",
        language="en",
        limit=100
    )
    examples = provider.load()
    templates = provider.to_templates()
"""

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

from cio_agent.data_providers.base import DatasetExample, DatasetProvider
from cio_agent.models import (
    FABQuestionTemplate,
    GroundTruth,
    TaskCategory,
    TaskDifficulty,
    TaskRubric,
)

logger = logging.getLogger(__name__)

# Global cache for HuggingFace datasets (avoid re-downloading)
_dataset_cache: Dict[str, Any] = {}


class BizFinBenchProvider(DatasetProvider):
    """
    Provider for HiThink BizFinBench dataset via HuggingFace API.

    Dynamically fetches data from:
    https://huggingface.co/datasets/HiThink-Research/BizFinBench

    Supports 9 task types (HuggingFace subset names):
        - Anomalous_Event_Attribution
        - Financial_Numerical_Computation
        - Financial_Time_Reasoning
        - Financial_Data_Description
        - Stock_Price_Prediction
        - Financial_Named_Entity_Recognition
        - Emotion_Recognition
        - Financial_Tool_Usage
        - Financial_Knowledge_QA
    """

    name = "bizfinbench"

    # HuggingFace dataset identifier (v2 has both English and Chinese)
    HF_DATASET_ID = "HiThink-Research/BizFinBench.v2"

    # Map task names to JSONL filenames in the dataset
    # Format: {lang}/{task_type}_{lang}.jsonl
    TASK_FILES = {
        "anomaly_information_tracing": "anomaly_information_tracing",
        "event_logic_reasoning": "event_logic_reasoning",
        "financial_data_description": "financial_data_description",
        "financial_quantitative_computation": "financial_quantitative_computation",
        "user_sentiment_analysis": "user_sentiment_analysis",
        "stock_price_predict": "stock_price_predict",
        "financial_multi_turn_perception": "financial_multi-turn_perception",
        # Note: conterfactual_en.jsonl has JSON parsing issues, skip for now
        # "conterfactual": "conterfactual",
    }

    # Map task types to TaskCategory enum
    TASK_CATEGORY_MAP = {
        "anomaly_information_tracing": TaskCategory.COMPLEX_RETRIEVAL,
        "event_logic_reasoning": TaskCategory.QUALITATIVE_RETRIEVAL,
        "financial_data_description": TaskCategory.QUANTITATIVE_RETRIEVAL,
        "financial_quantitative_computation": TaskCategory.NUMERICAL_REASONING,
        "user_sentiment_analysis": TaskCategory.QUALITATIVE_RETRIEVAL,
        "stock_price_predict": TaskCategory.MARKET_ANALYSIS,
        "financial_multi_turn_perception": TaskCategory.COMPLEX_RETRIEVAL,
    }

    # Difficulty mapping based on task complexity
    TASK_DIFFICULTY_MAP = {
        "anomaly_information_tracing": TaskDifficulty.HARD,
        "event_logic_reasoning": TaskDifficulty.MEDIUM,
        "financial_data_description": TaskDifficulty.EASY,
        "financial_quantitative_computation": TaskDifficulty.MEDIUM,
        "user_sentiment_analysis": TaskDifficulty.MEDIUM,
        "stock_price_predict": TaskDifficulty.EXPERT,
        "financial_multi_turn_perception": TaskDifficulty.HARD,
    }

    def __init__(
        self,
        task_type: str,
        language: str = "en",
        limit: Optional[int] = None,
        base_path: Optional[str] = None,  # Ignored, kept for backward compatibility
    ):
        """
        Initialize BizFinBench provider with HuggingFace API.

        Args:
            task_type: Task type name (e.g., "event_logic_reasoning")
            language: "en" for English or "cn" for Chinese
            limit: Optional limit on number of examples to load
            base_path: Ignored - kept for backward compatibility

        Raises:
            ValueError: If task_type is unknown
        """
        self.task_type = task_type
        self.language = language if language != "zh" else "cn"
        self.limit = limit

        # Validate task type
        if task_type not in self.TASK_FILES:
            raise ValueError(
                f"Unknown task type: {task_type}. "
                f"Valid types: {list(self.TASK_FILES.keys())}"
            )

        # Build the file path: {lang}/{task_file}_{lang}.jsonl
        task_file = self.TASK_FILES[task_type]
        self.hf_file_path = f"{self.language}/{task_file}_{self.language}.jsonl"

        # Update provider name to be unique per task
        self.name = f"bizfinbench_{task_type}_{self.language}"

        logger.info(
            f"Initialized BizFinBenchProvider (HuggingFace v2): "
            f"task={task_type}, lang={self.language}, file={self.hf_file_path}"
        )

    @classmethod
    def list_task_types(cls, language: str = None) -> List[str]:
        """
        Return list of available task types.

        Args:
            language: Optional filter (not used, all tasks available in en/cn)

        Returns:
            List of task type names
        """
        return list(cls.TASK_FILES.keys())

    def _fetch_from_huggingface(self) -> List[Dict[str, Any]]:
        """
        Fetch dataset from HuggingFace Hub with caching.

        Uses file-based loading to fetch specific JSONL files from BizFinBench.v2.

        Returns:
            List of records from the dataset
        """
        cache_key = f"{self.HF_DATASET_ID}:{self.hf_file_path}"

        if cache_key in _dataset_cache:
            logger.debug(f"Using cached dataset: {cache_key}")
            return _dataset_cache[cache_key]

        logger.info(f"Fetching from HuggingFace: {self.HF_DATASET_ID} / {self.hf_file_path}")

        try:
            from datasets import load_dataset

            # Load specific file from HuggingFace using data_files parameter
            dataset = load_dataset(
                self.HF_DATASET_ID,
                data_files={"train": self.hf_file_path},
                split="train",
            )

            data = list(dataset)

            # Cache for future use
            _dataset_cache[cache_key] = data
            logger.info(f"Fetched {len(data)} records from HuggingFace")

            return data

        except Exception as e:
            logger.error(f"Failed to fetch from HuggingFace: {e}")
            raise RuntimeError(
                f"Could not fetch BizFinBench data from HuggingFace. "
                f"File: {self.hf_file_path}. Error: {e}"
            )

    def _extract_question(self, item: Dict[str, Any]) -> str:
        """
        Extract question from BizFinBench record.

        Format:
        {
            "messages": [
                {"role": "user", "content": [{"text": "...", "type": "text"}]}
            ]
        }
        """
        try:
            messages = item.get("messages", [])
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                return c.get("text", "")
                            elif isinstance(c, str):
                                return c
                    elif isinstance(content, str):
                        return content
        except Exception as e:
            logger.warning(f"Failed to extract question: {e}")

        return ""

    def _extract_answer(self, item: Dict[str, Any]) -> str:
        """
        Extract answer from BizFinBench record.

        Format:
        {
            "choices": [
                {"message": {"role": "assistant", "content": [{"text": "..."}]}}
            ]
        }
        """
        try:
            choices = item.get("choices", [])
            for choice in choices:
                msg = choice.get("message", {})
                if msg.get("role") == "assistant":
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                return c.get("text", "")
                            elif isinstance(c, str):
                                return c
                    elif isinstance(content, str):
                        return content
        except Exception as e:
            logger.warning(f"Failed to extract answer: {e}")

        return ""

    def load(self) -> List[DatasetExample]:
        """Load examples from HuggingFace API."""
        import random

        raw_data = self._fetch_from_huggingface()
        examples: List[DatasetExample] = []

        for idx, item in enumerate(raw_data):
            try:
                question = self._extract_question(item)
                answer = self._extract_answer(item)

                if not question:
                    continue

                example_id = f"bizfinbench_{self.task_type}_{self.language}_{idx}"
                category = self.TASK_CATEGORY_MAP.get(
                    self.task_type, TaskCategory.QUALITATIVE_RETRIEVAL
                )
                difficulty = self.TASK_DIFFICULTY_MAP.get(
                    self.task_type, TaskDifficulty.MEDIUM
                )

                ground_truth = GroundTruth(
                    macro_thesis=answer,
                    key_themes=[self.task_type],
                )

                examples.append(
                    DatasetExample(
                        example_id=example_id,
                        question=question,
                        answer=answer,
                        category=category,
                        difficulty=difficulty,
                        ground_truth=ground_truth,
                        source="bizfinbench_hf_v2",
                        metadata={
                            "hf_file": self.hf_file_path,
                            "task_type": self.task_type,
                            "language": self.language,
                        },
                    )
                )

            except Exception as e:
                logger.warning(f"Failed to process item {idx}: {e}")
                continue

        # Apply limit
        if self.limit and len(examples) > self.limit:
            random.shuffle(examples)
            examples = examples[:self.limit]

        logger.info(
            f"Loaded {len(examples)} examples from BizFinBench.v2 "
            f"(task={self.task_type}, lang={self.language})"
        )
        return examples

    def to_templates(self) -> List[FABQuestionTemplate]:
        """Convert loaded examples to FAB question templates."""
        templates: List[FABQuestionTemplate] = []
        for ex in self.load():
            templates.append(
                FABQuestionTemplate(
                    template_id=ex.example_id,
                    category=ex.category,
                    template=ex.question,
                    difficulty=ex.difficulty,
                    metric="bizfinbench",
                    rubric=TaskRubric(criteria=[]),
                    requires_code_execution=False,
                )
            )
        return templates


def clear_cache():
    """Clear the dataset cache (useful for testing or memory management)."""
    global _dataset_cache
    _dataset_cache.clear()
    logger.info("BizFinBench dataset cache cleared")
