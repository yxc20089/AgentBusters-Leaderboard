"""
Options Alpha Challenge dataset provider.

Provides access to options trading evaluation questions covering:
- Options Pricing (Black-Scholes)
- Greeks Analysis
- Strategy Construction
- Volatility Trading
- P&L Attribution
- Risk Management
- Copy Trading
- Race to 10M
- Strategy Defense

Usage:
    provider = OptionsDatasetProvider(
        path="data/options/questions.json",
        categories=["Options Pricing", "Greeks Analysis"],
        limit=50
    )
    examples = provider.load()
    templates = provider.to_templates()
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cio_agent.data_providers.base import DatasetExample, DatasetProvider
from cio_agent.models import (
    FABQuestionTemplate,
    GroundTruth,
    TaskCategory,
    TaskDifficulty,
    TaskRubric,
)

logger = logging.getLogger(__name__)


class OptionsDatasetProvider(DatasetProvider):
    """
    Provider for Options Alpha Challenge questions.

    Supports all 9 options task categories:
        - Options Pricing
        - Greeks Analysis
        - Strategy Construction
        - Volatility Trading
        - P&L Attribution
        - Risk Management
        - Copy Trading
        - Race to 10M
        - Strategy Defense
    """

    name = "options"

    # Map string category names to TaskCategory enum
    CATEGORY_MAP = {
        "Options Pricing": TaskCategory.OPTIONS_PRICING,
        "Greeks Analysis": TaskCategory.GREEKS_ANALYSIS,
        "Strategy Construction": TaskCategory.STRATEGY_CONSTRUCTION,
        "Volatility Trading": TaskCategory.VOLATILITY_TRADING,
        "P&L Attribution": TaskCategory.PNL_ATTRIBUTION,
        "Risk Management": TaskCategory.RISK_MANAGEMENT,
        "Copy Trading": TaskCategory.COPY_TRADING,
        "Race to 10M": TaskCategory.RACE_TO_10M,
        "Strategy Defense": TaskCategory.STRATEGY_DEFENSE,
    }

    # Map difficulty strings to TaskDifficulty enum
    DIFFICULTY_MAP = {
        "easy": TaskDifficulty.EASY,
        "medium": TaskDifficulty.MEDIUM,
        "hard": TaskDifficulty.HARD,
        "expert": TaskDifficulty.EXPERT,
    }

    def __init__(
        self,
        path: Union[str, Path] = "data/options/questions.json",
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
        shuffle: bool = False,
    ):
        """
        Initialize Options dataset provider.

        Args:
            path: Path to options questions JSON file
            categories: List of category names to include. None means all.
            limit: Optional limit on number of examples to load
            shuffle: Whether to shuffle examples before limiting

        Raises:
            FileNotFoundError: If the questions file doesn't exist
            ValueError: If invalid category specified
        """
        self.path = Path(path)
        self.categories = categories
        self.limit = limit
        self.shuffle = shuffle

        # Validate categories if specified
        if categories:
            for cat in categories:
                if cat not in self.CATEGORY_MAP:
                    raise ValueError(
                        f"Unknown options category: {cat}. "
                        f"Valid categories: {list(self.CATEGORY_MAP.keys())}"
                    )

        logger.info(
            f"Initialized OptionsDatasetProvider: path={path}, "
            f"categories={categories}, limit={limit}"
        )

    @classmethod
    def list_categories(cls) -> List[str]:
        """Return list of available options categories."""
        return list(cls.CATEGORY_MAP.keys())

    def _load_json(self) -> List[Dict[str, Any]]:
        """Load and parse the questions JSON file."""
        if not self.path.exists():
            raise FileNotFoundError(f"Options questions file not found: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle wrapped format {"questions": [...]} or direct list
        if isinstance(data, dict) and "questions" in data:
            questions = data["questions"]
        elif isinstance(data, list):
            questions = data
        else:
            raise ValueError(f"Invalid options questions format: {self.path}")

        return questions

    def _get_category(self, item: Dict[str, Any]) -> TaskCategory:
        """Map item category string to TaskCategory enum."""
        cat_str = item.get("category", "Options Pricing")
        return self.CATEGORY_MAP.get(cat_str, TaskCategory.OPTIONS_PRICING)

    def _get_difficulty(self, item: Dict[str, Any]) -> TaskDifficulty:
        """Map item difficulty string to TaskDifficulty enum."""
        diff_str = item.get("difficulty", "medium").lower()
        return self.DIFFICULTY_MAP.get(diff_str, TaskDifficulty.MEDIUM)

    def _build_rubric(self, item: Dict[str, Any]) -> TaskRubric:
        """Build evaluation rubric from item's rubric field."""
        rubric_data = item.get("rubric", {})
        components = rubric_data.get("components", [])

        # Convert components to criteria strings
        criteria = []
        for comp in components:
            name = comp.get("name", "")
            weight = comp.get("weight", 0)
            desc = comp.get("description", "")
            criteria.append(f"{name} ({int(weight*100)}%): {desc}")

        if not criteria:
            # Default criteria based on category
            category = item.get("category", "")
            if "Pricing" in category:
                criteria = ["Price accuracy within 5%", "Correct methodology"]
            elif "Greeks" in category:
                criteria = ["Greeks accuracy within 10%", "Proper interpretation"]
            elif "Strategy" in category:
                criteria = ["Valid strategy structure", "Correct P&L analysis"]
            else:
                criteria = ["Accurate analysis", "Sound reasoning"]

        return TaskRubric(criteria=criteria, penalty_conditions=[])

    def _format_ground_truth(self, item: Dict[str, Any]) -> str:
        """Format ground truth as string for evaluation."""
        gt = item.get("ground_truth", {})
        if isinstance(gt, dict):
            # Format key-value pairs
            parts = []
            for k, v in gt.items():
                if isinstance(v, (list, dict)):
                    parts.append(f"{k}: {json.dumps(v)}")
                else:
                    parts.append(f"{k}: {v}")
            return "; ".join(parts)
        return str(gt)

    def load(self) -> List[DatasetExample]:
        """Load all examples from the options questions file."""
        import random

        questions = self._load_json()
        examples: List[DatasetExample] = []

        for item in questions:
            try:
                # Filter by category if specified
                cat_str = item.get("category", "")
                if self.categories and cat_str not in self.categories:
                    continue

                question_id = item.get("question_id", f"options_{len(examples)}")
                question = item.get("question", "")
                answer = self._format_ground_truth(item)
                category = self._get_category(item)
                difficulty = self._get_difficulty(item)
                rubric = self._build_rubric(item)

                ground_truth = GroundTruth(
                    macro_thesis=answer,
                    key_themes=[cat_str],
                )

                examples.append(
                    DatasetExample(
                        example_id=question_id,
                        question=question,
                        answer=answer,
                        rubric=rubric,
                        category=category,
                        difficulty=difficulty,
                        ground_truth=ground_truth,
                        source="options",
                        metadata={
                            "category": cat_str,
                            "ticker": item.get("ticker"),
                            "ground_truth_raw": item.get("ground_truth"),
                            "rubric_raw": item.get("rubric"),
                            **{k: v for k, v in item.items()
                               if k not in ["question_id", "question", "category",
                                           "difficulty", "ground_truth", "rubric"]}
                        },
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to process options question: {e}")
                continue

        # Shuffle if requested
        if self.shuffle:
            random.shuffle(examples)

        # Apply limit
        if self.limit and len(examples) > self.limit:
            examples = examples[:self.limit]

        logger.info(f"Loaded {len(examples)} options examples from {self.path}")
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
                    metric="options",
                    rubric=ex.rubric or TaskRubric(criteria=[]),
                    requires_code_execution=True,  # Options often need calculations
                )
            )
        return templates
