"""
CSV-backed dataset provider for finance Q/A (e.g., data/public.csv).

Expected columns:
    Question, Answer, Question Type, Expert time (mins), Rubric

Rubric format (JSON array):
    [
        {"type": "required", "criteria": "Must identify growth trend"},
        {"type": "penalty", "criteria": "Should not confuse metrics"}
    ]

    Types:
        - "required": Criteria the answer SHOULD satisfy (positive scoring)
        - "penalty": Criteria the answer should NOT violate (score deductions)
"""

import csv
import json
import ast
import logging
from enum import Enum
from pathlib import Path
from typing import List, Union

from cio_agent.data_providers.base import DatasetExample, DatasetProvider
from cio_agent.models import (
    FABQuestionTemplate,
    GroundTruth,
    TaskCategory,
    TaskDifficulty,
    TaskRubric,
)

logger = logging.getLogger(__name__)


class RubricCriterionType(str, Enum):
    """Type of rubric criterion for evaluating agent responses."""

    REQUIRED = "required"  # Answer SHOULD satisfy this (positive scoring)
    PENALTY = "penalty"  # Answer should NOT violate this (score deduction)

# Required columns for the CSV dataset
REQUIRED_COLUMNS = {"Question", "Question Type"}
OPTIONAL_COLUMNS = {
    "Answer",  # Text answer / macro thesis
    "Expert time (mins)",  # Used to determine difficulty
    "Rubric",  # JSON array of criteria
    "Numerical Answer",  # For calculation tasks: exact numeric answer
    "Expected Recommendation",  # For investment tasks: Buy/Sell/Hold
    "Tolerance",  # Acceptable error margin for numerical answers (default 0.01 = 1%)
}

# Map CSV "Question Type" to internal categories
QUESTION_TYPE_MAP = {
    "Quantitative Retrieval": TaskCategory.QUANTITATIVE_RETRIEVAL,
    "Qualitative Retrieval": TaskCategory.QUALITATIVE_RETRIEVAL,
    "Numerical Reasoning": TaskCategory.NUMERICAL_REASONING,
    "Complex Retrieval": TaskCategory.COMPLEX_RETRIEVAL,
    "Adjustments": TaskCategory.ADJUSTMENTS,
    "Beat or Miss": TaskCategory.BEAT_OR_MISS,
    "Trends": TaskCategory.TRENDS,
    "Financial Modeling": TaskCategory.FINANCIAL_MODELING,
    "Market Analysis": TaskCategory.MARKET_ANALYSIS,
}


def _map_difficulty(expert_minutes: float) -> TaskDifficulty:
    """Heuristic mapping from expert time to difficulty."""
    if expert_minutes <= 5:
        return TaskDifficulty.EASY
    if expert_minutes <= 15:
        return TaskDifficulty.MEDIUM
    if expert_minutes <= 30:
        return TaskDifficulty.HARD
    return TaskDifficulty.EXPERT


def _process_rubric_items(
    data: list, row_index: int = -1
) -> tuple[list[str], list[str]]:
    """
    Process a list of rubric items into required criteria and penalty conditions.
    
    This is a helper function used by _parse_rubric for both JSON and Python literal paths.
    """
    required_criteria: list[str] = []
    penalty_conditions: list[str] = []
    
    for item in data:
        if not isinstance(item, dict):
            continue
        crit = item.get("criteria")
        if not crit:
            continue

        # Support both 'type' and 'operator' field names
        criterion_type_str = item.get("type") or item.get("operator")
        if not criterion_type_str:
            # Treat items without type as 'required' by default
            required_criteria.append(crit)
            continue
        
        # Normalize operator mappings
        if criterion_type_str in ("correctness", "required"):
            required_criteria.append(crit)
        elif criterion_type_str in ("contradiction", "penalty"):
            penalty_conditions.append(crit)
        else:
            logger.warning(
                f"Row {row_index}: Unknown rubric type '{criterion_type_str}'. "
                f"Valid types: {[t.value for t in RubricCriterionType]}"
            )
    
    return required_criteria, penalty_conditions


def _parse_rubric(raw: str, row_index: int = -1) -> tuple[list[str], list[str]]:
    """
    Parse rubric JSON into required criteria and penalty conditions.

    Expected format: [{"type": "required"|"penalty", "criteria": "..."}]

    Falls back to a simple single-criteria list if parsing fails.

    Returns:
        Tuple of (required_criteria, penalty_conditions)
    """
    if not raw:
        return [], []

    try:
        data = json.loads(raw)
        return _process_rubric_items(data, row_index)

    except json.JSONDecodeError:
        # Try Python literal syntax (single quotes) as fallback
        try:
            data = ast.literal_eval(raw)
            return _process_rubric_items(data, row_index)
        except (ValueError, SyntaxError):
            # Final fallback: use raw string when Python literal parsing fails
            return [raw], []
    except Exception as e:
        logger.warning(f"Row {row_index}: Unexpected error parsing rubric: {e}. Using raw value.")
        return [raw], []


class CsvFinanceDatasetProvider(DatasetProvider):
    """Provider that reads finance Q/A rows from a CSV file."""

    name = "csv_finance"

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._validated = False

    def _validate_columns(self, fieldnames: List[str]) -> None:
        """Validate that required columns exist in the CSV."""
        if fieldnames is None:
            raise ValueError(f"CSV file {self.path} appears to be empty or has no header row")

        available = set(fieldnames)
        missing = REQUIRED_COLUMNS - available
        if missing:
            raise ValueError(
                f"CSV file {self.path} is missing required columns: {missing}. "
                f"Required: {REQUIRED_COLUMNS}, Found: {available}"
            )

        # Log warning for missing optional columns
        missing_optional = OPTIONAL_COLUMNS - available
        if missing_optional:
            logger.info(f"CSV file {self.path} is missing optional columns: {missing_optional}")

        self._validated = True

    def load(self) -> List[DatasetExample]:
        rows: List[DatasetExample] = []
        with self.path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Validate columns on first read
            if not self._validated:
                self._validate_columns(reader.fieldnames)

            for idx, row in enumerate(reader):
                category = QUESTION_TYPE_MAP.get(
                    row.get("Question Type", ""), TaskCategory.QUALITATIVE_RETRIEVAL
                )
                difficulty = _map_difficulty(float(row.get("Expert time (mins)") or 10))
                criteria, penalties = _parse_rubric(row.get("Rubric", ""), row_index=idx)
                rubric = TaskRubric(criteria=criteria, penalty_conditions=penalties)

                # Parse optional numerical answer
                numerical_answer = None
                numerical_str = row.get("Numerical Answer", "").strip()
                if numerical_str:
                    try:
                        numerical_answer = float(numerical_str)
                    except ValueError:
                        logger.warning(f"Row {idx}: Invalid numerical answer '{numerical_str}'")

                # Parse optional tolerance
                tolerance = 0.01  # Default 1%
                tolerance_str = row.get("Tolerance", "").strip()
                if tolerance_str:
                    try:
                        tolerance = float(tolerance_str)
                    except ValueError:
                        logger.warning(f"Row {idx}: Invalid tolerance '{tolerance_str}'")

                ground_truth = GroundTruth(
                    macro_thesis=row.get("Answer", ""),
                    key_themes=criteria,
                    expected_recommendation=row.get("Expected Recommendation", "").strip() or None,
                    numerical_answer=numerical_answer,
                    tolerance=tolerance,
                )

                rows.append(
                    DatasetExample(
                        example_id=f"{self.name}_{idx}",
                        question=row.get("Question", ""),
                        answer=row.get("Answer"),
                        rubric=rubric,
                        category=category,
                        difficulty=difficulty,
                        ground_truth=ground_truth,
                        source=self.name,
                        metadata={"row_index": idx, "raw": row},
                    )
                )
        return rows

    def to_templates(self) -> List[FABQuestionTemplate]:
        """Convert CSV rows directly into FAB templates."""
        templates: List[FABQuestionTemplate] = []
        for ex in self.load():
            templates.append(
                FABQuestionTemplate(
                    template_id=ex.example_id,
                    category=ex.category,
                    template=ex.question,
                    difficulty=ex.difficulty,
                    metric="custom",
                    rubric=ex.rubric or TaskRubric(criteria=[]),
                    requires_code_execution=False,
                )
            )
        return templates
