"""
Dataset interfaces for pluggable finance Q/A sources.

Providers convert arbitrary datasets (CSV, JSONL, APIs, LLM-generated) into
`DatasetExample` objects and optionally into `FABQuestionTemplate` so the
dynamic task generator can consume them uniformly.
"""

from typing import Any, List, Optional

from pydantic import BaseModel, Field

from cio_agent.models import (
    FABQuestionTemplate,
    GroundTruth,
    TaskCategory,
    TaskDifficulty,
    TaskRubric,
)


class DatasetExample(BaseModel):
    """Single row/question from an external dataset."""

    example_id: str
    question: str
    answer: Optional[str] = None
    rubric: Optional[TaskRubric] = None
    category: TaskCategory
    difficulty: TaskDifficulty
    ground_truth: Optional[GroundTruth] = None
    source: str = "custom"
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetProvider:
    """
    Base class for dataset providers.

    Implementations load data from any source and expose templates that the
    task generator can consume directly.
    """

    name: str = "base"

    def load(self) -> List[DatasetExample]:
        """Return all dataset examples."""
        raise NotImplementedError

    def to_templates(self) -> List[FABQuestionTemplate]:
        """
        Convert loaded examples to FAB question templates.

        Concrete providers should set `template_id` to a stable value derived
        from the underlying dataset (e.g., file name + row index).
        """
        raise NotImplementedError
