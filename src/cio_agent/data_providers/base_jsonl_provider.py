"""
Base JSONL provider for datasets stored in JSON Lines format.

This provides a reusable foundation for any JSONL-based dataset that can be
extended by specific dataset providers like BizFinBenchProvider.
"""

import json
import logging
from pathlib import Path
from typing import Iterator, List, Optional, Union

from cio_agent.data_providers.base import DatasetExample, DatasetProvider
from cio_agent.models import (
    FABQuestionTemplate,
    GroundTruth,
    TaskCategory,
    TaskDifficulty,
    TaskRubric,
)

logger = logging.getLogger(__name__)


class BaseJSONLProvider(DatasetProvider):
    """
    Base provider for JSONL format datasets.
    
    Subclasses should implement:
        - _extract_question(item): Extract question text from a JSONL item
        - _extract_answer(item): Extract answer text from a JSONL item
        - _get_category(item): Map item to TaskCategory
        - _get_difficulty(item): Map item to TaskDifficulty
    """

    name = "jsonl_base"

    def __init__(self, path: Union[str, Path], limit: Optional[int] = None):
        """
        Initialize the JSONL provider.
        
        Args:
            path: Path to the JSONL file
            limit: Optional limit on number of examples to load
        """
        self.path = Path(path)
        self.limit = limit
        self._validated = False

    def _parse_jsonl(self) -> Iterator[dict]:
        """Yield parsed JSON objects from JSONL file."""
        if not self.path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.path}")
        
        count = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    yield item
                    count += 1
                    if self.limit and count >= self.limit:
                        break
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Failed to parse JSON: {e}")
                    continue

    def _extract_question(self, item: dict) -> str:
        """
        Override in subclass to extract question from item.
        
        Args:
            item: A parsed JSON object from the JSONL file
            
        Returns:
            The question text
        """
        raise NotImplementedError("Subclass must implement _extract_question")

    def _extract_answer(self, item: dict) -> str:
        """
        Override in subclass to extract answer from item.
        
        Args:
            item: A parsed JSON object from the JSONL file
            
        Returns:
            The answer text
        """
        raise NotImplementedError("Subclass must implement _extract_answer")

    def _get_category(self, item: dict) -> TaskCategory:
        """
        Override in subclass to determine task category.
        
        Args:
            item: A parsed JSON object from the JSONL file
            
        Returns:
            The appropriate TaskCategory
        """
        return TaskCategory.QUALITATIVE_RETRIEVAL

    def _get_difficulty(self, item: dict) -> TaskDifficulty:
        """
        Override in subclass to determine task difficulty.
        
        Args:
            item: A parsed JSON object from the JSONL file
            
        Returns:
            The appropriate TaskDifficulty
        """
        return TaskDifficulty.MEDIUM

    def _build_rubric(self, item: dict, answer: str) -> TaskRubric:
        """
        Build rubric from item. Can be overridden for specific datasets.
        
        Args:
            item: A parsed JSON object from the JSONL file
            answer: The extracted answer text
            
        Returns:
            TaskRubric for evaluation
        """
        criteria = [f"Answer should match: {answer[:200]}..."] if len(answer) > 200 else [f"Answer should match: {answer}"]
        return TaskRubric(criteria=criteria, penalty_conditions=[])

    def load(self) -> List[DatasetExample]:
        """Load all examples from the JSONL file."""
        examples: List[DatasetExample] = []
        
        for idx, item in enumerate(self._parse_jsonl()):
            try:
                question = self._extract_question(item)
                answer = self._extract_answer(item)
                category = self._get_category(item)
                difficulty = self._get_difficulty(item)
                rubric = self._build_rubric(item, answer)
                
                ground_truth = GroundTruth(
                    macro_thesis=answer,
                    key_themes=[answer[:500]] if answer else [],
                )
                
                examples.append(
                    DatasetExample(
                        example_id=f"{self.name}_{idx}",
                        question=question,
                        answer=answer,
                        rubric=rubric,
                        category=category,
                        difficulty=difficulty,
                        ground_truth=ground_truth,
                        source=self.name,
                        metadata={"row_index": idx, "raw": item},
                    )
                )
            except Exception as e:
                logger.warning(f"Item {idx}: Failed to process: {e}")
                continue
        
        logger.info(f"Loaded {len(examples)} examples from {self.path}")
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
                    metric="custom",
                    rubric=ex.rubric or TaskRubric(criteria=[]),
                    requires_code_execution=False,
                )
            )
        return templates
