"""
Base class for dataset-specific evaluators.

Each dataset (public.csv, BizFinBench.v2, Options, etc.) has its own evaluation criteria.
This module provides the abstract base class that all dataset evaluators implement.

SCORING CONVENTION:
    - All evaluators should return scores normalized to 0.0 - 1.0 scale
    - BizFinBenchEvaluator: 0.0 or 1.0 (binary correct/incorrect)
    - PublicCsvEvaluator: 0.0 to 1.0 (correctness - penalties, normalized)
    - OptionsEvaluator: Returns 0-100 internally, caller should normalize to 0-1

    The GreenAgent normalizes all scores to 0-1 for consistent aggregation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EvalResult:
    """Result from a dataset-specific evaluation."""
    
    score: float  # 0.0 to 1.0
    max_score: float = 1.0
    correct_count: int = 0
    total_count: int = 0
    details: Dict[str, Any] = None
    feedback: str = ""
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
    
    @property
    def percentage(self) -> float:
        """Return score as percentage (0-100)."""
        return (self.score / self.max_score) * 100 if self.max_score > 0 else 0.0
    
    @property
    def is_correct(self) -> bool:
        """Return True if score equals max score."""
        return abs(self.score - self.max_score) < 0.001


class BaseDatasetEvaluator(ABC):
    """
    Abstract base class for dataset-specific evaluators.
    
    Each dataset has its own evaluation strategy:
    - public.csv: correctness/contradiction rubric operators
    - BizFinBench.v2: exact match, numerical tolerance, etc.
    
    Subclasses must implement:
        - evaluate(): Evaluate predicted answer against expected
    """
    
    name: str = "base"  # Evaluator identifier
    
    @abstractmethod
    def evaluate(
        self,
        predicted: str,
        expected: str,
        **kwargs
    ) -> EvalResult:
        """
        Evaluate predicted answer against expected.
        
        Args:
            predicted: The model's predicted answer
            expected: The ground truth answer
            **kwargs: Additional evaluation parameters (rubric, task_type, etc.)
            
        Returns:
            EvalResult with score and details
        """
        pass
    
    def evaluate_batch(
        self,
        predictions: List[str],
        expected_list: List[str],
        **kwargs
    ) -> List[EvalResult]:
        """
        Evaluate multiple predictions.
        
        Args:
            predictions: List of predicted answers
            expected_list: List of expected answers
            **kwargs: Additional parameters passed to evaluate()
            
        Returns:
            List of EvalResults
        """
        results = []
        for pred, exp in zip(predictions, expected_list):
            results.append(self.evaluate(pred, exp, **kwargs))
        return results
    
    def aggregate_results(self, results: List[EvalResult]) -> Dict[str, Any]:
        """
        Aggregate multiple evaluation results into summary statistics.
        
        Args:
            results: List of EvalResults
            
        Returns:
            Dictionary with aggregated statistics
        """
        if not results:
            return {"count": 0, "mean_score": 0.0, "accuracy": 0.0}
        
        total_score = sum(r.score for r in results)
        total_max = sum(r.max_score for r in results)
        correct_count = sum(1 for r in results if r.is_correct)
        
        return {
            "count": len(results),
            "mean_score": total_score / len(results) if results else 0.0,
            "total_score": total_score,
            "total_max_score": total_max,
            "accuracy": correct_count / len(results) if results else 0.0,
            "correct_count": correct_count,
        }
