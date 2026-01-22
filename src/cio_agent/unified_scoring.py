"""
Unified Scoring System for FAB++ Benchmark.

Normalizes all evaluator outputs to 0-100 scale and computes
a weighted overall score across sections.

Sections and Weights:
- Knowledge Retrieval (30%): bizfinbench, public_csv
- Analytical Reasoning (35%): synthetic
- Options Trading (35%): options
- Crypto Trading (20%): crypto
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class ScoreSection(str, Enum):
    """Benchmark sections for scoring."""
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    ANALYTICAL_REASONING = "analytical_reasoning"
    OPTIONS_TRADING = "options_trading"
    CRYPTO_TRADING = "crypto_trading"


# Dataset to section mapping
DATASET_SECTION_MAP: dict[str, ScoreSection] = {
    "bizfinbench": ScoreSection.KNOWLEDGE_RETRIEVAL,
    "public_csv": ScoreSection.KNOWLEDGE_RETRIEVAL,
    "synthetic": ScoreSection.ANALYTICAL_REASONING,
    "options": ScoreSection.OPTIONS_TRADING,
    "crypto": ScoreSection.CRYPTO_TRADING,
}

# Default section weights (sum to 1.0)
DEFAULT_SECTION_WEIGHTS: dict[ScoreSection, float] = {
    ScoreSection.KNOWLEDGE_RETRIEVAL: 0.30,
    ScoreSection.ANALYTICAL_REASONING: 0.35,
    ScoreSection.OPTIONS_TRADING: 0.35,
    ScoreSection.CRYPTO_TRADING: 0.25,
}

# Grade thresholds
GRADE_THRESHOLDS: dict[str, float] = {
    "A+": 97, "A": 93, "A-": 90,
    "B+": 87, "B": 83, "B-": 80,
    "C+": 77, "C": 73, "C-": 70,
    "D": 60, "F": 0,
}


@dataclass
class NormalizedTaskResult:
    """A single task result normalized to 0-100 scale."""
    task_id: str
    dataset_type: str
    section: ScoreSection
    raw_score: float
    normalized_score: float  # 0-100
    is_correct: bool
    feedback: str = ""
    sub_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SectionScore:
    """Score for a benchmark section."""
    score: float  # 0-100
    max_score: float = 100.0
    weight: float = 0.0
    weighted_contribution: float = 0.0
    datasets: list[str] = field(default_factory=list)
    task_count: int = 0
    accuracy: float = 0.0
    sub_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class OverallScore:
    """Final unified score."""
    score: float  # 0-100
    max_score: float = 100.0
    grade: str = ""


@dataclass
class UnifiedEvaluationResult:
    """Complete unified evaluation result."""
    schema_version: str = "2.0"
    benchmark: str = "FAB++ Finance Agent Benchmark"
    version: str = "2.0.0"
    overall_score: Optional[OverallScore] = None
    section_scores: dict[str, SectionScore] = field(default_factory=dict)
    evaluation_metadata: dict[str, Any] = field(default_factory=dict)
    detailed_results: list[NormalizedTaskResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "benchmark": self.benchmark,
            "version": self.version,
            "overall_score": {
                "score": round(self.overall_score.score, 2) if self.overall_score else 0,
                "max_score": self.overall_score.max_score if self.overall_score else 100,
            },
            "section_scores": {
                section_name: {
                    "score": round(ss.score, 2),
                    "weight": round(ss.weight, 4),
                    "weighted_contribution": round(ss.weighted_contribution, 2),
                    "datasets": ss.datasets,
                    "task_count": ss.task_count,
                    "accuracy": round(ss.accuracy, 4),
                    "sub_scores": {k: round(v, 2) for k, v in ss.sub_scores.items()} if ss.sub_scores else {},
                }
                for section_name, ss in self.section_scores.items()
            },
            "evaluation_metadata": self.evaluation_metadata,
            "detailed_results": [
                {
                    "task_id": r.task_id,
                    "dataset_type": r.dataset_type,
                    "section": r.section.value,
                    "raw_score": round(r.raw_score, 4),
                    "normalized_score": round(r.normalized_score, 2),
                    "is_correct": r.is_correct,
                    "feedback": r.feedback,
                }
                for r in self.detailed_results
            ],
        }


class UnifiedScorer:
    """
    Unified scoring system for FAB++ benchmark.

    Normalizes all evaluator outputs and computes weighted overall score.
    """

    def __init__(
        self,
        section_weights: Optional[dict[ScoreSection, float]] = None,
        grade_thresholds: Optional[dict[str, float]] = None,
    ):
        self.section_weights = section_weights or DEFAULT_SECTION_WEIGHTS
        self.grade_thresholds = grade_thresholds or GRADE_THRESHOLDS

    def normalize_score(
        self,
        score: float,
        dataset_type: str,
    ) -> float:
        """
        Normalize any score to 0-100 range.

        Args:
            score: Raw score from evaluator
            dataset_type: Type of dataset (bizfinbench, public_csv, options, synthetic)

        Returns:
            Normalized score in 0-100 range
        """
        if dataset_type in ("bizfinbench", "public_csv", "synthetic"):
            # These evaluators return 0.0-1.0
            return min(100.0, max(0.0, score * 100.0))
        elif dataset_type in ("options", "crypto"):
            # Options and crypto evaluators return 0-100
            return min(100.0, max(0.0, score))
        else:
            # Unknown - assume 0-1 if <= 1, otherwise cap at 100
            if score <= 1.0:
                return min(100.0, max(0.0, score * 100.0))
            return min(100.0, max(0.0, score))

    def get_section_for_dataset(self, dataset_type: str) -> Optional[ScoreSection]:
        """Get the section for a given dataset type."""
        return DATASET_SECTION_MAP.get(dataset_type)

    def create_normalized_result(
        self,
        task_id: str,
        dataset_type: str,
        raw_score: float,
        is_correct: bool,
        feedback: str = "",
        sub_scores: Optional[dict[str, float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[NormalizedTaskResult]:
        """
        Create a normalized task result.

        Args:
            task_id: Unique task identifier
            dataset_type: Type of dataset
            raw_score: Raw score from evaluator
            is_correct: Whether the answer was correct
            feedback: Evaluation feedback
            sub_scores: Optional sub-scores (e.g., for options: pnl_accuracy, etc.)
            metadata: Optional additional metadata

        Returns:
            NormalizedTaskResult or None if dataset type is unknown
        """
        section = self.get_section_for_dataset(dataset_type)
        if section is None:
            return None

        normalized_score = self.normalize_score(raw_score, dataset_type)

        return NormalizedTaskResult(
            task_id=task_id,
            dataset_type=dataset_type,
            section=section,
            raw_score=raw_score,
            normalized_score=normalized_score,
            is_correct=is_correct,
            feedback=feedback,
            sub_scores=sub_scores or {},
            metadata=metadata or {},
        )

    def compute_section_scores(
        self,
        task_results: list[NormalizedTaskResult],
    ) -> dict[ScoreSection, SectionScore]:
        """
        Compute scores for each section from task results.

        Args:
            task_results: List of normalized task results

        Returns:
            Dictionary mapping sections to their scores
        """
        # Group results by section
        section_results: dict[ScoreSection, list[NormalizedTaskResult]] = {}

        for result in task_results:
            section = result.section
            if section not in section_results:
                section_results[section] = []
            section_results[section].append(result)

        # Calculate active weights (redistribute if some sections are missing)
        active_sections = set(section_results.keys())
        weights = self._redistribute_weights(active_sections)

        # Calculate scores for each section
        section_scores: dict[ScoreSection, SectionScore] = {}

        for section, results in section_results.items():
            if not results:
                continue

            # Calculate average score
            avg_score = sum(r.normalized_score for r in results) / len(results)

            # Calculate accuracy (proportion of correct answers)
            correct_count = sum(1 for r in results if r.is_correct)
            accuracy = correct_count / len(results)

            # Get unique datasets in this section
            datasets = list(set(r.dataset_type for r in results))

            # Aggregate sub-scores if available (for options)
            sub_scores = self._aggregate_sub_scores(results)

            # Get weight and calculate contribution
            weight = weights.get(section, 0.0)
            weighted_contribution = avg_score * weight

            section_scores[section] = SectionScore(
                score=avg_score,
                weight=weight,
                weighted_contribution=weighted_contribution,
                datasets=datasets,
                task_count=len(results),
                accuracy=accuracy,
                sub_scores=sub_scores,
            )

        return section_scores

    def compute_overall_score(
        self,
        section_scores: dict[ScoreSection, SectionScore],
    ) -> OverallScore:
        """
        Compute weighted overall score from section scores.

        Args:
            section_scores: Dictionary of section scores

        Returns:
            Overall score with grade
        """
        if not section_scores:
            return OverallScore(score=0.0, grade="F")

        # Sum weighted contributions
        total = sum(s.weighted_contribution for s in section_scores.values())

        # Determine grade
        grade = self._score_to_grade(total)

        return OverallScore(
            score=round(total, 2),
            grade=grade,
        )

    def compute_unified_result(
        self,
        task_results: list[NormalizedTaskResult],
        purple_agent_url: str = "",
        conduct_debate: bool = False,
    ) -> UnifiedEvaluationResult:
        """
        Compute complete unified evaluation result.

        Args:
            task_results: List of normalized task results
            purple_agent_url: URL of the purple agent
            conduct_debate: Whether debate was conducted

        Returns:
            Complete unified evaluation result
        """
        # Compute section scores
        section_scores = self.compute_section_scores(task_results)

        # Compute overall score
        overall_score = self.compute_overall_score(section_scores)

        # Convert section scores dict to use string keys
        section_scores_dict = {
            section.value: score
            for section, score in section_scores.items()
        }

        # Build metadata
        num_successful = sum(1 for r in task_results if r.is_correct or r.normalized_score > 0)

        return UnifiedEvaluationResult(
            overall_score=overall_score,
            section_scores=section_scores_dict,
            evaluation_metadata={
                "purple_agent": purple_agent_url,
                "num_tasks": len(task_results),
                "num_successful": num_successful,
                "conduct_debate": conduct_debate,
            },
            detailed_results=task_results,
        )

    def _redistribute_weights(
        self,
        active_sections: set[ScoreSection],
    ) -> dict[ScoreSection, float]:
        """
        Redistribute weights for active sections only.

        When some sections are missing, proportionally redistribute
        their weights among the active sections.

        Args:
            active_sections: Set of sections with results

        Returns:
            Dictionary of redistributed weights
        """
        if not active_sections:
            return {}

        total_weight = sum(
            self.section_weights[s] for s in active_sections
        )

        if total_weight == 0:
            # Equal weights if all active sections have 0 weight
            equal_weight = 1.0 / len(active_sections)
            return {s: equal_weight for s in active_sections}

        return {
            s: self.section_weights[s] / total_weight
            for s in active_sections
        }

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        for grade, threshold in sorted(
            self.grade_thresholds.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if score >= threshold:
                return grade
        return "F"

    def _aggregate_sub_scores(
        self,
        results: list[NormalizedTaskResult],
    ) -> dict[str, float]:
        """
        Aggregate sub-scores across multiple results.

        For options tasks, this aggregates pnl_accuracy, greeks_accuracy, etc.

        Args:
            results: List of task results

        Returns:
            Dictionary of aggregated sub-scores
        """
        if not results:
            return {}

        # Collect all sub-score keys
        all_keys: set[str] = set()
        for r in results:
            all_keys.update(r.sub_scores.keys())

        if not all_keys:
            return {}

        # Aggregate each sub-score
        aggregated: dict[str, float] = {}
        for key in all_keys:
            values = [r.sub_scores.get(key, 0) for r in results if key in r.sub_scores]
            if values:
                aggregated[key] = sum(values) / len(values)

        return aggregated
