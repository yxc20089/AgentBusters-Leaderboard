"""
Robustness Integration for FAB++ Evaluation Pipeline.

Integrates adversarial robustness testing into the main evaluation flow.
When enabled, a sample of questions are tested with perturbations and
the robustness score is included in the final unified score.

Flow:
1. For each question in the robustness sample:
   a. Generate perturbations (paraphrase, typo, distraction, etc.)
   b. Send original + perturbations to Purple Agent
   c. Compare answers for consistency
   d. Calculate per-question robustness score
2. Aggregate into overall robustness score (0-100)
3. Include in unified scoring as ROBUSTNESS section
"""

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, List

import structlog

from cio_agent.adversarial_robustness import (
    AdversarialPerturbationEngine,
    AttackType,
    PerturbedQuestion,
)

logger = structlog.get_logger()


@dataclass
class RobustnessConfig:
    """Configuration for robustness testing."""
    enabled: bool = False
    sample_ratio: float = 0.2  # Test 20% of questions for robustness
    min_samples: int = 3  # Minimum questions to test
    max_samples: int = 10  # Maximum questions to test
    attack_types: List[str] = field(default_factory=lambda: [
        "paraphrase", "typo", "distraction"
    ])
    attack_intensity: float = 0.5
    seed: int = 42
    
    @classmethod
    def from_dict(cls, data: dict) -> "RobustnessConfig":
        """Create config from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            sample_ratio=data.get("sample_ratio", 0.2),
            min_samples=data.get("min_samples", 3),
            max_samples=data.get("max_samples", 10),
            attack_types=data.get("attack_types", ["paraphrase", "typo", "distraction"]),
            attack_intensity=data.get("attack_intensity", 0.5),
            seed=data.get("seed", 42),
        )


@dataclass
class QuestionRobustnessResult:
    """Robustness result for a single question."""
    question_id: str
    original_answer: str
    consistency_scores: dict[str, float]  # attack_type -> score
    avg_consistency: float
    is_robust: bool  # True if avg_consistency >= 0.7
    vulnerabilities: list[str]


@dataclass
class IntegratedRobustnessResult:
    """Aggregated robustness result for the evaluation."""
    enabled: bool
    questions_tested: int
    overall_score: float  # 0-100 scale
    grade: str  # A, B, C, D, F
    attack_type_scores: dict[str, float]
    vulnerable_questions: int
    question_results: list[QuestionRobustnessResult]
    
    def to_eval_result(self) -> dict:
        """Convert to evaluation result format for unified scoring."""
        return {
            "example_id": "robustness_aggregate",
            "dataset_type": "robustness",
            "score": self.overall_score / 100.0,  # Normalize to 0-1 for scorer
            "is_correct": self.overall_score >= 70,
            "feedback": f"Robustness Grade: {self.grade}, Score: {self.overall_score:.1f}/100",
            "questions_tested": self.questions_tested,
            "attack_type_scores": self.attack_type_scores,
            "vulnerable_questions": self.vulnerable_questions,
        }


class RobustnessIntegration:
    """
    Integrates robustness testing into the evaluation pipeline.
    
    Usage:
        integration = RobustnessIntegration(config)
        
        # During evaluation loop:
        for question in questions:
            answer = await get_answer(question)
            integration.record_answer(question.id, question.text, answer)
        
        # After all questions evaluated:
        robustness_result = await integration.evaluate_robustness(answer_func)
    """
    
    GRADE_THRESHOLDS = {
        "A": 90, "B": 80, "C": 70, "D": 60, "F": 0,
    }
    
    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.engine = AdversarialPerturbationEngine(
            seed=config.seed,
            attack_intensity=config.attack_intensity,
        )
        
        # Parse attack types
        self.attack_types = [
            AttackType(at) for at in config.attack_types
            if at in [e.value for e in AttackType]
        ]
        if not self.attack_types:
            self.attack_types = [AttackType.PARAPHRASE, AttackType.TYPO, AttackType.DISTRACTION]
        
        # Store answers for consistency comparison
        self._recorded_answers: dict[str, tuple[str, str]] = {}  # id -> (question, answer)
        self._rng = random.Random(config.seed)
    
    def record_answer(self, question_id: str, question_text: str, answer: str) -> None:
        """Record an answer for potential robustness testing."""
        self._recorded_answers[question_id] = (question_text, answer)
    
    def select_robustness_sample(self) -> list[str]:
        """Select which questions to test for robustness."""
        all_ids = list(self._recorded_answers.keys())
        
        if not all_ids:
            return []
        
        # Calculate sample size
        sample_size = int(len(all_ids) * self.config.sample_ratio)
        sample_size = max(self.config.min_samples, min(self.config.max_samples, sample_size))
        sample_size = min(sample_size, len(all_ids))
        
        # Random sample
        return self._rng.sample(all_ids, sample_size)
    
    async def evaluate_robustness(
        self,
        answer_func: Callable[[str], Any],  # async function: question -> answer
    ) -> IntegratedRobustnessResult:
        """
        Run robustness evaluation on sampled questions.
        
        Args:
            answer_func: Async function that takes a question and returns an answer
            
        Returns:
            IntegratedRobustnessResult with scores
        """
        if not self.config.enabled:
            return IntegratedRobustnessResult(
                enabled=False,
                questions_tested=0,
                overall_score=0.0,
                grade="N/A",
                attack_type_scores={},
                vulnerable_questions=0,
                question_results=[],
            )
        
        sample_ids = self.select_robustness_sample()
        
        if not sample_ids:
            logger.warning("robustness_no_samples", reason="No questions recorded")
            return IntegratedRobustnessResult(
                enabled=True,
                questions_tested=0,
                overall_score=100.0,  # No tests = assume robust
                grade="A",
                attack_type_scores={},
                vulnerable_questions=0,
                question_results=[],
            )
        
        question_results = []
        all_attack_scores: dict[str, list[float]] = {at.value: [] for at in self.attack_types}
        
        for qid in sample_ids:
            question_text, original_answer = self._recorded_answers[qid]
            
            result = await self._test_question_robustness(
                question_id=qid,
                question_text=question_text,
                original_answer=original_answer,
                answer_func=answer_func,
            )
            question_results.append(result)
            
            # Collect scores by attack type
            for attack_type, score in result.consistency_scores.items():
                if attack_type in all_attack_scores:
                    all_attack_scores[attack_type].append(score)
        
        # Calculate aggregate scores
        attack_type_avg = {
            at: sum(scores) / len(scores) if scores else 1.0
            for at, scores in all_attack_scores.items()
        }
        
        # Overall score (average of all question consistencies)
        if question_results:
            overall_score = sum(r.avg_consistency for r in question_results) / len(question_results)
            overall_score = overall_score * 100  # Convert to 0-100
        else:
            overall_score = 100.0
        
        # Count vulnerabilities
        vulnerable_questions = sum(1 for r in question_results if not r.is_robust)
        
        # Determine grade
        grade = "F"
        for g, threshold in sorted(self.GRADE_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
            if overall_score >= threshold:
                grade = g
                break
        
        logger.info(
            "robustness_evaluation_complete",
            questions_tested=len(question_results),
            overall_score=overall_score,
            grade=grade,
            vulnerable_questions=vulnerable_questions,
        )
        
        return IntegratedRobustnessResult(
            enabled=True,
            questions_tested=len(question_results),
            overall_score=overall_score,
            grade=grade,
            attack_type_scores={k: v * 100 for k, v in attack_type_avg.items()},
            vulnerable_questions=vulnerable_questions,
            question_results=question_results,
        )
    
    async def _test_question_robustness(
        self,
        question_id: str,
        question_text: str,
        original_answer: str,
        answer_func: Callable,
    ) -> QuestionRobustnessResult:
        """Test robustness for a single question."""
        
        # Generate perturbations
        perturbations = self.engine.generate_perturbations(
            question_text,
            attack_types=self.attack_types,
            num_variants=1,
        )
        
        consistency_scores = {}
        vulnerabilities = []
        
        for perturb in perturbations:
            try:
                # Get answer for perturbed question
                perturbed_answer = await answer_func(perturb.perturbed)
                perturbed_answer_str = str(perturbed_answer) if perturbed_answer else ""
                
                # Calculate consistency
                consistency = self._calculate_consistency(original_answer, perturbed_answer_str)
                
                attack_name = perturb.attack_type.value
                consistency_scores[attack_name] = consistency
                
                # Check if this is a vulnerability
                if not perturb.expected_answer_changed and consistency < 0.7:
                    vulnerabilities.append(
                        f"{attack_name}: Answer changed unexpectedly (consistency: {consistency:.2f})"
                    )
                elif perturb.expected_answer_changed and consistency > 0.8:
                    vulnerabilities.append(
                        f"{attack_name}: Answer didn't change when expected (consistency: {consistency:.2f})"
                    )
                    # For expected changes, low consistency is good
                    consistency_scores[attack_name] = 1.0 - consistency
                    
            except Exception as e:
                logger.warning(
                    "robustness_perturbation_failed",
                    question_id=question_id,
                    attack_type=perturb.attack_type.value,
                    error=str(e),
                )
                consistency_scores[perturb.attack_type.value] = 0.0
                vulnerabilities.append(f"{perturb.attack_type.value}: Agent crashed")
        
        # Calculate average consistency
        if consistency_scores:
            avg_consistency = sum(consistency_scores.values()) / len(consistency_scores)
        else:
            avg_consistency = 1.0  # No perturbations = assume robust
        
        return QuestionRobustnessResult(
            question_id=question_id,
            original_answer=original_answer[:100] + "..." if len(original_answer) > 100 else original_answer,
            consistency_scores=consistency_scores,
            avg_consistency=avg_consistency,
            is_robust=avg_consistency >= 0.7,
            vulnerabilities=vulnerabilities,
        )
    
    def _calculate_consistency(self, answer1: str, answer2: str) -> float:
        """Calculate consistency between two answers."""
        if not answer1 or not answer2:
            return 0.0
        
        a1_lower = answer1.lower().strip()
        a2_lower = answer2.lower().strip()
        
        # Exact match
        if a1_lower == a2_lower:
            return 1.0
        
        # Word overlap (Jaccard similarity)
        words1 = set(a1_lower.split())
        words2 = set(a2_lower.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


def create_robustness_integration(
    eval_config: Optional[Any] = None,
    enabled: bool = False,
) -> RobustnessIntegration:
    """
    Factory function to create RobustnessIntegration.
    
    Args:
        eval_config: Optional EvaluationConfig with robustness settings
        enabled: Override to enable/disable robustness
        
    Returns:
        Configured RobustnessIntegration
    """
    if eval_config and hasattr(eval_config, 'robustness'):
        config = RobustnessConfig.from_dict(vars(eval_config.robustness))
    else:
        config = RobustnessConfig(enabled=enabled)
    
    return RobustnessIntegration(config)
