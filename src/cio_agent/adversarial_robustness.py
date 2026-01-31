"""
Adversarial Robustness System for AgentBusters.

Tests agent robustness through various perturbation attacks:
1. Paraphrase Attack: Rephrase questions without changing meaning
2. Typo Attack: Inject realistic typos
3. Number Perturbation: Swap similar numbers
4. Distraction Attack: Add irrelevant but plausible context
5. Temporal Attack: Mix outdated and current information
6. Format Attack: Change question format (list, paragraph, etc.)

This module integrates with the evaluation pipeline to measure
consistency and robustness across perturbations.
"""

import random
import re
import string
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from enum import Enum

import structlog

logger = structlog.get_logger()


class AttackType(Enum):
    """Types of adversarial perturbations."""
    PARAPHRASE = "paraphrase"
    TYPO = "typo"
    NUMBER_SWAP = "number_swap"
    DISTRACTION = "distraction"
    TEMPORAL = "temporal"
    FORMAT = "format"
    NEGATION = "negation"
    SYNONYM = "synonym"


@dataclass
class PerturbedQuestion:
    """A perturbed version of an original question."""
    original: str
    perturbed: str
    attack_type: AttackType
    attack_params: dict[str, Any] = field(default_factory=dict)
    expected_answer_changed: bool = False  # True if perturbation should change answer


@dataclass
class RobustnessResult:
    """Result of robustness testing across perturbations."""
    original_score: float
    perturbed_scores: dict[AttackType, float]
    consistency_score: float  # How consistent are answers across perturbations
    stability_score: float  # How stable is performance under attack
    vulnerabilities: list[dict[str, Any]]
    total_perturbations: int
    passed_perturbations: int
    robustness_grade: str  # A, B, C, D, F


class AdversarialPerturbationEngine:
    """
    Engine for generating adversarial perturbations of questions.
    
    Implements multiple attack strategies to test agent robustness.
    """
    
    # Common financial synonyms for paraphrasing
    SYNONYMS = {
        "revenue": ["sales", "top-line", "income"],
        "earnings": ["profits", "net income", "bottom-line"],
        "growth": ["increase", "expansion", "gain"],
        "decline": ["decrease", "drop", "fall"],
        "margin": ["profitability", "spread"],
        "guidance": ["forecast", "outlook", "projection"],
        "beat": ["exceeded", "surpassed", "outperformed"],
        "miss": ["fell short of", "underperformed", "missed"],
        "quarterly": ["Q1/Q2/Q3/Q4", "three-month"],
        "annual": ["yearly", "full-year", "12-month"],
        "what": ["which", "identify"],
        "how much": ["what amount", "what value"],
        "calculate": ["compute", "determine", "find"],
    }
    
    # Common typo patterns
    TYPO_PATTERNS = [
        ("the", "teh"),
        ("and", "adn"),
        ("for", "fro"),
        ("with", "wiht"),
        ("from", "form"),
        ("their", "thier"),
        ("which", "whcih"),
    ]
    
    # Irrelevant but plausible distractors
    DISTRACTORS = [
        "Note that broader market conditions have been volatile recently.",
        "Industry analysts have varying opinions on sector trends.",
        "Macroeconomic factors may influence these figures.",
        "Historical data shows mixed patterns in this sector.",
        "Competitor performance has been inconsistent this quarter.",
        "Regulatory changes are being discussed but not finalized.",
    ]
    
    # Temporal confounders
    TEMPORAL_CONFOUNDERS = [
        "Based on last year's data,",
        "According to outdated reports,",
        "Historical figures from 2020 show",
        "Pre-pandemic metrics indicated",
        "Legacy financial statements reported",
    ]
    
    def __init__(
        self,
        seed: int = 42,
        attack_intensity: float = 0.5,
    ):
        """
        Initialize perturbation engine.
        
        Args:
            seed: Random seed for reproducibility
            attack_intensity: How aggressive perturbations are (0.0-1.0)
        """
        self.rng = random.Random(seed)
        self.attack_intensity = attack_intensity
    
    def generate_perturbations(
        self,
        question: str,
        attack_types: list[AttackType] = None,
        num_variants: int = 1,
    ) -> list[PerturbedQuestion]:
        """
        Generate perturbed versions of a question.
        
        Args:
            question: Original question
            attack_types: Types of attacks to apply (default: all)
            num_variants: Number of variants per attack type
            
        Returns:
            List of PerturbedQuestion objects
        """
        if attack_types is None:
            attack_types = list(AttackType)
        
        perturbations = []
        
        for attack_type in attack_types:
            for _ in range(num_variants):
                perturbed = self._apply_attack(question, attack_type)
                if perturbed and perturbed != question:
                    perturbations.append(perturbed)
        
        return perturbations
    
    def _apply_attack(
        self,
        question: str,
        attack_type: AttackType,
    ) -> Optional[PerturbedQuestion]:
        """Apply a specific attack type to a question."""
        
        attack_methods = {
            AttackType.PARAPHRASE: self._paraphrase_attack,
            AttackType.TYPO: self._typo_attack,
            AttackType.NUMBER_SWAP: self._number_swap_attack,
            AttackType.DISTRACTION: self._distraction_attack,
            AttackType.TEMPORAL: self._temporal_attack,
            AttackType.FORMAT: self._format_attack,
            AttackType.NEGATION: self._negation_attack,
            AttackType.SYNONYM: self._synonym_attack,
        }
        
        method = attack_methods.get(attack_type)
        if method:
            return method(question)
        return None
    
    def _paraphrase_attack(self, question: str) -> PerturbedQuestion:
        """Rephrase question while preserving meaning."""
        perturbed = question
        
        # Apply random synonym substitutions
        for original, synonyms in self.SYNONYMS.items():
            if original in perturbed.lower():
                if self.rng.random() < self.attack_intensity:
                    synonym = self.rng.choice(synonyms)
                    # Preserve case
                    if original[0].isupper():
                        synonym = synonym.capitalize()
                    perturbed = re.sub(
                        rf'\b{original}\b',
                        synonym,
                        perturbed,
                        flags=re.IGNORECASE,
                        count=1
                    )
        
        # Structural changes
        if perturbed.startswith("What"):
            alternatives = [
                f"Can you tell me {perturbed[5:]}",
                f"I need to know {perturbed[5:]}",
                f"Please provide {perturbed[5:]}",
            ]
            if self.rng.random() < self.attack_intensity * 0.5:
                perturbed = self.rng.choice(alternatives)
        
        return PerturbedQuestion(
            original=question,
            perturbed=perturbed,
            attack_type=AttackType.PARAPHRASE,
            attack_params={"synonyms_applied": True},
            expected_answer_changed=False,
        )
    
    def _typo_attack(self, question: str) -> PerturbedQuestion:
        """Inject realistic typos."""
        perturbed = question
        typos_applied = []
        
        # Apply known typo patterns
        for correct, typo in self.TYPO_PATTERNS:
            if correct in perturbed.lower() and self.rng.random() < self.attack_intensity:
                perturbed = re.sub(
                    rf'\b{correct}\b',
                    typo,
                    perturbed,
                    flags=re.IGNORECASE,
                    count=1
                )
                typos_applied.append((correct, typo))
        
        # Random character swaps
        if self.rng.random() < self.attack_intensity * 0.3:
            words = perturbed.split()
            if len(words) > 3:
                idx = self.rng.randint(1, len(words) - 2)
                word = words[idx]
                if len(word) > 3:
                    # Swap two adjacent characters
                    pos = self.rng.randint(1, len(word) - 2)
                    word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
                    words[idx] = word
                    perturbed = " ".join(words)
                    typos_applied.append(("char_swap", word))
        
        return PerturbedQuestion(
            original=question,
            perturbed=perturbed,
            attack_type=AttackType.TYPO,
            attack_params={"typos": typos_applied},
            expected_answer_changed=False,
        )
    
    def _number_swap_attack(self, question: str) -> PerturbedQuestion:
        """Swap numbers with similar values to test precision."""
        perturbed = question
        swaps = []
        
        # Find numbers in question
        numbers = re.findall(r'\b(\d{4})\b', perturbed)  # Years
        
        for num_str in numbers:
            if self.rng.random() < self.attack_intensity:
                num = int(num_str)
                # Swap year by ±1
                new_num = num + self.rng.choice([-1, 1])
                perturbed = perturbed.replace(num_str, str(new_num), 1)
                swaps.append((num_str, str(new_num)))
        
        return PerturbedQuestion(
            original=question,
            perturbed=perturbed,
            attack_type=AttackType.NUMBER_SWAP,
            attack_params={"swaps": swaps},
            expected_answer_changed=True,  # Different year = different answer expected
        )
    
    def _distraction_attack(self, question: str) -> PerturbedQuestion:
        """Add irrelevant but plausible context."""
        distractor = self.rng.choice(self.DISTRACTORS)
        
        # Insert at beginning or end
        if self.rng.random() < 0.5:
            perturbed = f"{distractor} {question}"
        else:
            perturbed = f"{question} {distractor}"
        
        return PerturbedQuestion(
            original=question,
            perturbed=perturbed,
            attack_type=AttackType.DISTRACTION,
            attack_params={"distractor": distractor},
            expected_answer_changed=False,
        )
    
    def _temporal_attack(self, question: str) -> PerturbedQuestion:
        """Add temporal confusion."""
        confounder = self.rng.choice(self.TEMPORAL_CONFOUNDERS)
        
        # Add misleading temporal context
        perturbed = f"{confounder} However, for current figures: {question}"
        
        return PerturbedQuestion(
            original=question,
            perturbed=perturbed,
            attack_type=AttackType.TEMPORAL,
            attack_params={"confounder": confounder},
            expected_answer_changed=False,
        )
    
    def _format_attack(self, question: str) -> PerturbedQuestion:
        """Change question format."""
        # Convert to different formats
        if "?" in question:
            # Remove question mark, make it a command
            perturbed = question.rstrip("?") + "."
            perturbed = perturbed.replace("What is", "Provide")
            perturbed = perturbed.replace("How much", "Calculate")
        else:
            # Add explicit question format
            perturbed = f"Question: {question}?"
        
        return PerturbedQuestion(
            original=question,
            perturbed=perturbed,
            attack_type=AttackType.FORMAT,
            attack_params={"format_change": "command_to_question"},
            expected_answer_changed=False,
        )
    
    def _negation_attack(self, question: str) -> PerturbedQuestion:
        """Test with negated questions (should change answer)."""
        perturbed = question
        negated = False
        
        # Add negation to specific patterns
        if "beat" in question.lower():
            perturbed = re.sub(r'\bbeat\b', "NOT beat", question, flags=re.IGNORECASE)
            negated = True
        elif "increase" in question.lower():
            perturbed = re.sub(r'\bincrease\b', "NOT increase", question, flags=re.IGNORECASE)
            negated = True
        elif " did " in question.lower():
            perturbed = question.replace(" did ", " did NOT ")
            negated = True
        
        return PerturbedQuestion(
            original=question,
            perturbed=perturbed,
            attack_type=AttackType.NEGATION,
            attack_params={"negated": negated},
            expected_answer_changed=negated,  # Negation should change the answer
        )
    
    def _synonym_attack(self, question: str) -> PerturbedQuestion:
        """Replace words with synonyms."""
        perturbed = question
        replacements = []
        
        for original, synonyms in self.SYNONYMS.items():
            if original in perturbed.lower():
                synonym = self.rng.choice(synonyms)
                perturbed = re.sub(
                    rf'\b{original}\b',
                    synonym,
                    perturbed,
                    flags=re.IGNORECASE,
                    count=1
                )
                replacements.append((original, synonym))
        
        return PerturbedQuestion(
            original=question,
            perturbed=perturbed,
            attack_type=AttackType.SYNONYM,
            attack_params={"replacements": replacements},
            expected_answer_changed=False,
        )


class RobustnessEvaluator:
    """
    Evaluates agent robustness across perturbations.
    
    Measures:
    - Consistency: Same answer across equivalent perturbations
    - Stability: Performance doesn't degrade under attack
    - Sensitivity: Correctly changes answer when perturbation changes meaning
    """
    
    # Thresholds for grading
    GRADE_THRESHOLDS = {
        "A": 0.9,
        "B": 0.8,
        "C": 0.7,
        "D": 0.6,
        "F": 0.0,
    }
    
    def __init__(
        self,
        perturbation_engine: AdversarialPerturbationEngine = None,
        answer_comparator: Callable[[str, str], float] = None,
    ):
        """
        Initialize robustness evaluator.
        
        Args:
            perturbation_engine: Engine for generating perturbations
            answer_comparator: Function to compare two answers (returns similarity 0-1)
        """
        self.engine = perturbation_engine or AdversarialPerturbationEngine()
        self.answer_comparator = answer_comparator or self._default_comparator
    
    def _default_comparator(self, answer1: str, answer2: str) -> float:
        """Default answer comparison using normalized overlap."""
        if not answer1 or not answer2:
            return 0.0
        
        a1_lower = answer1.lower().strip()
        a2_lower = answer2.lower().strip()
        
        if a1_lower == a2_lower:
            return 1.0
        
        # Word overlap
        words1 = set(a1_lower.split())
        words2 = set(a2_lower.split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        union = len(words1 | words2)
        
        return overlap / union if union > 0 else 0.0
    
    async def evaluate_robustness(
        self,
        question: str,
        answer_func: Callable[[str], str],
        expected_answer: str = None,
        score_func: Callable[[str, str], float] = None,
        attack_types: list[AttackType] = None,
    ) -> RobustnessResult:
        """
        Evaluate agent robustness on a question.
        
        Args:
            question: Original question
            answer_func: Async function that takes question and returns answer
            expected_answer: Ground truth answer (optional)
            score_func: Function to score answer against expected (optional)
            attack_types: Types of attacks to test
            
        Returns:
            RobustnessResult with comprehensive analysis
        """
        # Get original answer
        original_answer = await answer_func(question)
        original_score = 1.0
        
        if expected_answer and score_func:
            original_score = score_func(original_answer, expected_answer)
        
        # Generate perturbations
        perturbations = self.engine.generate_perturbations(
            question,
            attack_types=attack_types,
            num_variants=1,
        )
        
        # Test each perturbation
        perturbed_scores = {}
        perturbed_answers = {}
        vulnerabilities = []
        passed = 0
        
        for perturb in perturbations:
            try:
                perturbed_answer = await answer_func(perturb.perturbed)
                perturbed_answers[perturb.attack_type] = perturbed_answer
                
                # Calculate consistency with original
                consistency = self.answer_comparator(original_answer, perturbed_answer)
                
                # If perturbation shouldn't change answer, high consistency is good
                if not perturb.expected_answer_changed:
                    perturbed_scores[perturb.attack_type] = consistency
                    
                    if consistency < 0.7:
                        vulnerabilities.append({
                            "attack_type": perturb.attack_type.value,
                            "original_question": question[:100],
                            "perturbed_question": perturb.perturbed[:100],
                            "consistency_score": consistency,
                            "issue": "Answer changed when it shouldn't have",
                        })
                    else:
                        passed += 1
                else:
                    # If perturbation should change answer, low consistency is good
                    # (agent correctly recognized the change)
                    perturbed_scores[perturb.attack_type] = 1.0 - consistency
                    
                    if consistency > 0.8:
                        vulnerabilities.append({
                            "attack_type": perturb.attack_type.value,
                            "original_question": question[:100],
                            "perturbed_question": perturb.perturbed[:100],
                            "consistency_score": consistency,
                            "issue": "Answer didn't change when it should have",
                        })
                    else:
                        passed += 1
                        
            except Exception as e:
                logger.warning(
                    "perturbation_test_failed",
                    attack_type=perturb.attack_type.value,
                    error=str(e),
                )
                perturbed_scores[perturb.attack_type] = 0.0
                vulnerabilities.append({
                    "attack_type": perturb.attack_type.value,
                    "issue": f"Agent crashed: {str(e)[:100]}",
                })
        
        # Calculate aggregate metrics
        total_perturbations = len(perturbations)
        
        if perturbed_scores:
            avg_perturbed_score = sum(perturbed_scores.values()) / len(perturbed_scores)
        else:
            avg_perturbed_score = 0.0
        
        consistency_score = avg_perturbed_score
        stability_score = original_score * avg_perturbed_score
        
        # Determine grade
        robustness_grade = "F"
        for grade, threshold in sorted(
            self.GRADE_THRESHOLDS.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if consistency_score >= threshold:
                robustness_grade = grade
                break
        
        return RobustnessResult(
            original_score=original_score,
            perturbed_scores=perturbed_scores,
            consistency_score=consistency_score,
            stability_score=stability_score,
            vulnerabilities=vulnerabilities,
            total_perturbations=total_perturbations,
            passed_perturbations=passed,
            robustness_grade=robustness_grade,
        )
    
    def generate_robustness_report(
        self,
        results: list[RobustnessResult],
    ) -> dict[str, Any]:
        """
        Generate aggregate robustness report from multiple evaluations.
        
        Args:
            results: List of RobustnessResult from multiple questions
            
        Returns:
            Comprehensive robustness report
        """
        if not results:
            return {
                "total_questions": 0,
                "overall_robustness_score": 0.0,
                "overall_grade": "F",
            }
        
        # Aggregate metrics
        avg_consistency = sum(r.consistency_score for r in results) / len(results)
        avg_stability = sum(r.stability_score for r in results) / len(results)
        
        total_vulnerabilities = []
        attack_type_scores = {}
        
        for result in results:
            total_vulnerabilities.extend(result.vulnerabilities)
            
            for attack_type, score in result.perturbed_scores.items():
                if attack_type not in attack_type_scores:
                    attack_type_scores[attack_type] = []
                attack_type_scores[attack_type].append(score)
        
        # Calculate per-attack-type averages
        attack_type_avg = {
            attack_type.value: sum(scores) / len(scores)
            for attack_type, scores in attack_type_scores.items()
        }
        
        # Find weakest attack type
        weakest_attack = min(
            attack_type_avg.items(),
            key=lambda x: x[1],
            default=(None, 0.0)
        )
        
        # Grade distribution
        grade_counts = {}
        for result in results:
            grade = result.robustness_grade
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        # Overall grade
        overall_grade = "F"
        for grade, threshold in sorted(
            self.GRADE_THRESHOLDS.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if avg_consistency >= threshold:
                overall_grade = grade
                break
        
        return {
            "total_questions": len(results),
            "overall_robustness_score": avg_consistency,
            "overall_stability_score": avg_stability,
            "overall_grade": overall_grade,
            "grade_distribution": grade_counts,
            "attack_type_scores": attack_type_avg,
            "weakest_attack_type": weakest_attack[0],
            "weakest_attack_score": weakest_attack[1],
            "total_vulnerabilities": len(total_vulnerabilities),
            "sample_vulnerabilities": total_vulnerabilities[:10],
            "recommendations": self._generate_recommendations(
                attack_type_avg,
                total_vulnerabilities,
            ),
        }
    
    def _generate_recommendations(
        self,
        attack_scores: dict[str, float],
        vulnerabilities: list[dict],
    ) -> list[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Identify weak areas
        for attack_type, score in attack_scores.items():
            if score < 0.7:
                if attack_type == "paraphrase":
                    recommendations.append(
                        "Agent struggles with paraphrased questions. "
                        "Consider improving semantic understanding."
                    )
                elif attack_type == "typo":
                    recommendations.append(
                        "Agent is sensitive to typos. "
                        "Add input normalization or spell-checking."
                    )
                elif attack_type == "distraction":
                    recommendations.append(
                        "Agent is easily distracted by irrelevant context. "
                        "Improve focus on key question elements."
                    )
                elif attack_type == "temporal":
                    recommendations.append(
                        "Agent confuses temporal information. "
                        "Strengthen date/time awareness."
                    )
                elif attack_type == "number_swap":
                    recommendations.append(
                        "Agent doesn't detect number changes. "
                        "Improve numerical precision."
                    )
        
        if not recommendations:
            recommendations.append(
                "Agent shows good robustness across attack types. "
                "Continue monitoring for edge cases."
            )
        
        return recommendations


# Convenience function for quick robustness testing
async def quick_robustness_test(
    question: str,
    answer_func: Callable,
    expected_answer: str = None,
) -> dict[str, Any]:
    """
    Quick robustness test for a single question.
    
    Args:
        question: Question to test
        answer_func: Async function that returns answer for a question
        expected_answer: Expected ground truth
        
    Returns:
        Simple robustness summary
    """
    evaluator = RobustnessEvaluator()
    result = await evaluator.evaluate_robustness(
        question=question,
        answer_func=answer_func,
        expected_answer=expected_answer,
    )
    
    return {
        "robustness_grade": result.robustness_grade,
        "consistency_score": result.consistency_score,
        "vulnerabilities_found": len(result.vulnerabilities),
        "passed_tests": f"{result.passed_perturbations}/{result.total_perturbations}",
    }
