"""
Public CSV dataset evaluator.

Evaluates predictions against public.csv rubric using correctness/contradiction
operators as defined in the dataset.

Rubric format:
[
    {'operator': 'correctness', 'criteria': 'Must contain X'},
    {'operator': 'contradiction', 'criteria': 'Must not contradict Y'}
]
"""

import logging
import re
from typing import Any, Dict, List, Optional

from evaluators.base import BaseDatasetEvaluator, EvalResult
from evaluators.llm_utils import (
    build_llm_client_for_evaluator,
    call_llm,
    coerce_bool,
    extract_json,
    get_model_for_evaluator,
    get_temperature_for_evaluator,
    get_max_tokens_for_evaluator,
)

logger = logging.getLogger(__name__)

# Evaluator name for config lookup
EVALUATOR_NAME = "public_csv"


class PublicCsvEvaluator(BaseDatasetEvaluator):
    """
    Evaluator for public.csv (FAB++) dataset.
    
    Uses rubric operators:
        - 'correctness': Check if answer contains/matches the criteria (+1 point)
        - 'contradiction': Check if answer contradicts reference (-1 penalty)
    
    This evaluator can use either:
        1. Simple string matching (fast, no LLM)
        2. LLM-based judgment (accurate, requires LLM client)
    """
    
    name = "public_csv"
    
    # Scoring constants
    CONTRADICTION_PENALTY = 0.5  # Penalty per contradiction found
    MIN_WORD_LENGTH = 4  # Minimum word length for substring matching
    LLM_MAX_TOKENS = 1200
    
    def __init__(
        self,
        use_llm: bool = False,
        llm_client: Any = None,
        llm_model: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_max_tokens: Optional[int] = None,
    ):
        """
        Initialize Public CSV evaluator.

        Args:
            use_llm: Whether to use LLM for evaluation (more accurate)
            llm_client: LLM client for LLM-based evaluation
            llm_model: Optional LLM model override (default: from EvaluatorLLMConfig)
            llm_temperature: Optional temperature override (default: from EvaluatorLLMConfig)
            llm_max_tokens: Optional max_tokens override (default: from EvaluatorLLMConfig)
        """
        self.use_llm = use_llm
        self.llm_client = llm_client

        # Use per-evaluator config with optional overrides
        self.llm_model = llm_model or get_model_for_evaluator(EVALUATOR_NAME)
        self.llm_temperature = (
            llm_temperature if llm_temperature is not None
            else get_temperature_for_evaluator(EVALUATOR_NAME)
        )
        self._llm_max_tokens = (
            llm_max_tokens if llm_max_tokens is not None
            else get_max_tokens_for_evaluator(EVALUATOR_NAME)
        )
    
    def evaluate(
        self,
        predicted: str,
        expected: str = None,
        rubric: List[Dict[str, str]] = None,
        **kwargs
    ) -> EvalResult:
        """
        Evaluate predicted answer against rubric.
        
        Args:
            predicted: Model's predicted answer
            expected: Reference answer (optional, used for contradiction check)
            rubric: List of rubric items with 'operator' and 'criteria' keys
            **kwargs: Additional parameters
            
        Returns:
            EvalResult with score and details
        """
        if not rubric:
            # No rubric provided, fall back to simple comparison
            return self._simple_compare(predicted, expected or "")

        llm_failure = None
        llm_raw_output = None
        if self.use_llm:
            llm_result, llm_failure, llm_raw_output = self._llm_evaluate_rubric(
                predicted=predicted,
                expected=expected or "",
                rubric=rubric,
                question=kwargs.get("question"),
            )
            if llm_result is not None:
                return llm_result
        
        # Evaluate each rubric item
        correct_count = 0
        penalty_count = 0
        item_results = []
        
        correctness_items = [r for r in rubric if r.get("operator") == "correctness"]
        contradiction_items = [r for r in rubric if r.get("operator") == "contradiction"]
        
        # Evaluate correctness criteria
        for item in correctness_items:
            criteria = item.get("criteria", "")
            is_met = self._check_correctness(predicted, criteria)
            if is_met:
                correct_count += 1
            item_results.append({
                "operator": "correctness",
                "criteria": criteria[:100] + "..." if len(criteria) > 100 else criteria,
                "met": is_met,
            })
        
        # Evaluate contradiction criteria (check for hallucination/contradiction)
        for item in contradiction_items:
            criteria = item.get("criteria", "")
            has_contradiction = self._check_contradiction(predicted, criteria)
            if has_contradiction:
                penalty_count += 1
            item_results.append({
                "operator": "contradiction",
                "criteria": criteria[:100] + "..." if len(criteria) > 100 else criteria,
                "violated": has_contradiction,
            })
        
        # Calculate score
        total_correctness = len(correctness_items)
        max_score = total_correctness if total_correctness > 0 else 1
        score = correct_count - (penalty_count * self.CONTRADICTION_PENALTY)
        score = max(0.0, score)  # Don't go negative
        
        # Normalize to 0-1 range
        normalized_score = score / max_score if max_score > 0 else 0.0
        normalized_score = min(1.0, normalized_score)  # Cap at 1.0
        
        details = {
            "correctness_met": correct_count,
            "correctness_total": total_correctness,
            "contradictions_found": penalty_count,
            "item_results": item_results,
        }
        if self.use_llm:
            details["llm_used"] = False
            if llm_failure:
                details["llm_failure"] = llm_failure
            if llm_raw_output:
                details["llm_raw_output"] = llm_raw_output

        return EvalResult(
            score=normalized_score,
            max_score=1.0,
            correct_count=correct_count,
            total_count=total_correctness,
            feedback=f"Correctness: {correct_count}/{total_correctness}, Contradictions: {penalty_count}",
            details=details,
        )

    def _get_llm_client(self) -> Optional[Any]:
        if self.llm_client is None:
            self.llm_client = build_llm_client_for_evaluator(EVALUATOR_NAME)
        return self.llm_client

    def _llm_evaluate_rubric(
        self,
        predicted: str,
        expected: str,
        rubric: List[Dict[str, str]],
        question: Optional[str] = None,
    ) -> tuple[Optional[EvalResult], Optional[str], Optional[str]]:
        client = self._get_llm_client()
        if not client:
            return None, "llm_client_unavailable", None

        rubric_lines = []
        for idx, item in enumerate(rubric):
            operator = item.get("operator", "correctness")
            criteria = item.get("criteria", "")
            rubric_lines.append(f"{idx + 1}. [{operator}] {criteria}")

        system_prompt = (
            "You are a strict finance QA grader. "
            "Use only the provided candidate answer and rubric. "
            "Do not infer missing facts."
        )

        prompt = f"""QUESTION:
{question or "N/A"}

REFERENCE ANSWER (if provided):
{expected or "N/A"}

CANDIDATE ANSWER:
{predicted}

RUBRIC ITEMS (in order):
{chr(10).join(rubric_lines)}

Rules:
- For correctness: met=true only if the answer explicitly states or clearly implies the criterion.
- For contradiction: violated=true only if the answer directly contradicts the criterion. If the answer is silent, violated=false.

Return JSON only:
{{"items":[{{"operator":"correctness","met":true}},{{"operator":"contradiction","violated":false}}]}}
"""

        raw = None
        try:
            raw = call_llm(
                client=client,
                prompt=prompt,
                model=self.llm_model,
                system_prompt=system_prompt,
                temperature=self.llm_temperature,
                max_tokens=self._llm_max_tokens,
            )
            data = extract_json(raw)
            if not data or "items" not in data:
                logger.warning("llm_public_csv_invalid_json")
                return None, "llm_invalid_json", raw
        except Exception as e:
            logger.warning("llm_public_csv_failed: %s", e)
            return None, f"llm_call_failed: {e}", raw

        items = data.get("items", [])
        expected_count = len(rubric)
        actual_count = len(items)
        partial_mismatch = actual_count != expected_count
        if partial_mismatch:
            logger.warning(
                "llm_public_csv_item_mismatch: expected=%s got=%s",
                expected_count,
                actual_count,
            )

        correct_count = 0
        penalty_count = 0
        item_results = []

        for idx, rubric_item in enumerate(rubric):
            eval_item = items[idx] if idx < actual_count else None
            missing = not isinstance(eval_item, dict)
            if missing:
                eval_item = {}
            operator = rubric_item.get("operator", "correctness")
            criteria = rubric_item.get("criteria", "")
            if operator == "correctness":
                is_met = coerce_bool(eval_item.get("met"))
                if is_met is None:
                    is_met = False
                if is_met:
                    correct_count += 1
                item_result = {
                    "operator": "correctness",
                    "criteria": criteria[:100] + "..." if len(criteria) > 100 else criteria,
                    "met": is_met,
                }
                if missing:
                    item_result["missing"] = True
                item_results.append(item_result)
            elif operator == "contradiction":
                has_contradiction = coerce_bool(eval_item.get("violated"))
                if has_contradiction is None:
                    has_contradiction = False
                if has_contradiction:
                    penalty_count += 1
                item_result = {
                    "operator": "contradiction",
                    "criteria": criteria[:100] + "..." if len(criteria) > 100 else criteria,
                    "violated": has_contradiction,
                }
                if missing:
                    item_result["missing"] = True
                item_results.append(item_result)

        total_correctness = sum(1 for r in rubric if r.get("operator") == "correctness")
        max_score = total_correctness if total_correctness > 0 else 1
        score = correct_count - (penalty_count * self.CONTRADICTION_PENALTY)
        score = max(0.0, score)
        normalized_score = score / max_score if max_score > 0 else 0.0
        normalized_score = min(1.0, normalized_score)

        details = {
            "llm_used": True,
            "llm_model": self.llm_model,
            "correctness_met": correct_count,
            "correctness_total": total_correctness,
            "contradictions_found": penalty_count,
            "item_results": item_results,
        }
        if raw is not None:
            details["llm_raw_output"] = raw
        if partial_mismatch:
            details["llm_partial"] = True
            details["llm_item_count_expected"] = expected_count
            details["llm_item_count_actual"] = actual_count

        return EvalResult(
            score=normalized_score,
            max_score=1.0,
            correct_count=correct_count,
            total_count=total_correctness,
            feedback=f"LLM rubric scoring: Correctness {correct_count}/{total_correctness}, "
                     f"Contradictions {penalty_count}",
            details=details,
        ), None, raw
    
    def _check_correctness(self, predicted: str, criteria: str) -> bool:
        """
        Check if prediction satisfies correctness criteria.
        
        Simple approach: Check if key phrases from criteria appear in prediction.
        
        Args:
            predicted: The prediction to check
            criteria: The criteria to match
            
        Returns:
            True if criteria is satisfied
        """
        if not criteria:
            return True
        
        # Normalize both
        pred_lower = predicted.lower()
        crit_lower = criteria.lower()
        
        # Extract key elements from criteria
        # Look for numbers, names, key phrases
        key_elements = self._extract_key_elements(criteria)
        
        if not key_elements:
            # No extractable elements, check for substring
            return crit_lower in pred_lower or any(
                word in pred_lower for word in crit_lower.split() if len(word) > self.MIN_WORD_LENGTH
            )
        
        # Check if majority of key elements are present
        matches = sum(1 for elem in key_elements if elem in pred_lower)
        return matches >= len(key_elements) * 0.5
    
    def _check_contradiction(self, predicted: str, reference: str) -> bool:
        """
        Check if prediction contradicts the reference.
        
        Note: This heuristic method is conservative and returns False to avoid
        false positives. LLM-based contradiction detection is handled at the
        rubric level in _llm_evaluate_rubric.
        
        Args:
            predicted: The prediction to check
            reference: The reference that shouldn't be contradicted
            
        Returns:
            True if contradiction detected (always False without LLM)
        """
        if not reference or not predicted:
            return False
        
        # Be conservative and never flag contradictions without LLM
        return False
    
    def _extract_key_elements(self, text: str) -> List[str]:
        """
        Extract key elements (numbers, names, key phrases) from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            List of key elements (lowercase)
        """
        elements = []
        
        # Extract numbers
        numbers = re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:billion|million|%))?', text.lower())
        elements.extend(numbers)
        
        # Extract potential names (capitalized words in original)
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        elements.extend([n.lower() for n in names])
        
        # Extract stock tickers
        tickers = re.findall(r'\b[A-Z]{2,5}\b', text)
        elements.extend([t.lower() for t in tickers])
        
        return elements
    
    def _simple_compare(self, predicted: str, expected: str) -> EvalResult:
        """
        Simple comparison when no rubric provided.
        
        Args:
            predicted: Predicted answer
            expected: Expected answer
            
        Returns:
            EvalResult based on string similarity
        """
        pred_norm = predicted.lower().strip()
        exp_norm = expected.lower().strip()
        
        is_exact = pred_norm == exp_norm
        is_contained = exp_norm in pred_norm or pred_norm in exp_norm
        
        if is_exact:
            score = 1.0
        elif is_contained:
            score = 0.7
        else:
            score = 0.0
        
        return EvalResult(
            score=score,
            correct_count=1 if is_exact else 0,
            total_count=1,
            feedback="Simple comparison (no rubric provided)",
            details={"is_exact": is_exact, "is_contained": is_contained}
        )
