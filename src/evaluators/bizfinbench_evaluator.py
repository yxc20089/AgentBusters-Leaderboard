"""
BizFinBench.v2 dataset evaluator.

Evaluates predictions against BizFinBench.v2 ground truth using task-specific
evaluation strategies:
- Numerical tasks: Match with tolerance
- Ordering tasks: Exact sequence match
- Classification tasks: Label match
- Open-ended tasks: LLM-based judgment
"""

import logging
import re
from typing import Any, Optional

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
EVALUATOR_NAME = "bizfinbench"


class BizFinBenchEvaluator(BaseDatasetEvaluator):
    """
    Evaluator for HiThink BizFinBench.v2 dataset.
    
    Supports different evaluation strategies per task type:
        - financial_quantitative_computation: Numerical match ±tolerance
        - event_logic_reasoning: Exact sequence match
        - user_sentiment_analysis: Classification match
        - stock_price_predict: Numerical match
        - Others: Normalized string match
    """
    
    name = "bizfinbench"
    
    # Default numerical tolerance (1%)
    DEFAULT_TOLERANCE = 0.01
    LLM_MAX_TOKENS = 800
    STRUCTURED_TASKS = {
        "financial_quantitative_computation",
        "event_logic_reasoning",
        "user_sentiment_analysis",
        "stock_price_predict",
    }
    
    def __init__(
        self,
        tolerance: float = None,
        use_llm: bool = False,
        llm_client: Any = None,
        llm_model: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_max_tokens: Optional[int] = None,
    ):
        """
        Initialize BizFinBench evaluator.

        Args:
            tolerance: Numerical tolerance for quantitative tasks.
                       If None, uses DEFAULT_TOLERANCE (currently 0.01 = 1%).
            use_llm: Whether to use LLM for evaluation (more accurate)
            llm_client: LLM client for LLM-based evaluation
            llm_model: Optional LLM model override (default: from EvaluatorLLMConfig)
            llm_temperature: Optional temperature override (default: from EvaluatorLLMConfig)
            llm_max_tokens: Optional max_tokens override (default: from EvaluatorLLMConfig)
        """
        self.tolerance = tolerance or self.DEFAULT_TOLERANCE
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
        expected: str,
        task_type: str = None,
        **kwargs
    ) -> EvalResult:
        """
        Evaluate predicted answer against expected.
        
        Args:
            predicted: Model's predicted answer
            expected: Ground truth answer
            task_type: BizFinBench task type (determines evaluation strategy)
            **kwargs: Additional parameters
            
        Returns:
            EvalResult with score and details
        """
        if not predicted or not expected:
            return EvalResult(
                score=0.0,
                feedback="Empty prediction or expected answer",
                details={"predicted": predicted, "expected": expected}
            )
        
        # Normalize inputs
        predicted = predicted.strip()
        expected = expected.strip()

        # Route to task-specific evaluator
        if task_type == "financial_quantitative_computation":
            result = self._eval_numerical(predicted, expected)
        elif task_type == "event_logic_reasoning":
            result = self._eval_exact_sequence(predicted, expected)
        elif task_type == "user_sentiment_analysis":
            result = self._eval_classification(predicted, expected)
        elif task_type == "stock_price_predict":
            result = self._eval_numerical(predicted, expected)
        elif task_type == "conterfactual":
            result = self._eval_normalized_match(predicted, expected)
        else:
            # Default: normalized string match
            result = self._eval_normalized_match(predicted, expected)

        llm_failure = None
        llm_raw_output = None
        if self.use_llm:
            use_llm_now = task_type not in self.STRUCTURED_TASKS or result.score < 1.0
            if use_llm_now:
                llm_result, llm_failure, llm_raw_output = self._llm_evaluate(
                    predicted=predicted,
                    expected=expected,
                    task_type=task_type,
                    question=kwargs.get("question"),
                )
                if llm_result is not None:
                    return llm_result

        if self.use_llm:
            if result.details is None:
                result.details = {}
            result.details["llm_used"] = False
            if llm_failure:
                result.details["llm_failure"] = llm_failure
            if llm_raw_output:
                result.details["llm_raw_output"] = llm_raw_output

        return result

    def _get_llm_client(self) -> Optional[Any]:
        if self.llm_client is None:
            self.llm_client = build_llm_client_for_evaluator(EVALUATOR_NAME)
        return self.llm_client

    def _llm_evaluate(
        self,
        predicted: str,
        expected: str,
        task_type: Optional[str] = None,
        question: Optional[str] = None,
    ) -> tuple[Optional[EvalResult], Optional[str], Optional[str]]:
        client = self._get_llm_client()
        if not client:
            return None, "llm_client_unavailable", None

        pred_num = self._extract_number(predicted)
        exp_num = self._extract_number(expected)

        system_prompt = "You are a strict grader for BizFinBench answers."
        prompt = f"""TASK TYPE: {task_type or "unknown"}

QUESTION:
{question or "N/A"}

REFERENCE ANSWER:
{expected}

CANDIDATE ANSWER:
{predicted}

PARSED NUMBERS (for numeric tasks):
expected_number: {exp_num if exp_num is not None else "N/A"}
predicted_number: {pred_num if pred_num is not None else "N/A"}
relative_tolerance: {self.tolerance}

Rules:
- For financial_quantitative_computation and stock_price_predict, the answer is correct only if the numeric
  value matches within the tolerance.
- For event_logic_reasoning, the answer must match the expected ordered sequence (ignore whitespace and
  formatting differences).
- For user_sentiment_analysis, the label/sentiment must match.
- For other tasks, require semantic equivalence of key facts. Do not give credit for partially correct answers.

Return JSON only:
{{"correct": true, "score": 1, "reason": "short reason"}}
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
            if not data:
                return None, "llm_invalid_json", raw
        except Exception as e:
            logger.warning("llm_bizfinbench_failed: %s", e)
            return None, f"llm_call_failed: {e}", raw

        correct = coerce_bool(data.get("correct"))
        score_val = data.get("score")
        score = 0.0
        if isinstance(correct, bool):
            score = 1.0 if correct else 0.0
        else:
            try:
                score = float(score_val)
            except (TypeError, ValueError):
                score = 0.0
            score = 1.0 if score >= 0.5 else 0.0
            correct = score >= 1.0

        reason = data.get("reason") or "LLM evaluation"

        return EvalResult(
            score=score,
            correct_count=1 if correct else 0,
            total_count=1,
            feedback=reason,
            details={
                "llm_used": True,
                "llm_model": self.llm_model,
                "llm_correct": bool(correct),
                "llm_score_raw": score_val,
                "llm_raw_output": raw,
            }
        ), None, raw
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numerical value from text."""
        # Try to parse as float directly
        try:
            return float(text)
        except ValueError:
            pass
        
        # Look for numbers in text (handle percentages, commas, etc.)
        patterns = [
            r'(-?\d+\.?\d*)\s*%',  # Percentage
            r'(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?)',  # Number with commas
            r'(-?\d+\.?\d*)',  # Simple number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                num_str = match.group(1).replace(',', '')
                try:
                    return float(num_str)
                except ValueError:
                    continue
        
        return None
    
    def _eval_numerical(self, predicted: str, expected: str) -> EvalResult:
        """
        Evaluate numerical answer with tolerance.
        
        Args:
            predicted: Predicted numerical answer
            expected: Expected numerical answer
            
        Returns:
            EvalResult (1.0 if within tolerance, 0.0 otherwise)
        """
        pred_num = self._extract_number(predicted)
        exp_num = self._extract_number(expected)
        
        if pred_num is None:
            return EvalResult(
                score=0.0,
                feedback=f"Could not extract number from prediction: '{predicted}'",
                details={"predicted": predicted, "expected": expected}
            )
        
        if exp_num is None:
            return EvalResult(
                score=0.0,
                feedback=f"Could not extract number from expected: '{expected}'",
                details={"predicted": predicted, "expected": expected}
            )
        
        # Calculate relative error
        if exp_num == 0:
            is_correct = abs(pred_num) < self.tolerance
        else:
            relative_error = abs(pred_num - exp_num) / abs(exp_num)
            is_correct = relative_error <= self.tolerance
        
        return EvalResult(
            score=1.0 if is_correct else 0.0,
            correct_count=1 if is_correct else 0,
            total_count=1,
            feedback=f"Predicted: {pred_num}, Expected: {exp_num}, Tolerance: {self.tolerance}",
            details={
                "predicted_num": pred_num,
                "expected_num": exp_num,
                "is_correct": is_correct,
                "tolerance": self.tolerance,
            }
        )
    
    def _eval_exact_sequence(self, predicted: str, expected: str) -> EvalResult:
        """
        Evaluate sequence/ordering answer (exact match).
        
        Expected format: "2,1,4,3" or similar
        
        Args:
            predicted: Predicted sequence
            expected: Expected sequence
            
        Returns:
            EvalResult (1.0 if exact match, 0.0 otherwise)
        """
        # Normalize: remove spaces, extract comma-separated values
        pred_clean = re.sub(r'\s+', '', predicted)
        exp_clean = re.sub(r'\s+', '', expected)
        
        # Extract sequence (handle JSON format if present)
        pred_match = re.search(r'[\d,]+', pred_clean)
        exp_match = re.search(r'[\d,]+', exp_clean)
        
        if pred_match:
            pred_seq = pred_match.group()
        else:
            pred_seq = pred_clean
        
        if exp_match:
            exp_seq = exp_match.group()
        else:
            exp_seq = exp_clean
        
        is_correct = pred_seq == exp_seq
        
        return EvalResult(
            score=1.0 if is_correct else 0.0,
            correct_count=1 if is_correct else 0,
            total_count=1,
            feedback=f"Predicted: '{pred_seq}', Expected: '{exp_seq}'",
            details={
                "predicted_seq": pred_seq,
                "expected_seq": exp_seq,
                "is_correct": is_correct,
            }
        )
    
    def _eval_classification(self, predicted: str, expected: str) -> EvalResult:
        """
        Evaluate classification answer.
        
        Args:
            predicted: Predicted label
            expected: Expected label
            
        Returns:
            EvalResult (1.0 if match, 0.0 otherwise)
        """
        # Normalize: lowercase, strip
        pred_norm = predicted.lower().strip()
        exp_norm = expected.lower().strip()
        
        is_correct = pred_norm == exp_norm
        
        return EvalResult(
            score=1.0 if is_correct else 0.0,
            correct_count=1 if is_correct else 0,
            total_count=1,
            feedback=f"Predicted: '{pred_norm}', Expected: '{exp_norm}'",
            details={
                "predicted": pred_norm,
                "expected": exp_norm,
                "is_correct": is_correct,
            }
        )
    
    def _eval_normalized_match(self, predicted: str, expected: str) -> EvalResult:
        """
        Evaluate with normalized string matching.
        
        Handles minor formatting differences.
        
        Args:
            predicted: Predicted answer
            expected: Expected answer
            
        Returns:
            EvalResult with similarity score
        """
        # Normalize both strings
        pred_norm = self._normalize_text(predicted)
        exp_norm = self._normalize_text(expected)
        
        is_correct = pred_norm == exp_norm
        
        # Calculate partial match score using simple containment
        if is_correct:
            score = 1.0
        elif exp_norm in pred_norm or pred_norm in exp_norm:
            score = 0.5
        else:
            score = 0.0
        
        return EvalResult(
            score=score,
            correct_count=1 if is_correct else 0,
            total_count=1,
            feedback=f"Exact match: {is_correct}",
            details={
                "predicted_normalized": pred_norm[:100],
                "expected_normalized": exp_norm[:100],
                "is_exact_match": is_correct,
            }
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove common punctuation
        text = re.sub(r'[.,;:!?\'"()\[\]{}]', '', text)
        return text
