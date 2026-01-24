"""
GDPVal dataset evaluator.

Evaluates AI performance on real-world economically valuable tasks from OpenAI's
GDPVal benchmark (220 tasks across 44 occupations).

Since GDPVal tasks are open-ended with no ground truth answers, this evaluator
uses LLM-as-judge to assess deliverable quality based on:
- Task completion: Did the agent complete the requested task?
- Accuracy: Is the response factually correct and technically sound?
- Format: Is the deliverable in the expected format?
- Professionalism: Does it meet industry standards?

Reference: https://arxiv.org/abs/2510.04374
"""

import logging
from typing import Any, Dict, List, Optional

from evaluators.base import BaseDatasetEvaluator, EvalResult
from evaluators.llm_utils import (
    build_llm_client_for_evaluator,
    call_llm,
    extract_json,
    get_model_for_evaluator,
    get_temperature_for_evaluator,
    get_max_tokens_for_evaluator,
)

logger = logging.getLogger(__name__)

# Evaluator name for config lookup
EVALUATOR_NAME = "gdpval"


class GDPValEvaluator(BaseDatasetEvaluator):
    """
    Evaluator for OpenAI's GDPVal dataset.

    Uses LLM-as-judge to evaluate open-ended task completion across 4 dimensions:
        - Completion (0-25): Did the agent complete the requested task?
        - Accuracy (0-25): Is the response factually correct?
        - Format (0-25): Is the deliverable in the expected format?
        - Professionalism (0-25): Does it meet industry standards?

    Total score: 0-100, normalized to 0-1 for aggregation.
    """

    name = "gdpval"

    # Scoring weights (sum to 1.0)
    WEIGHTS = {
        "completion": 0.30,
        "accuracy": 0.30,
        "format": 0.20,
        "professionalism": 0.20,
    }

    LLM_MAX_TOKENS = 2000

    def __init__(
        self,
        use_llm: bool = True,  # LLM required for GDPVal
        llm_client: Any = None,
        llm_model: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_max_tokens: Optional[int] = None,
    ):
        """
        Initialize GDPVal evaluator.

        Args:
            use_llm: Whether to use LLM for evaluation (required for GDPVal)
            llm_client: LLM client for evaluation
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
        # Allow override but keep LLM_MAX_TOKENS as class-level default reference
        self._llm_max_tokens = (
            llm_max_tokens if llm_max_tokens is not None
            else get_max_tokens_for_evaluator(EVALUATOR_NAME)
        )

    def evaluate(
        self,
        predicted: str,
        expected: str = "",
        task_prompt: str = None,
        occupation: str = None,
        sector: str = None,
        reference_files: List[str] = None,
        **kwargs
    ) -> EvalResult:
        """
        Evaluate predicted answer for a GDPVal task.

        Args:
            predicted: Agent's response/deliverable
            expected: Not used (GDPVal tasks have no ground truth)
            task_prompt: The original task description
            occupation: The occupation category (e.g., "Accountants and Auditors")
            sector: The sector category (e.g., "Professional, Scientific, and Technical Services")
            reference_files: List of reference files provided (for context)
            **kwargs: Additional parameters

        Returns:
            EvalResult with score and details
        """
        if not task_prompt:
            task_prompt = kwargs.get("question", "")

        if not task_prompt:
            return EvalResult(
                score=0.0,
                feedback="No task prompt provided for evaluation",
                details={"error": "missing_task_prompt"},
            )

        if not predicted or len(predicted.strip()) < 10:
            return EvalResult(
                score=0.0,
                feedback="No meaningful response provided",
                details={"error": "empty_or_minimal_response"},
            )

        # Use LLM to evaluate
        if self.use_llm:
            result = self._llm_evaluate(
                predicted=predicted,
                task_prompt=task_prompt,
                occupation=occupation,
                sector=sector,
                reference_files=reference_files,
            )
            if result is not None:
                return result

        # Fallback: basic heuristic scoring (not recommended)
        return self._heuristic_evaluate(predicted, task_prompt)

    def _get_llm_client(self) -> Optional[Any]:
        if self.llm_client is None:
            self.llm_client = build_llm_client_for_evaluator(EVALUATOR_NAME)
        return self.llm_client

    def _llm_evaluate(
        self,
        predicted: str,
        task_prompt: str,
        occupation: Optional[str] = None,
        sector: Optional[str] = None,
        reference_files: Optional[List[str]] = None,
    ) -> Optional[EvalResult]:
        """
        Use LLM to evaluate task completion quality.

        Returns:
            EvalResult or None if LLM evaluation fails
        """
        client = self._get_llm_client()
        if not client:
            logger.warning("LLM client not available for GDPVal evaluation")
            return None

        # Build context about the task
        context_parts = []
        if occupation:
            context_parts.append(f"Occupation: {occupation}")
        if sector:
            context_parts.append(f"Sector: {sector}")
        if reference_files:
            context_parts.append(f"Reference files provided: {len(reference_files)}")
        context = "\n".join(context_parts) if context_parts else "No additional context"

        system_prompt = """You are an expert evaluator for professional work deliverables.
You are assessing an AI agent's response to a real-world professional task.
Evaluate the response objectively based on what a senior professional in this field would expect.
Be strict but fair - professional work requires accuracy and completeness."""

        prompt = f"""TASK CONTEXT:
{context}

TASK DESCRIPTION:
{task_prompt[:4000]}

AGENT'S RESPONSE:
{predicted[:8000]}

Evaluate the agent's response on these 4 dimensions (each 0-25 points):

1. COMPLETION (0-25): Did the agent complete all parts of the requested task?
   - 0-5: Did not attempt or completely off-topic
   - 6-12: Partially completed, missing major components
   - 13-18: Mostly complete, minor omissions
   - 19-25: Fully complete, addressed all requirements

2. ACCURACY (0-25): Is the response factually correct and technically sound?
   - 0-5: Major factual errors or technically incorrect
   - 6-12: Some errors or questionable reasoning
   - 13-18: Mostly accurate with minor issues
   - 19-25: Accurate and technically sound

3. FORMAT (0-25): Is the deliverable in an appropriate professional format?
   - 0-5: Wrong format or unstructured
   - 6-12: Poor formatting, hard to use
   - 13-18: Acceptable format with issues
   - 19-25: Professional format, ready to use

4. PROFESSIONALISM (0-25): Does it meet industry standards?
   - 0-5: Unprofessional, would not be acceptable
   - 6-12: Below industry standard
   - 13-18: Meets basic standards
   - 19-25: Meets or exceeds industry standards

Return ONLY valid JSON:
{{"completion": <0-25>, "accuracy": <0-25>, "format": <0-25>, "professionalism": <0-25>, "feedback": "<brief explanation>"}}"""

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
                logger.warning("GDPVal LLM returned invalid JSON")
                return None

            # Extract scores
            completion = min(25, max(0, int(data.get("completion", 0))))
            accuracy = min(25, max(0, int(data.get("accuracy", 0))))
            format_score = min(25, max(0, int(data.get("format", 0))))
            professionalism = min(25, max(0, int(data.get("professionalism", 0))))
            feedback = data.get("feedback", "")

            # Calculate weighted score (0-100)
            raw_total = completion + accuracy + format_score + professionalism

            # Normalize to 0-1
            normalized_score = raw_total / 100.0

            return EvalResult(
                score=normalized_score,
                max_score=1.0,
                feedback=feedback or f"GDPVal score: {raw_total}/100",
                details={
                    "llm_used": True,
                    "llm_model": self.llm_model,
                    "completion": completion,
                    "accuracy": accuracy,
                    "format": format_score,
                    "professionalism": professionalism,
                    "raw_total": raw_total,
                    "llm_raw_output": raw,
                },
            )

        except Exception as e:
            logger.warning(f"GDPVal LLM evaluation failed: {e}")
            return None

    def _heuristic_evaluate(self, predicted: str, task_prompt: str) -> EvalResult:
        """
        Basic heuristic evaluation when LLM is unavailable.

        This is a fallback and provides only rough estimates.
        """
        # Basic length and structure checks
        word_count = len(predicted.split())
        has_structure = any(c in predicted for c in ["\n", ":", "-", "1.", "2."])

        # Score based on response length and structure
        if word_count < 50:
            base_score = 0.1
        elif word_count < 200:
            base_score = 0.3
        elif word_count < 500:
            base_score = 0.5
        else:
            base_score = 0.6

        # Bonus for structure
        if has_structure:
            base_score = min(1.0, base_score + 0.2)

        return EvalResult(
            score=base_score,
            max_score=1.0,
            feedback="Heuristic evaluation (LLM unavailable)",
            details={
                "llm_used": False,
                "heuristic": True,
                "word_count": word_count,
                "has_structure": has_structure,
            },
        )
