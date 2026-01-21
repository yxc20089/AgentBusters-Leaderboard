"""
Execution Evaluator for action quality assessment.

Evaluates the quality of the agent's final recommendation and methodology,
using rubric-based LLM-as-judge and code execution validation.
"""

from typing import Any, Optional

import structlog

from cio_agent.models import (
    ExecutionScore,
    Task,
    AgentResponse,
    CodeExecution,
    TaskCategory,
)

logger = structlog.get_logger()


class ExecutionEvaluator:
    """
    Evaluates agent's execution quality and methodology.

    Scoring criteria (30% of Role Score):
    - Rubric-based assessment of response quality
    - Code execution validation for numerical tasks
    - Methodology quality scoring

    Penalties:
    - 50% penalty for missing code execution in numerical tasks
    """

    # Task categories requiring mandatory code execution
    NUMERICAL_CATEGORIES = [
        TaskCategory.NUMERICAL_REASONING,
        TaskCategory.ADJUSTMENTS,
        TaskCategory.FINANCIAL_MODELING,
    ]

    def __init__(
        self,
        task: Task,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize the execution evaluator.

        Args:
            task: The task being evaluated
            llm_client: Optional LLM client for rubric-based scoring
        """
        self.task = task
        self.llm_client = llm_client

    def _check_mandatory_elements(
        self,
        response: str,
        mandatory_elements: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Check for mandatory elements in the response.

        Returns:
            Tuple of (elements_found, elements_missing)
        """
        response_lower = response.lower()
        found = []
        missing = []

        for element in mandatory_elements:
            element_lower = element.lower()
            # Check for element phrase or key words
            element_words = element_lower.split()

            if element_lower in response_lower:
                found.append(element)
                continue

            # Check if majority of words appear
            words_found = sum(1 for w in element_words if w in response_lower)
            if words_found >= len(element_words) * 0.7:
                found.append(element)
            else:
                missing.append(element)

        return found, missing

    def _score_recommendation_clarity(self, recommendation: str) -> float:
        """
        Score the clarity of the recommendation.

        Looks for:
        - Clear action (Buy/Sell/Hold or specific answer)
        - Supporting rationale
        - Quantitative backing
        """
        score = 0.0
        rec_lower = recommendation.lower()

        # Clear action present
        actions = ["buy", "sell", "hold", "bullish", "bearish", "neutral", "recommend"]
        if any(action in rec_lower for action in actions):
            score += 30

        # Has numerical support
        import re
        numbers = re.findall(r"\d+\.?\d*%?", recommendation)
        if len(numbers) >= 1:
            score += 20
        if len(numbers) >= 3:
            score += 10

        # Has reasoning keywords
        reasoning_words = ["because", "due to", "since", "given", "therefore", "based on"]
        if any(word in rec_lower for word in reasoning_words):
            score += 20

        # Has metric references
        metric_words = ["revenue", "margin", "growth", "pe", "eps", "ebitda", "cash flow"]
        metric_count = sum(1 for word in metric_words if word in rec_lower)
        score += min(20, metric_count * 5)

        return min(100, score)

    def _evaluate_code_quality(
        self,
        code_executions: list[CodeExecution],
    ) -> tuple[float, str]:
        """
        Evaluate the quality of code executions.

        Returns:
            Tuple of (methodology_score, feedback)
        """
        if not code_executions:
            return 0.0, "No code execution provided."

        total_score = 0.0
        feedback_parts = []

        for i, execution in enumerate(code_executions):
            exec_score = 0.0

            # Check success
            if execution.success:
                exec_score += 30
            else:
                feedback_parts.append(f"Code execution {i+1} failed: {execution.error_message}")
                continue

            # Check libraries used
            good_libs = ["pandas", "numpy", "scipy"]
            libs_used = [lib for lib in execution.libraries_used if lib in good_libs]
            exec_score += len(libs_used) * 10  # Up to 30

            # Check code length (not too short, not too long)
            lines = execution.code.strip().split("\n")
            if 5 <= len(lines) <= 50:
                exec_score += 20
            elif len(lines) > 0:
                exec_score += 10

            # Check for output
            if execution.output:
                exec_score += 20

            total_score += exec_score

        avg_score = total_score / len(code_executions) if code_executions else 0.0

        if avg_score >= 80:
            feedback_parts.insert(0, "Excellent code methodology.")
        elif avg_score >= 50:
            feedback_parts.insert(0, "Adequate code methodology.")
        else:
            feedback_parts.insert(0, "Poor code methodology.")

        return min(100, avg_score), " ".join(feedback_parts)

    async def _llm_rubric_score(
        self,
        response: AgentResponse,
    ) -> tuple[float, str, Optional[str]]:
        """
        Use LLM to score response against rubric.

        Returns:
            Tuple of (score, feedback, raw_output)
        """
        if not self.llm_client:
            # Fallback to heuristic scoring
            score, feedback = self._heuristic_rubric_score(response)
            return score, feedback, None

        rubric = self.task.rubric
        prompt = f"""
        You are evaluating a finance agent's response against a scoring rubric.

        TASK: {self.task.question}
        CATEGORY: {self.task.category.value}

        RUBRIC CRITERIA:
        {chr(10).join(f'- {c}' for c in rubric.criteria)}

        MANDATORY ELEMENTS:
        {chr(10).join(f'- {e}' for e in rubric.mandatory_elements)}

        AGENT'S ANALYSIS:
        {response.analysis[:2000]}

        AGENT'S RECOMMENDATION:
        {response.recommendation}

        Score this response from 0 to 100. Consider:
        1. Does it meet all rubric criteria?
        2. Does it include mandatory elements?
        3. Is the reasoning clear and supported by data?
        4. Is the final recommendation justified?

        Respond with ONLY a JSON object:
        {{"score": <0-100>, "feedback": "<brief feedback>"}}
        """

        raw_output = None
        try:
            raw_output = await self.llm_client.generate(prompt)
            import json
            parsed = json.loads(raw_output)
            return parsed.get("score", 50), parsed.get("feedback", ""), raw_output
        except Exception as e:
            logger.warning("llm_rubric_failed", error=str(e))
            score, feedback = self._heuristic_rubric_score(response)
            return score, feedback, raw_output

    def _heuristic_rubric_score(
        self,
        response: AgentResponse,
    ) -> tuple[float, str]:
        """
        Heuristic rubric scoring without LLM.
        """
        score = 50.0  # Base score
        feedback_parts = []

        # Check mandatory elements
        full_response = f"{response.analysis} {response.recommendation}"
        found, missing = self._check_mandatory_elements(
            full_response,
            self.task.rubric.mandatory_elements,
        )

        if found:
            elements_ratio = len(found) / len(self.task.rubric.mandatory_elements)
            score += elements_ratio * 20
            feedback_parts.append(f"Found {len(found)}/{len(self.task.rubric.mandatory_elements)} mandatory elements.")

        if missing:
            feedback_parts.append(f"Missing: {', '.join(missing[:3])}.")

        # Check recommendation clarity
        clarity_score = self._score_recommendation_clarity(response.recommendation)
        score += clarity_score * 0.3  # Up to 30 additional points

        # Check response length (too short is bad)
        if len(response.analysis) > 500:
            score += 10
        if len(response.analysis) > 1000:
            score += 5

        return min(100, score), " ".join(feedback_parts)

    async def score(
        self,
        response: AgentResponse,
    ) -> ExecutionScore:
        """
        Score the agent's execution quality.

        Args:
            response: The agent's response to evaluate

        Returns:
            ExecutionScore with detailed breakdown
        """
        # Get rubric score
        rubric_score, rubric_feedback, llm_raw_output = await self._llm_rubric_score(response)

        # Check code execution requirement
        code_penalty = 0.0
        methodology_score = 50.0
        methodology_feedback = ""

        if self.task.category in self.NUMERICAL_CATEGORIES:
            if not response.code_executions:
                code_penalty = 0.5
                methodology_feedback = "Missing required code execution for numerical task."
                logger.warning(
                    "missing_code_execution",
                    task_id=self.task.question_id,
                    category=self.task.category.value,
                )
            else:
                methodology_score, methodology_feedback = self._evaluate_code_quality(
                    response.code_executions
                )
        else:
            # For non-numerical tasks, code is optional
            if response.code_executions:
                methodology_score, methodology_feedback = self._evaluate_code_quality(
                    response.code_executions
                )
            else:
                methodology_score = 70  # Default for non-numerical tasks
                methodology_feedback = "Code execution not required for this task category."

        # Calculate final score with penalty
        final_score = rubric_score * (1.0 - code_penalty)

        # Combine feedback
        all_feedback = [rubric_feedback, methodology_feedback]
        combined_feedback = " ".join(f for f in all_feedback if f)

        logger.info(
            "execution_evaluation",
            rubric_score=rubric_score,
            methodology_score=methodology_score,
            code_penalty=code_penalty,
            final_score=final_score,
        )

        return ExecutionScore(
            score=final_score,
            rubric_score=rubric_score,
            code_execution_penalty=code_penalty,
            methodology_score=methodology_score,
            feedback=combined_feedback,
            llm_raw_output=llm_raw_output,
        )
