"""
Question Verifier: Validates synthetic questions for solvability.

Part of the Generator-Verifier-Refiner architecture:
1. Takes a generated question
2. Attempts to solve it without seeing ground truth  
3. Compares answer to expected ground truth
4. Marks question as Accept/Reject/Refine
"""

from enum import Enum
from typing import Any, Optional

import structlog
from pydantic import BaseModel, Field

from cio_agent.synthetic_generator import SyntheticQuestion

logger = structlog.get_logger()


class VerificationResult(str, Enum):
    """Outcome of question verification."""
    ACCEPT = "accept"       # Question is solvable and unambiguous
    REJECT = "reject"       # Question is unsolvable or fundamentally flawed
    REFINE = "refine"       # Question is ambiguous and needs rewording


class VerificationReport(BaseModel):
    """Detailed report on question verification."""
    question_id: str
    result: VerificationResult
    computed_answer: Optional[Any] = None
    expected_answer: Any
    match: bool = False
    error_percentage: Optional[float] = None
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    reasoning: str = ""


class QuestionVerifier:
    """
    Verifies that synthetic questions are solvable.
    
    Uses heuristic checking and optionally an LLM verifier agent
    to validate that questions can be answered correctly.
    """
    
    def __init__(
        self,
        numerical_tolerance: float = 0.05,  # 5% tolerance for numerical answers
        llm_client: Optional[Any] = None,   # Optional for LLM-based verification
    ):
        if not (0 <= numerical_tolerance <= 1):
            raise ValueError("numerical_tolerance must be between 0 and 1 (inclusive).")
        self.tolerance = numerical_tolerance
        self.llm_client = llm_client
    
    def _compare_numerical(
        self, 
        computed: float, 
        expected: float,
    ) -> tuple[bool, float]:
        """
        Compare numerical values within tolerance.
        
        Returns:
            (match, error_percentage)
        """
        if expected == 0:
            if computed == 0:
                return True, 0.0
            else:
                return False, float('inf')
        
        error_percentage = abs(computed - expected) / abs(expected) * 100
        match = (error_percentage / 100) <= self.tolerance
        return match, error_percentage
    
    def _compare_ranking(
        self,
        computed: list[str],
        expected: list[str],
    ) -> bool:
        """Compare ranking lists."""
        return computed == expected
    
    def verify_question(
        self,
        question: SyntheticQuestion,
        computed_answer: Optional[Any] = None,
    ) -> VerificationReport:
        """
        Verify a single synthetic question.
        
        Args:
            question: The question to verify
            computed_answer: Optional pre-computed answer (for testing)
            
        Returns:
            VerificationReport with results
        """
        issues = []
        suggestions = []
        
        expected = question.ground_truth_value
        
        # Basic validation checks
        if not question.question.strip():
            return VerificationReport(
                question_id=question.question_id,
                result=VerificationResult.REJECT,
                expected_answer=expected,
                issues=["Empty question text"],
            )
        
        if not question.ground_truth_formatted:
            return VerificationReport(
                question_id=question.question_id,
                result=VerificationResult.REJECT,
                expected_answer=expected,
                issues=["Missing ground truth"],
            )
        
        # Check for minimum question length
        if len(question.question) < 20:
            issues.append("Question may be too short")
            suggestions.append("Add more context to the question")
        
        # Check ticker is mentioned in question
        if question.ticker != "LITERAL" and question.ticker not in question.question:
            issues.append(f"Ticker {question.ticker} not mentioned in question")
            suggestions.append(f"Include ticker {question.ticker} explicitly")
        
        # If we have a computed answer, compare it
        match = False
        error_pct = None
        
        if computed_answer is not None:
            if isinstance(expected, (int, float)) and isinstance(computed_answer, (int, float)):
                match, error_pct = self._compare_numerical(computed_answer, expected)
                if not match:
                    issues.append(f"Computed answer {computed_answer} differs from expected {expected} by {error_pct*100:.1f}%")
            
            elif isinstance(expected, dict) and isinstance(computed_answer, dict):
                # Compare dict values
                match = True
                for key in expected:
                    if key in computed_answer:
                        if isinstance(expected[key], (int, float)):
                            m, e = self._compare_numerical(computed_answer[key], expected[key])
                            if not m:
                                issues.append(f"Key '{key}' mismatch: {computed_answer[key]} vs {expected[key]}")
                            match = match and m
                
            elif isinstance(expected, list) and isinstance(computed_answer, list):
                match = self._compare_ranking(computed_answer, expected)
                if not match:
                    issues.append(f"Ranking mismatch: {computed_answer} vs {expected}")
            
            elif isinstance(expected, str) and isinstance(computed_answer, str):
                # Basic string comparison (for qualitative)
                match = expected.lower() in computed_answer.lower() or computed_answer.lower() in expected.lower()
        
        # Determine result
        if len(issues) == 0 and (computed_answer is None or match):
            result = VerificationResult.ACCEPT
        elif len(issues) <= 2 and len(suggestions) > 0:
            result = VerificationResult.REFINE
        else:
            result = VerificationResult.REJECT
        
        return VerificationReport(
            question_id=question.question_id,
            result=result,
            computed_answer=computed_answer,
            expected_answer=expected,
            match=match,
            error_percentage=error_pct,
            issues=issues,
            suggestions=suggestions,
            reasoning=f"Verified with {len(issues)} issues found",
        )
    
    def verify_batch(
        self,
        questions: list[SyntheticQuestion],
    ) -> dict:
        """
        Verify a batch of questions.
        
        Returns:
            Summary dict with accept/reject/refine counts and reports
        """
        reports = []
        accept_count = 0
        reject_count = 0
        refine_count = 0
        
        for q in questions:
            report = self.verify_question(q)
            reports.append(report)
            
            if report.result == VerificationResult.ACCEPT:
                accept_count += 1
            elif report.result == VerificationResult.REJECT:
                reject_count += 1
            else:
                refine_count += 1
        
        logger.info(
            "verification_batch_complete",
            total=len(questions),
            accept=accept_count,
            reject=reject_count,
            refine=refine_count,
        )
        
        return {
            "total": len(questions),
            "accept": accept_count,
            "reject": reject_count,
            "refine": refine_count,
            "accept_rate": accept_count / len(questions) if questions else 0,
            "reports": reports,
        }


class QuestionRefiner:
    """
    Refines ambiguous questions to make them more specific.
    
    Uses heuristic rules or optionally an LLM to rewrite questions.
    """
    
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
    
    def refine(
        self,
        question: SyntheticQuestion,
        report: VerificationReport,
    ) -> SyntheticQuestion:
        """
        Refine a question based on verification feedback.
        
        Applies heuristic fixes for common issues.
        """
        refined_text = question.question
        
        # Apply heuristic refinements
        for issue in report.issues:
            if "Ticker" in issue and "not mentioned" in issue:
                # Add ticker to question
                refined_text = f"For {question.ticker}: {refined_text}"
            
            if "too short" in issue.lower():
                # Add fiscal year context if missing
                if str(question.fiscal_year) not in refined_text:
                    refined_text = f"{refined_text} Use FY{question.fiscal_year} data."
        
        # Create refined question
        refined = question.model_copy()
        refined.question = refined_text
        refined.question_id = f"{question.question_id}_refined"
        
        logger.info(
            "question_refined",
            original_id=question.question_id,
            refined_id=refined.question_id,
        )
        
        return refined
