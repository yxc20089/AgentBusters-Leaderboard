"""
Fundamental Evaluator for data accuracy validation.

Validates extracted financial data against XBRL ground truth,
scoring on precision and completeness.
"""

from typing import Any, Optional

import structlog

from cio_agent.models import FundamentalScore, FinancialData, GroundTruth

logger = structlog.get_logger()


class FundamentalEvaluator:
    """
    Validates agent's extracted financial data against ground truth.

    Scoring criteria (40% of Role Score):
    - Accuracy of numerical extractions
    - Completeness of required fields
    - Use of correct filing source

    Tolerance: 1% for rounding differences in financial figures.
    """

    def __init__(
        self,
        ground_truth: GroundTruth,
        tolerance: float = 0.01,
    ):
        """
        Initialize the fundamental evaluator.

        Args:
            ground_truth: Ground truth containing verified financials
            tolerance: Acceptable relative error (default 1%)
        """
        self.ground_truth = ground_truth
        self.tolerance = tolerance

    def _extract_numeric_fields(self, financials: FinancialData) -> dict[str, Optional[float]]:
        """
        Extract numeric fields from FinancialData.
        """
        fields = {
            "revenue": financials.revenue,
            "gross_profit": financials.gross_profit,
            "operating_income": financials.operating_income,
            "net_income": financials.net_income,
            "ebitda": financials.ebitda,
            "eps": financials.eps,
            "total_assets": financials.total_assets,
            "total_liabilities": financials.total_liabilities,
            "shareholders_equity": financials.shareholders_equity,
            "operating_cash_flow": financials.operating_cash_flow,
            "free_cash_flow": financials.free_cash_flow,
            "gross_margin": financials.gross_margin,
            "operating_margin": financials.operating_margin,
            "net_margin": financials.net_margin,
            "pe_ratio": financials.pe_ratio,
            "market_cap": financials.market_cap,
        }
        return fields

    def _compare_values(
        self,
        agent_value: Optional[float],
        truth_value: Optional[float],
        field_name: str,
    ) -> tuple[bool, str]:
        """
        Compare a single value with tolerance.

        Returns:
            Tuple of (is_correct, reason)
        """
        # If ground truth is None, we can't validate
        if truth_value is None:
            return True, "no_ground_truth"

        # If agent didn't provide value
        if agent_value is None:
            return False, "missing"

        # Handle zero cases
        if truth_value == 0:
            if agent_value == 0:
                return True, "exact_match"
            else:
                return False, "wrong_value"

        # Calculate relative error
        relative_error = abs(agent_value - truth_value) / abs(truth_value)

        if relative_error <= self.tolerance:
            return True, "within_tolerance"
        else:
            return False, f"off_by_{relative_error*100:.1f}%"

    def score(
        self,
        agent_financials: FinancialData,
        required_fields: Optional[list[str]] = None,
    ) -> FundamentalScore:
        """
        Score the agent's extracted financial data.

        Args:
            agent_financials: Financials extracted by the agent
            required_fields: Optional list of fields that must be present

        Returns:
            FundamentalScore with detailed breakdown
        """
        truth_fields = self._extract_numeric_fields(self.ground_truth.financials)
        agent_fields = self._extract_numeric_fields(agent_financials)

        # Determine which fields to evaluate
        # Only evaluate fields that have ground truth values
        fields_to_check = [
            f for f, v in truth_fields.items()
            if v is not None
        ]

        if not fields_to_check:
            logger.warning("no_ground_truth_fields")
            return FundamentalScore(
                score=100.0,  # No fields to check, assume correct
                correct_fields=0,
                total_fields=0,
                field_accuracy={},
                feedback="No ground truth fields available for validation.",
            )

        # Evaluate each field
        field_accuracy = {}
        correct_count = 0

        for field in fields_to_check:
            is_correct, reason = self._compare_values(
                agent_fields.get(field),
                truth_fields.get(field),
                field,
            )
            field_accuracy[field] = is_correct

            if is_correct:
                correct_count += 1

            logger.debug(
                "field_comparison",
                field=field,
                agent=agent_fields.get(field),
                truth=truth_fields.get(field),
                correct=is_correct,
                reason=reason,
            )

        # Calculate score
        total_fields = len(fields_to_check)
        score = (correct_count / total_fields) * 100 if total_fields > 0 else 0.0

        # Check required fields
        missing_required = []
        if required_fields:
            for field in required_fields:
                if field in agent_fields and agent_fields[field] is None:
                    missing_required.append(field)

        # Generate feedback
        feedback_parts = []

        if score >= 90:
            feedback_parts.append(f"Excellent accuracy: {correct_count}/{total_fields} fields correct.")
        elif score >= 70:
            feedback_parts.append(f"Good accuracy: {correct_count}/{total_fields} fields correct.")
        elif score >= 50:
            feedback_parts.append(f"Moderate accuracy: {correct_count}/{total_fields} fields correct.")
        else:
            feedback_parts.append(f"Poor accuracy: {correct_count}/{total_fields} fields correct.")

        # List incorrect fields
        incorrect_fields = [f for f, v in field_accuracy.items() if not v]
        if incorrect_fields:
            feedback_parts.append(f"Incorrect fields: {', '.join(incorrect_fields[:5])}.")

        if missing_required:
            feedback_parts.append(f"Missing required fields: {', '.join(missing_required)}.")

        logger.info(
            "fundamental_evaluation",
            score=score,
            correct=correct_count,
            total=total_fields,
            incorrect_fields=incorrect_fields,
        )

        return FundamentalScore(
            score=score,
            correct_fields=correct_count,
            total_fields=total_fields,
            field_accuracy=field_accuracy,
            feedback=" ".join(feedback_parts),
        )

    def validate_specific_metric(
        self,
        agent_value: Optional[float],
        metric_name: str,
    ) -> tuple[bool, float, str]:
        """
        Validate a specific metric against ground truth.

        Useful for validating the primary metric of a task.

        Args:
            agent_value: Value provided by agent
            metric_name: Name of the metric to validate

        Returns:
            Tuple of (is_correct, error_percentage, feedback)
        """
        truth_fields = self._extract_numeric_fields(self.ground_truth.financials)
        truth_value = truth_fields.get(metric_name)

        if truth_value is None:
            return True, 0.0, "No ground truth available for validation."

        if agent_value is None:
            return False, 100.0, f"Agent did not provide {metric_name}."

        if truth_value == 0:
            if agent_value == 0:
                return True, 0.0, "Exact match."
            return False, 100.0, f"Agent provided {agent_value}, expected 0."

        relative_error = abs(agent_value - truth_value) / abs(truth_value)
        error_percentage = relative_error * 100

        if relative_error <= self.tolerance:
            return True, error_percentage, f"Within tolerance ({error_percentage:.2f}% error)."
        else:
            return False, error_percentage, f"Outside tolerance ({error_percentage:.2f}% error). Expected: {truth_value}, Got: {agent_value}."
