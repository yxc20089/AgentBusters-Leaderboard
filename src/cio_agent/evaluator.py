"""
Comprehensive Evaluator for CIO-Agent FAB++ System.

Orchestrates the complete evaluation pipeline:
1. Dynamic task generation
2. Agent execution monitoring
3. Adversarial debate
4. Multi-dimensional scoring
5. Alpha Score calculation
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from cio_agent.models import (
    Task,
    AgentResponse,
    DebateRebuttal,
    EvaluationResult,
    RoleScore,
    AlphaScore,
    LookAheadPenalty,
    TemporalViolation,
)
from cio_agent.task_generator import DynamicTaskGenerator
from cio_agent.debate import AdversarialDebateManager
from cio_agent.orchestrator import A2AOrchestrator
from evaluators.macro import MacroEvaluator
from evaluators.fundamental import FundamentalEvaluator
from evaluators.execution import ExecutionEvaluator
from evaluators.cost_tracker import CostTracker
from evaluators.options import OptionsEvaluator, OPTIONS_CATEGORIES

logger = structlog.get_logger()


class ComprehensiveEvaluator:
    """
    Complete evaluation pipeline for CIO-Agent FAB++.

    Combines all evaluation components:
    - MacroEvaluator: Strategic reasoning (30%)
    - FundamentalEvaluator: Data accuracy (40%)
    - ExecutionEvaluator: Action quality (30%)
    - AdversarialDebateManager: Robustness testing
    - CostTracker: Efficiency measurement
    """

    # Role score weights
    MACRO_WEIGHT = 0.30
    FUNDAMENTAL_WEIGHT = 0.40
    EXECUTION_WEIGHT = 0.30

    def __init__(
        self,
        task_generator: Optional[DynamicTaskGenerator] = None,
        debate_manager: Optional[AdversarialDebateManager] = None,
        orchestrator: Optional[A2AOrchestrator] = None,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize the comprehensive evaluator.

        Args:
            task_generator: Generator for dynamic FAB tasks
            debate_manager: Manager for adversarial debate
            orchestrator: A2A protocol orchestrator
            llm_client: LLM client for evaluation
        """
        self.task_generator = task_generator or DynamicTaskGenerator()
        self.debate_manager = debate_manager or AdversarialDebateManager(llm_client=llm_client)
        self.orchestrator = orchestrator or A2AOrchestrator()
        self.llm_client = llm_client

    def _generate_evaluation_id(self) -> str:
        """Generate a unique evaluation ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"eval_{timestamp}_{uuid.uuid4().hex[:8]}"

    def _calculate_role_score(
        self,
        macro_score: float,
        fundamental_score: float,
        execution_score: float,
    ) -> float:
        """
        Calculate weighted role score.

        Args:
            macro_score: Score from MacroEvaluator (0-100)
            fundamental_score: Score from FundamentalEvaluator (0-100)
            execution_score: Score from ExecutionEvaluator (0-100)

        Returns:
            Weighted role score (0-100)
        """
        return (
            self.MACRO_WEIGHT * macro_score +
            self.FUNDAMENTAL_WEIGHT * fundamental_score +
            self.EXECUTION_WEIGHT * execution_score
        )

    def _aggregate_temporal_violations(
        self,
        edgar_violations: list[TemporalViolation],
        yfinance_violations: list[TemporalViolation],
    ) -> LookAheadPenalty:
        """
        Aggregate temporal violations from all MCP tools.

        Args:
            edgar_violations: Violations from SEC EDGAR client
            yfinance_violations: Violations from Yahoo Finance client

        Returns:
            LookAheadPenalty with all violations
        """
        all_violations = edgar_violations + yfinance_violations
        total_days = sum(v.days_ahead for v in all_violations)

        # Cap penalty at 50%
        penalty = min(0.5, total_days / 365.0)

        return LookAheadPenalty(
            penalty=penalty,
            violations=all_violations,
            total_days_ahead=total_days,
        )

    async def evaluate_response(
        self,
        task: Task,
        agent_response: AgentResponse,
        agent_rebuttal: Optional[DebateRebuttal] = None,
        cost_tracker: Optional[CostTracker] = None,
        temporal_violations: Optional[list[TemporalViolation]] = None,
    ) -> EvaluationResult:
        """
        Perform comprehensive evaluation of an agent's response.

        Args:
            task: The evaluation task
            agent_response: The agent's response
            agent_rebuttal: The agent's rebuttal (if debate conducted)
            cost_tracker: Tracker with cost data
            temporal_violations: List of temporal violations

        Returns:
            Complete EvaluationResult
        """
        evaluation_id = self._generate_evaluation_id()
        cost_tracker = cost_tracker or CostTracker()
        temporal_violations = temporal_violations or []

        logger.info(
            "starting_evaluation",
            evaluation_id=evaluation_id,
            task_id=task.question_id,
            agent_id=agent_response.agent_id,
        )

        # Check if this is an options trading task
        is_options_task = task.category in OPTIONS_CATEGORIES

        if is_options_task:
            # Use OptionsEvaluator for options trading tasks
            logger.info("using_options_evaluator", category=task.category.value)
            options_evaluator = OptionsEvaluator(
                task=task,
                llm_client=self.llm_client,
            )
            options_result = await options_evaluator.score(agent_response)

            # Map options scores to role score dimensions
            # Options tasks: Strategy Quality -> Macro, P&L + Greeks -> Fundamental, Risk Mgmt -> Execution
            from cio_agent.models import MacroScore, FundamentalScore, ExecutionScore as ExecScore

            macro_result = MacroScore(
                score=options_result.strategy_quality,
                similarity_score=options_result.strategy_quality,
                theme_coverage=options_result.strategy_quality / 100,
                themes_identified=["options strategy"],
                themes_missed=[],
                feedback=f"Strategy quality: {options_result.feedback}",
            )

            fundamental_result = FundamentalScore(
                score=(options_result.pnl_accuracy + options_result.greeks_accuracy) / 2,
                correct_fields=2 if options_result.pnl_accuracy > 50 else 1,
                total_fields=4,
                field_accuracy={
                    "pnl": options_result.pnl_accuracy > 60,
                    "greeks": options_result.greeks_accuracy > 60,
                },
                feedback=f"P&L accuracy: {options_result.pnl_accuracy:.0f}, Greeks: {options_result.greeks_accuracy:.0f}",
            )

            execution_result = ExecScore(
                score=options_result.risk_management,
                rubric_score=options_result.score,
                code_execution_penalty=0.0,
                methodology_score=options_result.risk_management,
                feedback=f"Risk management: {options_result.risk_management:.0f}",
            )
        else:
            # Phase 1: Macro Score (standard FAB tasks)
            macro_evaluator = MacroEvaluator(
                ground_truth=task.ground_truth,
                use_embeddings=False,  # Set True if sentence-transformers available
            )
            macro_result = macro_evaluator.score(agent_response.analysis)

            # Phase 2: Fundamental Score
            fundamental_evaluator = FundamentalEvaluator(
                ground_truth=task.ground_truth,
            )
            fundamental_result = fundamental_evaluator.score(
                agent_response.extracted_financials
            )

            # Phase 3: Execution Score
            execution_evaluator = ExecutionEvaluator(
                task=task,
                llm_client=self.llm_client,
            )
            execution_result = await execution_evaluator.score(agent_response)

        # Combine into Role Score
        role_score_value = self._calculate_role_score(
            macro_score=macro_result.score,
            fundamental_score=fundamental_result.score,
            execution_score=execution_result.score,
        )

        role_score = RoleScore(
            total=role_score_value,
            macro=macro_result,
            fundamental=fundamental_result,
            execution=execution_result,
        )

        # Phase 4: Debate
        if agent_rebuttal:
            debate_result = await self.debate_manager.conduct_debate(
                task=task,
                agent_response=agent_response,
                agent_rebuttal=agent_rebuttal,
            )
        else:
            # No debate conducted, use neutral multiplier
            from cio_agent.models import DebateResult, ConvictionLevel
            debate_result = DebateResult(
                counter_argument="No debate conducted.",
                agent_rebuttal="No rebuttal provided.",
                debate_multiplier=1.0,
                conviction_level=ConvictionLevel.MEDIUM,
                feedback="Debate phase skipped.",
            )

        # Phase 5: Cost and Penalties
        cost_breakdown = cost_tracker.get_breakdown()

        lookahead_penalty = LookAheadPenalty(
            penalty=0.0,
            violations=temporal_violations,
            total_days_ahead=sum(v.days_ahead for v in temporal_violations),
        )
        if temporal_violations:
            lookahead_penalty.penalty = min(0.5, lookahead_penalty.total_days_ahead / 365.0)

        # Phase 6: Alpha Score
        alpha_score = AlphaScore.calculate(
            role_score=role_score_value,
            debate_multiplier=debate_result.debate_multiplier,
            cost_usd=cost_breakdown.total_cost_usd,
            lookahead_penalty=lookahead_penalty.penalty,
        )

        # Compile result
        result = EvaluationResult(
            evaluation_id=evaluation_id,
            task_id=task.question_id,
            agent_id=agent_response.agent_id,
            agent_analysis=agent_response.analysis,
            agent_recommendation=agent_response.recommendation,
            role_score=role_score,
            debate_result=debate_result,
            cost_breakdown=cost_breakdown,
            lookahead_penalty=lookahead_penalty,
            alpha_score=alpha_score,
            execution_feedback=execution_result.feedback,
            macro_feedback=macro_result.feedback if hasattr(macro_result, 'feedback') else "",
            tool_calls=agent_response.tool_calls,
            code_executions=agent_response.code_executions,
            total_execution_time_seconds=agent_response.execution_time_seconds,
            total_llm_calls=cost_breakdown.llm_calls,
            total_tokens=cost_breakdown.total_input_tokens + cost_breakdown.total_output_tokens,
        )

        logger.info(
            "evaluation_completed",
            evaluation_id=evaluation_id,
            alpha_score=alpha_score.score,
            role_score=role_score_value,
            debate_multiplier=debate_result.debate_multiplier,
        )

        return result

    async def run_full_evaluation(
        self,
        task: Task,
        agent_client: Any,
        conduct_debate: bool = True,
    ) -> EvaluationResult:
        """
        Run a complete evaluation cycle.

        Args:
            task: The evaluation task
            agent_client: Client for the agent being evaluated
            conduct_debate: Whether to conduct adversarial debate

        Returns:
            Complete EvaluationResult
        """
        cost_tracker = CostTracker()

        # Phase 1: Send task and get response
        logger.info("phase_1_task_assignment", task_id=task.question_id)
        agent_response = await agent_client.process_task(task)

        # Track tool costs
        cost_tracker.add_tool_calls(agent_response.tool_calls)

        # Simulate LLM costs (in real usage, these would come from the agent)
        cost_tracker.add_llm_call(
            model=agent_client.model,
            input_tokens=2000,
            output_tokens=1000,
            purpose="task_response",
        )

        # Phase 2: Conduct debate (optional)
        agent_rebuttal = None
        if conduct_debate:
            logger.info("phase_2_debate", task_id=task.question_id)

            # Generate counter-argument
            counter_argument = await self.debate_manager.generate_counter_argument(
                agent_thesis=agent_response.recommendation,
                financial_data=agent_response.extracted_financials,
                task=task,
            )

            # Get rebuttal
            agent_rebuttal = await agent_client.process_challenge(
                task_id=task.question_id,
                challenge=counter_argument,
                original_response=agent_response,
                ticker=task.ticker,
            )

            # Track debate costs
            cost_tracker.add_llm_call(
                model=agent_client.model,
                input_tokens=1500,
                output_tokens=800,
                purpose="rebuttal",
            )

        # Phase 3: Evaluate
        logger.info("phase_3_evaluation", task_id=task.question_id)
        result = await self.evaluate_response(
            task=task,
            agent_response=agent_response,
            agent_rebuttal=agent_rebuttal,
            cost_tracker=cost_tracker,
        )

        return result


class EvaluationReporter:
    """
    Generates comprehensive evaluation reports.

    Supports multiple output formats:
    - Markdown (detailed human-readable)
    - JSON (machine-readable)
    - Summary (brief overview)
    """

    @staticmethod
    def generate_markdown_report(
        task: Task,
        agent_response: AgentResponse,
        result: EvaluationResult,
    ) -> str:
        """
        Generate a detailed Markdown evaluation report.

        Args:
            task: The evaluation task
            agent_response: The agent's response
            result: The evaluation result

        Returns:
            Markdown-formatted report string
        """
        report = f"""# CIO-Agent Evaluation Report

## Task Information
- **Evaluation ID**: {result.evaluation_id}
- **Question ID**: {task.question_id}
- **Category**: {task.category.value}
- **Difficulty**: {task.difficulty.value}
- **Ticker**: {task.ticker}
- **Fiscal Year**: {task.fiscal_year}

### Question
{task.question}

## Agent Response Summary
- **Agent ID**: {result.agent_id}
- **Execution Time**: {result.total_execution_time_seconds:.2f} seconds

### Recommendation
{agent_response.recommendation[:500]}{"..." if len(agent_response.recommendation) > 500 else ""}

## Evaluation Scores

### Role Score: {result.role_score.total:.2f}/100

| Dimension | Score | Weight | Contribution |
|-----------|-------|--------|--------------|
| Macro Analysis | {result.role_score.macro.score:.2f} | 30% | {result.role_score.macro.score * 0.3:.2f} |
| Fundamental Accuracy | {result.role_score.fundamental.score:.2f} | 40% | {result.role_score.fundamental.score * 0.4:.2f} |
| Execution Quality | {result.role_score.execution.score:.2f} | 30% | {result.role_score.execution.score * 0.3:.2f} |

#### Macro Analysis Feedback
{result.role_score.macro.feedback}

#### Fundamental Accuracy Feedback
{result.role_score.fundamental.feedback}

#### Execution Quality Feedback
{result.role_score.execution.feedback}

### Adversarial Debate

**Challenge**:
{result.debate_result.counter_argument[:500]}{"..." if len(result.debate_result.counter_argument) > 500 else ""}

**Agent Rebuttal**:
{result.debate_result.agent_rebuttal[:500]}{"..." if len(result.debate_result.agent_rebuttal) > 500 else ""}

**Debate Multiplier**: {result.debate_result.debate_multiplier}x ({result.debate_result.conviction_level.value} conviction)

{result.debate_result.feedback}

### Efficiency Metrics

- **Total Cost**: ${result.cost_breakdown.total_cost_usd:.4f}
  - LLM Costs: ${result.cost_breakdown.llm_cost_usd:.4f}
  - Tool Costs: ${result.cost_breakdown.tool_cost_usd:.4f}
- **LLM Calls**: {result.cost_breakdown.llm_calls}
- **Tool Calls**: {result.cost_breakdown.tool_calls}
- **Total Tokens**: {result.total_tokens:,}

### Temporal Integrity

- **Look-Ahead Penalty**: {result.lookahead_penalty.penalty:.2f}
- **Violations**: {len(result.lookahead_penalty.violations)}
"""

        if result.lookahead_penalty.violations:
            report += "\n**Violation Details**:\n"
            for v in result.lookahead_penalty.violations[:5]:
                report += f"- {v.tool_name}: Requested {v.requested_date}, Allowed {v.simulation_date} ({v.days_ahead} days ahead)\n"

        report += f"""
## Final Alpha Score: {result.alpha_score.score:.2f}

**Formula**: Alpha = (Role Score × Debate Multiplier) / (ln(1 + Cost) × (1 + LookAhead Penalty))

**Calculation**:
```
Alpha = ({result.alpha_score.role_score:.2f} × {result.alpha_score.debate_multiplier}) / (ln(1 + {result.alpha_score.cost_usd:.4f}) × (1 + {result.alpha_score.lookahead_penalty:.2f}))
     = {result.alpha_score.score:.2f}
```

---

## Tool Usage Log
"""

        for i, call in enumerate(result.tool_calls[:10], 1):
            report += f"\n{i}. `{call.tool_name}` @ {call.timestamp.strftime('%H:%M:%S')}\n"
            report += f"   - Tokens: {call.response_tokens}, Duration: {call.duration_ms}ms\n"

        if len(result.tool_calls) > 10:
            report += f"\n... and {len(result.tool_calls) - 10} more tool calls\n"

        report += f"""
---
*Generated by CIO-Agent FAB++ Evaluator v1.0*
*Timestamp: {result.timestamp.isoformat()}*
"""

        return report

    @staticmethod
    def generate_json_report(result: EvaluationResult) -> dict:
        """
        Generate a JSON-serializable evaluation report.

        Args:
            result: The evaluation result

        Returns:
            Dictionary representation of the result
        """
        return {
            "evaluation_id": result.evaluation_id,
            "task_id": result.task_id,
            "agent_id": result.agent_id,
            "timestamp": result.timestamp.isoformat(),
            "scores": {
                "role_score": result.role_score.total,
                "macro_score": result.role_score.macro.score,
                "fundamental_score": result.role_score.fundamental.score,
                "execution_score": result.role_score.execution.score,
                "debate_multiplier": result.debate_result.debate_multiplier,
                "debate_conviction": result.debate_result.conviction_level.value,
                "total_cost_usd": result.cost_breakdown.total_cost_usd,
                "lookahead_penalty": result.lookahead_penalty.penalty,
                "alpha_score": result.alpha_score.score,
            },
            "debate": {
                "counter_argument": result.debate_result.counter_argument,
                "agent_rebuttal": result.debate_result.agent_rebuttal,
                "rebuttal_quality": result.debate_result.conviction_level.value,
            },
            "tool_usage": {
                "total_calls": len(result.tool_calls),
                "total_tokens": result.total_tokens,
                "temporal_violations": len(result.lookahead_penalty.violations),
            },
            "performance": {
                "execution_time_seconds": result.total_execution_time_seconds,
                "llm_calls": result.total_llm_calls,
            },
        }

    @staticmethod
    def generate_summary(result: EvaluationResult) -> str:
        """
        Generate a brief summary of the evaluation.

        Args:
            result: The evaluation result

        Returns:
            Brief summary string
        """
        return f"""
Evaluation Summary ({result.evaluation_id})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Task: {result.task_id}
Agent: {result.agent_id}

Alpha Score: {result.alpha_score.score:.2f}
├── Role Score: {result.role_score.total:.2f}/100
│   ├── Macro: {result.role_score.macro.score:.1f}
│   ├── Fundamental: {result.role_score.fundamental.score:.1f}
│   └── Execution: {result.role_score.execution.score:.1f}
├── Debate: {result.debate_result.debate_multiplier}x ({result.debate_result.conviction_level.value})
├── Cost: ${result.cost_breakdown.total_cost_usd:.4f}
└── Penalty: {result.lookahead_penalty.penalty:.2f}

Tool Calls: {len(result.tool_calls)} | Time: {result.total_execution_time_seconds:.1f}s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
