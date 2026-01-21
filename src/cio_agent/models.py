"""
Core data models for CIO-Agent FAB++ Evaluator.

These models define the structure for tasks, agent responses, evaluations,
and all related data types used throughout the system.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """Get current UTC time (Python 3.12+ compatible)."""
    return datetime.now(timezone.utc)


class TaskCategory(str, Enum):
    """FAB task categories covering financial analysis and options trading tasks."""
    # Original FAB categories (9 types)
    QUANTITATIVE_RETRIEVAL = "Quantitative Retrieval"
    QUALITATIVE_RETRIEVAL = "Qualitative Retrieval"
    NUMERICAL_REASONING = "Numerical Reasoning"
    COMPLEX_RETRIEVAL = "Complex Retrieval"
    ADJUSTMENTS = "Adjustments"
    BEAT_OR_MISS = "Beat or Miss"
    TRENDS = "Trends"
    FINANCIAL_MODELING = "Financial Modeling"
    MARKET_ANALYSIS = "Market Analysis"

    # Options Trading categories (Alpha Challenge)
    OPTIONS_PRICING = "Options Pricing"
    GREEKS_ANALYSIS = "Greeks Analysis"
    STRATEGY_CONSTRUCTION = "Strategy Construction"
    VOLATILITY_TRADING = "Volatility Trading"
    PNL_ATTRIBUTION = "P&L Attribution"
    RISK_MANAGEMENT = "Risk Management"
    COPY_TRADING = "Copy Trading"
    RACE_TO_10M = "Race to 10M"
    STRATEGY_DEFENSE = "Strategy Defense"


class TaskDifficulty(str, Enum):
    """Task difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class ConvictionLevel(str, Enum):
    """Agent conviction level during debate."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ViolationSeverity(str, Enum):
    """Severity of temporal violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# === Ground Truth Models ===

class FinancialData(BaseModel):
    """Financial data extracted from SEC filings."""
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    ebitda: Optional[float] = None
    eps: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    shareholders_equity: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    free_cash_flow: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    pe_ratio: Optional[float] = None
    market_cap: Optional[float] = None
    extra_fields: dict[str, Any] = Field(default_factory=dict)


class GroundTruth(BaseModel):
    """Ground truth data for task validation."""
    macro_thesis: str = Field(description="Expected macro analysis themes")
    financials: FinancialData = Field(default_factory=FinancialData)
    key_themes: list[str] = Field(default_factory=list)
    expected_recommendation: Optional[str] = None
    numerical_answer: Optional[float] = None
    tolerance: float = Field(default=0.01, description="Acceptable error tolerance (1%)")


# === Task Models ===

class TaskRubric(BaseModel):
    """Rubric for evaluating task responses."""
    criteria: list[str] = Field(default_factory=list)
    max_score: int = Field(default=100)
    mandatory_elements: list[str] = Field(default_factory=list)
    penalty_conditions: list[str] = Field(default_factory=list)


class Task(BaseModel):
    """A dynamic FAB++ evaluation task."""
    question_id: str = Field(description="Unique task identifier")
    category: TaskCategory
    question: str = Field(description="The task question for the agent")
    ticker: str = Field(description="Primary ticker symbol")
    fiscal_year: int = Field(description="Target fiscal year")
    simulation_date: datetime = Field(description="Simulated 'current' date for temporal locking")
    ground_truth: GroundTruth
    difficulty: TaskDifficulty = TaskDifficulty.MEDIUM
    rubric: TaskRubric = Field(default_factory=TaskRubric)
    available_tools: list[str] = Field(
        default_factory=lambda: ["sec-edgar-mcp", "yahoo-finance-mcp", "mcp-sandbox"]
    )
    deadline_seconds: int = Field(default=1800, description="Task deadline in seconds")
    requires_code_execution: bool = Field(default=False)

    @property
    def is_numerical_task(self) -> bool:
        """Check if this task requires numerical reasoning."""
        return self.category in [
            TaskCategory.NUMERICAL_REASONING,
            TaskCategory.ADJUSTMENTS,
            TaskCategory.FINANCIAL_MODELING,
        ]


class FABQuestionTemplate(BaseModel):
    """Template for FAB questions supporting dynamic generation."""
    template_id: str
    category: TaskCategory
    template: str = Field(description="Question template with placeholders like {ticker}, {year}")
    difficulty: TaskDifficulty
    metric: str = Field(description="Primary metric being queried")
    rubric: TaskRubric
    requires_code_execution: bool = False


# === Tool Usage Models ===

class ToolCall(BaseModel):
    """Record of a single tool invocation."""
    tool_name: str
    params: dict[str, Any]
    timestamp: datetime
    response_tokens: int = 0
    duration_ms: int = 0
    success: bool = True
    error_message: Optional[str] = None


class TemporalViolation(BaseModel):
    """Record of a temporal integrity violation."""
    ticker: str
    requested_date: str
    simulation_date: str
    days_ahead: int
    severity: ViolationSeverity
    tool_name: str
    timestamp: datetime


class CodeExecution(BaseModel):
    """Record of code execution in the sandbox."""
    code: str
    output: str
    execution_time_ms: int
    libraries_used: list[str] = Field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None


# === Agent Response Models ===

class AgentResponse(BaseModel):
    """Response from a White/Purple agent."""
    agent_id: str
    task_id: str
    analysis: str = Field(description="Agent's full analysis")
    recommendation: str = Field(description="Final recommendation (Buy/Sell/Hold or answer)")
    extracted_financials: FinancialData = Field(default_factory=FinancialData)
    code_executions: list[CodeExecution] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_utc_now)
    execution_time_seconds: float = 0.0


class DebateRebuttal(BaseModel):
    """Agent's rebuttal during adversarial debate."""
    agent_id: str
    task_id: str
    defense: str = Field(description="Agent's defense of their thesis")
    new_evidence_cited: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_utc_now)


# === Evaluation Models ===

class MacroScore(BaseModel):
    """Macro analysis dimension score."""
    score: float = Field(ge=0, le=100)
    similarity_score: float = Field(ge=0, le=100)
    theme_coverage: float = Field(ge=0, le=1)
    themes_identified: list[str] = Field(default_factory=list)
    themes_missed: list[str] = Field(default_factory=list)
    feedback: str = ""
    llm_raw_output: Optional[str] = None


class FundamentalScore(BaseModel):
    """Fundamental accuracy dimension score."""
    score: float = Field(ge=0, le=100)
    correct_fields: int = 0
    total_fields: int = 0
    field_accuracy: dict[str, bool] = Field(default_factory=dict)
    feedback: str = ""


class ExecutionScore(BaseModel):
    """Execution quality dimension score."""
    score: float = Field(ge=0, le=100)
    rubric_score: float = Field(ge=0, le=100)
    code_execution_penalty: float = Field(default=0.0, ge=0, le=1)
    methodology_score: float = Field(ge=0, le=100)
    feedback: str = ""
    llm_raw_output: Optional[str] = None


class RoleScore(BaseModel):
    """Combined hierarchical role assessment score."""
    total: float = Field(ge=0, le=100)
    macro: MacroScore
    fundamental: FundamentalScore
    execution: ExecutionScore
    weights: dict[str, float] = Field(
        default_factory=lambda: {"macro": 0.30, "fundamental": 0.40, "execution": 0.30}
    )


class DebateResult(BaseModel):
    """Result of adversarial debate phase."""
    counter_argument: str
    agent_rebuttal: str
    debate_multiplier: float = Field(ge=0.5, le=1.2)
    conviction_level: ConvictionLevel
    new_evidence_provided: bool = False
    hallucination_detected: bool = False
    immediate_concession: bool = False
    feedback: str = ""


class CostBreakdown(BaseModel):
    """Breakdown of evaluation costs."""
    llm_cost_usd: float = 0.0
    tool_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    tool_calls: int = 0
    tool_tokens: int = 0


class LookAheadPenalty(BaseModel):
    """Temporal integrity penalty calculation."""
    penalty: float = Field(default=0.0, ge=0, le=0.5)
    violations: list[TemporalViolation] = Field(default_factory=list)
    total_days_ahead: int = 0


class AlphaScore(BaseModel):
    """
    The Alpha Score metric.

    Formula: Alpha = (Role Score × Debate Multiplier) / (ln(1 + Cost) × (1 + LookAhead Penalty))
    """
    score: float
    role_score: float
    debate_multiplier: float
    cost_usd: float
    lookahead_penalty: float

    @classmethod
    def calculate(
        cls,
        role_score: float,
        debate_multiplier: float,
        cost_usd: float,
        lookahead_penalty: float
    ) -> "AlphaScore":
        """Calculate the Alpha Score from components."""
        import math
        numerator = role_score * debate_multiplier
        denominator = math.log(1 + cost_usd) * (1 + lookahead_penalty)

        # Avoid division by zero if cost is 0
        if denominator == 0:
            denominator = 0.001

        score = numerator / denominator

        return cls(
            score=score,
            role_score=role_score,
            debate_multiplier=debate_multiplier,
            cost_usd=cost_usd,
            lookahead_penalty=lookahead_penalty
        )


class EvaluationResult(BaseModel):
    """Complete evaluation result for a task."""
    evaluation_id: str
    task_id: str
    agent_id: str
    timestamp: datetime = Field(default_factory=_utc_now)

    # Agent's response
    agent_analysis: str = Field(default="", description="Purple agent's full analysis")
    agent_recommendation: str = Field(default="", description="Purple agent's final answer/recommendation")

    # Scores
    role_score: RoleScore
    debate_result: DebateResult
    cost_breakdown: CostBreakdown
    lookahead_penalty: LookAheadPenalty
    alpha_score: AlphaScore

    # Green agent evaluation feedback
    execution_feedback: str = Field(default="", description="Green agent's justification for execution score")
    macro_feedback: str = Field(default="", description="Green agent's feedback on macro analysis")

    # Tool usage details
    tool_calls: list[ToolCall] = Field(default_factory=list)
    code_executions: list[CodeExecution] = Field(default_factory=list)

    # Performance metrics
    total_execution_time_seconds: float = 0.0
    total_llm_calls: int = 0
    total_tokens: int = 0


# === A2A Protocol Models ===

class A2AMessageType(str, Enum):
    """Types of A2A protocol messages."""
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESPONSE = "task_response"
    CHALLENGE = "challenge"
    REBUTTAL = "rebuttal"
    EVALUATION_RESULT = "evaluation_result"


class A2AMessage(BaseModel):
    """Agent-to-Agent protocol message."""
    protocol_version: str = "1.0"
    message_type: A2AMessageType
    sender_id: str
    receiver_id: str
    timestamp: datetime = Field(default_factory=_utc_now)
    payload: dict[str, Any]

    @classmethod
    def task_assignment(
        cls,
        sender_id: str,
        receiver_id: str,
        task: Task
    ) -> "A2AMessage":
        """Create a task assignment message."""
        return cls(
            message_type=A2AMessageType.TASK_ASSIGNMENT,
            sender_id=sender_id,
            receiver_id=receiver_id,
            payload={
                "task_id": task.question_id,
                "question": task.question,
                "category": task.category.value,
                "simulation_date": task.simulation_date.isoformat(),
                "available_tools": task.available_tools,
                "deadline_seconds": task.deadline_seconds,
                "evaluation_criteria": "You will be scored on accuracy, efficiency, and robustness."
            }
        )

    @classmethod
    def challenge(
        cls,
        sender_id: str,
        receiver_id: str,
        task_id: str,
        counter_argument: str
    ) -> "A2AMessage":
        """Create an adversarial challenge message."""
        return cls(
            message_type=A2AMessageType.CHALLENGE,
            sender_id=sender_id,
            receiver_id=receiver_id,
            payload={
                "task_id": task_id,
                "challenge": counter_argument,
                "instructions": "Defend your thesis with additional evidence or revise your recommendation."
            }
        )
