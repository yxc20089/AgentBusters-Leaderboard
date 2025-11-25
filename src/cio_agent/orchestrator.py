"""
A2A (Agent-to-Agent) Orchestrator for CIO-Agent.

Manages communication between the Green Agent (CIO-Agent/Evaluator)
and White/Purple Agents (test agents) using the A2A Protocol.
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Optional

import structlog
from pydantic import BaseModel, Field

from cio_agent.models import (
    Task,
    AgentResponse,
    DebateRebuttal,
    A2AMessage,
    A2AMessageType,
    ToolCall,
    CodeExecution,
    FinancialData,
)

logger = structlog.get_logger()


class AgentConfig(BaseModel):
    """Configuration for an agent connection."""
    agent_id: str
    endpoint_url: str
    timeout_seconds: int = 1800  # 30 minutes default
    model: str = "gpt-4o"


class A2AOrchestrator:
    """
    Orchestrates communication between CIO-Agent and test agents.

    Responsibilities:
    - Send task assignments to agents
    - Receive and parse agent responses
    - Send adversarial challenges
    - Collect rebuttals
    - Track message timing and metadata
    """

    def __init__(
        self,
        cio_agent_id: str = "cio-agent-green",
        message_handler: Optional[Callable] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            cio_agent_id: ID of the CIO-Agent (Green Agent)
            message_handler: Optional custom message handler
        """
        self.cio_agent_id = cio_agent_id
        self.message_handler = message_handler
        self.message_log: list[A2AMessage] = []
        self._pending_responses: dict[str, asyncio.Future] = {}

    def _generate_message_id(self) -> str:
        """Generate a unique message ID."""
        return f"msg_{uuid.uuid4().hex[:12]}"

    def _log_message(self, message: A2AMessage) -> None:
        """Log a message for audit trail."""
        self.message_log.append(message)
        logger.info(
            "a2a_message",
            type=message.message_type.value,
            sender=message.sender_id,
            receiver=message.receiver_id,
            payload_keys=list(message.payload.keys()),
        )

    async def send_task_assignment(
        self,
        agent_id: str,
        task: Task,
    ) -> A2AMessage:
        """
        Send a task assignment to an agent.

        Args:
            agent_id: Target agent ID
            task: Task to assign

        Returns:
            The sent A2AMessage
        """
        message = A2AMessage.task_assignment(
            sender_id=self.cio_agent_id,
            receiver_id=agent_id,
            task=task,
        )

        self._log_message(message)

        if self.message_handler:
            await self.message_handler(message)

        logger.info(
            "task_assigned",
            agent_id=agent_id,
            task_id=task.question_id,
            deadline_seconds=task.deadline_seconds,
        )

        return message

    async def receive_task_response(
        self,
        agent_id: str,
        timeout_seconds: int = 1800,
    ) -> Optional[AgentResponse]:
        """
        Wait for and receive a task response from an agent.

        Args:
            agent_id: Agent ID to receive from
            timeout_seconds: Maximum wait time

        Returns:
            AgentResponse or None if timeout
        """
        response_key = f"response_{agent_id}"
        future = asyncio.get_event_loop().create_future()
        self._pending_responses[response_key] = future

        try:
            response = await asyncio.wait_for(future, timeout=timeout_seconds)
            return response
        except asyncio.TimeoutError:
            logger.warning("response_timeout", agent_id=agent_id, timeout=timeout_seconds)
            return None
        finally:
            self._pending_responses.pop(response_key, None)

    def deliver_response(
        self,
        agent_id: str,
        response: AgentResponse,
    ) -> None:
        """
        Deliver a response to a waiting receiver.

        Called by message handler when response is received.

        Args:
            agent_id: Agent ID that sent the response
            response: The response to deliver
        """
        response_key = f"response_{agent_id}"
        if response_key in self._pending_responses:
            future = self._pending_responses[response_key]
            if not future.done():
                future.set_result(response)

    async def send_challenge(
        self,
        agent_id: str,
        task_id: str,
        counter_argument: str,
    ) -> A2AMessage:
        """
        Send an adversarial challenge to an agent.

        Args:
            agent_id: Target agent ID
            task_id: ID of the task being challenged
            counter_argument: The counter-argument/challenge

        Returns:
            The sent A2AMessage
        """
        message = A2AMessage.challenge(
            sender_id=self.cio_agent_id,
            receiver_id=agent_id,
            task_id=task_id,
            counter_argument=counter_argument,
        )

        self._log_message(message)

        if self.message_handler:
            await self.message_handler(message)

        logger.info(
            "challenge_sent",
            agent_id=agent_id,
            task_id=task_id,
        )

        return message

    async def receive_rebuttal(
        self,
        agent_id: str,
        timeout_seconds: int = 600,  # 10 minutes for rebuttal
    ) -> Optional[DebateRebuttal]:
        """
        Wait for and receive a rebuttal from an agent.

        Args:
            agent_id: Agent ID to receive from
            timeout_seconds: Maximum wait time

        Returns:
            DebateRebuttal or None if timeout
        """
        rebuttal_key = f"rebuttal_{agent_id}"
        future = asyncio.get_event_loop().create_future()
        self._pending_responses[rebuttal_key] = future

        try:
            rebuttal = await asyncio.wait_for(future, timeout=timeout_seconds)
            return rebuttal
        except asyncio.TimeoutError:
            logger.warning("rebuttal_timeout", agent_id=agent_id, timeout=timeout_seconds)
            return None
        finally:
            self._pending_responses.pop(rebuttal_key, None)

    def deliver_rebuttal(
        self,
        agent_id: str,
        rebuttal: DebateRebuttal,
    ) -> None:
        """
        Deliver a rebuttal to a waiting receiver.

        Args:
            agent_id: Agent ID that sent the rebuttal
            rebuttal: The rebuttal to deliver
        """
        rebuttal_key = f"rebuttal_{agent_id}"
        if rebuttal_key in self._pending_responses:
            future = self._pending_responses[rebuttal_key]
            if not future.done():
                future.set_result(rebuttal)

    def get_message_history(
        self,
        agent_id: Optional[str] = None,
        message_type: Optional[A2AMessageType] = None,
    ) -> list[A2AMessage]:
        """
        Get filtered message history.

        Args:
            agent_id: Optional filter by agent ID
            message_type: Optional filter by message type

        Returns:
            Filtered list of messages
        """
        messages = self.message_log

        if agent_id:
            messages = [
                m for m in messages
                if m.sender_id == agent_id or m.receiver_id == agent_id
            ]

        if message_type:
            messages = [m for m in messages if m.message_type == message_type]

        return messages


class MockAgentClient:
    """
    Mock agent client for testing and local evaluation.

    Simulates responses from test agents without requiring
    actual agent deployments.
    """

    def __init__(
        self,
        agent_id: str,
        model: str = "gpt-4o",
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize mock agent.

        Args:
            agent_id: Agent identifier
            model: Model to simulate
            llm_client: Optional LLM client for generating responses
        """
        self.agent_id = agent_id
        self.model = model
        self.llm_client = llm_client

    async def process_task(self, task: Task) -> AgentResponse:
        """
        Process a task and generate a response.

        Args:
            task: Task to process

        Returns:
            Generated AgentResponse
        """
        start_time = time.time()

        if self.llm_client:
            # Use LLM to generate response
            response = await self._llm_process_task(task)
        else:
            # Use heuristic response
            response = self._heuristic_response(task)

        execution_time = time.time() - start_time

        response.agent_id = self.agent_id
        response.task_id = task.question_id
        response.execution_time_seconds = execution_time

        return response

    async def _llm_process_task(self, task: Task) -> AgentResponse:
        """Use LLM to generate a task response."""
        prompt = f"""
You are a finance analyst answering a benchmark question.

TASK: {task.question}
CATEGORY: {task.category.value}
TICKER: {task.ticker}
FISCAL YEAR: {task.fiscal_year}
SIMULATION DATE: {task.simulation_date.strftime('%Y-%m-%d')}

Provide a comprehensive analysis and answer. Include:
1. Your methodology
2. Key data points used
3. Any calculations performed
4. Your final answer/recommendation

Format your response as:
ANALYSIS: [Your detailed analysis]
RECOMMENDATION: [Your final answer/recommendation]
"""

        result = await self.llm_client.generate(prompt)

        # Parse response
        analysis = result
        recommendation = ""

        if "RECOMMENDATION:" in result:
            parts = result.split("RECOMMENDATION:")
            analysis = parts[0].replace("ANALYSIS:", "").strip()
            recommendation = parts[1].strip()
        elif "ANALYSIS:" in result:
            analysis = result.replace("ANALYSIS:", "").strip()

        return AgentResponse(
            agent_id=self.agent_id,
            task_id=task.question_id,
            analysis=analysis,
            recommendation=recommendation or analysis[:500],
            extracted_financials=FinancialData(),
            tool_calls=[],
            code_executions=[],
        )

    def _heuristic_response(self, task: Task) -> AgentResponse:
        """Generate a heuristic response for testing."""
        analysis = f"""
Analysis for {task.ticker} ({task.fiscal_year}):

Based on the available data for {task.ticker}, I analyzed the following:

1. Retrieved relevant financial data from SEC filings
2. Examined key metrics including revenue, margins, and cash flow
3. Considered macroeconomic factors affecting the sector

Key findings:
- Company operates in a competitive market
- Financial metrics appear stable year-over-year
- Management guidance indicates cautious optimism

The analysis was conducted using a combination of fundamental analysis
and comparison with industry benchmarks.
"""

        recommendation = f"""
Based on my analysis of {task.ticker}'s FY {task.fiscal_year} performance:

The company shows solid fundamentals with reasonable valuation metrics.
Given the current market conditions and company-specific factors,
my recommendation is HOLD with a neutral outlook.

Key metrics supporting this view:
- Stable revenue trajectory
- Consistent margin performance
- Manageable debt levels

Risk factors to monitor include competitive dynamics and macro conditions.
"""

        # Simulate tool calls
        tool_calls = [
            ToolCall(
                tool_name="sec-edgar-mcp:get_filing",
                params={"ticker": task.ticker, "form_type": "10-K", "fiscal_year": task.fiscal_year},
                timestamp=datetime.utcnow(),
                response_tokens=5000,
                duration_ms=500,
            ),
            ToolCall(
                tool_name="yahoo-finance-mcp:get_statistics",
                params={"ticker": task.ticker},
                timestamp=datetime.utcnow(),
                response_tokens=1000,
                duration_ms=200,
            ),
        ]

        # Simulate code execution for numerical tasks
        code_executions = []
        if task.requires_code_execution:
            code_executions.append(
                CodeExecution(
                    code=f"""
import pandas as pd
# Calculate financial metrics for {task.ticker}
revenue = 100000000000  # Example value
net_income = 20000000000
margin = (net_income / revenue) * 100
print(f"Net Margin: {{margin:.1f}}%")
""",
                    output="Net Margin: 20.0%",
                    execution_time_ms=150,
                    libraries_used=["pandas"],
                )
            )

        return AgentResponse(
            agent_id=self.agent_id,
            task_id=task.question_id,
            analysis=analysis,
            recommendation=recommendation,
            extracted_financials=FinancialData(
                revenue=100_000_000_000,
                net_income=20_000_000_000,
                net_margin=0.20,
            ),
            tool_calls=tool_calls,
            code_executions=code_executions,
        )

    async def process_challenge(
        self,
        task_id: str,
        challenge: str,
        original_response: AgentResponse,
    ) -> DebateRebuttal:
        """
        Process a challenge and generate a rebuttal.

        Args:
            task_id: ID of the task being challenged
            challenge: The counter-argument
            original_response: The original response being defended

        Returns:
            DebateRebuttal
        """
        if self.llm_client:
            rebuttal = await self._llm_process_challenge(
                task_id, challenge, original_response
            )
        else:
            rebuttal = self._heuristic_rebuttal(
                task_id, challenge, original_response
            )

        return rebuttal

    async def _llm_process_challenge(
        self,
        task_id: str,
        challenge: str,
        original_response: AgentResponse,
    ) -> DebateRebuttal:
        """Use LLM to generate a rebuttal."""
        prompt = f"""
You are defending your investment thesis against a challenge from a Risk Manager.

YOUR ORIGINAL THESIS:
{original_response.recommendation}

CHALLENGE FROM RISK MANAGER:
{challenge}

Defend your thesis by:
1. Addressing the specific concerns raised
2. Providing additional evidence from your analysis
3. Acknowledging valid risks while explaining why your thesis still holds
4. Being specific with data and rationale

Provide a strong, data-driven defense of your position.
"""

        defense = await self.llm_client.generate(prompt)

        return DebateRebuttal(
            agent_id=self.agent_id,
            task_id=task_id,
            defense=defense,
            new_evidence_cited=[],
            tool_calls=[],
        )

    def _heuristic_rebuttal(
        self,
        task_id: str,
        challenge: str,
        original_response: AgentResponse,
    ) -> DebateRebuttal:
        """Generate a heuristic rebuttal."""
        defense = f"""
Thank you for the challenging perspective. Let me address your concerns:

1. Regarding the competitive dynamics mentioned, while competition is indeed
   intensifying, the company maintains structural advantages including
   established market position and switching costs.

2. On valuation concerns, the current multiple reflects the company's
   growth trajectory and quality of earnings. Comparing to historical
   ranges suggests reasonable positioning.

3. For macro headwinds, management has demonstrated adaptability in
   previous cycles, and the balance sheet provides flexibility.

Additional supporting points:
- Recent guidance from the company suggests stable outlook
- Industry trends remain supportive despite near-term volatility
- Cash generation continues to exceed capital requirements

I maintain my position with confidence while acknowledging the risks
you've highlighted. These factors were considered in my original analysis.
"""

        return DebateRebuttal(
            agent_id=self.agent_id,
            task_id=task_id,
            defense=defense,
            new_evidence_cited=["guidance", "industry trends", "cash generation"],
            tool_calls=[],
        )
