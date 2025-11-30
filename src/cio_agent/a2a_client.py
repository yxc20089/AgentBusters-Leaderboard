"""
HTTP client to talk to the Purple Agent server.

Uses the Purple Agent's `/analyze` convenience endpoint to obtain an analysis
and wraps it into the Green-side AgentResponse / DebateRebuttal structures.
"""

from __future__ import annotations

import httpx
from datetime import datetime
from typing import Optional

from cio_agent.models import AgentResponse, DebateRebuttal, FinancialData, Task


class PurpleHTTPAgentClient:
    """Minimal HTTP client for the Purple Agent `/analyze` endpoint."""

    def __init__(
        self,
        base_url: str,
        agent_id: str = "purple-agent",
        model: str = "purple-http",
        timeout_seconds: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.agent_id = agent_id
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def process_task(self, task: Task) -> AgentResponse:
        """Send a task to purple `/analyze` and wrap the response."""
        url = f"{self.base_url}/analyze"
        payload = {
            "question": task.question,
            "ticker": task.ticker,
            "simulation_date": task.simulation_date.isoformat(),
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        analysis = data.get("analysis") or "No analysis returned."
        recommendation = analysis.splitlines()[0][:200] if analysis else "No recommendation."

        return AgentResponse(
            agent_id=self.agent_id,
            task_id=task.question_id,
            analysis=analysis,
            recommendation=recommendation,
            extracted_financials=FinancialData(),
            tool_calls=[],
            code_executions=[],
            timestamp=datetime.now(),
            execution_time_seconds=0.0,
        )

    async def process_challenge(
        self,
        task_id: str,
        challenge: str,
        original_response: Optional[AgentResponse] = None,
    ) -> DebateRebuttal:
        """Return a simple rebuttal acknowledging the challenge."""
        defense_lines = [
            "Acknowledging the challenge:",
            challenge,
            "",
            "I maintain the core thesis and will monitor the highlighted risks.",
        ]
        if original_response:
            defense_lines.append(f"Original recommendation: {original_response.recommendation}")

        return DebateRebuttal(
            agent_id=self.agent_id,
            task_id=task_id,
            defense="\n".join(defense_lines),
            new_evidence_cited=[],
            tool_calls=[],
        )
