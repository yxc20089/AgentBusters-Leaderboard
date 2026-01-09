"""
Agent Executor for Purple Agent

Implements the A2A AgentExecutor interface to handle incoming
finance analysis tasks from Green Agents.

Uses in-process MCP servers for controlled competition environment.
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Any
from uuid import uuid4

from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    Artifact,
    TextPart,
    Message,
    Role,
)

# Import MCP toolkit for in-process MCP servers
from purple_agent.mcp_toolkit import MCPToolkit


class FinanceAgentExecutor(AgentExecutor):
    """
    Finance Agent Executor implementing the A2A protocol.

    Handles finance analysis tasks including:
    - Earnings beat/miss analysis
    - SEC filing analysis
    - Financial ratio calculation
    - Investment recommendations
    """

    def __init__(
        self,
        llm_client: Any = None,
        model: str = "gpt-4o",
        simulation_date: datetime | None = None,
    ):
        """
        Initialize the Finance Agent Executor.

        Args:
            llm_client: LLM client for generating analysis (OpenAI or Anthropic)
            model: Model identifier for LLM calls
            simulation_date: Optional date for temporal locking
        """
        self.llm_client = llm_client
        self.model = model
        self.simulation_date = simulation_date

        # Always use in-process MCP servers for controlled environment
        self.toolkit = MCPToolkit(simulation_date=simulation_date)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Execute a finance analysis task.

        Args:
            context: Request context containing the task details
            event_queue: Queue for publishing task events
        """
        task_id = context.message.task_id if context.message else "unknown"
        context_id = context.message.context_id if context.message else "unknown"
        user_input = context.get_user_input()

        # Publish working status
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.working,
                    message=Message(
                        message_id=uuid4().hex,
                        role=Role.agent,
                        parts=[TextPart(text="Analyzing financial data...")],
                    ),
                ),
                final=False,
            )
        )

        try:
            # Parse the task and extract relevant information
            task_info = self._parse_task(user_input)

            # Gather financial data
            financial_data = await self._gather_data(task_info)

            # Generate analysis using LLM
            analysis = await self._generate_analysis(
                user_input=user_input,
                task_info=task_info,
                financial_data=financial_data,
            )

            # Create response artifact
            artifact = Artifact(
                name="financial_analysis",
                parts=[TextPart(text=analysis)],
            )

            # Publish artifact
            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    artifact=artifact,
                )
            )

            # Publish completed status
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.completed,
                        message=Message(
                            message_id=uuid4().hex,
                            role=Role.agent,
                            parts=[TextPart(text=analysis)],
                        ),
                    ),
                    final=True,
                )
            )

        except Exception as e:
            # Publish failed status
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=Message(
                            message_id=uuid4().hex,
                            role=Role.agent,
                            parts=[TextPart(text=f"Analysis failed: {str(e)}")],
                        ),
                    ),
                    final=True,
                )
            )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Cancel an ongoing task.

        Args:
            context: Request context with task to cancel
            event_queue: Queue for publishing cancellation event
        """
        task_id = context.message.task_id if context.message else "unknown"

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context.message.context_id if context.message else "unknown",
                status=TaskStatus(
                    state=TaskState.canceled,
                    message=Message(
                        message_id=uuid4().hex,
                        role=Role.agent,
                        parts=[TextPart(text="Task cancelled by request")],
                    ),
                ),
                final=True,
            )
        )

    def _parse_task(self, user_input: str) -> dict[str, Any]:
        """
        Parse the user input to extract task information.

        Args:
            user_input: Raw user input/question

        Returns:
            Dictionary with parsed task info
        """
        task_info: dict[str, Any] = {
            "raw_input": user_input,
            "tickers": [],
            "task_type": "general",
            "fiscal_year": None,
            "quarter": None,
        }

        # Extract ticker symbols (uppercase letters, 1-5 chars)
        tickers = re.findall(r'\b([A-Z]{1,5})\b', user_input)
        # Filter common words that might match
        common_words = {"Q", "FY", "EPS", "PE", "ROE", "YOY", "QOQ", "CEO", "CFO", "SEC", "AI", "US", "GDP"}
        task_info["tickers"] = [t for t in tickers if t not in common_words]

        # Detect task type
        user_lower = user_input.lower()
        if any(word in user_lower for word in ["beat", "miss", "earnings", "expectations"]):
            task_info["task_type"] = "beat_or_miss"
        elif any(word in user_lower for word in ["10-k", "10k", "annual report", "sec filing"]):
            task_info["task_type"] = "sec_filing"
        elif any(word in user_lower for word in ["ratio", "p/e", "pe ratio", "roe", "roa", "debt"]):
            task_info["task_type"] = "ratio_calculation"
        elif any(word in user_lower for word in ["recommend", "buy", "sell", "hold", "rating"]):
            task_info["task_type"] = "recommendation"
        elif any(word in user_lower for word in ["revenue", "income", "profit", "margin"]):
            task_info["task_type"] = "financial_metrics"

        # Extract fiscal year
        fy_match = re.search(r'FY\s*(\d{4})', user_input, re.IGNORECASE)
        if fy_match:
            task_info["fiscal_year"] = int(fy_match.group(1))
        else:
            year_match = re.search(r'20\d{2}', user_input)
            if year_match:
                task_info["fiscal_year"] = int(year_match.group())

        # Extract quarter
        q_match = re.search(r'Q(\d)', user_input, re.IGNORECASE)
        if q_match:
            task_info["quarter"] = int(q_match.group(1))

        return task_info

    async def _gather_data(self, task_info: dict[str, Any]) -> dict[str, Any]:
        """
        Gather financial data based on task requirements.

        Args:
            task_info: Parsed task information

        Returns:
            Dictionary with gathered financial data
        """
        data: dict[str, Any] = {"tickers": {}}

        for ticker in task_info.get("tickers", []):
            try:
                ticker_data = await self.toolkit.get_comprehensive_analysis(ticker)
                data["tickers"][ticker] = ticker_data
            except Exception as e:
                data["tickers"][ticker] = {"error": str(e)}

        return data

    async def _generate_analysis(
        self,
        user_input: str,
        task_info: dict[str, Any],
        financial_data: dict[str, Any],
    ) -> str:
        """
        Generate financial analysis using LLM.

        Args:
            user_input: Original user question
            task_info: Parsed task information
            financial_data: Gathered financial data

        Returns:
            Analysis text response
        """
        # Build the prompt
        system_prompt = self._get_system_prompt(task_info["task_type"])
        user_prompt = self._build_user_prompt(user_input, task_info, financial_data)

        # If no LLM client, return a structured response based on data
        if self.llm_client is None:
            return self._generate_fallback_response(task_info, financial_data)

        # Call LLM
        try:
            if hasattr(self.llm_client, "chat"):
                # OpenAI-style client
                response = await asyncio.to_thread(
                    lambda: self.llm_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.3,
                    )
                )
                return response.choices[0].message.content

            elif hasattr(self.llm_client, "messages"):
                # Anthropic-style client
                response = await asyncio.to_thread(
                    lambda: self.llm_client.messages.create(
                        model=self.model,
                        max_tokens=2000,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                    )
                )
                return response.content[0].text

        except Exception as e:
            return f"LLM analysis error: {str(e)}\n\n{self._generate_fallback_response(task_info, financial_data)}"

        return self._generate_fallback_response(task_info, financial_data)

    def _get_system_prompt(self, task_type: str) -> str:
        """Get system prompt based on task type."""
        base_prompt = """You are an expert financial analyst providing detailed, accurate analysis.
Your analysis should be:
- Data-driven and factual
- Include specific numbers and metrics
- Provide clear conclusions
- Be concise but comprehensive

"""
        type_specific = {
            "beat_or_miss": """Focus on:
- Clear beat/miss determination
- Actual vs expected figures (revenue, EPS)
- Key drivers of performance
- Forward guidance analysis""",

            "sec_filing": """Focus on:
- Key information from the filing
- Risk factors and disclosures
- Management discussion highlights
- Material changes from prior periods""",

            "ratio_calculation": """Focus on:
- Accurate ratio calculations
- Comparison to industry benchmarks
- Trend analysis
- What the ratios indicate about financial health""",

            "recommendation": """Focus on:
- Clear buy/hold/sell recommendation
- Supporting thesis with data
- Key risks to consider
- Target price rationale if applicable""",

            "financial_metrics": """Focus on:
- Accurate revenue, income, and margin figures
- Year-over-year and quarter-over-quarter changes
- Segment breakdown if available
- Key drivers of changes""",
        }

        return base_prompt + type_specific.get(task_type, "Provide comprehensive financial analysis.")

    def _build_user_prompt(
        self,
        user_input: str,
        task_info: dict[str, Any],
        financial_data: dict[str, Any],
    ) -> str:
        """Build the user prompt with context."""
        prompt_parts = [
            f"Question: {user_input}",
            "",
            "Financial Data:",
        ]

        for ticker, data in financial_data.get("tickers", {}).items():
            prompt_parts.append(f"\n{ticker}:")
            if "error" in data:
                prompt_parts.append(f"  Error: {data['error']}")
            else:
                # Quote (from Yahoo Finance MCP)
                if "quote" in data and "error" not in data["quote"]:
                    q = data["quote"]
                    if q.get('company_name'):
                        prompt_parts.append(f"  Name: {q.get('company_name')}")
                    if q.get('current_price') is not None:
                        prompt_parts.append(f"  Price: ${q.get('current_price')}")
                    if q.get('market_cap'):
                        prompt_parts.append(f"  Market Cap: ${q.get('market_cap'):,}")
                    if q.get('pe_ratio') is not None:
                        prompt_parts.append(f"  P/E Ratio: {q.get('pe_ratio')}")

                # Key Statistics (Yahoo Finance)
                if "statistics" in data and "error" not in data["statistics"]:
                    stats = data["statistics"]
                    if stats.get('beta') is not None:
                        prompt_parts.append(f"  Beta: {stats.get('beta')}")
                    if stats.get('profit_margin') is not None:
                        prompt_parts.append(f"  Profit Margin: {stats.get('profit_margin')*100:.1f}%")

                # Company info (SEC EDGAR)
                if "company_info" in data and "error" not in data["company_info"]:
                    company = data["company_info"]
                    if company.get('name'):
                        prompt_parts.append(f"  Company: {company.get('name')}")
                    if company.get('cik'):
                        prompt_parts.append(f"  CIK: {company.get('cik')}")

                # Recent filing (SEC EDGAR)
                if "recent_filing" in data and "error" not in data["recent_filing"]:
                    filing = data["recent_filing"]
                    if filing.get('form_type'):
                        prompt_parts.append(f"  Recent Filing: {filing.get('form_type')} on {filing.get('filing_date', 'N/A')}")

        prompt_parts.extend([
            "",
            "Please provide a detailed analysis addressing the question above.",
        ])

        return "\n".join(prompt_parts)

    def _generate_fallback_response(
        self,
        task_info: dict[str, Any],
        financial_data: dict[str, Any],
    ) -> str:
        """Generate a response without LLM based on available data."""
        response_parts = ["## Financial Analysis\n"]

        for ticker, data in financial_data.get("tickers", {}).items():
            response_parts.append(f"### {ticker}\n")

            if "error" in data:
                response_parts.append(f"Error retrieving data: {data['error']}\n")
                continue

            displayed_name = None

            # Quote summary
            if "quote" in data and "error" not in data["quote"]:
                q = data["quote"]
                displayed_name = q.get('company_name') or ticker
                response_parts.append(f"**{displayed_name}**\n")
                if q.get('current_price') is not None:
                    response_parts.append(f"- Current Price: ${q['current_price']}")
                if q.get('market_cap'):
                    response_parts.append(f"- Market Cap: ${q['market_cap']:,.0f}")
                if q.get('pe_ratio') is not None:
                    response_parts.append(f"- P/E Ratio: {q['pe_ratio']}")
                response_parts.append("")

            # Fallback company info (for tests or minimal data)
            if "stock_info" in data:
                stock = data["stock_info"]
                name = stock.get("name") or displayed_name or ticker
                if displayed_name is None:
                    response_parts.append(f"**{name}**\n")
                    displayed_name = name
                if stock.get("sector"):
                    response_parts.append(f"- Sector: {stock['sector']}")
                if stock.get("price") is not None and ("quote" not in data or data.get("quote", {}).get("current_price") is None):
                    response_parts.append(f"- Current Price: ${stock['price']}")
                response_parts.append("")

            # Financials summary when available
            if "financials" in data:
                fin = data["financials"]
                response_parts.append("**Financials:**")
                if fin.get("revenue") is not None:
                    response_parts.append(f"- Revenue: {fin['revenue']:,.0f}")
                if fin.get("net_income") is not None:
                    response_parts.append(f"- Net Income: {fin['net_income']:,.0f}")
                response_parts.append("")

            # Key statistics
            if "statistics" in data and "error" not in data["statistics"]:
                stats = data["statistics"]
                response_parts.append("**Key Statistics:**")
                if stats.get('beta') is not None:
                    response_parts.append(f"- Beta: {stats['beta']}")
                if stats.get('profit_margin') is not None:
                    response_parts.append(f"- Profit Margin: {stats['profit_margin']*100:.1f}%")
                response_parts.append("")

            # Recent filing
            if "recent_filing" in data and "error" not in data["recent_filing"]:
                filing = data["recent_filing"]
                response_parts.append("**Recent SEC Filing:**")
                if filing.get('form_type'):
                    response_parts.append(f"- Form: {filing['form_type']}")
                if filing.get('filing_date'):
                    response_parts.append(f"- Date: {filing['filing_date']}")
                if filing.get('accession_number'):
                    response_parts.append(f"- Accession: {filing['accession_number']}")
                response_parts.append("")

        # Task-specific conclusion
        task_type = task_info.get("task_type", "general")
        if task_type == "beat_or_miss":
            response_parts.append("\n**Note:** To determine beat/miss status, compare reported figures against analyst consensus estimates.")
        elif task_type == "recommendation":
            response_parts.append("\n**Note:** This is a data summary. Investment recommendations require additional analysis of market conditions, risk factors, and investment objectives.")

        return "\n".join(response_parts)
