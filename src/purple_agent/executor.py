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
    - Options trading and strategy construction
    - Greeks analysis and risk management
    """

    # Options trading task keywords
    OPTIONS_KEYWORDS = [
        "option", "options", "call", "put", "strike", "expiration", "expiry",
        "delta", "gamma", "theta", "vega", "greeks", "iron condor", "straddle",
        "strangle", "spread", "covered call", "protective put", "butterfly",
        "volatility", "iv", "implied volatility", "black-scholes", "premium",
    ]

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
            # Check if this is a crypto trading decision request
            trading_decision = self._try_parse_trading_decision(user_input)
            if trading_decision is not None:
                # Handle crypto trading decision
                response = await self._handle_trading_decision(trading_decision)

                # Publish completed status with response (no separate artifact needed)
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        task_id=task_id,
                        context_id=context_id,
                        status=TaskStatus(
                            state=TaskState.completed,
                            message=Message(
                                message_id=uuid4().hex,
                                role=Role.agent,
                                parts=[TextPart(text=response)],
                            ),
                        ),
                        final=True,
                    )
                )
                return

            # Parse the task and extract relevant information (LLM + keyword fallback)
            task_info = await self._parse_task(user_input)

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
                artifact_id=uuid4().hex,
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

    # Valid task types for classification
    VALID_TASK_TYPES = [
        "options_pricing",      # Black-Scholes, option valuation
        "options_greeks",       # Delta, gamma, theta, vega analysis
        "options_strategy",     # Iron condor, straddle, spreads
        "options_volatility",   # IV analysis, vol trading
        "options_risk",         # VaR, position sizing, stress tests
        "options_pnl",          # P&L attribution, decomposition
        "options_general",      # Other options questions
        "beat_or_miss",         # Earnings beat/miss analysis
        "sec_filing",           # 10-K, 10-Q analysis
        "ratio_calculation",    # Financial ratios (P/E, ROE, etc.)
        "recommendation",       # Buy/sell/hold recommendations
        "financial_metrics",    # Revenue, income, margins
        "event_logic",          # Event-driven reasoning (BizFinBench)
        "quantitative",         # Quantitative computation (BizFinBench)
        "general",              # General financial questions
    ]

    def _try_parse_trading_decision(self, user_input: str) -> dict | None:
        """
        Try to parse the input as a crypto trading decision request.

        Returns the parsed state dict if valid, None otherwise.
        """
        try:
            data = json.loads(user_input)
            if isinstance(data, dict) and data.get("type") == "trading_decision":
                return data.get("state", {})
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    async def _handle_trading_decision(self, state: dict) -> str:
        """
        Handle a crypto trading decision request.

        Uses LLM to analyze market state and make trading decisions.

        Args:
            state: Market state including ohlcv, indicators, account info

        Returns:
            JSON string with trading decision
        """
        # Extract market data
        symbol = state.get("symbol", "UNKNOWN")
        ohlcv = state.get("ohlcv", {})
        indicators = state.get("indicators", {})
        account = state.get("account", {})
        market_metrics = state.get("market_metrics", {})

        current_price = ohlcv.get("close", 0)
        balance = account.get("balance", 10000)
        equity = account.get("equity", balance)
        positions = account.get("positions", [])

        # Build prompt for LLM
        prompt = f"""You are a crypto trading AI. Analyze this market state and make a trading decision.

MARKET STATE:
Symbol: {symbol}
Current Price: {current_price}
OHLCV: Open={ohlcv.get('open')}, High={ohlcv.get('high')}, Low={ohlcv.get('low')}, Close={ohlcv.get('close')}, Volume={ohlcv.get('volume')}

INDICATORS:
- EMA 20: {indicators.get('ema_20')}
- EMA 50: {indicators.get('ema_50')}
- RSI: {indicators.get('rsi')}
- MACD: {indicators.get('macd')}
- ATR: {indicators.get('atr')}

MARKET METRICS:
- Funding Rate: {market_metrics.get('funding_rate')}
- Open Interest: {market_metrics.get('open_interest')}

ACCOUNT:
- Balance: ${balance:.2f}
- Equity: ${equity:.2f}
- Current Positions: {len(positions)}

RULES:
1. Respond ONLY with a valid JSON object
2. action must be one of: "BUY", "SELL", "HOLD", "CLOSE"
3. size is position size (0.0 to 1.0 of equity)
4. stop_loss and take_profit are price levels

Respond with ONLY this JSON format:
{{"action": "BUY|SELL|HOLD|CLOSE", "size": 0.1, "stop_loss": null, "take_profit": null, "reasoning": "brief reason"}}"""

        # Use LLM if available
        if self.llm_client is not None:
            try:
                if hasattr(self.llm_client, "chat"):
                    response = await asyncio.to_thread(
                        lambda: self.llm_client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                            max_tokens=200,
                        )
                    )
                    llm_response = response.choices[0].message.content.strip()
                elif hasattr(self.llm_client, "messages"):
                    response = await asyncio.to_thread(
                        lambda: self.llm_client.messages.create(
                            model=self.model,
                            max_tokens=200,
                            messages=[{"role": "user", "content": prompt}],
                        )
                    )
                    llm_response = response.content[0].text.strip()
                else:
                    llm_response = None

                if llm_response:
                    # Try to extract JSON from response
                    try:
                        # Handle markdown code blocks
                        if "```" in llm_response:
                            start = llm_response.find("{")
                            end = llm_response.rfind("}") + 1
                            if start != -1 and end > start:
                                llm_response = llm_response[start:end]
                        decision = json.loads(llm_response)
                        # Validate and normalize
                        return json.dumps({
                            "action": str(decision.get("action", "HOLD")).upper(),
                            "size": float(decision.get("size", 0.0)),
                            "stop_loss": decision.get("stop_loss"),
                            "take_profit": decision.get("take_profit"),
                            "reasoning": decision.get("reasoning", ""),
                        })
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass

            except Exception as e:
                # Log error but continue to fallback
                pass

        # Fallback: Simple rule-based decision
        rsi = indicators.get("rsi", 50)
        ema_20 = indicators.get("ema_20", current_price)
        ema_50 = indicators.get("ema_50", current_price)

        action = "HOLD"
        size = 0.0
        reasoning = "No clear signal"

        if rsi and ema_20 and ema_50:
            if rsi < 30 and current_price > ema_20:
                action = "BUY"
                size = 0.1
                reasoning = "RSI oversold with price above EMA20"
            elif rsi > 70 and current_price < ema_20:
                action = "SELL"
                size = 0.1
                reasoning = "RSI overbought with price below EMA20"
            elif ema_20 > ema_50 and current_price > ema_20:
                action = "BUY"
                size = 0.05
                reasoning = "Bullish EMA crossover"
            elif ema_20 < ema_50 and current_price < ema_20:
                action = "SELL"
                size = 0.05
                reasoning = "Bearish EMA crossover"

        atr = indicators.get("atr", current_price * 0.02)
        stop_loss = None
        take_profit = None

        if action == "BUY" and atr:
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 3)
        elif action == "SELL" and atr:
            stop_loss = current_price + (atr * 2)
            take_profit = current_price - (atr * 3)

        return json.dumps({
            "action": action,
            "size": size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reasoning": reasoning,
        })

    async def _extract_tickers_with_llm(self, user_input: str) -> list[str] | None:
        """
        Use LLM to extract stock ticker symbols from the question.

        Args:
            user_input: The user's question

        Returns:
            List of ticker symbols or None if extraction fails
        """
        if self.llm_client is None:
            return None

        extraction_prompt = f"""Extract stock ticker symbols from this financial question.

RULES:
- Only extract REAL stock ticker symbols (e.g., AAPL, MSFT, TSLA, NVDA, GOOG)
- Do NOT extract common words that happen to be uppercase (e.g., JSON, XML, API, CEO, CFO)
- Do NOT extract financial abbreviations (e.g., EPS, PE, ROE, CAGR, NPV)
- Do NOT extract currency codes (e.g., USD, EUR, GBP)
- If no real stock tickers are found, return "NONE"
- Return tickers as comma-separated list (e.g., "AAPL,MSFT,TSLA")

QUESTION: {user_input[:1000]}

TICKERS (or NONE):"""

        try:
            if hasattr(self.llm_client, "chat"):
                # OpenAI-style client
                response = await asyncio.to_thread(
                    lambda: self.llm_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": extraction_prompt}],
                        temperature=0,
                        max_tokens=50,
                    )
                )
                result = response.choices[0].message.content.strip().upper()
            elif hasattr(self.llm_client, "messages"):
                # Anthropic-style client
                response = await asyncio.to_thread(
                    lambda: self.llm_client.messages.create(
                        model=self.model,
                        max_tokens=50,
                        messages=[{"role": "user", "content": extraction_prompt}],
                    )
                )
                result = response.content[0].text.strip().upper()
            else:
                return None

            # Parse result
            if result == "NONE" or not result:
                return []

            # Clean and validate tickers
            tickers = [t.strip() for t in result.split(",") if t.strip()]
            # Filter to valid ticker format (1-5 uppercase letters)
            valid_tickers = [t for t in tickers if re.match(r'^[A-Z]{1,5}$', t)]
            return valid_tickers

        except Exception:
            return None

    async def _classify_task_with_llm(self, user_input: str) -> str | None:
        """
        Use LLM to classify the task type.

        Args:
            user_input: The user's question

        Returns:
            Classified task type or None if classification fails
        """
        if self.llm_client is None:
            return None

        classification_prompt = f"""Classify this financial question into exactly ONE of these task types:

TASK TYPES:
- options_pricing: Option valuation, Black-Scholes calculations, fair value
- options_greeks: Delta, gamma, theta, vega calculations, hedging with Greeks
- options_strategy: Building strategies like iron condor, straddle, spreads, butterfly
- options_volatility: Implied volatility analysis, IV rank, vol trading
- options_risk: VaR, position sizing, stress testing, risk management
- options_pnl: P&L attribution, profit/loss decomposition by Greeks
- options_general: Other options-related questions
- beat_or_miss: Earnings beat/miss analysis, actual vs expected
- sec_filing: SEC filing analysis (10-K, 10-Q, 8-K)
- ratio_calculation: Financial ratio calculations (P/E, ROE, ROA, debt ratios)
- recommendation: Investment recommendations (buy/sell/hold)
- financial_metrics: Revenue, income, profit margin analysis
- event_logic: Event-driven reasoning, cause-effect analysis
- quantitative: Numerical calculations, quantitative finance
- general: General financial questions

QUESTION: {user_input[:500]}

Respond with ONLY the task type (e.g., "options_pricing"). Nothing else."""

        try:
            if hasattr(self.llm_client, "chat"):
                # OpenAI-style client
                response = await asyncio.to_thread(
                    lambda: self.llm_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": classification_prompt}],
                        temperature=0,
                        max_tokens=20,
                    )
                )
                task_type = response.choices[0].message.content.strip().lower()
            elif hasattr(self.llm_client, "messages"):
                # Anthropic-style client
                response = await asyncio.to_thread(
                    lambda: self.llm_client.messages.create(
                        model=self.model,
                        max_tokens=20,
                        messages=[{"role": "user", "content": classification_prompt}],
                    )
                )
                task_type = response.content[0].text.strip().lower()
            else:
                return None

            # Validate the response
            if task_type in self.VALID_TASK_TYPES:
                return task_type
            # Try to extract valid type from response
            for valid_type in self.VALID_TASK_TYPES:
                if valid_type in task_type:
                    return valid_type
            return None

        except Exception:
            return None

    def _classify_task_with_keywords(self, user_input: str) -> str:
        """
        Fallback keyword-based task classification.

        Args:
            user_input: The user's question

        Returns:
            Classified task type
        """
        user_lower = user_input.lower()

        # Check for options trading tasks first (more specific)
        if any(word in user_lower for word in self.OPTIONS_KEYWORDS):
            if any(word in user_lower for word in ["iron condor", "straddle", "strangle", "spread", "butterfly", "construct", "build", "design"]):
                return "options_strategy"
            elif any(word in user_lower for word in ["delta", "gamma", "theta", "vega", "greeks", "hedge", "neutral"]):
                return "options_greeks"
            elif any(word in user_lower for word in ["volatility", "iv", "implied volatility", "vol", "iv rank", "iv percentile"]):
                return "options_volatility"
            elif any(word in user_lower for word in ["black-scholes", "theoretical", "fair value", "calculate the"]) and "price" in user_lower:
                return "options_pricing"
            elif any(word in user_lower for word in ["risk", "var", "stress", "drawdown", "position size", "max loss"]):
                return "options_risk"
            elif any(word in user_lower for word in ["p&l", "pnl", "attribution", "decompose"]):
                return "options_pnl"
            else:
                return "options_general"
        elif any(word in user_lower for word in ["beat", "miss", "earnings", "expectations", "surprise"]):
            return "beat_or_miss"
        elif any(word in user_lower for word in ["10-k", "10k", "10-q", "10q", "annual report", "sec filing", "8-k"]):
            return "sec_filing"
        elif any(word in user_lower for word in ["ratio", "p/e", "pe ratio", "roe", "roa", "debt", "current ratio"]):
            return "ratio_calculation"
        elif any(word in user_lower for word in ["recommend", "buy", "sell", "hold", "rating", "target price"]):
            return "recommendation"
        elif any(word in user_lower for word in ["revenue", "income", "profit", "margin", "growth"]):
            return "financial_metrics"
        elif any(word in user_lower for word in ["event", "impact", "affect", "consequence", "result in"]):
            return "event_logic"
        elif any(word in user_lower for word in ["calculate", "compute", "what is the", "how much"]):
            return "quantitative"

        return "general"

    async def _parse_task(self, user_input: str) -> dict[str, Any]:
        """
        Parse the user input to extract task information.
        Uses LLM for classification with keyword fallback.

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

        # Try LLM-based ticker extraction first (more accurate)
        llm_tickers = await self._extract_tickers_with_llm(user_input)
        if llm_tickers is not None:
            task_info["tickers"] = llm_tickers
            task_info["ticker_extraction_method"] = "llm"
        else:
            # Fallback to regex with common words filter
            tickers = re.findall(r'\b([A-Z]{1,5})\b', user_input)
            common_words = {
                "Q", "FY", "EPS", "PE", "ROE", "YOY", "QOQ", "CEO", "CFO", "SEC", "AI", "US", "GDP",
                "ATM", "OTM", "ITM", "DTE", "IV", "HV", "BS", "NPV", "IRR", "EBIT", "WACC", "DCF",
                "IPO", "ETF", "NAV", "AUM", "P", "E", "B", "S", "M", "K", "T", "USD", "EUR", "GBP",
                "JSON", "XML", "HTML", "CSV", "API", "HTTP", "URL", "SQL", "PHP", "CSS", "JS",
                "THE", "AND", "FOR", "NOT", "ARE", "BUT", "ALL", "CAN", "HER", "WAS", "ONE", "OUR",
                "OUT", "YOU", "HAD", "HAS", "HIS", "HOW", "ITS", "MAY", "NEW", "NOW", "OLD", "SEE",
                "WAY", "WHO", "DID", "GET", "LET", "PUT", "SAY", "SHE", "TOO", "USE", "YES", "NO",
                "IF", "IN", "IS", "IT", "OF", "ON", "OR", "TO", "UP", "SO", "BY", "AS", "AT", "AN",
                "DATA", "FILE", "FROM", "INTO", "NULL", "TRUE", "WITH", "YEAR", "DATE", "THIS",
            }
            task_info["tickers"] = [t for t in tickers if t not in common_words]
            task_info["ticker_extraction_method"] = "regex"

        # Try LLM classification first, fall back to keywords
        llm_task_type = await self._classify_task_with_llm(user_input)
        if llm_task_type:
            task_info["task_type"] = llm_task_type
            task_info["classification_method"] = "llm"
        else:
            task_info["task_type"] = self._classify_task_with_keywords(user_input)
            task_info["classification_method"] = "keywords"

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
        task_type = task_info.get("task_type", "general")

        for ticker in task_info.get("tickers", []):
            try:
                # Always get basic stock data
                ticker_data = await self.toolkit.get_comprehensive_analysis(ticker)
                data["tickers"][ticker] = ticker_data

                # For options tasks, gather options-specific data
                if task_type.startswith("options_"):
                    options_data = await self._gather_options_data(ticker, task_info)
                    data["tickers"][ticker]["options"] = options_data

            except Exception as e:
                data["tickers"][ticker] = {"error": str(e)}

        return data

    async def _gather_options_data(
        self,
        ticker: str,
        task_info: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Gather options-specific data for a ticker.

        Args:
            ticker: Stock ticker symbol
            task_info: Parsed task information

        Returns:
            Dictionary with options data
        """
        options_data: dict[str, Any] = {}

        try:
            # Get current stock quote for reference
            quote = await self.toolkit.get_quote(ticker)
            current_price = quote.get("current_price", 100.0)
            options_data["spot_price"] = current_price

            # Get volatility analysis
            vol_analysis = await self.toolkit.get_volatility_analysis(ticker)
            options_data["volatility"] = vol_analysis

            # Get available expirations
            expirations = await self.toolkit.get_option_expirations(ticker)
            options_data["expirations"] = expirations

            # Get options chain for nearest expiration
            chain = await self.toolkit.get_options_chain(
                ticker=ticker,
                expiration="nearest",
                min_strike=current_price * 0.85,
                max_strike=current_price * 1.15,
            )
            options_data["chain"] = chain

            # For strategy tasks, also get strike analysis
            task_type = task_info.get("task_type", "")
            if task_type == "options_strategy":
                # Calculate sample option prices at various strikes
                sample_strikes = [
                    current_price * 0.95,
                    current_price,
                    current_price * 1.05,
                ]
                sample_prices = []
                vol = vol_analysis.get("historical_volatility", 0.25)

                for strike in sample_strikes:
                    call_price = await self.toolkit.calculate_option_price(
                        spot_price=current_price,
                        strike_price=strike,
                        days_to_expiry=30,
                        volatility=vol,
                        option_type="call",
                    )
                    put_price = await self.toolkit.calculate_option_price(
                        spot_price=current_price,
                        strike_price=strike,
                        days_to_expiry=30,
                        volatility=vol,
                        option_type="put",
                    )
                    sample_prices.append({
                        "strike": strike,
                        "call": call_price,
                        "put": put_price,
                    })
                options_data["sample_prices"] = sample_prices

        except Exception as e:
            options_data["error"] = str(e)

        return options_data

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

            "options_strategy": """Focus on:
- Clear strategy identification (iron condor, spread, straddle, etc.)
- Strike selection rationale
- All four Greeks for each leg and net position
- Max profit, max loss, and breakeven points
- Probability of profit calculation
- Risk/reward analysis
- Entry and exit criteria""",

            "options_greeks": """Focus on:
- Accurate calculation of delta, gamma, theta, vega, rho
- Interpretation of each Greek for the position
- Hedging recommendations based on Greeks
- Portfolio-level Greek aggregation if multiple positions
- Sensitivity analysis""",

            "options_volatility": """Focus on:
- Historical volatility calculation
- Implied volatility analysis
- IV rank and IV percentile
- Comparison of IV to HV (premium/discount)
- Volatility trading opportunities
- Strategy recommendations based on volatility outlook""",

            "options_pricing": """Focus on:
- Black-Scholes model application
- Input parameters (spot, strike, time, vol, rate, dividend)
- Theoretical price calculation
- All Greeks values
- Intrinsic vs extrinsic value breakdown
- Put-call parity verification""",

            "options_risk": """Focus on:
- Position sizing methodology
- Value at Risk (VaR) calculation
- Maximum drawdown analysis
- Stress test scenarios
- Risk limits and stop-loss levels
- Portfolio hedging strategies""",

            "options_pnl": """Focus on:
- P&L attribution by Greek (delta, gamma, theta, vega)
- Realized vs unrealized P&L
- Impact of underlying price movement
- Impact of time decay
- Impact of volatility changes
- Performance metrics (Sharpe, Sortino)""",

            "options_general": """Focus on:
- Clear options analysis
- Relevant Greeks and risk metrics
- Strategy appropriateness for market conditions
- Risk management considerations
- Supporting data and calculations""",

            "event_logic": """Focus on:
- Clear cause-and-effect reasoning
- Logical chain of events and their financial impact
- Market implications of the event
- Historical precedents if applicable
- Short-term and long-term consequences
- Stakeholder impact analysis""",

            "quantitative": """Focus on:
- Precise numerical calculations
- Show all work and intermediate steps
- Use appropriate financial formulas
- Verify calculations with sanity checks
- Present results with appropriate precision
- Include units and context for all numbers""",
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

                # Options data (if present)
                if "options" in data:
                    opts = data["options"]
                    prompt_parts.append("\n  Options Data:")
                    if opts.get("spot_price"):
                        prompt_parts.append(f"    Spot Price: ${opts['spot_price']:.2f}")
                    if opts.get("volatility"):
                        vol = opts["volatility"]
                        if vol.get("historical_volatility"):
                            prompt_parts.append(f"    Historical Volatility: {vol['historical_volatility']*100:.1f}%")
                        if vol.get("iv_rank"):
                            prompt_parts.append(f"    IV Rank: {vol['iv_rank']:.1f}%")
                    if opts.get("expirations") and isinstance(opts["expirations"], dict):
                        exps = opts["expirations"].get("expirations", [])
                        if exps:
                            prompt_parts.append(f"    Available Expirations: {', '.join(exps[:5])}")
                    if opts.get("sample_prices"):
                        prompt_parts.append("    Sample Option Prices (30-day):")
                        for sp in opts["sample_prices"][:3]:
                            call_price = sp["call"].get("price", 0) if isinstance(sp["call"], dict) else 0
                            put_price = sp["put"].get("price", 0) if isinstance(sp["put"], dict) else 0
                            prompt_parts.append(f"      Strike ${sp['strike']:.0f}: Call ${call_price:.2f}, Put ${put_price:.2f}")

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

            # Options data
            if "options" in data:
                opts = data["options"]
                response_parts.append("**Options Analysis:**")
                if opts.get("spot_price"):
                    response_parts.append(f"- Spot Price: ${opts['spot_price']:.2f}")
                if opts.get("volatility"):
                    vol = opts["volatility"]
                    if vol.get("historical_volatility"):
                        response_parts.append(f"- Historical Volatility: {vol['historical_volatility']*100:.1f}%")
                    if vol.get("iv_rank"):
                        response_parts.append(f"- IV Rank: {vol['iv_rank']:.1f}%")
                    if vol.get("iv_percentile"):
                        response_parts.append(f"- IV Percentile: {vol['iv_percentile']:.1f}%")

                if opts.get("sample_prices"):
                    response_parts.append("\n**Sample Option Prices (30-day expiration):**")
                    for sp in opts["sample_prices"]:
                        strike = sp["strike"]
                        call_data = sp.get("call", {})
                        put_data = sp.get("put", {})

                        if isinstance(call_data, dict) and isinstance(put_data, dict):
                            response_parts.append(f"\nStrike ${strike:.0f}:")
                            response_parts.append(f"  Call: ${call_data.get('price', 0):.2f}")
                            response_parts.append(f"    Delta: {call_data.get('delta', 0):.3f}")
                            response_parts.append(f"    Gamma: {call_data.get('gamma', 0):.4f}")
                            response_parts.append(f"    Theta: {call_data.get('theta', 0):.3f}")
                            response_parts.append(f"    Vega: {call_data.get('vega', 0):.3f}")
                            response_parts.append(f"  Put: ${put_data.get('price', 0):.2f}")
                            response_parts.append(f"    Delta: {put_data.get('delta', 0):.3f}")
                            response_parts.append(f"    Gamma: {put_data.get('gamma', 0):.4f}")
                            response_parts.append(f"    Theta: {put_data.get('theta', 0):.3f}")
                            response_parts.append(f"    Vega: {put_data.get('vega', 0):.3f}")
                response_parts.append("")

        # Task-specific conclusion
        task_type = task_info.get("task_type", "general")
        if task_type == "beat_or_miss":
            response_parts.append("\n**Note:** To determine beat/miss status, compare reported figures against analyst consensus estimates.")
        elif task_type == "recommendation":
            response_parts.append("\n**Note:** This is a data summary. Investment recommendations require additional analysis of market conditions, risk factors, and investment objectives.")
        elif task_type.startswith("options_"):
            response_parts.append("\n**Options Trading Considerations:**")
            response_parts.append("- Greeks shown are calculated using Black-Scholes model")
            response_parts.append("- Actual market prices may vary due to bid/ask spreads")
            response_parts.append("- Consider position sizing based on portfolio risk limits")
            response_parts.append("- Always define max loss and exit criteria before entering trades")

        return "\n".join(response_parts)
