"""
Base MCP client with common functionality.
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

import httpx
import structlog
from pydantic import BaseModel, Field

from cio_agent.models import ToolCall

logger = structlog.get_logger()


class MCPConfig(BaseModel):
    """Configuration for MCP server connection."""
    base_url: str
    timeout_seconds: int = 30
    max_retries: int = 3
    token_cost_per_1k: float = 0.001  # $0.001 per 1K tokens


class BaseMCPClient(ABC):
    """
    Base class for MCP server clients with common functionality.

    Features:
    - Request logging and timing
    - Token counting
    - Cost tracking
    - Error handling with retries
    """

    def __init__(self, config: MCPConfig, simulation_date: Optional[datetime] = None):
        self.config = config
        self.simulation_date = simulation_date
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout_seconds
        )
        self.call_log: list[ToolCall] = []
        self.total_tokens: int = 0
        self.total_cost: float = 0.0

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text (rough approximation)."""
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4

    def _log_call(
        self,
        tool_name: str,
        params: dict[str, Any],
        response_tokens: int,
        duration_ms: int,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> ToolCall:
        """Log a tool call and update metrics."""
        call = ToolCall(
            tool_name=tool_name,
            params=params,
            timestamp=datetime.utcnow(),
            response_tokens=response_tokens,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message
        )
        self.call_log.append(call)
        self.total_tokens += response_tokens
        self.total_cost += (response_tokens / 1000) * self.config.token_cost_per_1k

        logger.info(
            "mcp_tool_call",
            tool=tool_name,
            params=params,
            tokens=response_tokens,
            duration_ms=duration_ms,
            success=success
        )

        return call

    async def _request(
        self,
        method: str,
        endpoint: str,
        tool_name: str,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Make an HTTP request with logging and error handling."""
        start_time = time.time()
        error_message = None
        success = True
        response_text = ""

        try:
            if method.upper() == "GET":
                response = await self._client.get(endpoint, params=params)
            elif method.upper() == "POST":
                response = await self._client.post(endpoint, json=json_body)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            response_text = response.text
            return response.json()

        except httpx.HTTPStatusError as e:
            success = False
            error_message = f"HTTP {e.response.status_code}: {e.response.text}"
            raise
        except httpx.RequestError as e:
            success = False
            error_message = str(e)
            raise
        finally:
            duration_ms = int((time.time() - start_time) * 1000)
            response_tokens = self._estimate_tokens(response_text)
            self._log_call(
                tool_name=tool_name,
                params=params or json_body or {},
                response_tokens=response_tokens,
                duration_ms=duration_ms,
                success=success,
                error_message=error_message
            )

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics for this client."""
        return {
            "total_calls": len(self.call_log),
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
            "successful_calls": sum(1 for c in self.call_log if c.success),
            "failed_calls": sum(1 for c in self.call_log if not c.success),
        }

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the MCP server is healthy."""
        pass
