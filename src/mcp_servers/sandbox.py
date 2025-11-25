"""
Sandbox MCP Server client for isolated Python code execution.

This client wraps the mcp-sandbox server with:
- Pre-loaded financial libraries
- Code trace analysis
- Mandatory code execution validation for numerical tasks
"""

import re
from datetime import datetime
from typing import Any, Optional

import structlog
from pydantic import BaseModel, Field

from mcp_servers.base import BaseMCPClient, MCPConfig
from cio_agent.models import CodeExecution

logger = structlog.get_logger()


class ExecutionResult(BaseModel):
    """Result of code execution in the sandbox."""
    execution_id: str
    code: str
    stdout: str = ""
    stderr: str = ""
    return_value: Optional[Any] = None
    execution_time_ms: int = 0
    success: bool = True
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class CodeTraceAnalysis(BaseModel):
    """Analysis of code execution trace."""
    libraries_used: list[str] = Field(default_factory=list)
    calculation_steps: list[str] = Field(default_factory=list)
    intermediate_values: dict[str, Any] = Field(default_factory=dict)
    final_result: Optional[Any] = None
    methodology_score: float = Field(default=0.0, ge=0, le=100)
    uses_vectorization: bool = False
    uses_proper_types: bool = False
    has_error_handling: bool = False


class QuantSandboxClient(BaseMCPClient):
    """
    Python sandbox MCP client for financial calculations.

    Features:
    - Pre-loaded heavy financial libraries (numpy, pandas, scipy, etc.)
    - Code trace analysis for methodology validation
    - Mandatory code execution validation for numerical tasks
    """

    # Pre-loaded libraries in the sandbox
    PRELOADED_LIBS = [
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "yfinance",
        "quantlib",
    ]

    # Recognized financial calculation patterns
    FINANCIAL_PATTERNS = {
        "dcf": r"(discounted|cash\s*flow|npv|irr|wacc)",
        "valuation": r"(pe_ratio|price.*earning|market_cap|ev.*ebitda)",
        "margin": r"(gross_margin|operating_margin|net_margin|profit_margin)",
        "growth": r"(yoy|year.*over.*year|cagr|growth_rate)",
        "ratio": r"(current_ratio|debt.*equity|roe|roa|roce)",
    }

    def __init__(
        self,
        config: MCPConfig,
        simulation_date: Optional[datetime] = None,
        execution_timeout_seconds: int = 30
    ):
        super().__init__(config, simulation_date)
        self.execution_timeout_seconds = execution_timeout_seconds
        self.executions: list[CodeExecution] = []

    async def health_check(self) -> bool:
        """Check if the sandbox MCP server is healthy."""
        try:
            await self._request("GET", "/health", "sandbox:health_check")
            return True
        except Exception:
            return False

    def _extract_imports(self, code: str) -> list[str]:
        """Extract imported libraries from code."""
        imports = []

        # Match 'import X' and 'from X import Y'
        import_pattern = r"^(?:import|from)\s+(\w+)"
        for match in re.finditer(import_pattern, code, re.MULTILINE):
            lib = match.group(1)
            if lib not in imports:
                imports.append(lib)

        return imports

    def _extract_print_statements(self, output: str) -> dict[str, Any]:
        """Extract intermediate values from print statements in output."""
        values = {}

        # Pattern for labeled output like "value_name: 123.45"
        labeled_pattern = r"(\w+):\s*([\d.]+)"
        for match in re.finditer(labeled_pattern, output):
            name = match.group(1)
            try:
                value = float(match.group(2))
                values[name] = value
            except ValueError:
                values[name] = match.group(2)

        return values

    def _extract_final_value(self, output: str) -> Optional[Any]:
        """Extract the final value from output."""
        if not output:
            return None

        lines = output.strip().split("\n")
        if not lines:
            return None

        last_line = lines[-1].strip()

        # Try to parse as number
        try:
            if "." in last_line:
                return float(last_line)
            return int(last_line)
        except ValueError:
            pass

        # Try to extract number from labeled output
        match = re.search(r"[\d.]+", last_line)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass

        return last_line if last_line else None

    def _score_methodology(self, code: str) -> float:
        """
        Score the methodology quality of the code.

        Scoring criteria:
        - Uses pandas/numpy for calculations: +30
        - Uses appropriate financial functions: +20
        - Has clear variable naming: +15
        - Has comments explaining logic: +10
        - Uses vectorized operations: +10
        - Has proper error handling: +10
        - Code is readable/structured: +5
        """
        score = 0.0

        # Uses pandas/numpy
        if "pandas" in code.lower() or "numpy" in code.lower() or "pd." in code or "np." in code:
            score += 30

        # Uses financial calculation patterns
        for pattern_name, pattern in self.FINANCIAL_PATTERNS.items():
            if re.search(pattern, code, re.IGNORECASE):
                score += 5  # Up to 25 points for various patterns

        # Has comments
        if "#" in code or '"""' in code or "'''" in code:
            score += 10

        # Uses vectorized operations
        vectorized_patterns = [r"\.apply\(", r"\.map\(", r"np\.\w+\(", r"\.sum\(", r"\.mean\("]
        for pattern in vectorized_patterns:
            if re.search(pattern, code):
                score += 2  # Up to 10 points

        # Has proper error handling
        if "try:" in code and "except" in code:
            score += 10

        # Clear variable naming (lowercase with underscores)
        var_pattern = r"\b([a-z][a-z0-9_]*)\s*="
        vars_found = re.findall(var_pattern, code)
        if vars_found and len(vars_found) >= 3:
            score += 10

        return min(100, score)

    def analyze_code_trace(self, code: str, output: str) -> CodeTraceAnalysis:
        """
        Analyze the code execution trace to verify calculation methodology.

        Args:
            code: The executed code
            output: The execution output

        Returns:
            CodeTraceAnalysis with detailed methodology assessment
        """
        libraries = self._extract_imports(code)
        intermediate = self._extract_print_statements(output)
        final = self._extract_final_value(output)
        methodology_score = self._score_methodology(code)

        # Check for vectorization
        uses_vectorization = any(
            pattern in code for pattern in ["np.", "pd.", ".apply(", ".map("]
        )

        # Check for proper types
        uses_proper_types = "float" in code or "int" in code or "Decimal" in code

        # Check for error handling
        has_error_handling = "try:" in code and "except" in code

        # Extract calculation steps (lines with assignments)
        calculation_steps = []
        for line in code.split("\n"):
            line = line.strip()
            if "=" in line and not line.startswith("#") and not line.startswith("import"):
                calculation_steps.append(line)

        return CodeTraceAnalysis(
            libraries_used=libraries,
            calculation_steps=calculation_steps,
            intermediate_values=intermediate,
            final_result=final,
            methodology_score=methodology_score,
            uses_vectorization=uses_vectorization,
            uses_proper_types=uses_proper_types,
            has_error_handling=has_error_handling
        )

    async def execute_code(
        self,
        code: str,
        timeout_seconds: Optional[int] = None
    ) -> ExecutionResult:
        """
        Execute Python code in the sandbox.

        Args:
            code: Python code to execute
            timeout_seconds: Execution timeout (defaults to client setting)

        Returns:
            ExecutionResult with output and metadata
        """
        timeout = timeout_seconds or self.execution_timeout_seconds

        json_body = {
            "code": code,
            "timeout": timeout,
            "libraries": self.PRELOADED_LIBS,
        }

        try:
            result = await self._request(
                "POST",
                "/execute",
                "sandbox:execute",
                json_body=json_body
            )

            execution_result = ExecutionResult(
                execution_id=result.get("execution_id", ""),
                code=code,
                stdout=result.get("stdout", ""),
                stderr=result.get("stderr", ""),
                return_value=result.get("return_value"),
                execution_time_ms=result.get("execution_time_ms", 0),
                success=result.get("success", True),
                error_type=result.get("error_type"),
                error_message=result.get("error_message"),
            )

        except Exception as e:
            execution_result = ExecutionResult(
                execution_id="",
                code=code,
                success=False,
                error_type="client_error",
                error_message=str(e),
            )

        # Log execution for tracking
        code_execution = CodeExecution(
            code=code,
            output=execution_result.stdout or execution_result.stderr,
            execution_time_ms=execution_result.execution_time_ms,
            libraries_used=self._extract_imports(code),
            success=execution_result.success,
            error_message=execution_result.error_message,
        )
        self.executions.append(code_execution)

        logger.info(
            "sandbox_execution",
            success=execution_result.success,
            execution_time_ms=execution_result.execution_time_ms,
            libraries=code_execution.libraries_used,
        )

        return execution_result

    async def get_execution_trace(self, execution_id: str) -> dict[str, Any]:
        """
        Get detailed execution trace for a previous execution.

        Args:
            execution_id: ID of the execution to retrieve

        Returns:
            Execution trace details
        """
        result = await self._request(
            "GET",
            f"/trace/{execution_id}",
            "sandbox:get_trace",
            params={"execution_id": execution_id}
        )
        return result

    def get_executions(self) -> list[CodeExecution]:
        """Get all code executions performed by this client."""
        return self.executions

    def validate_numerical_task_execution(
        self,
        task_category: str,
        agent_response_has_code: bool
    ) -> tuple[bool, float]:
        """
        Validate that numerical tasks used code execution.

        For FAB categories requiring numerical reasoning, agents MUST
        use the sandbox for calculations. Failure to do so results in
        a 50% penalty on Execution Score.

        Args:
            task_category: The FAB task category
            agent_response_has_code: Whether the agent executed code

        Returns:
            Tuple of (is_valid, penalty)
        """
        numerical_categories = [
            "Numerical Reasoning",
            "Adjustments",
            "Financial Modeling",
        ]

        if task_category in numerical_categories:
            if not agent_response_has_code:
                logger.warning(
                    "missing_code_execution",
                    task_category=task_category,
                    penalty=0.5
                )
                return False, 0.5

        return True, 0.0
