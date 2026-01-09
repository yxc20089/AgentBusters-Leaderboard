# CIO-Agent: FAB++ Architecture Design
## Dynamic Finance Agent Benchmark for AgentBeats Competition

**Team:** AgentBusters
**Competition:** AgentBeats (https://rdi.berkeley.edu/agentx-agentbeats)
**Agent Type:** Green Agent (Judge/Evaluator)
**Project Codename:** FAB++ (Dynamic Finance Agent Benchmark)
**Strategic Position:** Evolution and expansion of Finance Agent Benchmark (arXiv:2508.00828)

---

## Executive Summary

### Strategic Pivot: From Static Benchmark to Dynamic Simulation

Our goal for the AgentBeats Finance Track is to build the **definitive Green Agent (Evaluator)** by transforming the Finance Agent Benchmark (FAB) from a static question-answering dataset into **FAB++**, a dynamic, adversarial financial simulation environment.

While the FAB paper (arXiv:2508.00828) provides excellent content with 537 expert-authored questions across 9 task categories, it suffers from the **"Static Benchmark Problem"**:
- **Data Contamination**: Static questions with hardcoded answers (e.g., "What was Apple's 2023 revenue?")
- **Binary Scoring**: A slight math error fails the entire task
- **No Time Sensitivity**: Questions assume a fixed "now" without temporal constraints
- **Passive Evaluation**: The judge simply compares answers without testing robustness

### Key Innovation: The "Chief Investment Officer" Evaluator

Inspired by the TradingAgents framework (arXiv:2412.20138), our Green Agent acts as a **"Chief Investment Officer"**, engaging test agents in:
- **Adversarial Debate**: Counter-thesis generation to test conviction
- **Noise Injection**: Distractor documents mixed with ground truth data
- **Temporal Locking**: Strict enforcement of "time travel" constraints to prevent look-ahead bias
- **Hierarchical Assessment**: Multi-dimensional scoring across Macro Analysis, Fundamental Accuracy, and Execution Quality

---

## 1. The FAB++ Upgrade Strategy

We directly address the limitations in `vals-ai/finance_agent_benchmark` to meet AgentBeats' "Innovation" and "Realism" criteria:

| FAB Limitation (Current State) | AgentBusters "FAB++" Solution |
|-------------------------------|-------------------------------|
| **Static Ground Truth**: Answers are hardcoded (e.g., "Apple's 2023 Revenue"). | **Dynamic Injection**: We use FAB questions as *templates*. We dynamically swap tickers (e.g., AAPL → MSFT) and fetch live ground truth via our MCP tools during generation. |
| **Binary Accuracy**: A slight math error fails the whole task. | **Hierarchical Role Assessment**: Inspired by TradingAgents, we score distinct layers: *Macro Analysis* (Reasoning), *Fundamental Accuracy* (Math), and *Execution* (Action). |
| **No Time Sensitivity**: Questions assume a fixed "now." | **"Time Travel" Environment**: The Green Agent sets a simulated "Current Date." If an agent retrieves data *after* this date (look-ahead bias), it receives a penalty. |
| **Passive Evaluation**: The judge just reads the answer. | **Adversarial Debate**: The Green Agent challenges the White Agent's thesis (e.g., "Justify this P/E ratio given the sector risk"), forcing the agent to defend its reasoning. |

---

## 2. Finance Agent Benchmark (FAB) Foundation

### 2.1 FAB Task Categories (Our Curriculum)

FAB provides 537 expert-authored questions across **9 task categories**. Our FAB++ implementation maintains full compatibility with this taxonomy:

| Category | Description | Example Task | FAB++ Enhancement |
|----------|-------------|--------------|-------------------|
| **1. Quantitative Retrieval** | Extract specific numerical data from filings | "What was Microsoft's total revenue in FY 2023?" | Dynamic ticker substitution, temporal locking |
| **2. Qualitative Retrieval** | Extract textual information from documents | "What are the primary risk factors disclosed in Tesla's latest 10-K?" | Noise injection with distractor documents |
| **3. Numerical Reasoning** | Perform calculations on retrieved data | "Calculate the year-over-year revenue growth rate for NVDA from 2022 to 2023." | Mandatory code execution via mcp-sandbox |
| **4. Complex Retrieval** | Multi-step retrieval across documents | "Compare the operating margins of Apple and Samsung in 2023." | Cross-ticker validation, time-locked data |
| **5. Adjustments** | Apply accounting adjustments to raw data | "Calculate the adjusted EBITDA for Company X given one-time charges." | Rubric validation of adjustment methodology |
| **6. Beat or Miss** | Determine if actual results exceeded expectations | "Did Amazon beat or miss Q3 2023 earnings expectations?" | Real-time consensus data via yahoo-finance-mcp |
| **7. Trends** | Identify patterns across time periods | "What is the 5-year trend in Alphabet's R&D spending as % of revenue?" | Multi-period temporal integrity checks |
| **8. Financial Modeling** | Build projections or valuation models | "Build a DCF model to value Netflix stock." | Full code execution trace, assumption validation |
| **9. Market Analysis** | Synthesize macro and micro factors | "Given current Fed policy and tech sector trends, recommend a position on semiconductor stocks." | Adversarial debate on macro thesis |

### 2.2 FAB Evaluation Methodology (Our Starting Point)

The original FAB uses:
- **Rubric-Based LLM-as-Judge**: GPT-4 scores answers against expert-authored rubrics
- **Tool Harness**: Provides Google Search, EDGAR Search, ParseHTML, and RetrieveInformation tools
- **Contradiction Detection**: Uses LLM to identify internal inconsistencies in agent responses

**FAB++ Enhancements**:
- Replace generic web search with **controlled MCP tool suite** (sec-edgar-mcp, yahoo-finance-mcp, mcp-sandbox)
- Add **Debate Multiplier** to rubric scoring (rewards robust defense under adversarial critique)
- Introduce **Alpha Score** metric combining accuracy, efficiency, and temporal compliance

---

## 3. Technical Architecture: The "MCP Trinity"

To ensure a **fair and reproducible environment**, we do not allow agents to browse the open web. Instead, we provide a **"Virtual Bloomberg Terminal"** composed of three specific, open-source MCP servers.

### 3.1 Compliance & Filings: `sec-edgar-mcp`

**Source**: Fork of `stefanoamorelli/sec-edgar-mcp`
**Underlying Engine**: EdgarTools for precise XBRL parsing

#### Key Features
- Fetch 10-K, 10-Q, 8-K filings by ticker and date range
- Extract specific sections (e.g., "Item 1A: Risk Factors", "Item 7: MD&A")
- Parse XBRL financial statements (Balance Sheet, Income Statement, Cash Flow)

#### FAB++ Customization: "The Metered Middleware"

**A. Token Cost Tracking**
```python
class MeteredEDGARServer(MCPServer):
    """
    Intercepts every SEC EDGAR request to calculate token cost.
    Penalizes inefficient query patterns (e.g., fetching full 10-K when only revenue is needed).
    """
    def __init__(self):
        self.query_log = []
        self.token_costs = {}

    async def get_filing_section(self, ticker: str, form_type: str, section: str, date: str):
        # Calculate token cost based on section size
        section_text = await edgar_tools.fetch_section(ticker, form_type, section, date)
        token_count = count_tokens(section_text)

        self.query_log.append({
            "ticker": ticker,
            "form": form_type,
            "section": section,
            "date": date,
            "tokens": token_count,
            "timestamp": time.time()
        })

        return section_text
```

**B. Noise Injection: "Distractor Documents"**

As suggested by TradingAgents, we inject **signal-vs-noise testing**:

```python
class NoiseInjectionMiddleware:
    """
    Randomly injects distractor documents (e.g., press releases, unaudited earnings calls)
    alongside official 10-K results to test if the agent can filter signal from noise.
    """
    def __init__(self, noise_ratio: float = 0.3):
        self.noise_ratio = noise_ratio

    async def get_filing_with_noise(self, ticker: str, form_type: str):
        # Fetch legitimate 10-K filing
        official_filing = await edgar_tools.fetch_filing(ticker, form_type)

        # Inject distractor documents
        if random.random() < self.noise_ratio:
            distractor = generate_realistic_press_release(ticker)
            return {
                "official": official_filing,
                "distractors": [distractor],
                "test": "agent must identify which document is the official 10-K"
            }

        return {"official": official_filing, "distractors": []}
```

**C. Temporal Integrity Enforcement**

```python
class TemporalLockMiddleware:
    """
    Enforces 'time travel' constraints. If simulation_date is 2022-01-01,
    agent cannot access filings dated after this date.
    """
    def __init__(self, simulation_date: datetime):
        self.simulation_date = simulation_date

    async def get_filing(self, ticker: str, form_type: str, filing_date: str):
        requested_date = datetime.fromisoformat(filing_date)

        if requested_date > self.simulation_date:
            raise TemporalViolationError(
                f"Look-ahead bias detected: Cannot access {filing_date} data "
                f"when simulation date is {self.simulation_date}"
            )

        return await edgar_tools.fetch_filing(ticker, form_type, filing_date)
```

### 3.2 Market Data: `yahoo-finance-mcp`

**Source**: Fork of `Alex2Yang97/yahoo-finance-mcp`
**Underlying Engine**: yfinance for OHLCV data

#### Key Features
- Historical price data (Open, High, Low, Close, Volume)
- Financial statements (Income Statement, Balance Sheet, Cash Flow)
- Key statistics (P/E ratio, Market Cap, Beta, etc.)
- Analyst estimates and recommendations

#### FAB++ Customization: "The Time Machine"

**A. Simulation Date Enforcement**

```python
class TimeMachineMiddleware:
    """
    Accepts a simulation_date from the Green Agent. All data requests are filtered
    to only return information available as of that date.
    """
    def __init__(self, simulation_date: datetime):
        self.simulation_date = simulation_date

    async def get_price(self, ticker: str, start_date: str = None, end_date: str = None):
        # If no end_date specified, default to simulation_date
        if end_date is None:
            end_date = self.simulation_date.strftime("%Y-%m-%d")

        requested_end = datetime.fromisoformat(end_date)

        # Enforce temporal boundary
        if requested_end > self.simulation_date:
            raise TemporalViolationError(
                f"403 Future Data Forbidden: Cannot access {end_date} data "
                f"when simulation date is {self.simulation_date}"
            )

        # Fetch historical data up to simulation_date
        return await yfinance.fetch_ohlcv(ticker, start_date, end_date)
```

**B. Look-Ahead Penalty Tracking**

```python
class LookAheadDetector:
    """
    Tracks temporal violations and calculates penalty multiplier for final score.
    """
    def __init__(self):
        self.violations = []

    def log_violation(self, ticker: str, requested_date: str, simulation_date: str):
        days_ahead = (datetime.fromisoformat(requested_date) -
                     datetime.fromisoformat(simulation_date)).days

        self.violations.append({
            "ticker": ticker,
            "requested": requested_date,
            "allowed": simulation_date,
            "days_ahead": days_ahead,
            "severity": "high" if days_ahead > 90 else "medium"
        })

    def calculate_penalty(self) -> float:
        """
        Returns a penalty multiplier for the Alpha Score.
        No violations = 0.0 penalty
        Severe violations = up to 0.5 penalty
        """
        if not self.violations:
            return 0.0

        total_days_ahead = sum(v["days_ahead"] for v in self.violations)
        return min(0.5, total_days_ahead / 365.0)  # Cap at 50% penalty
```

### 3.3 Analyst Sandbox: `mcp-sandbox`

**Source**: Fork of `JohanLi233/mcp-sandbox`
**Underlying Engine**: Dockerized Python execution environment

#### Key Features
- Isolated code execution for financial calculations
- Pre-installed libraries: numpy, pandas, scipy, matplotlib
- Timeout protection (max 30 seconds per execution)

#### FAB++ Customization: "Pre-loaded Quants"

**A. Financial Library Pre-installation**

```python
class QuantSandbox(MCPSandbox):
    """
    Pre-loads heavy financial libraries to save installation time during evaluation.
    """
    PRELOADED_LIBS = [
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "yfinance",
        "pandas-datareader",
        "financial-modeling-prep",  # FMP API wrapper
        "quantlib",  # Options pricing
    ]

    def __init__(self):
        super().__init__(preload=self.PRELOADED_LIBS)
```

**B. Mandatory Code Execution for Numerical Tasks**

```python
class NumericalTaskValidator:
    """
    For FAB task categories requiring numerical reasoning (Categories 3, 5, 8),
    agents MUST use the sandbox to perform calculations.

    If an agent provides a numerical answer without code execution,
    it receives a 50% penalty on Execution Score.
    """
    NUMERICAL_CATEGORIES = ["Numerical Reasoning", "Adjustments", "Financial Modeling"]

    def validate_execution(self, task_category: str, agent_response: dict) -> bool:
        if task_category in self.NUMERICAL_CATEGORIES:
            if "code_execution" not in agent_response:
                return False  # Missing required code execution
        return True
```

**C. Code Trace Analysis**

```python
class CodeTraceAnalyzer:
    """
    Analyzes the agent's code execution trace to verify calculation methodology.
    """
    def analyze_trace(self, code: str, output: str) -> dict:
        return {
            "libraries_used": extract_imports(code),
            "calculation_steps": parse_calculation_flow(code),
            "intermediate_values": extract_print_statements(output),
            "final_result": extract_final_value(output),
            "methodology_score": score_methodology(code)  # 0-100
        }
```

---

## 4. Evaluation Methodology: The "Alpha Score"

We introduce a **composite metric** that aligns with the "Efficiency" and "Realism" criteria of AgentBeats, incorporating the TradingAgents "Debate" and "Role" concepts.

### 4.1 Alpha Score Formula

$$
\text{Alpha Score} = \frac{\text{Role Score} \times \text{Debate Multiplier}}{\ln(1 + \text{Cost}_{\text{USD}}) \times (1 + \text{LookAhead Penalty})}
$$

**Components**:
- **Role Score** (0-100): Hierarchical assessment across Macro, Fundamental, and Execution dimensions
- **Debate Multiplier** (0.5x-1.2x): Adversarial defense performance
- **Cost (USD)**: Total LLM inference + tool usage costs
- **LookAhead Penalty** (0.0-0.5): Temporal integrity violation severity

### 4.2 The Role Score (Hierarchical Assessment)

Inspired by TradingAgents' multi-agent architecture, we score **three distinct dimensions** instead of binary 0/1:

#### **1. Macro Score (30%): Strategic Reasoning**

**What we evaluate**:
- Did the agent identify correct sector trends?
- Did it recognize relevant macroeconomic factors (Fed policy, inflation, etc.)?
- Did it assess industry-specific risks (regulation, competition, disruption)?

**Evaluation Method**: Keyword matching and semantic similarity against ground truth macro thesis

```python
class MacroEvaluator:
    """
    Scores the agent's macro analysis against expert-authored ground truth.
    """
    def __init__(self, ground_truth_thesis: str):
        self.ground_truth = ground_truth_thesis
        self.key_themes = extract_themes(ground_truth_thesis)

    def score(self, agent_macro_analysis: str) -> float:
        # Semantic similarity to ground truth
        similarity_score = semantic_similarity(agent_macro_analysis, self.ground_truth)

        # Key theme coverage (e.g., "rising interest rates", "AI chip demand")
        theme_coverage = count_themes_mentioned(agent_macro_analysis, self.key_themes)
        coverage_score = theme_coverage / len(self.key_themes)

        # Combined score (0-100)
        return 0.6 * similarity_score * 100 + 0.4 * coverage_score * 100
```

#### **2. Fundamental Score (40%): Data Accuracy**

**What we evaluate**:
- Is the extracted data (Revenue, EBITDA, EPS, etc.) precise?
- Did the agent use the correct filing (10-K vs 10-Q)?
- Did it apply proper accounting adjustments?

**Evaluation Method**: Direct comparison with XBRL ground truth from sec-edgar-mcp

```python
class FundamentalEvaluator:
    """
    Validates extracted financial data against XBRL ground truth.
    """
    def __init__(self, ground_truth_financials: dict):
        self.ground_truth = ground_truth_financials

    def score(self, agent_financials: dict) -> float:
        total_fields = len(self.ground_truth)
        correct_fields = 0

        for field, true_value in self.ground_truth.items():
            agent_value = agent_financials.get(field)

            if agent_value is None:
                continue  # Missing field = 0 points

            # Allow 1% tolerance for rounding differences
            if abs(agent_value - true_value) / true_value < 0.01:
                correct_fields += 1

        # Score based on accuracy percentage
        return (correct_fields / total_fields) * 100
```

#### **3. Execution Score (30%): Action Quality**

**What we evaluate**:
- Is the final recommendation (Buy/Sell/Hold) supported by the analysis?
- Did the agent provide a clear investment thesis?
- Did it use proper code execution for numerical tasks?

**Evaluation Method**: Rubric-based LLM-as-judge + code execution validation

```python
class ExecutionEvaluator:
    """
    Assesses the quality of the agent's final recommendation and methodology.
    """
    def __init__(self, task_category: str):
        self.task_category = task_category
        self.rubric = load_rubric(task_category)

    async def score(self, agent_response: dict, code_trace: dict = None) -> float:
        # Rubric-based LLM scoring (following FAB methodology)
        rubric_score = await llm_judge(self.rubric, agent_response)

        # Code execution penalty (if required but missing)
        code_penalty = 0.0
        if self.task_category in ["Numerical Reasoning", "Adjustments", "Financial Modeling"]:
            if code_trace is None:
                code_penalty = 0.5  # 50% penalty for missing code

        # Combined score
        return rubric_score * (1.0 - code_penalty)
```

#### **Combined Role Score Calculation**

```python
def calculate_role_score(macro_score: float, fundamental_score: float, execution_score: float) -> float:
    """
    Weighted average of the three role dimensions.
    """
    return (
        0.30 * macro_score +
        0.40 * fundamental_score +
        0.30 * execution_score
    )
```

### 4.3 The Debate Multiplier (Adversarial Defense)

After the agent provides its initial answer, the **Green Agent** (acting as "Risk Manager") generates a **counter-argument** to test the agent's conviction and robustness.

#### Debate Flow

1. **Initial Response**: Agent submits analysis (e.g., "Buy TSLA based on strong Q3 deliveries")
2. **Green Agent Challenge**: "Margins are compressing by 5% YoY due to price cuts. Defend your thesis."
3. **Agent Rebuttal**: Agent must provide evidence-based defense
4. **Scoring**: Multiplier applied based on rebuttal quality

#### Scoring Logic

```python
class DebateEvaluator:
    """
    Evaluates the agent's ability to defend its thesis under adversarial critique.
    """
    async def generate_counter_argument(self, agent_thesis: str, financial_data: dict) -> str:
        """
        Uses LLM to generate a plausible counter-thesis based on real financial data.
        """
        prompt = f"""
        You are a skeptical Risk Manager reviewing an investment recommendation.

        Agent's Thesis: {agent_thesis}

        Financial Data: {financial_data}

        Generate a critical counter-argument that challenges the thesis using data-driven reasoning.
        Focus on risks the agent may have overlooked (margin compression, competitive threats,
        regulatory risks, valuation concerns, etc.).
        """

        return await llm.generate(prompt)

    async def score_rebuttal(self, counter_argument: str, agent_rebuttal: str,
                            financial_data: dict) -> float:
        """
        Scores the agent's rebuttal quality.

        Returns:
            1.2x: Agent provides NEW evidence and successfully defends thesis
            1.0x: Agent repeats previous evidence without new insights
            0.5x: Agent hallucinates, contradicts itself, or immediately concedes
        """
        prompt = f"""
        Evaluate the quality of this rebuttal:

        Challenge: {counter_argument}
        Rebuttal: {agent_rebuttal}
        Available Data: {financial_data}

        Score the rebuttal:
        - 1.2x: Agent provides NEW evidence from data and successfully defends thesis
        - 1.0x: Agent repeats previous evidence without new insights
        - 0.5x: Agent hallucinates (cites non-existent data), contradicts itself, or immediately concedes

        Return only the multiplier value (0.5, 1.0, or 1.2).
        """

        multiplier = await llm.generate(prompt)
        return float(multiplier)
```

#### Example Debate Sequence

```
Agent Initial Response:
"Recommendation: BUY NVDA
Thesis: NVDA's data center revenue grew 171% YoY in Q2 2024, driven by AI chip demand.
Gross margin remains strong at 70.1%. P/E of 45x is justified by 3-year revenue CAGR of 38%."

Green Agent Challenge:
"The semiconductor industry is cyclical. NVDA's valuation of 45x P/E assumes sustained hyper-growth,
but major customers (MSFT, META, GOOGL) are developing in-house AI chips, threatening NVDA's moat.
Additionally, export restrictions to China represent 20% revenue risk. Defend your BUY thesis
given these structural headwinds."

Agent Rebuttal (Strong - 1.2x multiplier):
"While customer in-house chip development is real, NVDA maintains 3-5 year technology lead in
interconnect (NVLink) and software stack (CUDA). Meta's custom chips still use NVDA for training.
On China risk: Management guided China revenue already down to 10% in latest call (Q2 2024 10-Q,
page 32), with Taiwan/SEA offsetting. Margin stability at 70%+ shows pricing power persists."

Agent Rebuttal (Weak - 0.5x multiplier):
"I agree those are valid risks. Perhaps HOLD is more appropriate than BUY."
[Immediate concession without defense]
```

### 4.4 Cost Efficiency Denominator

Following FAB's approach, we track **total cost** of agent execution:

```python
class CostTracker:
    """
    Tracks all costs incurred during agent evaluation.
    """
    def __init__(self):
        self.llm_costs = 0.0
        self.tool_costs = 0.0

    def add_llm_call(self, model: str, input_tokens: int, output_tokens: int):
        """
        Calculates cost based on model pricing.
        """
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
        }

        rate = pricing.get(model, pricing["gpt-4"])
        cost = (input_tokens / 1000) * rate["input"] + (output_tokens / 1000) * rate["output"]
        self.llm_costs += cost

    def add_tool_call(self, tool_name: str, token_count: int):
        """
        Token-based pricing for tool usage.
        $0.001 per 1K tokens retrieved from SEC EDGAR or Yahoo Finance.
        """
        self.tool_costs += (token_count / 1000) * 0.001

    def total_cost(self) -> float:
        return self.llm_costs + self.tool_costs
```

The **logarithmic cost penalty** ensures diminishing returns from throwing compute at the problem:

```python
def cost_penalty(total_cost_usd: float) -> float:
    """
    Logarithmic penalty to prevent brute-force compute scaling.

    Examples:
    - $1 cost → penalty of 0.69
    - $5 cost → penalty of 1.79
    - $10 cost → penalty of 2.40

    This means doubling the cost does NOT double the denominator,
    allowing efficient models to compete with expensive ones.
    """
    return math.log(1 + total_cost_usd)
```

### 4.6 Persistence Layer (New in v1.0)
To ensure reliable evaluation over long periods, the CIO-Agent uses a persistent storage mechanism:

- **Database**: SQLite (`tasks.db`)
- **Persistence**: All task states, messages, and results are saved to disk.
- **Resumability**: Evaluations can pick up where they left off if the server is restarted.
- **Scalability**: Capable of handling hundreds of concurrent evaluation tasks without memory exhaustion.

### 4.5 Look-Ahead Penalty (Temporal Integrity)

Calculated by the `LookAheadDetector` from section 3.2:

```python
def calculate_alpha_score(role_score: float, debate_multiplier: float,
                         cost_usd: float, lookahead_penalty: float) -> float:
    """
    Final Alpha Score calculation.

    Args:
        role_score: 0-100 from hierarchical assessment
        debate_multiplier: 0.5-1.2 from adversarial defense
        cost_usd: Total USD spent on LLM + tools
        lookahead_penalty: 0.0-0.5 from temporal violations

    Returns:
        Alpha Score (higher is better)
    """
    numerator = role_score * debate_multiplier
    denominator = math.log(1 + cost_usd) * (1 + lookahead_penalty)

    return numerator / denominator
```

#### Example Calculations

**Scenario 1: High-Quality, Efficient Agent**
- Role Score: 85
- Debate Multiplier: 1.2 (strong rebuttal)
- Cost: $2.50
- LookAhead Penalty: 0.0 (no violations)

```
Alpha Score = (85 * 1.2) / (ln(1 + 2.5) * 1.0)
            = 102 / (1.25 * 1.0)
            = 81.6
```

**Scenario 2: High-Quality, Expensive Agent**
- Role Score: 90
- Debate Multiplier: 1.2
- Cost: $10.00
- LookAhead Penalty: 0.0

```
Alpha Score = (90 * 1.2) / (ln(1 + 10) * 1.0)
            = 108 / (2.40 * 1.0)
            = 45.0
```

**Scenario 3: Low-Quality, Efficient Agent**
- Role Score: 60
- Debate Multiplier: 1.0 (weak rebuttal)
- Cost: $1.00
- LookAhead Penalty: 0.0

```
Alpha Score = (60 * 1.0) / (ln(1 + 1) * 1.0)
            = 60 / (0.69 * 1.0)
            = 87.0
```

**Key Insight**: Scenario 3 (efficient but mediocre) beats Scenario 2 (expensive excellence), incentivizing cost-effective solutions.

---

## 5. CIO-Agent Orchestration Logic

The Green Agent orchestrates the entire evaluation process through a multi-phase workflow.

### 5.1 Evaluation Phases

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Task Generation (Dynamic FAB Question Injection)      │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Agent Execution (White/Purple Agent responds)         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: Adversarial Debate (Green Agent challenges thesis)    │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 4: Multi-Dimensional Scoring (Role + Debate + Cost)      │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 5: Alpha Score Calculation & Reporting                   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Phase 1: Dynamic Task Generation

Instead of using static FAB questions, we **dynamically generate variants**:

```python
class DynamicTaskGenerator:
    """
    Takes FAB questions as templates and dynamically generates variants.
    """
    def __init__(self, fab_dataset: list):
        self.fab_questions = fab_dataset

    async def generate_task(self, task_id: str, simulation_date: datetime) -> dict:
        """
        Generates a dynamic variant of a FAB question.

        Example:
        Original FAB: "What was Apple's total revenue in FY 2023?"
        Dynamic Variant: "What was Microsoft's total revenue in FY 2022?"
        """
        # Select base question from FAB
        base_question = self.get_question_by_id(task_id)

        # Extract parameters (ticker, fiscal year, metric)
        original_ticker = base_question["ticker"]
        original_year = base_question["fiscal_year"]
        original_metric = base_question["metric"]

        # Randomly substitute ticker (within same sector for fairness)
        new_ticker = self.sample_similar_company(original_ticker)

        # Adjust fiscal year based on simulation date
        available_years = self.get_available_years(new_ticker, simulation_date)
        new_year = random.choice(available_years)

        # Generate new question text
        new_question_text = base_question["template"].format(
            ticker=new_ticker,
            year=new_year,
            metric=original_metric
        )

        # Fetch ground truth using MCP tools
        ground_truth = await self.fetch_ground_truth(new_ticker, new_year, original_metric)

        return {
            "question_id": f"{task_id}_variant_{new_ticker}_{new_year}",
            "category": base_question["category"],
            "question": new_question_text,
            "ground_truth": ground_truth,
            "simulation_date": simulation_date,
            "difficulty": base_question["difficulty"],
            "rubric": base_question["rubric"]
        }
```

### 5.3 Phase 2: Agent Execution with Tool Monitoring

The White/Purple agent receives the task via A2A Protocol:

```python
class A2AOrchestrator:
    """
    Manages communication between Green Agent (CIO-Agent) and White/Purple Agent.
    """
    async def send_task(self, agent_id: str, task: dict):
        """
        Sends task to agent via A2A Protocol.
        """
        message = {
            "type": "task_assignment",
            "task_id": task["question_id"],
            "question": task["question"],
            "simulation_date": task["simulation_date"].isoformat(),
            "available_tools": [
                "sec-edgar-mcp",
                "yahoo-finance-mcp",
                "mcp-sandbox"
            ],
            "deadline": "30 minutes",
            "evaluation_criteria": "You will be scored on accuracy, efficiency, and robustness."
        }

        await self.send_a2a_message(agent_id, message)

    async def receive_response(self, agent_id: str, timeout: int = 1800) -> dict:
        """
        Receives agent's response with timeout protection.
        """
        response = await self.wait_for_a2a_message(agent_id, timeout=timeout)

        return {
            "agent_id": agent_id,
            "task_id": response["task_id"],
            "analysis": response["analysis"],
            "recommendation": response["recommendation"],
            "code_trace": response.get("code_execution", None),
            "tool_calls": response["tool_usage_log"],
            "timestamp": response["timestamp"]
        }
```

While the agent executes, we monitor tool usage:

```python
class ToolUsageMonitor:
    """
    Monitors all MCP tool calls made by the agent.
    """
    def __init__(self):
        self.tool_log = []
        self.cost_tracker = CostTracker()
        self.lookahead_detector = LookAheadDetector()

    async def intercept_tool_call(self, tool_name: str, params: dict, simulation_date: datetime):
        """
        Intercepts and logs every tool call.
        """
        if tool_name == "yahoo-finance-mcp:get_price":
            # Check for temporal violations
            if "end_date" in params:
                end_date = datetime.fromisoformat(params["end_date"])
                if end_date > simulation_date:
                    self.lookahead_detector.log_violation(
                        ticker=params["ticker"],
                        requested_date=params["end_date"],
                        simulation_date=simulation_date.isoformat()
                    )
                    raise TemporalViolationError("Future data access forbidden")

        # Log tool call for cost tracking
        self.tool_log.append({
            "tool": tool_name,
            "params": params,
            "timestamp": time.time()
        })
```

### 5.4 Phase 3: Adversarial Debate

```python
class AdversarialDebateManager:
    """
    Conducts the debate phase to test agent robustness.
    """
    async def conduct_debate(self, agent_response: dict, task: dict,
                            financial_data: dict) -> dict:
        """
        Runs adversarial debate to test agent conviction.
        """
        # Generate counter-argument using Green Agent LLM
        counter_argument = await self.generate_counter_argument(
            agent_thesis=agent_response["recommendation"],
            financial_data=financial_data
        )

        # Send challenge to agent via A2A
        challenge_message = {
            "type": "adversarial_challenge",
            "task_id": task["question_id"],
            "challenge": counter_argument,
            "instructions": "Defend your thesis with additional evidence or revise your recommendation."
        }

        await self.send_a2a_message(agent_response["agent_id"], challenge_message)

        # Receive agent's rebuttal
        rebuttal = await self.receive_a2a_message(agent_response["agent_id"], timeout=600)

        # Score the rebuttal
        debate_evaluator = DebateEvaluator()
        debate_multiplier = await debate_evaluator.score_rebuttal(
            counter_argument=counter_argument,
            agent_rebuttal=rebuttal["defense"],
            financial_data=financial_data
        )

        return {
            "counter_argument": counter_argument,
            "agent_rebuttal": rebuttal["defense"],
            "debate_multiplier": debate_multiplier,
            "conviction_level": "high" if debate_multiplier == 1.2 else "medium" if debate_multiplier == 1.0 else "low"
        }
```

### 5.5 Phase 4: Multi-Dimensional Scoring

```python
class ComprehensiveEvaluator:
    """
    Orchestrates all evaluation components.
    """
    async def evaluate(self, task: dict, agent_response: dict,
                      debate_result: dict, tool_monitor: ToolUsageMonitor) -> dict:
        """
        Runs complete evaluation pipeline.
        """
        # Extract ground truth
        ground_truth = task["ground_truth"]

        # Phase 1: Macro Score
        macro_evaluator = MacroEvaluator(ground_truth["macro_thesis"])
        macro_score = macro_evaluator.score(agent_response["analysis"])

        # Phase 2: Fundamental Score
        fundamental_evaluator = FundamentalEvaluator(ground_truth["financials"])
        fundamental_score = fundamental_evaluator.score(
            extract_financials(agent_response["analysis"])
        )

        # Phase 3: Execution Score
        execution_evaluator = ExecutionEvaluator(task["category"])
        execution_score = await execution_evaluator.score(
            agent_response=agent_response,
            code_trace=agent_response.get("code_trace")
        )

        # Combine into Role Score
        role_score = calculate_role_score(macro_score, fundamental_score, execution_score)

        # Debate Multiplier
        debate_multiplier = debate_result["debate_multiplier"]

        # Cost Calculation
        total_cost = tool_monitor.cost_tracker.total_cost()

        # Look-Ahead Penalty
        lookahead_penalty = tool_monitor.lookahead_detector.calculate_penalty()

        # Final Alpha Score
        alpha_score = calculate_alpha_score(
            role_score=role_score,
            debate_multiplier=debate_multiplier,
            cost_usd=total_cost,
            lookahead_penalty=lookahead_penalty
        )

        return {
            "role_score": role_score,
            "macro_score": macro_score,
            "fundamental_score": fundamental_score,
            "execution_score": execution_score,
            "debate_multiplier": debate_multiplier,
            "total_cost_usd": total_cost,
            "lookahead_penalty": lookahead_penalty,
            "alpha_score": alpha_score
        }
```

### 5.6 Phase 5: Detailed Reporting

```python
class EvaluationReporter:
    """
    Generates comprehensive evaluation reports.
    """
    def generate_report(self, task: dict, agent_response: dict,
                       evaluation: dict, debate: dict, tool_log: list) -> str:
        """
        Creates detailed markdown report.
        """
        report = f"""
# CIO-Agent Evaluation Report

## Task Information
- **Question ID**: {task['question_id']}
- **Category**: {task['category']}
- **Difficulty**: {task['difficulty']}
- **Question**: {task['question']}

## Agent Response Summary
- **Agent ID**: {agent_response['agent_id']}
- **Recommendation**: {agent_response['recommendation']}
- **Execution Time**: {agent_response['timestamp']}

## Evaluation Scores

### Role Score: {evaluation['role_score']:.2f}/100

| Dimension | Score | Weight | Contribution |
|-----------|-------|--------|--------------|
| Macro Analysis | {evaluation['macro_score']:.2f} | 30% | {evaluation['macro_score']*0.3:.2f} |
| Fundamental Accuracy | {evaluation['fundamental_score']:.2f} | 40% | {evaluation['fundamental_score']*0.4:.2f} |
| Execution Quality | {evaluation['execution_score']:.2f} | 30% | {evaluation['execution_score']*0.3:.2f} |

### Adversarial Debate

**Challenge**: {debate['counter_argument']}

**Agent Rebuttal**: {debate['agent_rebuttal']}

**Debate Multiplier**: {debate['debate_multiplier']}x ({debate['conviction_level']} conviction)

### Efficiency Metrics

- **Total Cost**: ${evaluation['total_cost_usd']:.4f}
- **Tool Calls**: {len(tool_log)}
- **Look-Ahead Penalty**: {evaluation['lookahead_penalty']:.2f}

### Final Alpha Score: {evaluation['alpha_score']:.2f}

## Detailed Analysis

### Tool Usage Log
```

        for call in tool_log:
            report += f"\n- {call['tool']} @ {call['timestamp']}: {call['params']}"

        report += f"""
```

### Ground Truth Comparison
{self._generate_comparison_table(task['ground_truth'], agent_response)}

---
*Generated by CIO-Agent FAB++ Evaluator*
"""
        return report
```

---

## 6. Implementation Roadmap

Aligned with the 6-week timeline from the FAB++ Protocol:

### Phase 1: Infrastructure & Tools (Weeks 1-2)

**Week 1: MCP Server Setup**
- [ ] Fork `stefanoamorelli/sec-edgar-mcp`
  - Implement `MeteredEDGARServer` with token tracking
  - Implement `NoiseInjectionMiddleware` for distractor documents
  - Implement `TemporalLockMiddleware` for filing date enforcement
  - Add comprehensive logging for all tool calls
  - Deploy to Docker container

- [ ] Fork `Alex2Yang97/yahoo-finance-mcp`
  - Implement `TimeMachineMiddleware` with simulation_date parameter
  - Implement `LookAheadDetector` for temporal violation tracking
  - Add historical data caching for performance
  - Deploy to Docker container

- [ ] Fork `JohanLi233/mcp-sandbox`
  - Pre-install financial libraries (numpy, pandas, scipy, yfinance, quantlib)
  - Configure timeout limits (30 seconds per execution)
  - Set up isolated execution environment
  - Deploy to Docker container

**Week 2: MCP Integration Testing**
- [ ] Test MCP Trinity end-to-end with sample queries
- [ ] Validate temporal locking across different simulation dates
- [ ] Verify noise injection correctly mixes distractors with official filings
- [ ] Benchmark tool response times and optimize caching
- [ ] Create MCP server documentation and API specs

### Phase 2: CIO-Agent Logic (Weeks 3-4)

**Week 3: Core Evaluation Pipeline**
- [ ] Implement `DynamicTaskGenerator`
  - Parse FAB dataset (537 questions)
  - Create ticker substitution logic with sector matching
  - Implement ground truth fetching via MCP tools
  - Test dynamic question generation for all 9 categories

- [ ] Implement `A2AOrchestrator`
  - Set up A2A Protocol message handlers
  - Create task assignment workflow
  - Implement response collection with timeout handling
  - Test communication with Purple agents via HTTP

- [ ] Implement `ToolUsageMonitor`
  - Create real-time tool call interception
  - Implement cost tracking for all tools
  - Add temporal violation detection hooks
  - Create tool usage logging system

**Week 4: Scoring & Debate**
- [ ] Implement Hierarchical Evaluators
  - `MacroEvaluator`: Semantic similarity + theme coverage
  - `FundamentalEvaluator`: XBRL data validation
  - `ExecutionEvaluator`: Rubric-based LLM-as-judge + code validation

- [ ] Implement `AdversarialDebateManager`
  - Create counter-argument generation prompts
  - Implement debate flow with A2A messaging
  - Build rebuttal quality scoring logic
  - Test debate multiplier calculations

- [ ] Implement `ComprehensiveEvaluator`
  - Integrate all scoring components
  - Implement Alpha Score calculation
  - Create evaluation result aggregation
  - Build detailed reporting system

### Phase 3: Calibration & Submission (Weeks 5-6)

**Week 5: Testing & Calibration**
- [ ] Create "Purple Agent" baseline solver
  - Implement basic FAB question answering
  - Use GPT-4o-mini for cost efficiency
  - Test against all 9 FAB categories

- [ ] Run Purple Agent through FAB++ gauntlet
  - Execute 50+ dynamic task variants
  - Collect performance metrics across categories
  - Identify common failure modes

- [ ] Tune Alpha Score weights
  - Analyze cost vs. accuracy tradeoffs
  - Adjust role score weights (Macro/Fundamental/Execution)
  - Calibrate debate multiplier thresholds
  - Validate that efficient models can compete with expensive ones

- [ ] Edge case testing
  - Test temporal violation handling
  - Test noise injection effectiveness
  - Test debate phase with various rebuttal qualities
  - Validate all error handling paths

**Week 6: Finalization & Submission**
- [ ] Create comprehensive documentation
  - Architecture overview
  - MCP tool usage guide
  - Evaluation methodology specification
  - Sample evaluation reports

- [ ] Prepare demo video materials
  - Record evaluation walkthrough
  - Show dynamic task generation
  - Demonstrate debate phase
  - Highlight Alpha Score calculation

- [ ] Containerize entire system
  - Create Docker Compose setup for MCP Trinity
  - Package CIO-Agent orchestrator
  - Create deployment scripts
  - Test full system in isolated environment

- [ ] Submit to AgentBeats
  - Upload Docker container to registry
  - Submit documentation and demo video
  - Provide sample evaluation reports
  - Complete competition registration

---

## 7. Competition Checklist Alignment

Mapping FAB++ design to AgentBeats evaluation criteria:

| AgentBeats Criterion | FAB++ Implementation | Competitive Advantage |
|---------------------|---------------------|----------------------|
| **Technical Correctness** | Rigorous multi-dimensional scoring (Macro/Fundamental/Execution) with ground truth validation via XBRL data from SEC EDGAR | Unlike binary pass/fail, we provide granular assessment that identifies specific failure modes |
| **Implementation Quality** | Production-grade MCP server architecture using battle-tested libraries (EdgarTools, yfinance). Full Docker containerization with comprehensive logging | Reproducible, maintainable, and extensible codebase following software engineering best practices |
| **Benchmark Design Quality** | Builds on FAB's 537 expert-authored questions but adds dynamic generation, preventing memorization. Covers all 9 financial task categories with balanced difficulty | Addresses "static benchmark problem" while maintaining FAB's domain expertise and comprehensive coverage |
| **Evaluation Methodology** | Novel "Alpha Score" combining accuracy, efficiency, and robustness. Adversarial debate tests agent conviction under critique | Incentivizes cost-effective solutions and penalizes brute-force approaches. Debate phase tests real-world scenario where analysts defend theses |
| **Innovation & Impact** | Three breakthrough features: (1) Time Travel environment preventing look-ahead bias, (2) Noise Injection testing signal-vs-noise filtering, (3) Hierarchical Role Assessment separating reasoning from execution | Directly addresses fundamental flaws in financial AI evaluation: temporal leakage, information overload, and lack of robustness testing |
| **Realism** | Uses real SEC EDGAR filings and live market data via MCP tools. Enforces temporal constraints matching real-world information availability | Evaluates agents under realistic conditions where historical analysis must respect information horizons |
| **Agentic Capabilities** | Explicitly tests tool use (via MCP logs), reasoning (via Macro Score), and adaptation (via Debate phase) | Goes beyond simple QA to test true agentic behavior: planning, tool selection, and dynamic response to challenges |
| **Reliability** | Leverages stable, open-source MCP servers (stefanoamorelli, Alex2Yang97, JohanLi233) with comprehensive error handling and timeout protection | Minimizes infrastructure risk by building on proven components. Extensive testing ensures consistent evaluation |

---

## 8. Technical Specifications

### 8.1 System Requirements

**Hardware**:
- CPU: 8+ cores (for parallel evaluation)
- RAM: 32GB minimum (for large filing processing)
- Storage: 500GB SSD (for SEC EDGAR filing cache)
- Network: Stable internet for MCP tool data fetching

**Software**:
- Docker Engine 24.0+
- Python 3.11+
- Node.js 18+ (for MCP server runtime)

### 8.2 MCP Server Endpoints

#### sec-edgar-mcp
```typescript
// Get company filings
GET /filings/{ticker}/{form_type}?date={yyyy-mm-dd}

// Get specific filing section
GET /filings/{ticker}/{form_type}/section/{section_name}?date={yyyy-mm-dd}

// Parse XBRL financials
GET /financials/{ticker}?filing_date={yyyy-mm-dd}&statement_type={BS|IS|CF}

// Search filings by keyword
POST /search
Body: {
  "ticker": "AAPL",
  "form_type": "10-K",
  "keywords": ["revenue", "risk factors"],
  "date_range": {"start": "2022-01-01", "end": "2023-12-31"}
}
```

#### yahoo-finance-mcp
```typescript
// Get historical prices
GET /price/{ticker}?start={yyyy-mm-dd}&end={yyyy-mm-dd}&simulation_date={yyyy-mm-dd}

// Get financial statements
GET /financials/{ticker}/{statement_type}?simulation_date={yyyy-mm-dd}

// Get key statistics
GET /statistics/{ticker}?simulation_date={yyyy-mm-dd}

// Get analyst estimates
GET /estimates/{ticker}?simulation_date={yyyy-mm-dd}
```

#### mcp-sandbox
```typescript
// Execute Python code
POST /execute
Body: {
  "code": "import pandas as pd\ndf = pd.DataFrame(...)",
  "timeout": 30,
  "libraries": ["pandas", "numpy", "scipy"]
}

// Get execution trace
GET /trace/{execution_id}
```

### 8.3 A2A Protocol Message Format

```json
{
  "protocol_version": "1.0",
  "message_type": "task_assignment | task_response | challenge | rebuttal",
  "sender_id": "cio-agent-green",
  "receiver_id": "white-agent-123",
  "timestamp": "2024-01-15T10:30:00Z",
  "payload": {
    "task_id": "FAB_001_variant_MSFT_2022",
    "question": "What was Microsoft's total revenue in FY 2022?",
    "category": "Quantitative Retrieval",
    "simulation_date": "2023-01-01",
    "available_tools": ["sec-edgar-mcp", "yahoo-finance-mcp", "mcp-sandbox"],
    "deadline": "2024-01-15T11:00:00Z"
  }
}
```

### 8.4 Evaluation Output Format

```json
{
  "evaluation_id": "eval_20240115_103045",
  "task_id": "FAB_001_variant_MSFT_2022",
  "agent_id": "white-agent-123",
  "timestamp": "2024-01-15T11:05:00Z",
  "scores": {
    "role_score": 85.3,
    "macro_score": 82.0,
    "fundamental_score": 90.0,
    "execution_score": 84.0,
    "debate_multiplier": 1.2,
    "debate_conviction": "high",
    "total_cost_usd": 2.34,
    "lookahead_penalty": 0.0,
    "alpha_score": 81.6
  },
  "debate": {
    "counter_argument": "Microsoft's cloud growth is decelerating...",
    "agent_rebuttal": "While Azure revenue growth slowed from 50% to 38%...",
    "rebuttal_quality": "strong"
  },
  "tool_usage": {
    "total_calls": 8,
    "sec_edgar_calls": 3,
    "yahoo_finance_calls": 4,
    "sandbox_executions": 1,
    "token_cost": 0.15,
    "temporal_violations": 0
  },
  "performance": {
    "execution_time_seconds": 145,
    "llm_calls": 12,
    "total_tokens": 45000
  }
}
```

---

## 9. Demo Video Script

**Duration**: 5 minutes
**Format**: Screen recording with voiceover

### Scene 1: Introduction (30 seconds)
**Visual**: Title slide with AgentBusters logo
**Voiceover**:
> "Welcome to the CIO-Agent demonstration. We're Team AgentBusters, and we've built FAB++: a dynamic, adversarial financial benchmark that transforms the static Finance Agent Benchmark into a realistic Chief Investment Officer evaluation environment."

### Scene 2: The Problem (45 seconds)
**Visual**: Show FAB paper statistics, highlight limitations
**Voiceover**:
> "The Finance Agent Benchmark is excellent content—537 expert questions across 9 categories. But it has a critical flaw: static questions with hardcoded answers lead to data contamination. Even top models like o3 only achieve 46% accuracy. We need dynamic evaluation."

### Scene 3: FAB++ Solution Overview (60 seconds)
**Visual**: Architecture diagram, highlight MCP Trinity
**Voiceover**:
> "FAB++ introduces three innovations. First, Dynamic Task Generation: we use FAB questions as templates but swap tickers and fetch live ground truth. Second, Time Travel: agents can't access future data—if they try, we catch them. Third, Adversarial Debate: our Green Agent challenges their thesis like a real Risk Manager would."

### Scene 4: Live Demonstration - Task Generation (45 seconds)
**Visual**: Screen recording of dynamic task generation
**Voiceover**:
> "Watch as we generate a dynamic variant. Starting with FAB's Apple revenue question, we substitute Microsoft, set simulation date to January 2023, and fetch ground truth from SEC EDGAR. The agent now gets a fresh question it couldn't have memorized."

### Scene 5: Live Demonstration - Evaluation (90 seconds)
**Visual**: Screen recording of full evaluation cycle
**Voiceover**:
> "Now let's evaluate an agent. The agent receives the task, uses our MCP tools to fetch Microsoft's 10-K, extracts revenue data, and provides an answer. Notice it uses the sandbox to calculate year-over-year growth—this is mandatory for numerical tasks.

> Now comes the debate phase. Our Green Agent challenges: 'Cloud growth is decelerating—defend your bullish thesis.' The agent rebuts with new evidence about margin expansion. Strong rebuttal—1.2x multiplier.

> Finally, scoring. Macro Score: 82 for identifying cloud trends. Fundamental Score: 90 for precise revenue extraction. Execution Score: 84 for proper methodology. Total cost: $2.34. No temporal violations. Final Alpha Score: 81.6—excellent."

### Scene 6: Competitive Advantages (30 seconds)
**Visual**: Comparison table with other benchmarks
**Voiceover**:
> "Why FAB++ wins: We prevent memorization through dynamic generation. We test robustness through adversarial debate. We enforce realism through temporal locking. And we incentivize efficiency through the Alpha Score formula."

### Scene 7: Closing (30 seconds)
**Visual**: GitHub repo link, team contact info
**Voiceover**:
> "FAB++ is production-ready, fully containerized, and built on stable open-source MCP servers. We're excited to provide the AgentBeats community with the definitive financial agent evaluator. Visit our GitHub for full documentation. Thank you."

---

## 10. References

### Core Papers
1. **Finance Agent Benchmark** (arXiv:2508.00828)
   *Source of 537 expert-authored questions, 9 task categories, and rubric-based evaluation methodology*

2. **TradingAgents** (arXiv:2412.20138)
   *Inspiration for hierarchical role assessment, adversarial debate, and noise injection concepts*

### MCP Servers (Open Source)
3. **sec-edgar-mcp**: `stefanoamorelli/sec-edgar-mcp`
   *SEC EDGAR filing retrieval and XBRL parsing*

4. **yahoo-finance-mcp**: `Alex2Yang97/yahoo-finance-mcp`
   *Historical market data and financial statements via yfinance*

5. **mcp-sandbox**: `JohanLi233/mcp-sandbox`
   *Isolated Python code execution environment*

### Competition Resources
6. **AgentBeats Competition**: https://rdi.berkeley.edu/agentx-agentbeats
   *Official competition page with judging criteria and submission guidelines*

7. **A2A Protocol Specification**
   *Agent-to-agent communication standard for message exchange*

---

## Appendix A: FAB Dataset Structure

The Finance Agent Benchmark dataset contains **537 questions** organized as follows:

| Category | Count | Avg Difficulty | Key Focus |
|----------|-------|----------------|-----------|
| Quantitative Retrieval | 89 | 2.1/5 | Extracting specific numbers from filings |
| Qualitative Retrieval | 76 | 2.3/5 | Extracting textual information |
| Numerical Reasoning | 112 | 3.4/5 | Calculations on retrieved data |
| Complex Retrieval | 45 | 3.8/5 | Multi-document information synthesis |
| Adjustments | 38 | 4.1/5 | Accounting adjustments and normalization |
| Beat or Miss | 52 | 2.9/5 | Comparing actuals vs. expectations |
| Trends | 61 | 3.2/5 | Multi-period pattern identification |
| Financial Modeling | 34 | 4.5/5 | Building valuation and projection models |
| Market Analysis | 30 | 4.3/5 | Macro + micro factor synthesis |

**Total**: 537 questions
**Average Difficulty**: 3.2/5
**Date Range**: 2019-2024 SEC filings

---

## Appendix B: Sample Evaluation Report

```markdown
# CIO-Agent Evaluation Report

## Task Information
- **Question ID**: FAB_032_variant_NVDA_2023
- **Category**: Numerical Reasoning
- **Difficulty**: 3.5/5
- **Question**: "Calculate NVIDIA's gross margin percentage for FY 2023 and compare it to FY 2022. Explain the trend."

## Agent Response Summary
- **Agent ID**: white-agent-gpt4o
- **Recommendation**: "NVDA's gross margin improved from 64.9% (FY22) to 72.7% (FY23), driven by data center revenue mix shift and H100 pricing power."
- **Execution Time**: 2024-01-15T10:45:23Z

## Evaluation Scores

### Role Score: 88.5/100

| Dimension | Score | Weight | Contribution |
|-----------|-------|--------|--------------|
| Macro Analysis | 85.0 | 30% | 25.5 |
| Fundamental Accuracy | 95.0 | 40% | 38.0 |
| Execution Quality | 83.3 | 30% | 25.0 |

**Macro Analysis (85.0)**: Agent correctly identified AI chip demand surge and data center mix shift as primary drivers. Missed mention of crypto mining revenue decline as offsetting factor.

**Fundamental Accuracy (95.0)**: Gross margin calculations are precise. FY22: $15.01B / $23.14B = 64.9% ✓. FY23: $18.07B / $24.85B = 72.7% ✓. Minor rounding difference (-0.1pp) from XBRL ground truth.

**Execution Quality (83.3)**: Used mcp-sandbox for calculation ✓. Clear methodology ✓. Could have provided more granular breakdown by segment.

### Adversarial Debate

**Challenge**: "NVIDIA's margin expansion appears unsustainable. H100 GPU pricing is at artificial highs due to supply constraints. Once TSMC ramps 3nm production and AMD MI300 gains traction, NVIDIA will face pricing pressure. Additionally, hyperscaler customers (MSFT, META, GOOGL) are developing in-house AI chips. Defend your thesis that this margin trend will persist."

**Agent Rebuttal**: "While competition is real, NVIDIA maintains structural advantages: (1) CUDA software moat—10M+ developers, switching cost is high, (2) NVLink interconnect technology outperforms AMD by 3x on multi-GPU workloads per MLPerf benchmarks, (3) Hyperscaler in-house chips (TPU, Trainium) complement rather than replace NVIDIA for general-purpose workloads. Management guided Q4 FY24 gross margin at 73%+, indicating pricing power persists despite supply normalization. MI300 benchmark data shows 20% performance gap vs. H100 on LLM training."

**Debate Multiplier**: 1.2x (high conviction)
**Reasoning**: Agent provided NEW evidence (MLPerf benchmarks, Q4 guidance, MI300 performance gap) beyond initial response. Demonstrated deep understanding of competitive dynamics.

### Efficiency Metrics

- **Total Cost**: $3.12
  - LLM Calls: $2.85 (GPT-4o)
  - Tool Usage: $0.27 (8 tool calls, 270K tokens)
- **Tool Calls**: 8
  - sec-edgar-mcp: 3 calls (10-K FY22, 10-K FY23, XBRL financials)
  - yahoo-finance-mcp: 4 calls (historical prices for context)
  - mcp-sandbox: 1 execution (gross margin calculation)
- **Look-Ahead Penalty**: 0.0 (no violations)

### Final Alpha Score: 74.2

```
Alpha Score = (88.5 * 1.2) / (ln(1 + 3.12) * 1.0)
            = 106.2 / (1.42 * 1.0)
            = 74.8
```

## Detailed Analysis

### Tool Usage Log
```
1. sec-edgar-mcp:get_filing(ticker='NVDA', form='10-K', fiscal_year=2023) @ 10:42:15
2. sec-edgar-mcp:get_filing(ticker='NVDA', form='10-K', fiscal_year=2022) @ 10:42:18
3. sec-edgar-mcp:parse_xbrl(ticker='NVDA', statement='IS', fiscal_year=2023) @ 10:42:45
4. yahoo-finance-mcp:get_statistics(ticker='NVDA', date='2023-01-29') @ 10:43:02
5. mcp-sandbox:execute(code='gross_margin_2023 = 18.07 / 24.85 ...') @ 10:43:30
6. yahoo-finance-mcp:get_price(ticker='NVDA', start='2023-01-01', end='2023-12-31') @ 10:44:10
7. sec-edgar-mcp:get_filing_section(ticker='NVDA', form='10-K', section='MD&A', year=2023) @ 10:44:45
8. yahoo-finance-mcp:get_estimates(ticker='NVDA', date='2024-01-15') @ 10:45:05
```

### Ground Truth Comparison

| Metric | Ground Truth (XBRL) | Agent Response | Accuracy |
|--------|---------------------|----------------|----------|
| FY23 Revenue | $24.85B | $24.85B | ✓ Exact |
| FY23 Gross Profit | $18.07B | $18.07B | ✓ Exact |
| FY23 Gross Margin | 72.7% | 72.7% | ✓ Exact |
| FY22 Revenue | $23.14B | $23.14B | ✓ Exact |
| FY22 Gross Profit | $15.01B | $15.01B | ✓ Exact |
| FY22 Gross Margin | 64.9% | 64.9% | ✓ Exact |
| YoY Margin Change | +7.8pp | +7.8pp | ✓ Exact |

### Code Execution Trace
```python
# Agent's calculation code (mcp-sandbox execution)
import pandas as pd

# FY 2023
revenue_2023 = 24.85  # billion USD
gross_profit_2023 = 18.07
gross_margin_2023 = (gross_profit_2023 / revenue_2023) * 100

# FY 2022
revenue_2022 = 23.14
gross_profit_2022 = 15.01
gross_margin_2022 = (gross_profit_2022 / revenue_2022) * 100

# YoY Change
margin_change = gross_margin_2023 - gross_margin_2022

print(f"FY23 Gross Margin: {gross_margin_2023:.1f}%")
print(f"FY22 Gross Margin: {gross_margin_2022:.1f}%")
print(f"YoY Change: +{margin_change:.1f} percentage points")

# Output:
# FY23 Gross Margin: 72.7%
# FY22 Gross Margin: 64.9%
# YoY Change: +7.8 percentage points
```

### Strengths
1. **Precise Data Extraction**: Perfect accuracy on all financial metrics
2. **Proper Methodology**: Used code execution for calculations, reducing hallucination risk
3. **Strong Debate Defense**: Provided new evidence (benchmarks, guidance) under challenge
4. **Comprehensive Analysis**: Identified both demand-side (AI) and supply-side (mix shift) drivers

### Areas for Improvement
1. **Macro Completeness**: Missed crypto revenue decline as offsetting factor
2. **Execution Depth**: Could have segmented margin analysis (Data Center vs Gaming vs Automotive)
3. **Cost Efficiency**: 8 tool calls slightly high for this task (optimal: 4-5)

### Recommendation
**Strong Performance**: This agent demonstrates high-quality financial analysis capabilities with precise data extraction, robust reasoning, and strong conviction under adversarial challenge. The Alpha Score of 74.2 places it in the top quartile of evaluated agents.

---
*Generated by CIO-Agent FAB++ Evaluator v1.0*
*Evaluation Time: 2024-01-15T10:45:23Z*
*Report ID: eval_20240115_104523*
```

---

## Appendix C: Deployment Architecture

### Docker Compose Configuration

```yaml
version: '3.8'

services:
  # MCP Server 1: SEC EDGAR
  sec-edgar-mcp:
    build: ./mcp-servers/sec-edgar-mcp
    container_name: fab-plus-edgar
    environment:
      - EDGAR_CACHE_DIR=/data/edgar_cache
      - TOKEN_TRACKING_ENABLED=true
      - NOISE_INJECTION_RATIO=0.3
      - TEMPORAL_LOCK_ENABLED=true
    volumes:
      - edgar-cache:/data/edgar_cache
    ports:
      - "8001:8000"
    restart: unless-stopped

  # MCP Server 2: Yahoo Finance
  yahoo-finance-mcp:
    build: ./mcp-servers/yahoo-finance-mcp
    container_name: fab-plus-yfinance
    environment:
      - YFINANCE_CACHE_DIR=/data/yf_cache
      - TIME_MACHINE_ENABLED=true
      - LOOKAHEAD_DETECTION=true
    volumes:
      - yfinance-cache:/data/yf_cache
    ports:
      - "8002:8000"
    restart: unless-stopped

  # MCP Server 3: Sandbox
  mcp-sandbox:
    build: ./mcp-servers/mcp-sandbox
    container_name: fab-plus-sandbox
    environment:
      - EXECUTION_TIMEOUT=30
      - PRELOAD_LIBS=numpy,pandas,scipy,yfinance,quantlib
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    ports:
      - "8003:8000"
    restart: unless-stopped

  # CIO-Agent Orchestrator
  cio-agent:
    build: ./cio-agent
    container_name: fab-plus-orchestrator
    environment:
      - MCP_EDGAR_URL=http://sec-edgar-mcp:8000
      - MCP_YFINANCE_URL=http://yahoo-finance-mcp:8000
      - MCP_SANDBOX_URL=http://mcp-sandbox:8000
      - A2A_PROTOCOL_VERSION=1.0
      - LLM_PROVIDER=openai  # or anthropic
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - sec-edgar-mcp
      - yahoo-finance-mcp
      - mcp-sandbox
    ports:
      - "8080:8080"
    volumes:
      - evaluation-results:/data/results
    restart: unless-stopped

  # PostgreSQL for evaluation storage
  postgres:
    image: postgres:15
    container_name: fab-plus-db
    environment:
      - POSTGRES_DB=fab_plus
      - POSTGRES_USER=fab_admin
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  edgar-cache:
  yfinance-cache:
  evaluation-results:
  postgres-data:
```

### Environment Variables

Create a `.env` file:
```bash
# LLM Provider
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=openai  # or anthropic

# Database
DB_PASSWORD=secure_password_here

# MCP Configuration
TEMPORAL_LOCK_ENABLED=true
NOISE_INJECTION_RATIO=0.3
TOKEN_TRACKING_ENABLED=true

# Evaluation Settings
SIMULATION_DATE=2023-01-01
DEBATE_ENABLED=true
ALPHA_SCORE_ENABLED=true
```

---

## Conclusion

FAB++ transforms the Finance Agent Benchmark from a static QA dataset into a **dynamic, adversarial Chief Investment Officer evaluation environment**. By introducing:

1. **Dynamic Task Generation** (preventing memorization)
2. **Time Travel Enforcement** (eliminating look-ahead bias)
3. **Adversarial Debate** (testing robustness and conviction)
4. **Hierarchical Role Assessment** (granular capability evaluation)
5. **Alpha Score Metric** (balancing accuracy, efficiency, and temporal integrity)

We directly address fundamental flaws in financial AI evaluation while maintaining full compatibility with FAB's comprehensive curriculum of 537 expert-authored questions across 9 task categories.

Our **"MCP Trinity"** architecture—sec-edgar-mcp, yahoo-finance-mcp, and mcp-sandbox—provides a realistic "Virtual Bloomberg Terminal" that enforces proper tool use, prevents data contamination, and enables reproducible evaluation.

FAB++ is production-ready, fully containerized, and positioned to become the definitive benchmark for financial agent capabilities in the AgentBeats competition and beyond.

---

**Team AgentBusters**
*Building the future of financial agent evaluation*

**Last Updated**: 2026-01-08
**Document Version**: 1.1
**Architecture Status**: Production Ready (A2A Server Implemented)
