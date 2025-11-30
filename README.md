# AgentBusters - CIO-Agent FAB++ System

A dynamic finance agent benchmark system for the [AgentBeats Competition](https://rdi.berkeley.edu/agentx-agentbeats). This project implements both **Green Agent** (Evaluator) and **Purple Agent** (Finance Analyst) using the A2A (Agent-to-Agent) protocol.

## Overview

The CIO-Agent FAB++ system evaluates AI agents on financial analysis tasks using:

- **FAB++ (Finance Agent Benchmark)**: Dynamic variant with 537 questions across 9 categories
- **MCP Trinity**: SEC EDGAR, Yahoo Finance, and Python Sandbox servers
- **Adversarial Debate**: Counter-argument generation to test conviction
- **Alpha Score**: Comprehensive evaluation metric

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AgentBusters System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐         A2A Protocol        ┌───────────┐ │
│  │   Green Agent   │◄──────────────────────────►│  Purple   │ │
│  │   (Evaluator)   │                             │   Agent   │ │
│  │   CIO-Agent     │                             │ (Analyst) │ │
│  └────────┬────────┘                             └─────┬─────┘ │
│           │                                            │       │
│           ▼                                            ▼       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    MCP Trinity                          │   │
│  │  ┌──────────┐   ┌──────────────┐   ┌──────────────┐    │   │
│  │  │  SEC     │   │   Yahoo      │   │   Python     │    │   │
│  │  │  EDGAR   │   │   Finance    │   │   Sandbox    │    │   │
│  │  │  MCP     │   │   MCP        │   │   MCP        │    │   │
│  │  └──────────┘   └──────────────┘   └──────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Docker (optional, for full stack deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/yxc20089/AgentBusters.git
cd AgentBusters

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Running the Green Agent (Evaluator)

```bash
# List available tasks
cio-agent list-tasks

# Run evaluation on a specific task
cio-agent evaluate --task-id FAB_001 --purple-endpoint http://localhost:8001

# Run the NVIDIA Q3 FY2026 test
python scripts/test_nvidia.py
```

### Running the Purple Agent (Finance Analyst)

```bash
# Start the A2A server
purple-agent serve --host 0.0.0.0 --port 8001

# Or run a direct analysis
purple-agent analyze "Did NVIDIA beat or miss Q3 FY2026 expectations?" --ticker NVDA

# Get stock information
purple-agent info NVDA

# Display the Agent Card
purple-agent card
```

## MCP Server Configuration

The Purple Agent connects to MCP servers for real financial data:

| Server | Default URL | Purpose |
|--------|-------------|---------|
| SEC EDGAR MCP | `http://localhost:8001` | SEC filings, XBRL data |
| Yahoo Finance MCP | `http://localhost:8002` | Market data, statistics |
| Sandbox MCP | `http://localhost:8003` | Python code execution |

Configure via environment variables:

```bash
export MCP_EDGAR_URL=http://localhost:8001
export MCP_YFINANCE_URL=http://localhost:8002
export MCP_SANDBOX_URL=http://localhost:8003
```

## Docker Deployment

MCP servers build from `src/mcp_servers/*.py` using the provided Dockerfiles. Use targeted commands instead of a blanket `docker-compose up`.

### Build images
```bash
docker compose build --no-cache sec-edgar-mcp yahoo-finance-mcp mcp-sandbox purple-agent
```

#### MCP endpoints and connectivity tips
- MCP servers run on HTTP with `/mcp` prefix: EDGAR `http://localhost:8101/mcp`, Yahoo `http://localhost:8102/mcp`, Sandbox `http://localhost:8103/mcp`.
- Root (`/`) and `/health` return 404 by design; use SSE ping instead:
  - Host: `curl -H "Accept: text/event-stream" -H "x-session-id: test" http://localhost:8101/mcp` (same for 8102/8103)
  - Inside container (avoids quoting issues): `docker exec -it fab-plus-edgar curl -H "Accept: text/event-stream" -H "x-session-id: test" http://localhost:8000/mcp`
- List tools via FastMCP inspect (inside containers):
  - `docker exec -it fab-plus-edgar fastmcp inspect "/app/src/mcp_servers/sec_edgar.py:create_edgar_server"`
  - `docker exec -it fab-plus-yfinance fastmcp inspect "/app/src/mcp_servers/yahoo_finance.py:create_yahoo_finance_server"`
  - `docker exec -it fab-plus-sandbox fastmcp inspect "/app/src/mcp_servers/sandbox.py:create_sandbox_server"`

### Run MCP + Purple together
```bash
docker compose up -d sec-edgar-mcp yahoo-finance-mcp mcp-sandbox purple-agent
docker compose logs -f purple-agent
```
External ports: 8101/8102/8103 -> 8000 inside. Internal URLs: `http://sec-edgar-mcp:8000`, `http://yahoo-finance-mcp:8000`, `http://mcp-sandbox:8000`.

### Green Agent (calling Purple)
- Purple must be running.
- From another container on the same network:
```bash
docker compose run --rm --no-deps cio-agent \
  cio-agent evaluate --task-id FAB_050 --date 2024-01-01 --output summary \
  --purple-endpoint http://fab-plus-purple-agent:8001
```
- From host to purple container: use `--purple-endpoint http://localhost:8001`.

### Batch Evaluation (run all CSV tasks headless)

Use the provided CSV dataset (mounted at `/app/data/public.csv`) to run a full, headless Green Agent evaluation and write aggregated results to a volume:

```bash
# Run full CSV batch evaluation and print summary (use root to ensure write access)
docker compose run --rm --user root cio-agent \
  sh -c "python -m scripts.run_csv_eval \
    --dataset-path /app/data/public.csv \
    --simulation-date 2024-12-31 \
    --difficulty medium \
    --output /data/results/summary.json \
    && cat /data/results/summary.json"
```

What it does:
- Loads all rows from the CSV (optional `--difficulty` filter, `--limit N`, `--seed` for reproducibility).
- Generates dynamic tasks via `DynamicTaskGenerator` (ticker/year substitutions with fixed seed if provided).
- Runs full evaluation with `ComprehensiveEvaluator` (debate optional via `--no-debate`).
- Records per-task Alpha Score/cost and writes an aggregated JSON summary to `/data/results/summary.json`.
- Continues on individual task errors and logs them instead of aborting.

Inspect outputs:
```bash
# Recommended one-shot (prints summary immediately):
docker compose run --rm --user root cio-agent \
  sh -c "python -m scripts.run_csv_eval --dataset-path /app/data/public.csv --simulation-date 2024-12-31 --difficulty medium --output /data/results/summary.json && cat /data/results/summary.json"

# Persist to host:
mkdir -p results
docker compose run --rm --user root -v ${PWD}/results:/data/results cio-agent \
  python -m scripts.run_csv_eval --dataset-path /app/data/public.csv --simulation-date 2024-12-31 --difficulty medium --output /data/results/summary.json
cat results/summary.json
```
Note: `docker compose run` creates a fresh container each time; `/tmp` is per-run. Reading `/data/results/summary.json` in a separate `run` only works if you wrote it to a persistent volume (e.g., the host bind above). If `/data/results` is not writable, use `--user root` or `--output /tmp/summary.json` within the same one-shot command.

Options:
- `--difficulty` can be repeated to filter (easy/medium/hard/expert).
- `--limit N` to cap number of rows.
- `--seed` to fix randomness (ticker/year substitution).
- `--no-debate` to skip debate phase.
- `--output` target JSON path; if `/data/results` not writable, use `--user root` or `--output /tmp/summary.json`.

Notes on fields:
- `debate_multiplier` comes from `EvaluationResult.debate_result`.
- `cost` comes from `EvaluationResult.cost_breakdown.total_cost_usd` (may be null if not available).

Notes:
- Dependencies (EDGAR/YFinance/Sandbox MCP) should be up via `docker compose up -d`.
- API keys are read from `.env` (e.g., `OPENAI_API_KEY`). Anthropic is optional.
- Purple baseline is exposed at `http://localhost:8010/health` (see MCP tips above).

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM | - |
| `ANTHROPIC_API_KEY` | Anthropic API key for LLM | - |
| `LLM_MODEL` | Model to use | `gpt-4o` |
| `SIMULATION_DATE` | Date for temporal locking (YYYY-MM-DD) | Current date |
| `MCP_EDGAR_URL` | SEC EDGAR MCP server URL | `http://localhost:8001` |
| `MCP_YFINANCE_URL` | Yahoo Finance MCP server URL | `http://localhost:8002` |
| `MCP_SANDBOX_URL` | Sandbox MCP server URL | `http://localhost:8003` |

### Purple Agent Options

```bash
# Run with simulation date (temporal locking)
purple-agent serve --simulation-date 2025-11-20

# Run without MCP (direct API access - for testing)
# Set USE_MCP=false in code or use direct APIs
```

## Project Structure

```
AgentBusters/
├── src/
│   ├── cio_agent/           # Green Agent (Evaluator)
│   │   ├── models.py        # Core data models
│   │   ├── evaluator.py     # Comprehensive evaluator
│   │   ├── debate.py        # Adversarial debate manager
│   │   ├── task_generator.py # Dynamic task generation
│   │   ├── orchestrator.py  # A2A orchestrator
│   │   └── cli.py           # CLI interface
│   │
│   ├── purple_agent/        # Purple Agent (Finance Analyst)
│   │   ├── agent.py         # Main agent class
│   │   ├── executor.py      # A2A executor implementation
│   │   ├── card.py          # Agent Card definition
│   │   ├── tools.py         # Direct API tools (fallback)
│   │   ├── mcp_tools.py     # MCP-based tools
│   │   ├── server.py        # A2A FastAPI server
│   │   └── cli.py           # CLI interface
│   │
│   ├── mcp_clients/         # MCP client wrappers
│   │   ├── edgar.py         # SEC EDGAR MCP client
│   │   ├── yahoo_finance.py # Yahoo Finance MCP client
│   │   └── sandbox.py       # Python Sandbox MCP client
│   │
│   ├── mcp_servers/         # Actual MCP servers (FastMCP)
│   │   ├── sec_edgar.py     # SEC EDGAR server (edgartools)
│   │   ├── yahoo_finance.py # Yahoo Finance server (yfinance)
│   │   └── sandbox.py       # Python execution sandbox
│   │
│   └── evaluators/          # Evaluation components
│       ├── macro.py         # Macro thesis evaluator
│       ├── fundamental.py   # Fundamental analysis evaluator
│       └── cost_tracker.py  # Cost tracking
│
├── tests/
│   ├── test_evaluator.py    # Unit tests
│   ├── test_e2e.py          # E2E tests with real NVIDIA data
│   └── test_purple_agent.py # Purple Agent tests
│
├── scripts/
│   ├── test_nvidia.py       # NVIDIA Q3 FY2026 demo
│   └── run_demo.py          # Full pipeline demo
│
├── docker-compose.yml       # Full stack deployment
├── Dockerfile               # Green Agent container
├── Dockerfile.purple        # Purple Agent container
└── pyproject.toml           # Project configuration
```

## Alpha Score Formula

The evaluation uses the Alpha Score metric:

```
Alpha Score = (RoleScore × DebateMultiplier) / (ln(1 + Cost) × (1 + LookaheadPenalty))
```

Where:
- **RoleScore**: Weighted combination of Macro (30%), Fundamental (40%), Execution (30%)
- **DebateMultiplier**: 0.5x - 1.2x based on conviction in adversarial debate
- **Cost**: Total USD cost of LLM and tool calls
- **LookaheadPenalty**: Penalty for temporal violations (accessing future data)

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_purple_agent.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## API Reference

### Purple Agent A2A Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/.well-known/agent.json` | GET | Agent Card (A2A discovery) |
| `/health` | GET | Health check |
| `/analyze` | POST | Direct analysis (non-A2A) |
| `/` | POST | A2A JSON-RPC endpoint |

### Agent Card Skills

1. **earnings_analysis** - Earnings beat/miss analysis
2. **sec_filing_analysis** - SEC 10-K, 10-Q analysis
3. **financial_ratio_calculation** - P/E, ROE, debt ratios
4. **market_analysis** - Sector and macro trends
5. **investment_recommendation** - Buy/hold/sell recommendations

## Competition Info

This project is built for the [AgentBeats Finance Track](https://rdi.berkeley.edu/agentx-agentbeats):

- **Phase 1** (Dec 2025): Green Agent submissions
- **Phase 2** (Feb 2026): Purple Agent submissions

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `python -m pytest tests/ -v`
4. Submit a pull request

## Acknowledgments

- [AgentBeats Competition](https://rdi.berkeley.edu/agentx-agentbeats) by Berkeley RDI
- [A2A Protocol](https://a2a-protocol.org/) by Google
- [FAB Benchmark](https://github.com/financial-agent-benchmark/FAB) for task templates
