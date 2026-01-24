<p align="center">
  <img src="docs/agentbusters-logo.jpg" alt="AgentBusters Logo" width="200"/>
</p>

# AgentBusters - CIO-Agent FAB++ System

A dynamic finance agent benchmark system for the [AgentBeats Competition](https://rdi.berkeley.edu/agentx-agentbeats). This project implements both **Green Agent** (Evaluator) and **Purple Agent** (Finance Analyst) using the A2A (Agent-to-Agent) protocol.

**[Technical Report (PDF)](paper/agentbusters.pdf)**

## 🚀 AgentBeats Platform Submission

This codebase is designed to work with the [AgentBeats platform](https://agentbeats.dev). The Green Agent follows the official [green-agent-template](https://github.com/RDI-Foundation/green-agent-template).

### Submission Readiness (Reproducibility & Transparency)

- **Deterministic configs**: use fixed `sampling.seed` in eval configs and avoid ad‑hoc overrides during runs.
- **LLM‑as‑judge determinism**: set `llm_eval.temperature: 0.0` and pin `llm_eval.model` (e.g., `gpt-4o-mini`) in configs such as `config/eval_all.yaml` and `config/eval_gdpval.yaml`.
- **Crypto reproducibility**: use anonymized hidden scenarios plus fixed seeds; for fully deterministic runs you can set `EVAL_SCENARIO_SEED`.
- **Transparent evaluation logic**: scoring lives in `src/evaluators/` and `src/cio_agent/unified_scoring.py`; crypto scoring in `src/cio_agent/crypto_benchmark.py`.

### End-to-End (AgentBeats) Flow

1. Build & push Green/Purple images to GHCR.
2. Register agents on AgentBeats and copy the agent IDs.
3. Update the leaderboard `scenario.toml` (set `EVAL_CONFIG=config/eval_all.yaml` and agent IDs).
4. Add secrets to the leaderboard repo (`OPENAI_API_KEY`, `EVAL_DATA_REPO`, `EVAL_DATA_PAT`, optional `HF_TOKEN`).
5. Push `scenario.toml` → workflow runs → merge PR to publish results.

### Resource Requirements

- **CPU/RAM**: 4 vCPU / 16 GB RAM recommended for multi‑dataset runs.
- **Storage**: local datasets + hidden crypto windows (size depends on your private repo).
- **Network**: required for HuggingFace datasets (BizFinBench/GDPVal) and LLM APIs.
- **LLM**: external API or local LLM; pin model + temperature for reproducibility.

### Submission Checklist

- [ ] `scenario.toml` has real AgentBeats IDs (no placeholders)
- [ ] `config/eval_all.yaml` (or your target config) uses fixed seeds + `llm_eval.temperature: 0.0`
- [ ] Hidden crypto data is private and only mounted into Green (not visible to Purple)
- [ ] README + DEPLOYMENT docs match the exact run steps
- [ ] End-to-end dry run completed from a clean clone

### Quick Start

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -e ".[dev]"

# Start Green Agent (A2A server)
# For all datasets: use config/eval_all.yaml
python src/cio_agent/a2a_server.py --host 0.0.0.0 --port 9109 --eval-config .\config\eval_all.yaml --store-predicted --predicted-max-chars 200

# Start Purple Agent
purple-agent serve --host 0.0.0.0 --port 9110 --card-url http://127.0.0.1:9110

# Start MCP server
python -m src.mcp_servers.sec_edgar --transport http --host 0.0.0.0 --port 8101
python -m src.mcp_servers.yahoo_finance --transport http --host 0.0.0.0 --port 8102
python -m src.mcp_servers.sandbox --transport http --host 0.0.0.0 --port 8103

# Run the evaluation
python scripts/run_a2a_eval.py --green-url http://127.0.0.1:9109 --purple-url http://127.0.0.1:9110 --num-tasks 25 -v -o results/eval_output.json
# gpt 4o total score: 42.44

# with debate
python scripts/run_a2a_eval.py --green-url http://127.0.0.1:9109 --purple-url http://127.0.0.1:9110 --num-tasks 25 --conduct-debate -v -o results/eval_output_debate.json
# gpt 4o total score: 43.65
```

## Prerequisites

- Python 3.13 (recommended for AgentBeats)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- vLLM, Ollama, or LM Studio (for local LLM deployment)

## Installation

```bash
# Clone the repository
git clone https://github.com/yxc20089/AgentBusters.git
cd AgentBusters

# Option 1: Using uv (recommended)
uv sync

# Option 2: Using pip
pip install -e ".[dev]"

# Option 3: Create .env file from template
cp .env.example .env
# Edit .env with your API keys and configuration
```

---

## Overview

The CIO-Agent FAB++ system evaluates AI agents on financial analysis tasks using a **unified scoring system** across three weighted sections:

| Section | Weight | Datasets | Skills Tested |
|---------|--------|----------|---------------|
| **Knowledge Retrieval** | 30% | BizFinBench, Public CSV | Data extraction, financial facts |
| **Analytical Reasoning** | 35% | Synthetic Questions | Logic puzzles, multi-step calculations |
| **Options Trading** | 35% | Options Alpha | Derivatives, Greeks, strategies |

### Final Score Calculation

```
OverallScore = 0.30 × Knowledge + 0.35 × Analysis + 0.35 × Options
```

All section scores are normalized to 0-100 scale. Example: Knowledge (83.33) + Analysis (50.00) + Options (51.25) → **Overall: 60.44/100**

### Benchmark Datasets

1. **BizFinBench v2** (Knowledge): Event logic reasoning, quantitative computation
2. **Public CSV** (Knowledge): Beat/miss analysis, market analysis from FAB benchmark
3. **Synthetic Questions** (Analysis): 20 olympiad-style finance logic problems covering:
   - Capital budgeting (NPV, IRR)
   - Portfolio theory (beta, leverage)
   - Fixed income (duration, immunization)
   - Corporate finance (FCFF, M&M)
   - Options & derivatives (put-call parity, swaps)
4. **Options Alpha** (Options): Greeks analysis, strategy construction, P&L analysis
5. **Crypto Trading Scenarios** (Optional): Multi-round trading evaluation on market states
6. **GDPVal** (Optional): Open‑ended professional tasks scored by LLM‑as‑judge

### Key Features

- **Unified Scoring**: All evaluators normalized to 0-100, weighted by section
- **MCP Servers**: 6 servers for financial data, options pricing, and trading simulation
- **Options Alpha Challenge**: Black-Scholes pricing, Greeks analysis, multi-leg strategies
- **Adversarial Debate**: Optional counter-argument generation to test conviction
- **Dynamic Weight Redistribution**: When sections are disabled, weights redistribute proportionally

## Crypto Trading Benchmark (Optional Track)

The repo includes an optional crypto trading benchmark that evaluates
multi-round trading decisions on historical scenarios (baseline, noisy,
adversarial, meta-consistency). Use `config/eval_crypto.yaml` to run it
and see `docs/CRYPTO_BENCHMARK.md` for data format and integration
details.

> Do not commit hidden seeds or evaluation data. Keep `~/.agentbusters/hidden_seeds.yaml` and `data/crypto/hidden/` private.

### Anti-Overfitting Design

The crypto benchmark implements a **Hidden Windows** strategy to prevent overfitting:

| Protection Layer | Mechanism |
|------------------|-----------|
| **Private Seed** | Master seed stored in `~/.agentbusters/` (not in repo) |
| **Dynamic Selection** | Evaluation windows selected deterministically from seed |
| **Anonymous IDs** | Scenario IDs are SHA256 hashes (cannot reverse to timestamps) |
| **Quarterly Rotation** | Seeds refreshed periodically to prevent long-term optimization |

For production deployment with PostgreSQL and hidden windows, see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AgentBusters System                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐           A2A Protocol          ┌───────────────┐      │
│  │   Green Agent   │◄──────────────────────────────► │  Purple Agent │      │
│  │   (Evaluator)   │                                 │   (Analyst)   │      │
│  │   Port: 9109    │                                 │  Port: 9110   │      │
│  └───────┬─────────┘                                 └───────┬───────┘      │
│          │                                                   │              │
│          │  ┌──────────────────────────────────────────────┐ │              │
│          │  │              6 MCP Servers                   │ │              │
│          │  ├──────────────────────────────────────────────┤ │              │
│          │  │  ┌────────────┐ ┌────────────┐ ┌──────────┐  │ │              │
│          │  │  │ SEC EDGAR  │ │  Yahoo     │ │  Python  │  │ │              │
│          │  │  │  :8101     │ │ Finance    │ │ Sandbox  │  │ │              │
│          │  │  │            │ │  :8102     │ │  :8103   │  │ │              │
│          │  │  └────────────┘ └────────────┘ └──────────┘  │ │              │
│          │  │  ┌────────────┐ ┌────────────┐ ┌──────────┐  │ │              │
│          │  │  │  Options   │ │  Trading   │ │   Risk   │  │ │              │
│          │  │  │   Chain    │ │    Sim     │ │ Metrics  │  │ │              │
│          │  │  │  :8104     │ │  :8105     │ │  :8106   │  │ │              │
│          │  │  └────────────┘ └────────────┘ └──────────┘  │ │              │
│          │  └──────────────────────────────────────────────┘ │              │
│          │                                                   │              │
│          ▼                                                   ▼              │
│  ┌───────────────┐                               ┌───────────────┐          │
│  │    SQLite     │                               │    SQLite     │          │
│  │   tasks.db    │                               │ purple_tasks  │          │
│  └───────────────┘                               └───────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### One-Page Quick Start (Full Stack: 5 Terminals + Tests)

Use these exact commands to run the whole stack locally with openai/gpt-oss-20b. Each terminal runs one long-lived process; keep them open.

```bash
# (Optional) Terminal 1 — Local LLM (vLLM: openai/gpt-oss-20b)
# conda activate /chronos_data/conda_envs/py313
# Install vLLM
# pip install vllm

#export LIBRARY_PATH="/chronos_data/huixu/libcuda_stub:$LIBRARY_PATH"
#export LD_LIBRARY_PATH="/chronos_data/huixu/libcuda_stub:$LD_LIBRARY_PATH"
vllm serve openai/gpt-oss-20b --port 8000
# For multi-GPU: add --tensor-parallel-size=2

You can also set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `.env`.

# Terminal 2–4 — MCP Servers (OPTIONAL - can skip these!)
# Purple Agent can run MCP servers in-process (no external servers needed).
# Only start these if you want separate processes for debugging or multi-agent scenarios.

# Option A: Skip Terminals 2-4 entirely (recommended, uses in-process MCP)
#   → Just comment out MCP_*_URL in .env

# Option B: Run external MCP servers (for debugging/multi-agent)
# Terminal 2 — SEC EDGAR MCP
# python -m src.mcp_servers.sec_edgar --transport http --host 0.0.0.0 --port 8101

# # Terminal 3 — Yahoo Finance MCP
# python -m src.mcp_servers.yahoo_finance --transport http --host 0.0.0.0 --port 8102

# # Terminal 4 — Sandbox MCP
# python -m src.mcp_servers.sandbox --transport http --host 0.0.0.0 --port 8103

# Terminal 5 — Purple Agent (Finance Analyst, A2A server for AgentBeats)
# Recommended: Production-grade A2A server with full LLM support
purple-agent serve --host 0.0.0.0 --port 9110 --card-url http://localhost:9110

# Alternatively: Simple test agent (minimal A2A + REST)
# python src/simple_purple_agent.py --host 0.0.0.0 --port 9110 --card-url http://localhost:9110

# Quick one-off analysis (no server needed)
# purple-agent analyze "Did NVIDIA beat or miss Q3 FY2026 expectations?" --ticker NVDA

# Terminal 6 — Green Agent (Evaluator, A2A server)
# No CLI wrapper for serve command—start the server directly
# If you use gpt-oss-20b, set 
#```
# llm_eval:
#   enabled: true
#   model: "openai/gpt-oss-20b"
#   temperature: 0.0
#```
# If you use local dataset, update 
# - type: crypto
#     path: data/crypto/scenarios/sample_btc_window
# in `eval_all.yaml`
python src/cio_agent/a2a_server.py --host 0.0.0.0 --port 9109 --eval-config config/eval_all.yaml

# Terminal 7 Run Evaluation:
python scripts/run_a2a_eval.py --green-url http://localhost:9109 --purple-url http://localhost:9110 --num-tasks 1 --timeout 300 -v
```

#### More Useful Commands (Optional)

```bash
################################################################################
# 1. QUICK START - Most Common Commands
################################################################################
# Quick smoke checks (discovery/health)
curl http://localhost:9109/.well-known/agent.json   # Green agent card
curl http://localhost:9110/health                   # Purple agent health

# Tests and end-to-end run
# Run all tests
python -m pytest tests/ -v

# Run A2A conformance tests
python -m pytest tests/test_a2a_green.py -v --agent-url http://localhost:9109

# Run A2A tests with synthetic questions (integration test)
python -m pytest tests/test_a2a_green.py::test_synthetic_questions_evaluation -v \
    --agent-url http://localhost:9109 --purple-url http://localhost:9110

# Run synthetic question unit tests (no server required)
python -m pytest tests/test_synthetic.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Trigger a manual evaluation (Green → Purple via A2A)
# List available tasks
cio-agent list-tasks

# Run evaluation on a specific task
cio-agent evaluate --task-id FAB_001 --purple-endpoint http://localhost:9110

# Demo: NVIDIA Q3 FY2026 evaluation
python scripts/run_demo.py
# Optional: override Purple endpoint
# PURPLE_ENDPOINT=http://localhost:9110 python scripts/run_demo.py

# Purple Agent utilities
purple-agent info NVDA                    # Pulls quote/statistics/SEC snapshot via MCP
purple-agent card                         # Prints the Purple Agent Card JSON

# Green Evaluator power tools
cio-agent list-tasks                      # View all FAB++ templates
cio-agent lake-status                     # Check Financial Lake cache status

################################################################################
# 2. RUN EVALUATION (recommended workflow)
################################################################################

# Step 1: Start Green Agent A2A Server (choose one):

# RECOMMENDED: Multi-dataset config (production-ready)
python src/cio_agent/a2a_server.py --host 0.0.0.0 --port 9109 \
    --eval-config config/eval_quick.yaml   # Quick test (10 examples)
# Or:
#   --eval-config config/eval_full.yaml    # Full evaluation (100+ examples)

# Step 2: Trigger evaluation
python scripts/run_a2a_eval.py --num-tasks 5 -v

# With custom options:
python scripts/run_a2a_eval.py \
    --green-url http://localhost:9109 \
    --purple-url http://localhost:9110 \
    --num-tasks 100 \
    --conduct-debate \
    -o results/eval_output.json

################################################################################
# 3. EVALUATION RESULTS STORAGE
################################################################################

# Results are stored in TWO places:

# 1. SQLite Database (persistent, auto-created)
#    File: tasks.db
#    Contains: task status, context_id, artifacts (full evaluation results)
#    Note: predicted/predicted_full are empty unless you start Green with --store-predicted

# 2. JSON file (optional, via -o flag)
python scripts/run_a2a_eval.py --num-tasks 10 -o results/eval_output.json

# View stored results from database:
sqlite3 tasks.db "SELECT artifacts FROM tasks ORDER BY id DESC LIMIT 1;" | python3 -m json.tool

# Query task status by ID:
curl -X POST http://localhost:9109/ -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tasks/get","id":"q1","params":{"id":"TASK_ID"}}'

# Reset database (clear all history):
rm tasks.db

# Result storage comparison:
# ┌─────────────────────────────┬──────────────────────────────────────────┐
# │ Method                      │ Results Storage                          │
# ├─────────────────────────────┼──────────────────────────────────────────┤
# │ evaluate-synthetic --output │ Saved to JSON file (persistent)          │
# │ A2A Server                  │ SQLite Database (persistent in tasks.db) │
# └─────────────────────────────┴──────────────────────────────────────────┘

################################################################################
# 4. GENERATE SYNTHETIC DATA (optional)
################################################################################

# Financial Lake + Synthetic benchmark (requires ALPHAVANTAGE_API_KEY)
# Rate limiting: Free tier allows 5 calls/min, 25 calls/day
# Each ticker needs 5 API calls, so harvest 1 ticker at a time

cio-agent harvest --tickers NVDA         # ~1.5 min per ticker
cio-agent harvest --tickers AAPL         # Run after first completes

# Generate synthetic questions from Financial Lake data
cio-agent generate-synthetic -n 10 -o data/synthetic_questions/questions.json
cio-agent verify-questions data/synthetic_questions/questions.json -o /tmp/verify.json

# Troubleshooting: If cache files are empty, delete and re-harvest
# rm data/alphavantage_cache/AAPL_EARNINGS.json  # Delete empty file
# cio-agent harvest --tickers AAPL --force       # Force re-fetch

################################################################################
# 5. ALTERNATIVE: Local Testing (no A2A server needed)
################################################################################

# For quick local testing, use evaluate-synthetic (simpler, faster):
cio-agent evaluate-synthetic data/synthetic_questions/questions.json \
    --purple-endpoint http://localhost:9110 \
    --output data/synthetic_questions/results.json \
    --limit 5 --no-debate

# This directly calls Purple Agent's /analyze endpoint, no A2A server needed
# Recommendation: Use this with --output for local dev

################################################################################
# 6. ARCHITECTURE: Local Dev vs AgentBeats Evaluation
################################################################################

# Option A: Local Testing (evaluate-synthetic uses HTTP REST, faster)
# ┌─────────────────────┐    HTTP POST /analyze    ┌───────────┐
# │   cio-agent CLI     │─────────────────────────►│  Purple   │
# │   evaluate-synthetic│◄─────────────────────────│   Agent   │
# └─────────────────────┘                          └───────────┘
# 
# Option B: AgentBeats Evaluation (uses full A2A Protocol)
#                     ┌─────────────────────────┐
#                     │  AgentBeats Platform    │
#                     │  (or curl test request) │
#                     └───────────┬─────────────┘
#                                 │ A2A JSON-RPC
#                                 ▼
# ┌─────────────────────────────────────────────────────────────┐
# │  Green Agent A2A Server (:9109)                             │◄──┐
# │  --eval-config config/eval_*.yaml                           │   │
# │  (Loads datasets from config file)                          │   │
# └───────────────────────────┬─────────────────────────────────┘   │
#                             │ Evaluates Purple Agent              │
#                             ▼                                     │
#                     ┌───────────────────┐                         │
#                     │  Purple Agent     │                  ┌────────────┐
#                     │  (:9110)          │                  │  SQLite    │
#                     └───────────────────┘                  │ (tasks.db) │
#                                                            └────────────┘

# Quick testing recommendation:
# ┌────────────────────────────────────────────────────────────────────────────┐
# │ Method                      │ Use Case           │ Protocol  │ Speed      │
# ├────────────────────────────────────────────────────────────────────────────┤
# │ cio-agent evaluate-synthetic│ Local dev testing  │ HTTP REST │ Fast       │
# │ A2A Server + run_a2a_eval   │ AgentBeats official│ A2A JSON-RPC│ Full stack│
# └────────────────────────────────────────────────────────────────────────────┘

################################################################################
# 7. ADVANCED: Raw Curl Testing & Legacy Mode
################################################################################

# Test A2A evaluation with curl (alternative to run_a2a_eval.py script):
curl -X POST http://localhost:9109/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "id": "test-'$(date +%s)'",
    "params": {
      "message": {
        "messageId": "'$(uuidgen || cat /proc/sys/kernel/random/uuid)'",
        "role": "user",
        "parts": [{"type": "text", "text": "{\"participants\": {\"purple_agent\": \"http://localhost:9110\"}, \"config\": {\"num_tasks\": 1}}"}]
      }
    }
  }'

# NOTE: A2A SDK tracks tasks by session context. 
# Error "Task already in terminal state" means the task ID was reused.
# Solution: Use dynamic UUID (shown above), or rm tasks.db to reset.

# Legacy: Single dataset mode (use --eval-config instead):
# python src/cio_agent/a2a_server.py --host 0.0.0.0 --port 9109 \
#     --dataset-type bizfinbench --dataset-path data/BizFinBench.v2 \
#     --task-type event_logic_reasoning --limit 10

# Config file example (config/eval_full.yaml):
# ---
# name: "FAB++ Full Evaluation"
# datasets:
#   - type: synthetic
#     path: data/synthetic_questions/questions.json
#     limit: 10
#   - type: bizfinbench
#     path: data/BizFinBench.v2
#     task_types: [event_logic_reasoning, user_sentiment_analysis]
#     languages: [en, cn]
#     limit_per_task: 20
#   - type: public_csv
#     path: finance-agent/data/public.csv
#     limit: 100
# sampling:
#   strategy: stratified  # Options: sequential, random, stratified, weighted
#   total_limit: 100
#   seed: 42
llm_eval:
  enabled: true
  model: gpt-4o-mini
  temperature: 0.0


# MCP helpers and CSV batch eval
# Note: start_mcp_servers.py uses stdio transport by default (for local dev or MCP Inspector testing)
# For network HTTP (used in Quick Start above), add --transport http: python -m src.mcp_servers.XXXX --transport http --host 0.0.0.0 --port PORT
python scripts/start_mcp_servers.py --server edgar      # Stdio/SSE transport mode (dev only)
python scripts/test_mcp_live.py                         # Smoke test MCP servers
python -m scripts.run_csv_eval \
	--dataset-path finance-agent/data/public.csv \
	--purple-endpoint http://localhost:9110 \
	--output /tmp/summary.json --no-debate --limit 5

# BizFinBench.v2 evaluation (29,578 Q&A pairs across 9 task types)
# English (8 tasks): anomaly_information_tracing, conterfactual, event_logic_reasoning,
#   financial_data_description, financial_multi_turn_perception, financial_quantitative_computation,
#   stock_price_predict, user_sentiment_analysis
# Chinese (9 tasks): all above + financial_report_analysis
python -m scripts.run_bizfin_eval \
	--dataset-path data/BizFinBench.v2 \
	--task-type event_logic_reasoning \
	--language en \
	--purple-endpoint http://localhost:9110 \
	--output /tmp/bizfin_summary.json --limit 10

# List task types by language:
python -c "from cio_agent.local_datasets import BizFinBenchProvider; print(BizFinBenchProvider.list_task_types_by_language())"

# Dataset-specific evaluators (exact-match scoring by default, optional LLM grading):
# BizFinBench: numerical matching (+/-1% tolerance), sequence matching, classification
python -m scripts.run_bizfin_simple \
	--dataset-path data/BizFinBench.v2 \
	--task-type financial_quantitative_computation \
	--language en \
	--purple-endpoint http://localhost:9110 \
	--output /tmp/bizfin_results.json --limit 5
# Optional: --eval-llm --eval-llm-model gpt-4o-mini

# public.csv: correctness/contradiction rubric evaluation
python -m scripts.run_csv_simple \
	--dataset-path finance-agent/data/public.csv \
	--purple-endpoint http://localhost:9110 \
	--output /tmp/csv_results.json --limit 5
# Optional: --eval-llm --eval-llm-model gpt-4o-mini


# Alternative direct startup (stdio by default)
# Default: stdio transport (not accessible via HTTP). Add --transport http for network access.
python src/mcp_servers/sec_edgar.py                                      # Stdio only
python src/mcp_servers/sec_edgar.py --transport http --port 8101         # HTTP on :8101
python src/mcp_servers/yahoo_finance.py --transport http --port 8102     # HTTP on :8102
python src/mcp_servers/sandbox.py --transport http --port 8103           # HTTP on :8103

# Purple Agent startup methods (all use HTTP/Uvicorn, differ in features):
python src/simple_purple_agent.py --host 0.0.0.0 --port 9110     # Minimal A2A + REST test agent
python src/purple_agent/server.py           # Full A2A server (read .env for LLM config)
purple-agent serve --host 0.0.0.0 --port 9110                   # CLI wrapper for src/purple_agent/server.py
```

Tip: Using hosted APIs instead of local vLLM? You can skip Terminal 1 and just configure your `.env`:

```dotenv
# OpenAI (skip Terminal 1)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-REDACTED
LLM_MODEL=gpt-4o
# Do not set OPENAI_API_BASE or OPENAI_BASE_URL when using OpenAI's hosted API
```

```dotenv
# Anthropic (skip Terminal 1)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-REDACTED
LLM_MODEL=claude-3.5-sonnet
```

Tip: For vLLM-backed LLM calls, set these in `.env` (auto-loaded):

```dotenv
LLM_PROVIDER=openai
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_BASE_URL=http://localhost:8000/v1  # alias for OPENAI_API_BASE
OPENAI_API_KEY=dummy
LLM_MODEL=openai/gpt-oss-20b
```

## MCP Server Configuration

The system uses 6 MCP servers for financial data and options trading:

| Server | Port | Purpose |
|--------|------|---------|
| SEC EDGAR MCP | 8101 | SEC filings, XBRL data, temporal locking |
| Yahoo Finance MCP | 8102 | Market data, statistics, lookahead detection |
| Sandbox MCP | 8103 | Python code execution |
| **Options Chain MCP** | 8104 | Black-Scholes pricing, Greeks, IV surface |
| **Trading Sim MCP** | 8105 | Paper trading, slippage simulation, P&L |
| **Risk Metrics MCP** | 8106 | VaR, Sharpe/Sortino, stress testing |

Configure via environment variables or `.env` file:

```dotenv
# Core MCP Servers
MCP_EDGAR_URL=http://localhost:8101
MCP_YFINANCE_URL=http://localhost:8102
MCP_SANDBOX_URL=http://localhost:8103

# Options Alpha MCP Servers
MCP_OPTIONS_URL=http://localhost:8104
MCP_TRADING_URL=http://localhost:8105
MCP_RISK_URL=http://localhost:8106
```

Tip: If MCP URLs are unset, the Purple Agent falls back to in-process MCP servers.

## Docker Deployment

### Published Docker Images

| Image | URL |
|-------|-----|
| Green Agent | `ghcr.io/yxc20089/agentbusters-green:latest` |
| Purple Agent | `ghcr.io/yxc20089/agentbusters-purple:latest` |

### Quick Start with Docker

```bash
# Pull and run Green Agent
docker pull ghcr.io/yxc20089/agentbusters-green:latest
docker run -p 9109:9109 ghcr.io/yxc20089/agentbusters-green:latest --host 0.0.0.0

# Pull and run Purple Agent
docker pull ghcr.io/yxc20089/agentbusters-purple:latest
docker run -p 9110:9110 -e OPENAI_API_KEY=sk-xxx ghcr.io/yxc20089/agentbusters-purple:latest
```

### Full Stack with Docker Compose

```bash
# Start all services (6 MCP servers + Green + Purple agents)
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f green-agent purple-agent
```

### Building from Source

```bash
# Build Green Agent
docker build -f Dockerfile.green -t cio-agent-green .
docker run -p 9109:9109 cio-agent-green --host 0.0.0.0

# Build Purple Agent
docker build -f Dockerfile.purple -t purple-agent .
docker run -p 9110:9110 -e OPENAI_API_KEY=sk-xxx purple-agent
```

### Individual Service Build & Run

```bash
# Green Agent
docker build -f Dockerfile -t cio-agent-green .
docker run -p 9109:9109 cio-agent-green

# Purple Agent
docker build -f Dockerfile.purple -t purple-agent .
docker run -p 9110:9110 purple-agent

# MCP Servers
docker build -f Dockerfile.mcp-edgar -t mcp-edgar .
docker run -p 8101:8000 mcp-edgar

docker build -f Dockerfile.mcp-yahoo -t mcp-yahoo .
docker run -p 8102:8000 mcp-yahoo

docker build -f Dockerfile.mcp-sandbox -t mcp-sandbox .
docker run -p 8103:8000 mcp-sandbox
```

Port mapping: Green Agent `9109`, Purple Agent `9110`, EDGAR `8101`, YFinance `8102`, Sandbox `8103`.

## Configuration

### Environment Setup with `.env` File

```bash
# 1. Create .env from template
cp .env.example .env

# 2. Edit .env with your LLM configuration
```

**For local vLLM (openai/gpt-oss-20b):**
```dotenv
LLM_PROVIDER=openai
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_API_KEY=dummy
LLM_MODEL=openai/gpt-oss-20b
```

**For OpenAI API:**
```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-REDACTED
LLM_MODEL=gpt-4o
```

**For Anthropic API:**
```dotenv
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-REDACTED
LLM_MODEL=claude-3.5-sonnet
```

**MCP Servers (optional):**
```dotenv
MCP_EDGAR_URL=http://localhost:8101
MCP_YFINANCE_URL=http://localhost:8102
MCP_SANDBOX_URL=http://localhost:8103
```

**LLM grading for dataset evaluators (bizfinbench/public_csv):**
```dotenv
EVAL_USE_LLM=true
EVAL_LLM_MODEL=gpt-4o-mini
EVAL_LLM_TEMPERATURE=0.0
```
Uses `OPENAI_API_KEY` + `OPENAI_BASE_URL`/`OPENAI_API_BASE` (OpenAI-compatible) or `ANTHROPIC_API_KEY`.

CLI override example:
```bash
python src/cio_agent/a2a_server.py --host 0.0.0.0 --port 9109 \
  --eval-llm --eval-llm-model gpt-4o-mini --eval-llm-temperature 0.0
```

**Predicted output storage (recommended for memory control):**
```bash
python src/cio_agent/a2a_server.py --host 0.0.0.0 --port 9109 \
  --eval-config config/eval_config.yaml \
  --store-predicted --predicted-max-chars 200
```
By default, predicted outputs are omitted from results (fields are empty).  
Use `--store-predicted` to include them, and `--no-truncate-predicted` to keep full outputs.

The agents will automatically load `.env` on startup. Alternatively, you can use `export` commands instead of `.env` file.

### Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider | `openai`, `anthropic` |
| `LLM_MODEL` | Model name | `gpt-4o`, `claude-3.5-sonnet`, `openai/gpt-oss-20b` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `OPENAI_API_BASE` | Custom API endpoint (for local vLLM) | `http://localhost:8000/v1` |
| `OPENAI_BASE_URL` | Alias for `OPENAI_API_BASE` | `http://localhost:8000/v1` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `EVAL_USE_LLM` | Enable LLM grading for dataset evaluators | `true` |
| `EVAL_LLM_MODEL` | Model override for LLM grading | `gpt-4o-mini` |
| `EVAL_LLM_TEMPERATURE` | Temperature for LLM grading | `0.0` |
| `MCP_EDGAR_URL` | SEC EDGAR MCP server | `http://localhost:8101` |
| `MCP_YFINANCE_URL` | Yahoo Finance MCP server | `http://localhost:8102` |
| `MCP_SANDBOX_URL` | Sandbox MCP server | `http://localhost:8103` |
| `DATABASE_URL` | SQLite database URL (Green Agent) | `sqlite+aiosqlite:///tasks.db` |
| `PURPLE_DATABASE_URL` | SQLite database URL (Purple Agent) | `sqlite+aiosqlite:///purple_tasks.db` |

### Database Maintenance

The Green Agent uses SQLite for persistent task storage. The database file (`tasks.db`) is created automatically on first use.

**Backup:**
```bash
# Simple file copy (stop server first for consistency)
cp tasks.db tasks.db.backup

# Or with timestamp
cp tasks.db "tasks_$(date +%Y%m%d_%H%M%S).db"
```

**Reset database:**
```bash
# Delete to start fresh (all task history will be lost)
rm tasks.db
```

**Migrations:** The A2A SDK handles schema internally. If you encounter schema errors after upgrading `a2a-sdk`, delete `tasks.db` to regenerate with the new schema.

**Troubleshooting:**
- "Database is locked" → Ensure only one server instance is running
- "Disk I/O error" → Check disk space and file permissions


## Project Structure

```
AgentBusters/
├── src/
│   ├── cio_agent/           # Green Agent (Evaluator)
│   │   ├── a2a_server.py    # A2A server entry point (AgentBeats)
│   │   ├── green_executor.py # A2A protocol executor
│   │   ├── green_agent.py   # FAB++ evaluation logic
│   │   ├── messenger.py     # A2A messaging utilities
│   │   ├── models.py        # Core data models (18 TaskCategories)
│   │   ├── evaluator.py     # Comprehensive evaluator
│   │   ├── debate.py        # Adversarial debate manager
│   │   ├── task_generator.py # Dynamic task generation (18 templates)
│   │   └── cli.py           # CLI interface
│   │
│   ├── purple_agent/        # Purple Agent (Finance Analyst)
│   │   ├── server.py        # A2A FastAPI server
│   │   ├── executor.py      # A2A executor (options support)
│   │   ├── mcp_toolkit.py   # MCP client toolkit (21 methods)
│   │   └── cli.py           # CLI interface
│   │
│   ├── mcp_servers/         # MCP servers (FastMCP)
│   │   ├── sec_edgar.py     # SEC EDGAR server (:8101)
│   │   ├── yahoo_finance.py # Yahoo Finance server (:8102)
│   │   ├── sandbox.py       # Python execution sandbox (:8103)
│   │   ├── options_chain.py # Black-Scholes pricing (:8104)
│   │   ├── trading_sim.py   # Paper trading simulator (:8105)
│   │   └── risk_metrics.py  # VaR, Sharpe, stress tests (:8106)
│   │
│   └── evaluators/          # Evaluation components
│       ├── macro.py         # Macro thesis evaluator
│       ├── fundamental.py   # Fundamental analysis evaluator
│       ├── execution.py     # Execution quality evaluator
│       └── options.py       # Options-specific evaluator (P&L, Greeks)
│
├── scripts/
│   ├── run_a2a_eval.py      # A2A evaluation trigger
│   ├── run_options_demo.py  # Options Alpha Challenge demo
│   └── run_csv_eval.py      # CSV dataset evaluation
│
├── data/
│   ├── synthetic_questions/ # Generated evaluation tasks
│   ├── BizFinBench.v2/      # HiThink benchmark dataset
│   └── financial_lake/      # Cached financial data
│
├── docs/
│   └── ARCHITECTURE_OPTIONS.md  # Options system design
│
├── Dockerfile.green         # Green Agent container
├── Dockerfile.purple        # Purple Agent container
├── docker-compose.yml       # Full stack deployment
└── ABSTRACT.md              # Competition abstract
```

## Options Alpha Challenge

The benchmark includes a comprehensive options trading evaluation:

### Options Task Types
- **Iron Condor**: Construct neutral strategies with defined risk
- **Volatility Trading**: IV rank/percentile analysis
- **Greeks Hedging**: Delta neutralization strategies
- **Risk Management**: VaR-based position sizing

### Options Evaluation Scoring
| Dimension | Weight | Description |
|-----------|--------|-------------|
| P&L Accuracy | 25% | Profit/loss calculations |
| Greeks Accuracy | 25% | Delta, gamma, theta, vega |
| Strategy Quality | 25% | Structure and rationale |
| Risk Management | 25% | Position sizing, hedging |

### Running Options Demo

```bash
# Single task
python scripts/run_options_demo.py --task iron_condor --ticker SPY

# All task types
python scripts/run_options_demo.py --task all --ticker SPY
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

# Run A2A conformance tests
python -m pytest tests/test_a2a_green.py -v --agent-url http://localhost:9109

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## API Reference

### Green Agent A2A Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/.well-known/agent.json` | GET | Agent Card (A2A discovery) |
| `/` | POST | A2A JSON-RPC endpoint |

### Purple Agent A2A Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/.well-known/agent.json` | GET | Agent Card (A2A discovery) |
| `/health` | GET | Health check |
| `/analyze` | POST | Direct analysis (non-A2A) |
| `/` | POST | A2A JSON-RPC endpoint |

## Competition Info

This project is built for the [AgentBeats Finance Track](https://rdi.berkeley.edu/agentx-agentbeats):

- **Phase 1** (Jan 15, 2026): Green Agent submissions
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
- [green-agent-template](https://github.com/RDI-Foundation/green-agent-template) for A2A implementation reference

