# AgentBusters Production Deployment Guide

This guide covers deploying the AgentBusters crypto benchmark with anti-overfitting protections for production use on platforms like GitHub Actions or AgentBeats.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Local Machine (One-time Setup)                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │ PostgreSQL      │───→│ generate_hidden │───→│ Anonymized JSON │  │
│  │ 100GB+ data     │    │ _windows.py     │    │ (no timestamps) │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│           ↑                                            │            │
│  ~/.agentbusters/hidden_seeds.yaml                     │            │
│  (Private, NOT in repo)                                │            │
└────────────────────────────────────────────────────────│────────────┘
                                                         │ Upload
                                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Private Storage (Choose One)                                       │
│  • GitHub Private Repository                                        │
│  • HuggingFace Private Dataset                                      │
│  • S3/R2 Private Bucket                                             │
└─────────────────────────────────────────────────────────────────────┘
                                                         │ Download
                                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  GitHub Actions / AgentBeats                                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │ Download Data   │───→│ Green Agent     │───→│ Evaluate Purple │  │
│  │ (anonymized)    │    │ Evaluator       │    │ Agent           │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- PostgreSQL with TimescaleDB (for market data)
- Python 3.11+
- GitHub CLI (`gh`) for private repo uploads
- 100GB+ of historical crypto market data

## Step 1: Configure Hidden Seeds

Create the private seed configuration file:

```bash
mkdir -p ~/.agentbusters
cat > ~/.agentbusters/hidden_seeds.yaml << 'EOF'
# AgentBusters Hidden Seeds - DO NOT COMMIT TO GIT
# Change seed quarterly to refresh evaluation windows

crypto_benchmark_v1:
  master_seed: 0x7A3F2E1D  # Generate new: python -c "import secrets; print(f'0x{secrets.token_hex(4).upper()}')"
  created: "2025-01-21"
  expires: "2025-04-21"
EOF
```

**Security Notes:**
- This file must NEVER be committed to version control
- Add `~/.agentbusters/` to your global `.gitignore`
- Rotate seeds quarterly for fresh evaluation windows

## Step 2: Generate Evaluation Windows

Generate anonymized evaluation scenarios from PostgreSQL:

```bash
# Using production config
python tools/generate_hidden_windows.py \
    --config config/eval_crypto_production.yaml \
    --output data/crypto/hidden

# Or with explicit parameters
python tools/generate_hidden_windows.py \
    --hidden-seed-config crypto_benchmark_v1 \
    --window-count 12 \
    --symbols BTCUSDT,ETHUSDT \
    --output data/crypto/hidden
```

**What gets generated:**
```
data/crypto/hidden/
├── manifest.json                    # List of scenarios (no timestamps)
├── scenario_a1b2c3d4e5f6/
│   ├── market_data.json            # OHLCV + indicators
│   └── metadata.json               # Symbol, bar count (no timestamps)
├── scenario_f6e5d4c3b2a1/
│   ├── market_data.json
│   └── metadata.json
└── ...
```

**Anonymization:**
- Scenario IDs are SHA256 hashes of `seed|index|symbol|timestamp`
- No actual timestamps appear in the output
- Cannot reverse-engineer which time windows were selected

## Step 3: Upload to Private Storage

### Option A: GitHub Private Repository (Recommended)

```bash
# Create private repo and upload
python tools/upload_eval_data.py \
    --source data/crypto/hidden \
    --backend github \
    --repo your-org/agentbusters-eval-data
```

### Option B: HuggingFace Private Dataset

```bash
# Login first
huggingface-cli login

# Upload as private dataset
python tools/upload_eval_data.py \
    --source data/crypto/hidden \
    --backend huggingface \
    --repo your-org/agentbusters-eval-data
```

### Option C: S3/R2 Private Bucket

```bash
# Configure AWS credentials first
python tools/upload_eval_data.py \
    --source data/crypto/hidden \
    --backend s3 \
    --bucket your-bucket-name \
    --prefix eval/crypto/v1
```

## Step 4: Configure GitHub Actions

### Repository Secrets

Add these secrets to your AgentBusters repository:

| Secret Name | Description |
|-------------|-------------|
| `EVAL_DATA_REPO` | Private repo name (e.g., `your-org/agentbusters-eval-data`) |
| `EVAL_DATA_PAT` | GitHub PAT with `repo` scope for reading private repos |
| `HF_TOKEN` | (Optional) HuggingFace token if using HF storage |
| `HF_EVAL_DATASET` | (Optional) HF dataset name |

### Workflow File

The workflow at `.github/workflows/crypto_eval.yaml` handles:

1. Downloading evaluation data from private storage
2. Starting the Green Agent
3. Evaluating the Purple Agent
4. Uploading results as artifacts

Trigger manually:
```bash
gh workflow run crypto_eval.yaml \
    -f purple_agent_url=http://your-agent:9110 \
    -f num_tasks=12
```

## Step 5: Quarterly Seed Rotation

To prevent long-term optimization against fixed evaluation windows:

```bash
# 1. Generate new seed
python -c "import secrets; print(f'0x{secrets.token_hex(4).upper()}')"
# Output: 0x8B2E4F1A

# 2. Update ~/.agentbusters/hidden_seeds.yaml
crypto_benchmark_v1:
  master_seed: 0x8B2E4F1A  # New seed
  created: "2025-04-01"
  expires: "2025-07-01"

# 3. Regenerate evaluation windows
python tools/generate_hidden_windows.py \
    --config config/eval_crypto_production.yaml \
    --output data/crypto/hidden

# 4. Re-upload to private storage
python tools/upload_eval_data.py \
    --source data/crypto/hidden \
    --backend github \
    --repo your-org/agentbusters-eval-data
```

## Security Guarantees

| Attack Vector | Protection |
|---------------|------------|
| View evaluation data | Data in private storage, requires authorization |
| Infer time from scenario ID | IDs are SHA256 hashes, cannot reverse |
| LLM memorization | 100GB+ data + random windows = infeasible to memorize |
| Long-term optimization | Quarterly seed rotation refreshes windows |
| Pre-compute answers | Windows unknown until evaluation runs |

## Configuration Files

> Do not commit hidden seeds or evaluation data. Keep `~/.agentbusters/hidden_seeds.yaml` and `data/crypto/hidden/` private.

### Production Config (`config/eval_crypto_production.yaml`)

```yaml
datasets:
  - type: crypto
    path: data/crypto/hidden  # Private fallback only (do not commit)
    # remote_manifest: https://example.com/private/crypto/manifest.json
    download_on_missing: false

    pg_enabled: true
    pg_host: localhost
    pg_dbname: market_data
    pg_ohlcv_table: market_data.candles_1m

    # Hidden windows
    hidden_seed_config: crypto_benchmark_v1
    window_count: 12
    symbols: [BTCUSDT, ETHUSDT]
    date_range_start: "2020-01-01"
    date_range_end: "2025-12-31"

    evaluation:
      metric_weights:
        sharpe: 0.50      # Primary ranking metric
        total_return: 0.25
        max_drawdown: 0.15
        win_rate: 0.10
```

### Testing Config (`config/eval_crypto.yaml`)

For local development/testing with static data (not for production):

```yaml
datasets:
  - type: crypto
    path: data/crypto/hidden  # Static test data (private)
    pg_enabled: false
    limit: 1
    max_steps: 20
```

## Troubleshooting

### "Could not load seed from config"

```bash
# Check if hidden_seeds.yaml exists
cat ~/.agentbusters/hidden_seeds.yaml

# Verify config name matches
grep -A2 "crypto_benchmark_v1" ~/.agentbusters/hidden_seeds.yaml
```

### "PostgreSQL connection failed"

```bash
# Test connection
psql -h localhost -U postgres -d market_data -c "SELECT 1"

# Check table exists
psql -h localhost -U postgres -d market_data -c "\dt market_data.*"
```

### "No scenarios generated"

```bash
# Check date range has data
psql -h localhost -U postgres -d market_data -c "
  SELECT MIN(bucket_ts), MAX(bucket_ts)
  FROM market_data.candles_1m
  WHERE symbol = 'BTCUSDT'
"
```

## Local Testing (Without Production Setup)

For quick local testing without PostgreSQL:

```bash
# Use pre-generated test data
python -m cio_agent serve \
    --host 0.0.0.0 --port 9109 \
    --config config/eval_crypto.yaml

# Run evaluation
python scripts/run_a2a_eval.py \
    --green-url http://localhost:9109 \
    --purple-url http://localhost:9110 \
    --num-tasks 1
```

Note: This uses static test data and should NOT be used for official benchmarking.
