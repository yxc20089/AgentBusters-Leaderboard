# AgentBusters Crypto Trading Benchmark

This benchmark evaluates crypto trading agents with multi-round A2A
interactions over historical market scenarios. It is designed to
complement the FAB++ finance tasks with a trading-focused track.

## What It Evaluates

- Strategy performance across multiple market regimes
- Robustness under noise and adversarial perturbations
- Consistency under data transformations
- Risk control (drawdown, win rate, Sharpe)

## Data Format

Each scenario is a directory containing `market_data.json` and (optional)
`metadata.json`. The JSON structure is:

```json
{
  "metadata": {
    "source": "nofx",
    "exchange": "binance",
    "timeframe": "1m",
    "description": "2022 bear market initial phase"
  },
  "market_states": [
    {
      "timestamp": "2022-01-01T00:00:00Z",
      "symbol": "BTCUSDT",
      "ohlcv": {
        "open": 47000.0,
        "high": 47150.0,
        "low": 46800.0,
        "close": 46950.0,
        "volume": 1234.5,
        "quoteVolume": 58000000.0,
        "trades": 2345,
        "takerBuyBaseVolume": 650.0,
        "takerBuyQuoteVolume": 30500000.0
      },
      "indicators": {
        "ema_20": 46850.0,
        "ema_50": 47010.0,
        "rsi": 45.2,
        "macd": -12.3,
        "atr": 210.5
      },
      "market_metrics": {
        "funding_rate": 0.0001,
        "open_interest": 1500000000,
        "cvd": 0.0
      }
    }
  ]
}
```

The extra kline fields (`quoteVolume`, `trades`, `takerBuyBaseVolume`,
`takerBuyQuoteVolume`) are optional and may be omitted or set to zero if
the source does not provide them.

## Agent Interface (Purple Agent)

The Green Agent sends a JSON payload as text via A2A:

```json
{
  "type": "trading_decision",
  "state": {
    "timestamp": "2022-01-01T00:00:00Z",
    "symbol": "BTCUSDT",
    "ohlcv": {
      "open": 47000,
      "high": 47150,
      "low": 46800,
      "close": 46950,
      "volume": 1234.5,
      "quoteVolume": 58000000.0,
      "trades": 2345,
      "takerBuyBaseVolume": 650.0,
      "takerBuyQuoteVolume": 30500000.0
    },
    "indicators": { "ema_20": 46850, "ema_50": 47010, "rsi": 45.2, "macd": -12.3, "atr": 210.5 },
    "market_metrics": { "funding_rate": 0.0001, "open_interest": 1500000000, "cvd": 0.0 },
    "account": {
      "balance": 10000.0,
      "equity": 10000.0,
      "positions": []
    }
  }
}
```

Expected response format:

```json
{
  "action": "BUY",
  "symbol": "BTCUSDT",
  "size": 0.1,
  "reasoning": "RSI oversold and price below EMA20",
  "confidence": 0.75,
  "stop_loss": 46500,
  "take_profit": 48500
}
```

## Evaluation Dimensions

1. Baseline (clean historical data)
2. Noisy (Gaussian price noise + randomized slippage)
3. Adversarial (flash crash/pump and liquidity shocks)
4. Meta-consistency (price scaling + inverted returns)

Scores are weighted by `score_weights` in the dataset config. Each
dimension uses portfolio metrics to compute a 0-100 score.

## Funding Settlement

If `market_metrics.funding_rate` is present, the simulator applies funding
every `funding_interval_hours` (default: 8). Funding cashflow is computed
on position notional per interval: `notional * funding_rate`. Longs pay
when the rate is positive, shorts pay when the rate is negative. If
timestamps are available, settlement uses real time deltas; otherwise it
falls back to bar counts based on the scenario timeframe.

## Configuration

Use `config/eval_crypto.yaml`:

```bash
python src/cio_agent/a2a_server.py --host 0.0.0.0 --port 9109 \
  --eval-config config/eval_crypto.yaml
```

## Remote Scenario Manifest & Cache

To avoid shipping large datasets in the Docker image, configure a remote
manifest. Missing scenarios are downloaded into `cache_dir` (or `path` if
`cache_dir` is not set) and reused until `cache_ttl_hours` expires.

Manifest example:

```json
{
  "version": "1",
  "base_url": "https://example.com/agentbusters/crypto",
  "scenarios": [
    {
      "id": "window_2_ftx_crash",
      "name": "FTX collapse and market crisis",
      "description": "2022-07-01 to 2022-12-31",
      "url": "window_2_ftx_crash.zip",
      "sha256": "0123456789abcdef...",
      "metadata": {
        "exchange": "binance",
        "timeframe": "1m",
        "period": "2022-07-01 to 2022-12-31"
      }
    }
  ]
}
```

Config example:

```yaml
datasets:
  - type: crypto
    path: data/crypto/cache
    remote_manifest: https://example.com/agentbusters/crypto/manifest.json
    cache_ttl_hours: 24
    download_on_missing: true
```

Artifacts can be `.zip` files containing `market_data.json` (+ optional
`metadata.json`), or a direct `.json` file for `market_data.json`.

## Data Source Guidance (Klines vs Trades)

For AgentBeats-style evaluation, minute-level OHLCV (K-line) data is the
recommended default because:

- It is compact and reproducible.
- It keeps A2A evaluation time bounded.
- It matches the multi-round decision cadence.

Trade-level (tick) data is significantly larger and would require
microstructure simulation (order book, latency, queue position). That
is valuable for market making, but it is not necessary for this
benchmark's goal of strategy robustness.

If you need trade-level evaluation, add a separate dataset track with
coarser sampling or a dedicated microstructure simulator.

## Multi-Round Interaction

Yes. The benchmark is multi-round: each scenario is a sequence of states
and the Green Agent maintains A2A context for the duration of that run.
Each evaluation mode (baseline/noisy/adversarial/meta) starts a fresh
conversation context.

## Generating Scenarios (NOFX)

NOFX provides historical K-line data that can be converted into
`market_data.json` format. Use the NOFX toolkit to download candles and
compute indicators, then export into the structure above. The Green
Agent only requires the JSON format and does not depend on NOFX itself.

### NOFX CLI Export Script

This repo includes a helper script that calls the NOFX CLI (Go) to fetch
K-lines and converts them into AgentBusters scenarios:

```bash
python tools/nofx_klines_to_market_data.py \
  --nofx-dir nofx \
  --name window_1_bear_market \
  --symbol BTCUSDT \
  --timeframe 1m \
  --start 2022-01-01T00:00:00Z \
  --end 2022-01-02T00:00:00Z \
  --description "2022 bear market initial phase"
```

For multiple scenarios, use a config file (JSON or YAML):

```json
{
  "nofx_dir": "nofx",
  "output_root": "data/crypto/scenarios",
  "scenarios": [
    {
      "name": "window_1_bear_market",
      "symbol": "BTCUSDT",
      "timeframe": "1m",
      "start": "2022-01-01T00:00:00Z",
      "end": "2022-01-02T00:00:00Z",
      "description": "2022 bear market initial phase",
      "exchange": "binance",
      "market_metrics": { "funding_rate": 0.0001, "open_interest": 1500000000, "cvd": 0 }
    }
  ]
}
```

Note: the NOFX CLI helper uses `market.GetKlinesRange`, which currently
fetches Binance futures K-lines. The `exchange` field is stored as
metadata only.

### HuggingFace OHLCV Export Script

If you prefer the HuggingFace dataset, use `tools/hf_ohlcv_to_market_data.py`
to slice windows directly from HF OHLCV data. For file-based HF datasets,
pass the filename with `--data-file` (no dataset config needed).

```bash
python tools/hf_ohlcv_to_market_data.py \
  --dataset 123olp/binance-futures-ohlcv-2018-2026 \
  --data-file candles_1m.csv.gz \
  --name window_1_bear_market \
  --symbol BTCUSDT \
  --timeframe 1m \
  --start 2020-01-01T00:00:00Z \
  --end 2020-01-07T00:00:00Z \
  --description "2020-01 first week BTC"
```

To include open interest from a futures metrics table (5m cadence),
pass the metrics dataset/config. Open interest is forward-filled onto
the candle timeline:

```bash
python tools/hf_ohlcv_to_market_data.py \
  --dataset 123olp/binance-futures-ohlcv-2018-2026 \
  --data-file candles_1m.csv.gz \
  --name window_1_bear_market \
  --symbol BTCUSDT \
  --timeframe 1m \
  --start 2020-01-01T00:00:00Z \
  --end 2020-01-07T00:00:00Z \
  --metrics-dataset 123olp/binance-futures-ohlcv-2018-2026 \
  --metrics-data-file futures_metrics_5m.csv.gz \
  --metrics-time-field create_time \
  --metrics-open-interest-field sum_open_interest_value
```

Funding rates are not in the HF dataset. If you want funding in
`market_metrics`, the exporter can pull Binance funding history and
forward-fill it onto the candles. Funding responses are cached locally
to avoid repeated API calls.

```bash
python tools/hf_ohlcv_to_market_data.py \
  --dataset 123olp/binance-futures-ohlcv-2018-2026 \
  --data-file candles_1m.csv.gz \
  --name window_1_bear_market \
  --symbol BTCUSDT \
  --timeframe 1m \
  --start 2020-01-01T00:00:00Z \
  --end 2020-01-07T00:00:00Z \
  --funding-source binance \
  --funding-cache-dir data/crypto/cache/funding \
  --funding-cache-ttl-hours 24
```

### Postgres/TimescaleDB Export Script

If you restored the `.bin.zst` files into Postgres/TimescaleDB, use
`tools/pg_ohlcv_to_market_data.py` to export windows without loading
the large CSVs:

```bash
python tools/pg_ohlcv_to_market_data.py \
  --host localhost \
  --port 5433 \
  --dbname market_data \
  --user postgres \
  --ohlcv-table market_data.candles_1m \
  --metrics-table market_data.binance_futures_metrics_5m \
  --name window_1_bear_market \
  --symbol BTCUSDT \
  --timeframe 1m \
  --start 2020-01-01T00:00:00Z \
  --end 2020-01-07T00:00:00Z
```

You can also enable funding fetch + cache the same way:

```bash
python tools/pg_ohlcv_to_market_data.py \
  --host localhost \
  --port 5433 \
  --dbname market_data \
  --user postgres \
  --ohlcv-table market_data.candles_1m \
  --metrics-table market_data.binance_futures_metrics_5m \
  --name window_1_bear_market \
  --symbol BTCUSDT \
  --timeframe 1m \
  --start 2020-01-01T00:00:00Z \
  --end 2020-01-07T00:00:00Z \
  --funding-source binance \
  --funding-cache-dir data/crypto/cache/funding \
  --funding-cache-ttl-hours 24
```
