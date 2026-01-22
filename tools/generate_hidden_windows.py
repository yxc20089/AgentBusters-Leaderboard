#!/usr/bin/env python3
"""
Generate hidden evaluation windows from PostgreSQL.

This script pre-generates evaluation windows and exports them to JSON files
that can be used during AgentBeats evaluation without requiring database access.

Usage:
    # Generate windows using hidden seed
    python tools/generate_hidden_windows.py --config config/eval_crypto.yaml --output data/crypto/hidden

    # Generate with explicit seed (for testing)
    python tools/generate_hidden_windows.py --seed 12345 --output data/crypto/test_windows

The generated files use hashed scenario IDs that don't reveal time window information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import yaml


def _hash_scenario_id(master_seed: int, window_index: int, symbol: str) -> str:
    """Generate anonymous scenario ID from window params."""
    text = f"{master_seed}|{window_index}|{symbol}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"scenario_{digest}"


def generate_windows_from_postgres(
    master_seed: int,
    window_count: int,
    symbols: List[str],
    date_range_start: str,
    date_range_end: str,
    min_bars: int,
    max_bars: int,
    pg_config: dict,
    output_dir: Path,
) -> List[dict]:
    """
    Generate evaluation windows from PostgreSQL and export to JSON.

    Returns list of generated scenario metadata (without revealing time windows).
    """
    from cio_agent.crypto_benchmark import PostgresMarketDataLoader
    from cio_agent.hidden_windows import select_evaluation_windows

    # Connect to PostgreSQL
    loader = PostgresMarketDataLoader(
        dsn=pg_config.get("pg_dsn"),
        host=pg_config.get("pg_host", "localhost"),
        port=pg_config.get("pg_port", 5432),
        dbname=pg_config.get("pg_dbname", "market_data"),
        user=pg_config.get("pg_user", "postgres"),
        password=pg_config.get("pg_password", "postgres"),
        ohlcv_table=pg_config.get("pg_ohlcv_table", "market_data.candles_1m"),
        funding_table=pg_config.get("pg_funding_table"),
    )

    try:
        conn = loader._connect()

        # Select windows using hidden seed
        windows = select_evaluation_windows(
            master_seed=master_seed,
            window_count=window_count,
            symbols=symbols,
            date_range=(date_range_start, date_range_end),
            min_bars=min_bars,
            max_bars=max_bars,
            conn=conn,
            ohlcv_table=pg_config.get("pg_ohlcv_table", "market_data.candles_1m"),
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        generated_scenarios = []

        for window in windows:
            # Parse timestamps
            start_dt = datetime.fromisoformat(window["start"].replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(window["end"].replace("Z", "+00:00"))

            # Generate anonymous scenario ID
            anon_id = _hash_scenario_id(
                master_seed, window["window_index"], window["symbol"]
            )

            # Load market data
            scenario_data = loader.load_scenario(
                scenario_id=anon_id,
                symbol=window["symbol"],
                start=start_dt,
                end=end_dt,
                timeframe="1m",
                description=f"Evaluation scenario {window['window_index'] + 1}",
            )

            # Remove time-revealing metadata
            scenario_data["metadata"] = {
                "scenario_id": anon_id,
                "symbol": window["symbol"],
                "data_points": len(scenario_data.get("market_states", [])),
                "timeframe": "1m",
                # Deliberately omit start/end timestamps
            }

            # Save scenario
            scenario_dir = output_dir / anon_id
            scenario_dir.mkdir(parents=True, exist_ok=True)

            (scenario_dir / "market_data.json").write_text(
                json.dumps(scenario_data, indent=2), encoding="utf-8"
            )
            (scenario_dir / "metadata.json").write_text(
                json.dumps(scenario_data["metadata"], indent=2), encoding="utf-8"
            )

            generated_scenarios.append({
                "scenario_id": anon_id,
                "symbol": window["symbol"],
                "bars": len(scenario_data.get("market_states", [])),
            })

            print(f"  Generated: {anon_id} ({window['symbol']}, {len(scenario_data.get('market_states', []))} bars)")

        # Save manifest (without revealing window details)
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_count": len(generated_scenarios),
            "scenarios": [
                {"id": s["scenario_id"], "symbol": s["symbol"]}
                for s in generated_scenarios
            ],
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        return generated_scenarios

    finally:
        loader.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate hidden evaluation windows from PostgreSQL"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to evaluation config YAML with crypto dataset",
    )
    parser.add_argument(
        "--seed",
        type=str,
        help="Master seed (hex like 0x5F3A2B1C or decimal). Overrides config.",
    )
    parser.add_argument(
        "--hidden-seed-config",
        type=str,
        help="Name of hidden seed config in ~/.agentbusters/hidden_seeds.yaml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/crypto/hidden",
        help="Output directory for generated scenarios",
    )
    parser.add_argument(
        "--window-count",
        type=int,
        default=12,
        help="Number of windows to generate",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT",
        help="Comma-separated list of symbols",
    )

    args = parser.parse_args()

    # Determine master seed
    master_seed: Optional[int] = None

    if args.seed:
        if args.seed.startswith("0x") or args.seed.startswith("0X"):
            master_seed = int(args.seed, 16)
        else:
            master_seed = int(args.seed)
    elif args.hidden_seed_config:
        from cio_agent.hidden_windows import load_hidden_seed
        master_seed = load_hidden_seed(args.hidden_seed_config)
        if master_seed is None:
            print(f"Error: Could not load seed from config '{args.hidden_seed_config}'")
            print("Make sure ~/.agentbusters/hidden_seeds.yaml exists with this config.")
            return 1

    # Load config if provided
    pg_config = {}
    window_count = args.window_count
    symbols = args.symbols.split(",")
    date_range_start = "2020-01-01"
    date_range_end = "2025-12-31"
    min_bars = 1440
    max_bars = 10080

    if args.config:
        config_data = yaml.safe_load(Path(args.config).read_text())
        for dataset in config_data.get("datasets", []):
            if dataset.get("type") != "crypto":
                continue

            # Extract PostgreSQL config
            pg_config = {
                "pg_dsn": dataset.get("pg_dsn"),
                "pg_host": dataset.get("pg_host", "localhost"),
                "pg_port": dataset.get("pg_port", 5432),
                "pg_dbname": dataset.get("pg_dbname", "market_data"),
                "pg_user": dataset.get("pg_user", "postgres"),
                "pg_password": dataset.get("pg_password"),
                "pg_ohlcv_table": dataset.get("pg_ohlcv_table", "market_data.candles_1m"),
                "pg_funding_table": dataset.get("pg_funding_table"),
            }

            # Extract window config
            window_count = dataset.get("window_count", window_count)
            symbols = dataset.get("symbols", symbols)
            date_range_start = dataset.get("date_range_start", date_range_start)
            date_range_end = dataset.get("date_range_end", date_range_end)
            min_bars = dataset.get("window_min_bars", min_bars)
            max_bars = dataset.get("window_max_bars", max_bars)

            # Get seed from config if not overridden
            if master_seed is None and dataset.get("hidden_seed_config"):
                from cio_agent.hidden_windows import load_hidden_seed
                master_seed = load_hidden_seed(dataset["hidden_seed_config"])

            break

    if master_seed is None:
        print("Error: No master seed provided.")
        print("Use --seed, --hidden-seed-config, or configure hidden_seed_config in YAML.")
        return 1

    if not pg_config:
        # Default PostgreSQL config
        pg_config = {
            "pg_host": "localhost",
            "pg_port": 5432,
            "pg_dbname": "market_data",
            "pg_user": "postgres",
            "pg_ohlcv_table": "market_data.candles_1m",
        }

    print(f"Generating {window_count} hidden windows...")
    print(f"  Symbols: {symbols}")
    print(f"  Date range: {date_range_start} to {date_range_end}")
    print(f"  Output: {args.output}")
    print()

    try:
        scenarios = generate_windows_from_postgres(
            master_seed=master_seed,
            window_count=window_count,
            symbols=symbols,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            min_bars=min_bars,
            max_bars=max_bars,
            pg_config=pg_config,
            output_dir=Path(args.output),
        )

        print()
        print(f"Generated {len(scenarios)} scenarios to {args.output}/")
        print("These can be used for AgentBeats evaluation without database access.")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
