"""
Hidden windows strategy for anti-overfitting in crypto benchmark evaluation.

The master seed is stored privately in ~/.agentbusters/hidden_seeds.yaml
and is NOT committed to the repository. This ensures evaluation windows
are unpredictable and cannot be gamed.
"""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

import yaml


def get_hidden_config_path() -> Path:
    """Get path to hidden seeds config file."""
    return Path.home() / ".agentbusters" / "hidden_seeds.yaml"


def load_hidden_config() -> dict[str, Any]:
    """
    Load hidden configuration from ~/.agentbusters/hidden_seeds.yaml.

    Returns empty dict if file doesn't exist.
    """
    config_path = get_hidden_config_path()
    if not config_path.exists():
        return {}
    try:
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def load_hidden_seed(config_name: str) -> Optional[int]:
    """
    Load master seed for a specific benchmark config.

    Args:
        config_name: Name of the config (e.g., "crypto_benchmark_v1")

    Returns:
        Master seed as integer, or None if not found
    """
    config = load_hidden_config()
    if config_name not in config:
        return None

    seed_config = config[config_name]
    if isinstance(seed_config, dict):
        seed_value = seed_config.get("master_seed")
    else:
        seed_value = seed_config

    if seed_value is None:
        return None

    # Handle hex string format (e.g., "0x5F3A2B1C")
    if isinstance(seed_value, str):
        if seed_value.startswith("0x") or seed_value.startswith("0X"):
            return int(seed_value, 16)
        return int(seed_value)

    return int(seed_value)


def save_hidden_seed(config_name: str, master_seed: int) -> None:
    """
    Save master seed to hidden config file.

    Args:
        config_name: Name of the config (e.g., "crypto_benchmark_v1")
        master_seed: Master seed value
    """
    config_path = get_hidden_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_hidden_config()
    config[config_name] = {"master_seed": f"0x{master_seed:08X}"}

    config_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")


def generate_random_seed() -> int:
    """Generate a cryptographically random seed."""
    return int.from_bytes(random.randbytes(4), "big")


def _derive_window_seed(master_seed: int, window_index: int, symbol: str) -> int:
    """Derive a deterministic seed for a specific window."""
    text = f"{master_seed}|{window_index}|{symbol}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _parse_date(date_str: str) -> datetime:
    """Parse a date string to datetime."""
    if "T" in date_str:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    else:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _get_available_range(
    conn,
    symbol: str,
    table: str,
    time_field: str = "bucket_ts",
    symbol_field: str = "symbol",
) -> Optional[Tuple[datetime, datetime]]:
    """Query the available date range for a symbol in the database."""
    import psycopg2.extras

    query = f"""
        SELECT MIN({time_field}) as min_ts, MAX({time_field}) as max_ts
        FROM {table}
        WHERE {symbol_field} = %s
    """

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, (symbol,))
        row = cur.fetchone()
        if row and row["min_ts"] and row["max_ts"]:
            min_ts = row["min_ts"]
            max_ts = row["max_ts"]
            if isinstance(min_ts, datetime) and min_ts.tzinfo is None:
                min_ts = min_ts.replace(tzinfo=timezone.utc)
            if isinstance(max_ts, datetime) and max_ts.tzinfo is None:
                max_ts = max_ts.replace(tzinfo=timezone.utc)
            return (min_ts, max_ts)
    return None


def select_evaluation_windows(
    master_seed: int,
    window_count: int,
    symbols: List[str],
    date_range: Tuple[str, str],
    min_bars: int,
    max_bars: int,
    conn=None,
    ohlcv_table: str = "market_data.candles_1m",
) -> List[dict]:
    """
    Deterministically select evaluation windows based on seed.

    The selection is deterministic given the same inputs, ensuring
    reproducibility while keeping the actual windows hidden.

    Args:
        master_seed: Master random seed (kept private)
        window_count: Number of windows to generate
        symbols: List of symbols to distribute windows across
        date_range: (start_date, end_date) as ISO strings
        min_bars: Minimum number of bars per window
        max_bars: Maximum number of bars per window
        conn: Optional PostgreSQL connection for verifying data availability
        ohlcv_table: Table name for OHLCV data

    Returns:
        List of window dicts with keys:
        - scenario_id: Unique identifier for the window
        - symbol: Trading symbol
        - start: Window start datetime (ISO string)
        - end: Window end datetime (ISO string)
        - bars: Expected number of bars
    """
    start_dt = _parse_date(date_range[0])
    end_dt = _parse_date(date_range[1])

    # Get available ranges per symbol from database if connection provided
    available_ranges: dict[str, Tuple[datetime, datetime]] = {}
    if conn is not None:
        for symbol in symbols:
            range_result = _get_available_range(conn, symbol, ohlcv_table)
            if range_result:
                available_ranges[symbol] = range_result

    windows: List[dict] = []
    rng = random.Random(master_seed)

    for window_idx in range(window_count):
        # Distribute windows across symbols round-robin with some randomness
        symbol_idx = (window_idx + rng.randint(0, len(symbols) - 1)) % len(symbols)
        symbol = symbols[symbol_idx]

        # Derive window-specific seed
        window_seed = _derive_window_seed(master_seed, window_idx, symbol)
        window_rng = random.Random(window_seed)

        # Determine effective date range for this symbol
        eff_start = start_dt
        eff_end = end_dt
        if symbol in available_ranges:
            db_start, db_end = available_ranges[symbol]
            eff_start = max(start_dt, db_start)
            eff_end = min(end_dt, db_end)

        # Calculate window duration in minutes
        window_bars = window_rng.randint(min_bars, max_bars)
        window_duration = timedelta(minutes=window_bars)

        # Calculate available window for random start selection
        total_range = (eff_end - eff_start).total_seconds()
        window_seconds = window_duration.total_seconds()

        if total_range < window_seconds:
            # Date range too small, use full range
            window_start = eff_start
            window_end = eff_end
            actual_bars = int((window_end - window_start).total_seconds() / 60)
        else:
            # Random start within valid range
            max_start_offset = int(total_range - window_seconds)
            start_offset = window_rng.randint(0, max_start_offset)
            window_start = eff_start + timedelta(seconds=start_offset)
            window_end = window_start + window_duration
            actual_bars = window_bars

        # Generate scenario ID from window params (deterministic)
        id_hash = hashlib.sha256(
            f"{master_seed}|{window_idx}|{symbol}|{window_start.isoformat()}".encode()
        ).hexdigest()[:12]
        scenario_id = f"{symbol.lower()}_{id_hash}"

        windows.append({
            "scenario_id": scenario_id,
            "symbol": symbol,
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
            "bars": actual_bars,
            "window_index": window_idx,
        })

    return windows


def log_evaluation_windows(
    windows: List[dict],
    output_path: Optional[Path] = None,
    config_name: Optional[str] = None,
) -> None:
    """
    Log selected windows AFTER evaluation completes.

    This provides an audit trail of which windows were evaluated
    without revealing them beforehand.

    Args:
        windows: List of window dicts from select_evaluation_windows
        output_path: Optional path to write log file
        config_name: Optional config name for metadata
    """
    log_entry = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "config_name": config_name,
        "window_count": len(windows),
        "windows": windows,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Append to existing log file if present
        existing = []
        if output_path.exists():
            try:
                import json
                existing = [json.loads(line) for line in output_path.read_text().strip().split("\n") if line]
            except Exception:
                existing = []

        import json
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")


def create_example_hidden_config() -> str:
    """
    Generate example hidden_seeds.yaml content.

    Returns:
        YAML string for example config
    """
    return """# AgentBusters Hidden Seeds Configuration
# This file should be stored in ~/.agentbusters/hidden_seeds.yaml
# DO NOT commit this file to version control!

crypto_benchmark_v1:
  master_seed: 0x5F3A2B1C  # Change quarterly for fresh evaluation windows
  # Created: 2025-01-01
  # Expires: 2025-04-01

# You can add multiple benchmark configs:
# crypto_benchmark_v2:
#   master_seed: 0x12345678
"""
