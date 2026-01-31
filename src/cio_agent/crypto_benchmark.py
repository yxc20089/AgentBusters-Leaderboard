"""
Crypto trading benchmark utilities and evaluator.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator


DEFAULT_SCORE_WEIGHTS = {
    "baseline": 0.40,
    "noisy": 0.30,
    "adversarial": 0.20,
    "meta": 0.10,
}

DEFAULT_METRIC_WEIGHTS = {
    "sharpe": 0.50,
    "total_return": 0.25,
    "max_drawdown": 0.15,
    "win_rate": 0.10,
}


class CryptoEvaluationConfig(BaseModel):
    """Evaluation settings for the crypto trading benchmark."""

    initial_balance: float = 10000.0
    max_leverage: float = 3.0
    trading_fee: float = 0.0004
    price_noise_level: float = 0.001
    slippage_range: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    adversarial_injection_rate: float = 0.05
    decision_interval: int = 1
    funding_interval_hours: float = 8.0
    seed: Optional[int] = None
    score_weights: dict[str, float] = Field(default_factory=lambda: DEFAULT_SCORE_WEIGHTS.copy())
    metric_weights: dict[str, float] = Field(default_factory=lambda: DEFAULT_METRIC_WEIGHTS.copy())
    meta_transforms: list[str] = Field(
        default_factory=lambda: ["identity", "scale_1_1", "invert_returns"]
    )
    return_cap: float = 0.50
    sharpe_floor: float = -1.0
    sharpe_cap: float = 3.0
    drawdown_cap: float = 0.50
    win_rate_floor: float = 0.30
    win_rate_cap: float = 0.70

    @field_validator("slippage_range")
    @classmethod
    def validate_slippage_range(cls, value: list[float]) -> list[float]:
        if len(value) != 2:
            raise ValueError("slippage_range must have two values [min, max]")
        if value[0] < 0 or value[1] < 0 or value[0] > value[1]:
            raise ValueError("slippage_range must be non-negative and min <= max")
        return value

    @field_validator("decision_interval")
    @classmethod
    def validate_decision_interval(cls, value: int) -> int:
        if value < 1:
            raise ValueError("decision_interval must be >= 1")
        return value

    @field_validator("funding_interval_hours")
    @classmethod
    def validate_funding_interval_hours(cls, value: float) -> float:
        if value < 0:
            raise ValueError("funding_interval_hours must be >= 0")
        return value


@dataclass
class CryptoScenarioIndex:
    """Index metadata for a crypto scenario."""

    scenario_id: str
    name: str
    description: str
    data_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeRecord:
    """Completed trade record for win-rate statistics."""

    entry_price: float
    exit_price: float
    size: float
    pnl: float
    reason: str


def stable_seed(*parts: str) -> int:
    """Create a deterministic seed from text parts."""
    text = "|".join(parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _parse_iso_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _get_github_auth_header() -> Optional[dict[str, str]]:
    """Get GitHub authorization header from environment if available."""
    pat = os.environ.get("EVAL_DATA_PAT")
    if pat:
        return {"Authorization": f"token {pat}"}
    return None


def _build_github_raw_url(repo: str, path: str, branch: str = "main") -> str:
    """Build raw GitHub URL from repo and path.

    Args:
        repo: GitHub repo in format 'owner/repo'
        path: Path within the repo
        branch: Branch name (default: main)

    Returns:
        Raw GitHub URL for the file
    """
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"


def _create_url_request(url: str) -> urllib.request.Request:
    """Create a URL request with GitHub auth if applicable."""
    request = urllib.request.Request(url)
    if "github" in url.lower() or "githubusercontent" in url.lower():
        auth_header = _get_github_auth_header()
        if auth_header:
            for key, value in auth_header.items():
                request.add_header(key, value)
    return request


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_manifest(
    manifest_ref: str,
    cache_dir: Path,
    cache_ttl_hours: int,
) -> tuple[dict[str, Any], Optional[Path]]:
    if _is_url(manifest_ref):
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / ".crypto_manifest_cache.json"
        if cache_path.exists():
            try:
                payload = _read_json(cache_path)
                cached_at = _parse_iso_timestamp(payload.get("_cached_at"))
                if cached_at:
                    if cache_ttl_hours <= 0:
                        manifest = payload.get("manifest")
                        if isinstance(manifest, dict):
                            return manifest, None
                    else:
                        age_hours = (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600
                        if age_hours < cache_ttl_hours:
                            manifest = payload.get("manifest")
                            if isinstance(manifest, dict):
                                return manifest, None
            except Exception:
                pass

        request = _create_url_request(manifest_ref)
        with urllib.request.urlopen(request, timeout=60) as response:
            manifest = json.loads(response.read().decode("utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a JSON object")
        _write_json(
            cache_path,
            {"_cached_at": datetime.now(timezone.utc).isoformat(), "manifest": manifest},
        )
        return manifest, None

    manifest_path = Path(manifest_ref)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest file not found: {manifest_path}")
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    return manifest, manifest_path.parent


def _manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = manifest.get("scenarios")
    if isinstance(entries, list):
        return entries
    entries = manifest.get("data")
    if isinstance(entries, list):
        return entries
    raise ValueError("manifest must contain a scenarios list")


def _resolve_ref(ref: str, base_url: Optional[str], manifest_parent: Optional[Path]) -> str:
    if _is_url(ref):
        return ref
    if base_url:
        return urllib.parse.urljoin(base_url.rstrip("/") + "/", ref)
    if manifest_parent:
        return str((manifest_parent / ref).resolve())
    return ref


def _sha256_file(path: Path) -> str:
    hash_obj = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def _cache_valid(cache_meta: Path, cache_ttl_hours: int) -> bool:
    if cache_ttl_hours <= 0:
        return True
    if not cache_meta.exists():
        return False
    try:
        payload = _read_json(cache_meta)
        cached_at = _parse_iso_timestamp(payload.get("downloaded_at"))
        if not cached_at:
            return False
        age_hours = (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600
        return age_hours < cache_ttl_hours
    except Exception:
        return False


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            target = dest_dir / member.filename
            if not str(target.resolve()).startswith(str(dest_dir.resolve())):
                raise RuntimeError("zip archive contains unsafe paths")
        archive.extractall(dest_dir)


def _download_to_path(url: str, dest_path: Path) -> None:
    """Download a file from URL to local path, with GitHub auth if available."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    request = _create_url_request(url)
    with urllib.request.urlopen(request, timeout=120) as response:
        with dest_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def prepare_crypto_scenarios(
    path: Path,
    remote_manifest: Optional[str],
    scenarios: Optional[List[str]],
    cache_dir: Optional[Path],
    cache_ttl_hours: int,
    download_on_missing: bool = True,
) -> Path:
    if not remote_manifest:
        return path

    cache_root = cache_dir or path
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest, manifest_parent = _load_manifest(remote_manifest, cache_root, cache_ttl_hours)
    base_url = manifest.get("base_url") or manifest.get("base_uri")

    # If no base_url in manifest but we have a remote URL, derive base_url from manifest URL
    # e.g., https://.../crypto/eval_hidden/manifest.json → https://.../crypto/eval_hidden/
    if not base_url and _is_url(remote_manifest):
        base_url = remote_manifest.rsplit("/", 1)[0] + "/"

    scenario_filter = {s for s in scenarios} if scenarios else None
    for entry in _manifest_entries(manifest):
        scenario_id = entry.get("id") or entry.get("scenario_id") or entry.get("name")
        if not scenario_id:
            continue
        if scenario_filter and scenario_id not in scenario_filter:
            continue

        artifact_meta = entry.get("artifact") if isinstance(entry.get("artifact"), dict) else {}
        ref = entry.get("url") or artifact_meta.get("url") or entry.get("artifact")

        # If no explicit URL, assume scenario data is in a subdirectory named after the ID
        # with market_data.json inside (common pattern for directory-based manifests)
        if not ref:
            if base_url:
                # Construct URL: {base_url}/{scenario_id}/market_data.json
                ref = f"{scenario_id}/market_data.json"
            elif manifest_parent:
                # Local path: {manifest_parent}/{scenario_id}/market_data.json
                ref = f"{scenario_id}/market_data.json"
            else:
                raise ValueError(f"manifest entry missing url for scenario {scenario_id}")

        resolved_ref = _resolve_ref(str(ref), base_url, manifest_parent)
        scenario_dir = cache_root / scenario_id
        market_path = scenario_dir / "market_data.json"
        cache_meta = scenario_dir / ".cache.json"

        if market_path.exists() and _cache_valid(cache_meta, cache_ttl_hours):
            continue
        if not download_on_missing and not market_path.exists():
            raise FileNotFoundError(f"scenario {scenario_id} missing in cache: {scenario_dir}")

        scenario_dir.mkdir(parents=True, exist_ok=True)
        if _is_url(resolved_ref):
            artifact_path = scenario_dir / "artifact.download"
            _download_to_path(resolved_ref, artifact_path)
        else:
            artifact_path = Path(resolved_ref)
            if not artifact_path.exists():
                raise FileNotFoundError(f"scenario artifact not found: {artifact_path}")

        expected_sha = entry.get("sha256") or entry.get("artifact_sha256") or artifact_meta.get("sha256")
        if expected_sha:
            actual_sha = _sha256_file(artifact_path)
            if actual_sha.lower() != str(expected_sha).lower():
                raise RuntimeError(f"scenario {scenario_id} failed sha256 check")

        # Detect artifact format by extension or content
        artifact_str = str(artifact_path)
        is_zip = artifact_str.endswith(".zip")
        is_json = artifact_str.endswith(".json")

        # For downloaded files without proper extension, detect by content
        if artifact_path.name == "artifact.download":
            with artifact_path.open("rb") as f:
                header = f.read(4)
            if header[:2] == b"PK":  # ZIP magic bytes
                is_zip = True
            elif header[:1] in (b"{", b"["):  # JSON starts with { or [
                is_json = True

        if is_zip:
            _safe_extract_zip(artifact_path, scenario_dir)
            if artifact_path.name == "artifact.download":
                artifact_path.unlink(missing_ok=True)
        elif is_json:
            if artifact_path != market_path:
                shutil.copyfile(artifact_path, market_path)
            if artifact_path.name == "artifact.download":
                artifact_path.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"unsupported scenario artifact format: {artifact_path}")

        if entry.get("metadata"):
            _write_json(scenario_dir / "metadata.json", entry["metadata"])
        else:
            metadata = {
                "name": entry.get("name", scenario_id),
                "description": entry.get("description", ""),
                "exchange": entry.get("exchange"),
                "timeframe": entry.get("timeframe"),
                "period": entry.get("period"),
                "source": entry.get("source"),
            }
            _write_json(scenario_dir / "metadata.json", {k: v for k, v in metadata.items() if v})

        _write_json(
            cache_meta,
            {
                "scenario_id": scenario_id,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "source": resolved_ref,
            },
        )

    return cache_root


def discover_crypto_scenarios(
    path: Path,
    scenarios: Optional[List[str]] = None,
    limit: Optional[int] = None,
    shuffle: bool = True,
) -> List[CryptoScenarioIndex]:
    """Discover crypto scenarios from a directory or single JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Crypto scenarios path not found: {path}")

    indices: List[CryptoScenarioIndex] = []
    scenario_filter = {s for s in scenarios} if scenarios else None

    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if "market_states" not in data:
            raise ValueError(f"Crypto scenario file missing market_states: {path}")
        metadata = data.get("metadata", {})
        scenario_id = metadata.get("name") or path.stem
        if scenario_filter and scenario_id not in scenario_filter:
            return []
        indices.append(
            CryptoScenarioIndex(
                scenario_id=scenario_id,
                name=metadata.get("name", scenario_id),
                description=metadata.get("description", ""),
                data_path=path,
                metadata=metadata,
            )
        )
    else:
        scenario_dirs = [p for p in path.iterdir() if p.is_dir()]
        if scenario_filter is None:
            scenario_named = [p for p in scenario_dirs if p.name.startswith("scenario_")]
            if scenario_named:
                scenario_dirs = scenario_named
        for scenario_dir in sorted(scenario_dirs):
            scenario_id = scenario_dir.name
            if scenario_filter and scenario_id not in scenario_filter:
                continue

            data_path = scenario_dir / "market_data.json"
            if not data_path.exists():
                continue

            metadata_path = scenario_dir / "metadata.json"
            metadata: dict[str, Any] = {}
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            else:
                try:
                    data = json.loads(data_path.read_text(encoding="utf-8"))
                    metadata = data.get("metadata", {})
                except Exception:
                    metadata = {}

            indices.append(
                CryptoScenarioIndex(
                    scenario_id=scenario_id,
                    name=metadata.get("name", scenario_id),
                    description=metadata.get("description", ""),
                    data_path=data_path,
                    metadata=metadata,
                )
            )

    if shuffle:
        random.shuffle(indices)

    if limit:
        indices = indices[:limit]

    return indices


def load_market_states(
    data_path: Path,
    max_steps: Optional[int] = None,
    stride: int = 1,
) -> List[dict]:
    """Load and optionally downsample market states."""
    data = json.loads(data_path.read_text(encoding="utf-8"))
    states = data.get("market_states", [])

    if stride > 1:
        states = states[::stride]
    if max_steps:
        states = states[:max_steps]
    return states


def _scale_linear(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.0
    if value <= min_value:
        return 0.0
    if value >= max_value:
        return 1.0
    return (value - min_value) / (max_value - min_value)


def _parse_timeframe_minutes(timeframe: Optional[str]) -> Optional[int]:
    if not timeframe:
        return None
    tf = timeframe.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    return None


def _annualization_factor(timeframe: Optional[str]) -> float:
    minutes = _parse_timeframe_minutes(timeframe)
    if not minutes:
        return 1.0
    periods_per_year = (365.0 * 24.0 * 60.0) / minutes
    return math.sqrt(max(periods_per_year, 1.0))


def _apply_price_noise(states: List[dict], noise_level: float, rng: random.Random) -> List[dict]:
    if noise_level <= 0:
        return states
    noisy = copy.deepcopy(states)
    for state in noisy:
        ohlcv = state.get("ohlcv", {})
        if not ohlcv:
            continue
        noise = rng.gauss(0, noise_level)
        for key in ("open", "high", "low", "close"):
            if key in ohlcv and ohlcv[key] is not None:
                ohlcv[key] = float(ohlcv[key]) * (1.0 + noise)
        open_p = float(ohlcv.get("open", 0))
        close_p = float(ohlcv.get("close", 0))
        high_p = max(float(ohlcv.get("high", 0)), open_p, close_p)
        low_p = min(float(ohlcv.get("low", 0)), open_p, close_p)
        ohlcv["high"] = high_p
        ohlcv["low"] = max(low_p, 0.0001)
    return noisy


def _inject_adversarial_events(
    states: List[dict],
    injection_rate: float,
    rng: random.Random,
) -> tuple[List[dict], list[dict]]:
    if injection_rate <= 0:
        return states, []
    adv = copy.deepcopy(states)
    events: list[dict] = []
    for idx, state in enumerate(adv):
        if rng.random() > injection_rate:
            continue
        ohlcv = state.get("ohlcv", {})
        if not ohlcv:
            continue
        event_type = rng.choice(["flash_crash", "flash_pump", "liquidity_crisis"])
        close_p = float(ohlcv.get("close", 0))
        if close_p <= 0:
            continue

        if event_type == "flash_crash":
            close_p *= 0.85
        elif event_type == "flash_pump":
            close_p *= 1.12
        elif event_type == "liquidity_crisis":
            ohlcv["volume"] = float(ohlcv.get("volume", 0)) * 0.10

        ohlcv["close"] = close_p
        open_p = float(ohlcv.get("open", close_p))
        high_p = max(float(ohlcv.get("high", close_p)), open_p, close_p)
        low_p = min(float(ohlcv.get("low", close_p)), open_p, close_p)
        ohlcv["high"] = high_p
        ohlcv["low"] = max(low_p, 0.0001)

        events.append({"index": idx, "type": event_type})
    return adv, events


def _transform_scale(states: List[dict], factor: float) -> List[dict]:
    scaled = copy.deepcopy(states)
    for state in scaled:
        ohlcv = state.get("ohlcv", {})
        for key in ("open", "high", "low", "close"):
            if key in ohlcv and ohlcv[key] is not None:
                ohlcv[key] = float(ohlcv[key]) * factor
    return scaled


def _transform_invert_returns(states: List[dict]) -> List[dict]:
    inverted = copy.deepcopy(states)
    prev_close = None
    for idx, state in enumerate(inverted):
        ohlcv = state.get("ohlcv", {})
        close_p = float(ohlcv.get("close", 0))
        if idx == 0 or prev_close is None or prev_close <= 0:
            prev_close = close_p if close_p > 0 else 1.0
            continue
        new_close = max(0.0001, 2.0 * prev_close - close_p)
        ratio = new_close / close_p if close_p > 0 else 1.0

        for key in ("open", "high", "low", "close"):
            if key in ohlcv and ohlcv[key] is not None:
                ohlcv[key] = float(ohlcv[key]) * ratio

        open_p = float(ohlcv.get("open", new_close))
        high_p = max(float(ohlcv.get("high", new_close)), open_p, new_close)
        low_p = min(float(ohlcv.get("low", new_close)), open_p, new_close)
        ohlcv["high"] = high_p
        ohlcv["low"] = max(low_p, 0.0001)
        prev_close = new_close
    return inverted


def _extract_json(text: str) -> Optional[dict]:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data:
            return data[0] if isinstance(data[0], dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _parse_decision(response_text: str, symbol: str) -> dict[str, Any]:
    data = _extract_json(response_text) or {}
    action = str(data.get("action", "")).strip().upper()
    if action not in {"BUY", "SELL", "HOLD", "CLOSE"}:
        lowered = response_text.lower()
        if "buy" in lowered or "long" in lowered:
            action = "BUY"
        elif "sell" in lowered or "short" in lowered:
            action = "SELL"
        elif "close" in lowered or "exit" in lowered:
            action = "CLOSE"
        else:
            action = "HOLD"

    size = data.get("size", data.get("quantity", 0.0))
    try:
        size_value = float(size)
    except (TypeError, ValueError):
        size_value = 0.0

    stop_loss = data.get("stop_loss")
    take_profit = data.get("take_profit")
    try:
        stop_loss_val = float(stop_loss) if stop_loss is not None else None
    except (TypeError, ValueError):
        stop_loss_val = None
    try:
        take_profit_val = float(take_profit) if take_profit is not None else None
    except (TypeError, ValueError):
        take_profit_val = None

    confidence = data.get("confidence", 0.0)
    try:
        confidence_val = float(confidence)
    except (TypeError, ValueError):
        confidence_val = 0.0

    return {
        "action": action,
        "symbol": data.get("symbol", symbol),
        "size": size_value,
        "reasoning": data.get("reasoning", ""),
        "confidence": confidence_val,
        "stop_loss": stop_loss_val,
        "take_profit": take_profit_val,
    }


class TradingSimulator:
    """Simple single-asset trading simulator."""

    def __init__(self, config: CryptoEvaluationConfig, rng: random.Random):
        self.config = config
        self.rng = rng
        self.cash = config.initial_balance
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.trades: list[TradeRecord] = []
        # Seed equity curve with initial equity
        self.equity_curve: list[float] = [self.cash]
        self.realized_pnl = 0.0
        self.funding_paid = 0.0
        self.last_funding_time: Optional[datetime] = None
        self.last_funding_step = 0

    def _apply_funding_payment(self, funding_rate: float, price: float, intervals: int) -> None:
        if intervals <= 0 or price <= 0:
            return
        notional = abs(self.position_size) * price
        if notional <= 0:
            return
        payment = notional * funding_rate * intervals
        cashflow = -payment if self.position_size > 0 else payment
        self.cash += cashflow
        self.funding_paid += cashflow

    def apply_funding(
        self,
        timestamp: Optional[str],
        funding_rate: float,
        price: float,
        timeframe: Optional[str],
        step_index: int,
    ) -> None:
        if self.position_size == 0 or funding_rate == 0.0:
            return
        if self.config.funding_interval_hours <= 0:
            return

        current_time = _parse_iso_timestamp(timestamp)
        interval_seconds = self.config.funding_interval_hours * 3600.0
        if current_time:
            if self.last_funding_time is None:
                self.last_funding_time = current_time
                return
            elapsed = (current_time - self.last_funding_time).total_seconds()
            if elapsed < interval_seconds:
                return
            intervals = int(elapsed // interval_seconds)
            self._apply_funding_payment(funding_rate, price, intervals)
            self.last_funding_time = self.last_funding_time + timedelta(
                seconds=intervals * interval_seconds
            )
            return

        minutes = _parse_timeframe_minutes(timeframe)
        if not minutes:
            return
        bars_per_interval = max(int(round((self.config.funding_interval_hours * 60.0) / minutes)), 1)
        steps_since = step_index - self.last_funding_step
        if steps_since < bars_per_interval:
            return
        intervals = steps_since // bars_per_interval
        self._apply_funding_payment(funding_rate, price, intervals)
        self.last_funding_step += intervals * bars_per_interval

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        slip_min, slip_max = self.config.slippage_range
        slippage = self.rng.uniform(slip_min, slip_max)
        return price * (1.0 + slippage) if is_buy else price * (1.0 - slippage)

    def _max_position_size(self, price: float) -> float:
        equity = self.cash + self.position_size * price
        max_notional = max(equity, 0.0) * self.config.max_leverage
        if max_notional <= 0 or price <= 0:
            return 0.0
        return max_notional / price

    def _record_trade(self, exit_price: float, size: float, reason: str) -> None:
        if size == 0:
            return
        if self.position_size > 0:
            pnl = (exit_price - self.entry_price) * size
        else:
            pnl = (self.entry_price - exit_price) * size
        self.realized_pnl += pnl
        self.trades.append(
            TradeRecord(
                entry_price=self.entry_price,
                exit_price=exit_price,
                size=size,
                pnl=pnl,
                reason=reason,
            )
        )

    def _close_position(self, price: float, reason: str) -> None:
        if self.position_size == 0:
            return
        size = abs(self.position_size)
        is_buy = self.position_size < 0
        exec_price = self._apply_slippage(price, is_buy=is_buy)
        notional = size * exec_price
        fee = notional * self.config.trading_fee
        if self.position_size > 0:
            self.cash += notional - fee
        else:
            self.cash -= notional + fee
        self._record_trade(exit_price=exec_price, size=size, reason=reason)
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_loss = None
        self.take_profit = None

    def _execute_trade(self, trade_size: float, price: float, decision: dict[str, Any]) -> None:
        if trade_size == 0:
            return
        is_buy = trade_size > 0
        exec_price = self._apply_slippage(price, is_buy=is_buy)
        notional = abs(trade_size) * exec_price
        fee = notional * self.config.trading_fee

        if is_buy:
            self.cash -= notional + fee
        else:
            self.cash += notional - fee

        if self.position_size == 0 or (self.position_size > 0 and trade_size > 0) or (
            self.position_size < 0 and trade_size < 0
        ):
            new_size = self.position_size + trade_size
            if self.position_size == 0:
                self.entry_price = exec_price
            else:
                total_notional = abs(self.position_size) * self.entry_price + abs(trade_size) * exec_price
                self.entry_price = total_notional / abs(new_size)
            self.position_size = new_size
        else:
            closed_size = min(abs(trade_size), abs(self.position_size))
            self._record_trade(exit_price=exec_price, size=closed_size, reason="signal")
            remaining_size = self.position_size + trade_size
            if remaining_size == 0:
                self.position_size = 0.0
                self.entry_price = 0.0
                self.stop_loss = None
                self.take_profit = None
            elif self.position_size * remaining_size > 0:
                self.position_size = remaining_size
            else:
                self.position_size = remaining_size
                self.entry_price = exec_price

        if self.position_size != 0:
            if decision.get("stop_loss") is not None:
                self.stop_loss = decision.get("stop_loss")
            if decision.get("take_profit") is not None:
                self.take_profit = decision.get("take_profit")

    def apply_decision(
        self,
        decision: dict[str, Any],
        price: Optional[float] = None,
        current_price: Optional[float] = None,
    ) -> None:
        if current_price is not None:
            price = current_price
        if price is None or price <= 0:
            return
        action = decision.get("action", "HOLD")
        size = decision.get("size", 0.0)

        if action == "HOLD":
            return
        if action == "CLOSE":
            self._close_position(price, reason="close")
            return

        max_size = self._max_position_size(price)
        if max_size <= 0:
            return

        if action == "BUY":
            trade_size = max(0.0, size)
        elif action == "SELL":
            trade_size = -max(0.0, size)
        else:
            return

        desired_size = self.position_size + trade_size
        if abs(desired_size) > max_size:
            allowed_size = math.copysign(max_size, desired_size)
            trade_size = allowed_size - self.position_size
        self._execute_trade(trade_size, price, decision)

    def update_equity(self, price: float) -> float:
        # In unit tests we only have a single price; use it for stop/take checks.
        if self.position_size != 0 and (self.stop_loss is not None or self.take_profit is not None):
            self.check_stops(price, price)
        equity = self.cash + self.position_size * price
        self.equity_curve.append(equity)
        return equity

    def check_stops(self, high: float, low: float) -> bool:
        if self.position_size == 0:
            return False
        if self.stop_loss is None and self.take_profit is None:
            return False

        if self.position_size > 0:
            if self.stop_loss is not None and low <= self.stop_loss:
                self._close_position(self.stop_loss, reason="stop_loss")
                return True
            if self.take_profit is not None and high >= self.take_profit:
                self._close_position(self.take_profit, reason="take_profit")
                return True
        else:
            if self.stop_loss is not None and high >= self.stop_loss:
                self._close_position(self.stop_loss, reason="stop_loss")
                return True
            if self.take_profit is not None and low <= self.take_profit:
                self._close_position(self.take_profit, reason="take_profit")
                return True
        return False


class PostgresMarketDataLoader:
    """Load market data directly from PostgreSQL/TimescaleDB for crypto benchmark."""

    def __init__(
        self,
        dsn: Optional[str] = None,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "market_data",
        user: str = "postgres",
        password: Optional[str] = None,
        ohlcv_table: str = "market_data.candles_1m",
        funding_table: Optional[str] = "market_data.funding_rates",
    ):
        self.dsn = dsn
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.ohlcv_table = ohlcv_table
        self.funding_table = funding_table
        self._conn = None

    def _connect(self):
        """Establish database connection."""
        if self._conn is not None:
            return self._conn
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError as exc:
            raise RuntimeError("psycopg2 is required (pip install psycopg2-binary)") from exc

        if self.dsn:
            self._conn = psycopg2.connect(self.dsn)
        else:
            conn_kwargs = {"host": self.host, "port": self.port, "dbname": self.dbname, "user": self.user}
            if self.password:
                conn_kwargs["password"] = self.password
            self._conn = psycopg2.connect(**conn_kwargs)
        return self._conn

    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @staticmethod
    def _ema(values: List[float], period: int) -> List[Optional[float]]:
        """Compute Exponential Moving Average."""
        result: List[Optional[float]] = [None] * len(values)
        if period <= 0 or len(values) < period:
            return result
        alpha = 2.0 / (period + 1.0)
        ema_value = sum(values[:period]) / period
        result[period - 1] = ema_value
        for idx in range(period, len(values)):
            ema_value = alpha * values[idx] + (1.0 - alpha) * ema_value
            result[idx] = ema_value
        return result

    @staticmethod
    def _rsi(values: List[float], period: int) -> List[Optional[float]]:
        """Compute Relative Strength Index."""
        result: List[Optional[float]] = [None] * len(values)
        if period <= 0 or len(values) <= period:
            return result
        gains = []
        losses = []
        for i in range(1, period + 1):
            delta = values[i] - values[i - 1]
            gains.append(max(delta, 0.0))
            losses.append(max(-delta, 0.0))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else float("inf")
        result[period] = 100.0 - (100.0 / (1.0 + rs))
        for idx in range(period + 1, len(values)):
            delta = values[idx] - values[idx - 1]
            gain = max(delta, 0.0)
            loss = max(-delta, 0.0)
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            rs = avg_gain / avg_loss if avg_loss > 0 else float("inf")
            result[idx] = 100.0 - (100.0 / (1.0 + rs))
        return result

    @staticmethod
    def _atr(highs: List[float], lows: List[float], closes: List[float], period: int) -> List[Optional[float]]:
        """Compute Average True Range."""
        result: List[Optional[float]] = [None] * len(highs)
        if period <= 0 or len(highs) <= period:
            return result
        trs = []
        for idx in range(1, period + 1):
            tr = max(
                highs[idx] - lows[idx],
                abs(highs[idx] - closes[idx - 1]),
                abs(lows[idx] - closes[idx - 1]),
            )
            trs.append(tr)
        atr_value = sum(trs) / period
        result[period] = atr_value
        for idx in range(period + 1, len(highs)):
            tr = max(
                highs[idx] - lows[idx],
                abs(highs[idx] - closes[idx - 1]),
                abs(lows[idx] - closes[idx - 1]),
            )
            atr_value = (atr_value * (period - 1) + tr) / period
            result[idx] = atr_value
        return result

    @staticmethod
    def _macd(values: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> List[Optional[float]]:
        """Compute MACD line."""
        ema_fast = PostgresMarketDataLoader._ema(values, fast)
        ema_slow = PostgresMarketDataLoader._ema(values, slow)
        macd_line: List[Optional[float]] = [None] * len(values)
        for idx in range(len(values)):
            if ema_fast[idx] is None or ema_slow[idx] is None:
                continue
            macd_line[idx] = ema_fast[idx] - ema_slow[idx]
        return macd_line

    def _query_ohlcv(
        self,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
        symbol_field: str = "symbol",
        time_field: str = "bucket_ts",
    ) -> List[dict]:
        """Query OHLCV data from database."""
        import psycopg2.extras

        conn = self._connect()
        columns = [time_field, "open", "high", "low", "close", "volume", "quote_volume", "trade_count"]
        column_list = ", ".join(columns)
        query = (
            f"SELECT {column_list} FROM {self.ohlcv_table} "
            f"WHERE {symbol_field} = %s AND {time_field} >= %s AND {time_field} <= %s "
            f"ORDER BY {time_field} ASC"
        )

        rows: List[dict] = []
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (symbol, start_dt, end_dt))
            for row in cur:
                ts = row.get(time_field)
                if ts is None:
                    continue
                if isinstance(ts, datetime):
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                rows.append({
                    "timestamp": ts,
                    "open": float(row.get("open", 0)),
                    "high": float(row.get("high", 0)),
                    "low": float(row.get("low", 0)),
                    "close": float(row.get("close", 0)),
                    "volume": float(row.get("volume", 0)),
                    "quote_volume": float(row.get("quote_volume", 0)),
                    "trade_count": int(row.get("trade_count", 0) or 0),
                })
        return rows

    def _query_funding_rates(
        self,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> List[dict]:
        """Query funding rates from database."""
        if not self.funding_table:
            return []
        import psycopg2.extras

        conn = self._connect()
        query = (
            f"SELECT funding_time, funding_rate FROM {self.funding_table} "
            f"WHERE symbol = %s AND funding_time >= %s AND funding_time <= %s "
            f"ORDER BY funding_time ASC"
        )

        rates: List[dict] = []
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (symbol, start_dt, end_dt))
            for row in cur:
                ts = row.get("funding_time")
                if ts is None:
                    continue
                if isinstance(ts, datetime) and ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                rates.append({
                    "timestamp": ts,
                    "funding_rate": float(row.get("funding_rate", 0)),
                })
        return rates

    def load_window(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1m",
    ) -> List[dict]:
        """
        Load market states for a time window.

        Returns list of market state dicts compatible with CryptoTradingEvaluator.
        """
        rows = self._query_ohlcv(symbol, start, end)
        if not rows:
            return []

        funding_rates = self._query_funding_rates(symbol, start, end)

        opens = [r["open"] for r in rows]
        highs = [r["high"] for r in rows]
        lows = [r["low"] for r in rows]
        closes = [r["close"] for r in rows]

        ema_20 = self._ema(closes, 20)
        ema_50 = self._ema(closes, 50)
        rsi_14 = self._rsi(closes, 14)
        macd_line = self._macd(closes, 12, 26, 9)
        atr_14 = self._atr(highs, lows, closes, 14)

        funding_idx = 0
        current_funding: Optional[float] = None

        states = []
        for idx, row in enumerate(rows):
            if (
                ema_20[idx] is None
                or ema_50[idx] is None
                or rsi_14[idx] is None
                or macd_line[idx] is None
                or atr_14[idx] is None
            ):
                continue

            ts = row["timestamp"]
            while funding_idx < len(funding_rates) and funding_rates[funding_idx]["timestamp"] <= ts:
                current_funding = funding_rates[funding_idx].get("funding_rate")
                funding_idx += 1

            state = {
                "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
                "symbol": symbol,
                "ohlcv": {
                    "open": opens[idx],
                    "high": highs[idx],
                    "low": lows[idx],
                    "close": closes[idx],
                    "volume": row["volume"],
                    "quoteVolume": row["quote_volume"],
                    "trades": row["trade_count"],
                },
                "indicators": {
                    "ema_20": float(ema_20[idx]),
                    "ema_50": float(ema_50[idx]),
                    "rsi": float(rsi_14[idx]),
                    "macd": float(macd_line[idx]),
                    "atr": float(atr_14[idx]),
                },
                "market_metrics": {
                    "funding_rate": float(current_funding) if current_funding is not None else 0.0,
                    "open_interest": 0.0,
                },
            }
            states.append(state)

        return states

    def load_scenario(
        self,
        scenario_id: str,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1m",
        description: str = "",
    ) -> dict:
        """
        Load a complete scenario dict compatible with existing JSON format.

        Returns a dict with 'metadata' and 'market_states' keys.
        """
        states = self.load_window(symbol, start, end, timeframe)
        return {
            "metadata": {
                "source": "postgres",
                "scenario_id": scenario_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "description": description,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "data_points": len(states),
            },
            "market_states": states,
        }


class CryptoTradingEvaluator:
    """Runs multi-dimensional crypto trading evaluation with A2A agents."""

    def __init__(self, messenger: Any, timeout_seconds: int = 120):
        self.messenger = messenger
        self.timeout_seconds = timeout_seconds

    def _score_metrics(self, metrics: dict[str, float], config: CryptoEvaluationConfig) -> float:
        weights = config.metric_weights or {}
        total_weight = sum(weights.values()) or 1.0

        return_score = _scale_linear(
            metrics["total_return"],
            -config.return_cap,
            config.return_cap,
        )
        sharpe_score = _scale_linear(
            metrics["sharpe"],
            config.sharpe_floor,
            config.sharpe_cap,
        )
        drawdown_score = 1.0 - _scale_linear(
            metrics["max_drawdown"],
            0.0,
            config.drawdown_cap,
        )
        win_rate_score = _scale_linear(
            metrics["win_rate"],
            config.win_rate_floor,
            config.win_rate_cap,
        )

        weighted = (
            weights.get("total_return", 0.0) * return_score +
            weights.get("sharpe", 0.0) * sharpe_score +
            weights.get("max_drawdown", 0.0) * drawdown_score +
            weights.get("win_rate", 0.0) * win_rate_score
        )
        return max(0.0, min(100.0, (weighted / total_weight) * 100.0))

    def _grade(self, score: float) -> str:
        if score >= 90:
            return "S"
        if score >= 80:
            return "A"
        if score >= 70:
            return "B"
        if score >= 60:
            return "C"
        if score >= 50:
            return "D"
        return "F"

    async def _run_episode(
        self,
        states: List[dict],
        purple_agent_url: str,
        config: CryptoEvaluationConfig,
        new_conversation: bool,
        seed: int,
        timeframe: Optional[str],
    ) -> dict[str, Any]:
        rng = random.Random(seed)
        simulator = TradingSimulator(config, rng)
        decision_interval = config.decision_interval
        last_price = None
        decision_errors = 0

        for idx, state in enumerate(states):
            ohlcv = state.get("ohlcv", {})
            close_p = float(ohlcv.get("close", 0))
            high_p = float(ohlcv.get("high", close_p))
            low_p = float(ohlcv.get("low", close_p))
            if close_p <= 0:
                continue
            last_price = close_p

            simulator.check_stops(high_p, low_p)
            funding_rate = 0.0
            market_metrics = state.get("market_metrics", {}) or {}
            try:
                funding_rate = float(market_metrics.get("funding_rate", 0.0))
            except (TypeError, ValueError):
                funding_rate = 0.0
            simulator.apply_funding(
                timestamp=state.get("timestamp"),
                funding_rate=funding_rate,
                price=close_p,
                timeframe=timeframe,
                step_index=idx,
            )

            if idx % decision_interval == 0:
                positions = []
                if simulator.position_size != 0:
                    positions.append({
                        "symbol": state.get("symbol", ""),
                        "size": simulator.position_size,
                        "entry_price": simulator.entry_price,
                        "stop_loss": simulator.stop_loss,
                        "take_profit": simulator.take_profit,
                    })
                account = {
                    "balance": simulator.cash,
                    "equity": simulator.cash + simulator.position_size * close_p,
                    "positions": positions,
                }
                payload = {
                    "type": "trading_decision",
                    "state": {
                        "timestamp": state.get("timestamp"),
                        "symbol": state.get("symbol"),
                        "ohlcv": state.get("ohlcv", {}),
                        "indicators": state.get("indicators", {}),
                        "market_metrics": state.get("market_metrics", {}),
                        "account": account,
                    },
                }
                try:
                    response = await self.messenger.talk_to_agent(
                        message=json.dumps(payload),
                        url=purple_agent_url,
                        new_conversation=new_conversation if idx == 0 else False,
                        timeout=self.timeout_seconds,
                    )
                    decision = _parse_decision(response, symbol=state.get("symbol", ""))
                except Exception:
                    decision_errors += 1
                    decision = {"action": "HOLD", "size": 0.0}
            else:
                decision = {"action": "HOLD", "size": 0.0}

            simulator.apply_decision(decision, close_p)
            simulator.update_equity(close_p)

        if simulator.position_size != 0 and last_price is not None:
            simulator._close_position(last_price, reason="final")
            simulator.update_equity(last_price)

        metrics = self._compute_metrics(simulator.equity_curve, simulator.trades, timeframe)
        score = self._score_metrics(metrics, config)

        return {
            "score": score,
            "metrics": metrics,
            "trade_count": len(simulator.trades),
            "decision_errors": decision_errors,
        }

    def _compute_metrics(
        self,
        equity_curve: list[float],
        trades: list[TradeRecord],
        timeframe: Optional[str],
    ) -> dict[str, float]:
        if not equity_curve:
            return {
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe": 0.0,
                "win_rate": 0.0,
            }

        if equity_curve[0] <= 0:
            return {
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe": 0.0,
                "win_rate": 0.0,
            }
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        peak = equity_curve[0]
        max_drawdown = 0.0
        returns = []
        for i, equity in enumerate(equity_curve):
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak if peak > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)
            if i > 0 and equity_curve[i - 1] > 0:
                returns.append((equity - equity_curve[i - 1]) / equity_curve[i - 1])

        sharpe = 0.0
        if len(returns) > 1:
            mean_ret = statistics.mean(returns)
            std_ret = statistics.pstdev(returns)
            if std_ret > 0:
                sharpe = (mean_ret / std_ret) * _annualization_factor(timeframe)

        win_rate = 0.0
        if trades:
            wins = sum(1 for t in trades if t.pnl > 0)
            win_rate = wins / len(trades)

        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "win_rate": win_rate,
        }

    async def evaluate_scenario(
        self,
        scenario_meta: dict[str, Any],
        purple_agent_url: str,
        seed: int,
    ) -> dict[str, Any]:
        max_steps = scenario_meta.get("max_steps")
        stride = scenario_meta.get("stride", 1)
        timeframe = scenario_meta.get("metadata", {}).get("timeframe")
        config = CryptoEvaluationConfig.model_validate(scenario_meta.get("evaluation", {}))

        # Support inline market_states (from PostgreSQL mode) or file-based loading
        if "market_states" in scenario_meta:
            states = scenario_meta["market_states"]
            if stride > 1:
                states = states[::stride]
            if max_steps:
                states = states[:max_steps]
        else:
            data_path = Path(scenario_meta["data_path"])
            states = load_market_states(data_path, max_steps=max_steps, stride=stride)

        if not states:
            scenario_id = scenario_meta.get("scenario_id", "unknown")
            return {"error": f"No market states found for scenario {scenario_id}"}

        if config.seed is not None:
            seed = config.seed

        baseline = await self._run_episode(
            states=states,
            purple_agent_url=purple_agent_url,
            config=config,
            new_conversation=True,
            seed=seed,
            timeframe=timeframe,
        )

        noisy_states = _apply_price_noise(states, config.price_noise_level, random.Random(seed + 1))
        noisy = await self._run_episode(
            states=noisy_states,
            purple_agent_url=purple_agent_url,
            config=config,
            new_conversation=True,
            seed=seed + 1,
            timeframe=timeframe,
        )

        adversarial_states, events = _inject_adversarial_events(
            states,
            config.adversarial_injection_rate,
            random.Random(seed + 2),
        )
        adversarial = await self._run_episode(
            states=adversarial_states,
            purple_agent_url=purple_agent_url,
            config=config,
            new_conversation=True,
            seed=seed + 2,
            timeframe=timeframe,
        )

        meta_scores: list[float] = []
        meta_details: list[dict[str, Any]] = []
        for offset, transform in enumerate(config.meta_transforms):
            if transform == "identity":
                transformed = states
            elif transform == "scale_1_1":
                transformed = _transform_scale(states, factor=1.10)
            elif transform == "invert_returns":
                transformed = _transform_invert_returns(states)
            else:
                continue
            meta_result = await self._run_episode(
                states=transformed,
                purple_agent_url=purple_agent_url,
                config=config,
                new_conversation=True,
                seed=seed + 10 + offset,
                timeframe=timeframe,
            )
            meta_scores.append(meta_result["score"])
            meta_details.append({"transform": transform, **meta_result})

        meta_avg = statistics.mean(meta_scores) if meta_scores else 0.0
        meta_std = statistics.pstdev(meta_scores) if len(meta_scores) > 1 else 0.0
        meta_score = max(0.0, min(100.0, meta_avg - meta_std * 0.5))

        score_weights = config.score_weights or DEFAULT_SCORE_WEIGHTS
        weight_total = sum(score_weights.values()) or 1.0
        final_score = (
            score_weights.get("baseline", 0.0) * baseline["score"] +
            score_weights.get("noisy", 0.0) * noisy["score"] +
            score_weights.get("adversarial", 0.0) * adversarial["score"] +
            score_weights.get("meta", 0.0) * meta_score
        ) / weight_total

        return {
            "baseline": baseline,
            "noisy": noisy,
            "adversarial": adversarial,
            "meta": {
                "score": meta_score,
                "average": meta_avg,
                "dispersion": meta_std,
                "details": meta_details,
            },
            "final_score": round(final_score, 2),
            "grade": self._grade(final_score),
            "events": events,
            "random_seed": seed,
        }
