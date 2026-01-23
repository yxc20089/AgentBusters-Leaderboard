"""
Unit tests for crypto trading benchmark.

Tests cover:
- Hidden windows selection
- Trading decision parsing
- Trading simulator
- Crypto benchmark evaluator
- Market data loading
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestHiddenWindows:
    """Test hidden windows anti-overfitting strategy."""

    def test_load_hidden_seed(self):
        """Test loading hidden seed from config file."""
        from cio_agent.hidden_windows import load_hidden_seed, save_hidden_seed, get_hidden_config_path

        # Create temp config
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                # Save a seed
                save_hidden_seed("test_config", 0x12345678)

                # Load it back
                seed = load_hidden_seed("test_config")
                assert seed == 0x12345678

    def test_load_hidden_seed_hex_format(self):
        """Test loading seed in hex string format."""
        from cio_agent.hidden_windows import load_hidden_config
        import yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / ".agentbusters" / "hidden_seeds.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)

            config_path.write_text(yaml.dump({
                "test_v1": {"master_seed": "0x5F3A2B1C"}
            }))

            with patch('cio_agent.hidden_windows.get_hidden_config_path', return_value=config_path):
                from cio_agent.hidden_windows import load_hidden_seed
                seed = load_hidden_seed("test_v1")
                assert seed == 0x5F3A2B1C

    def test_select_evaluation_windows_deterministic(self):
        """Test that window selection is deterministic with same seed."""
        from cio_agent.hidden_windows import select_evaluation_windows

        windows1 = select_evaluation_windows(
            master_seed=12345,
            window_count=5,
            symbols=["BTCUSDT", "ETHUSDT"],
            date_range=("2023-01-01", "2023-12-31"),
            min_bars=100,
            max_bars=500,
        )

        windows2 = select_evaluation_windows(
            master_seed=12345,
            window_count=5,
            symbols=["BTCUSDT", "ETHUSDT"],
            date_range=("2023-01-01", "2023-12-31"),
            min_bars=100,
            max_bars=500,
        )

        # Same seed should produce same windows
        assert len(windows1) == len(windows2) == 5
        assert [w["scenario_id"] for w in windows1] == [w["scenario_id"] for w in windows2]

    def test_select_evaluation_windows_different_seeds(self):
        """Test that different seeds produce different windows."""
        from cio_agent.hidden_windows import select_evaluation_windows

        windows1 = select_evaluation_windows(
            master_seed=11111,
            window_count=3,
            symbols=["BTCUSDT"],
            date_range=("2023-01-01", "2023-12-31"),
            min_bars=100,
            max_bars=500,
        )

        windows2 = select_evaluation_windows(
            master_seed=22222,
            window_count=3,
            symbols=["BTCUSDT"],
            date_range=("2023-01-01", "2023-12-31"),
            min_bars=100,
            max_bars=500,
        )

        # Different seeds should produce different windows
        assert [w["scenario_id"] for w in windows1] != [w["scenario_id"] for w in windows2]

    def test_scenario_id_is_hashed(self):
        """Test that scenario IDs are anonymized hashes."""
        from cio_agent.hidden_windows import select_evaluation_windows

        windows = select_evaluation_windows(
            master_seed=99999,
            window_count=2,
            symbols=["BTCUSDT"],
            date_range=("2023-01-01", "2023-12-31"),
            min_bars=100,
            max_bars=500,
        )

        for w in windows:
            # Scenario ID should be hashed format
            assert w["scenario_id"].startswith("btcusdt_")
            # Should be hex characters after prefix
            hash_part = w["scenario_id"].split("_")[1]
            assert len(hash_part) == 12
            assert all(c in "0123456789abcdef" for c in hash_part)


class TestTradingDecisionParsing:
    """Test trading decision parsing in Purple Agent."""

    def test_parse_valid_trading_decision(self):
        """Test parsing valid trading decision request."""
        from purple_agent.executor import FinanceAgentExecutor

        executor = FinanceAgentExecutor(llm_client=None)

        input_json = json.dumps({
            "type": "trading_decision",
            "state": {
                "symbol": "BTCUSDT",
                "ohlcv": {"open": 50000, "high": 50500, "low": 49500, "close": 50200, "volume": 1000},
                "indicators": {"ema_20": 49800, "rsi": 55},
                "account": {"balance": 10000, "equity": 10000, "positions": []}
            }
        })

        result = executor._try_parse_trading_decision(input_json)

        assert result is not None
        assert result["symbol"] == "BTCUSDT"
        assert result["ohlcv"]["close"] == 50200
        assert result["indicators"]["rsi"] == 55

    def test_parse_non_trading_decision(self):
        """Test that non-trading requests return None."""
        from purple_agent.executor import FinanceAgentExecutor

        executor = FinanceAgentExecutor(llm_client=None)

        # Regular question
        result = executor._try_parse_trading_decision("What is AAPL's P/E ratio?")
        assert result is None

        # Different type
        result = executor._try_parse_trading_decision(json.dumps({"type": "question", "text": "test"}))
        assert result is None

    def test_parse_invalid_json(self):
        """Test handling of invalid JSON."""
        from purple_agent.executor import FinanceAgentExecutor

        executor = FinanceAgentExecutor(llm_client=None)

        result = executor._try_parse_trading_decision("not valid json {")
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_trading_decision_fallback(self):
        """Test rule-based fallback trading decision."""
        from purple_agent.executor import FinanceAgentExecutor

        executor = FinanceAgentExecutor(llm_client=None)  # No LLM = use fallback

        state = {
            "symbol": "BTCUSDT",
            "ohlcv": {"open": 50000, "high": 50500, "low": 49500, "close": 50200, "volume": 1000},
            "indicators": {"ema_20": 49800, "ema_50": 49500, "rsi": 25, "atr": 500},  # RSI oversold
            "account": {"balance": 10000, "equity": 10000, "positions": []}
        }

        response = await executor._handle_trading_decision(state)
        decision = json.loads(response)

        assert decision["action"] in ["BUY", "SELL", "HOLD", "CLOSE"]
        assert "size" in decision
        assert "reasoning" in decision


class TestTradingSimulator:
    """Test trading simulator logic."""

    def _create_simulator(self, initial_balance=10000, max_leverage=3.0):
        """Helper to create simulator with config."""
        import random
        from cio_agent.eval_config import CryptoEvaluationConfig
        from cio_agent.crypto_benchmark import TradingSimulator

        config = CryptoEvaluationConfig(
            initial_balance=initial_balance,
            max_leverage=max_leverage,
        )
        rng = random.Random(42)
        return TradingSimulator(config, rng)

    def test_simulator_initialization(self):
        """Test simulator initializes correctly."""
        sim = self._create_simulator(initial_balance=10000, max_leverage=3.0)

        assert sim.cash == 10000
        assert sim.position_size == 0
        assert len(sim.trades) == 0
        assert len(sim.equity_curve) == 1  # Initial equity

    def test_simulator_buy_decision(self):
        """Test simulator handles BUY decision."""
        sim = self._create_simulator(initial_balance=10000, max_leverage=1.0)

        decision = {"action": "BUY", "size": 0.5, "stop_loss": 48000, "take_profit": 55000}
        sim.apply_decision(decision, current_price=50000)

        assert sim.position_size > 0
        assert sim.entry_price == 50000
        assert sim.stop_loss == 48000
        assert sim.take_profit == 55000

    def test_simulator_sell_decision(self):
        """Test simulator handles SELL (short) decision."""
        sim = self._create_simulator(initial_balance=10000, max_leverage=1.0)

        decision = {"action": "SELL", "size": 0.3}
        sim.apply_decision(decision, current_price=50000)

        assert sim.position_size < 0  # Short position
        assert sim.entry_price == 50000

    def test_simulator_hold_decision(self):
        """Test simulator handles HOLD decision."""
        sim = self._create_simulator(initial_balance=10000)
        initial_cash = sim.cash

        decision = {"action": "HOLD", "size": 0}
        sim.apply_decision(decision, current_price=50000)

        assert sim.cash == initial_cash
        assert sim.position_size == 0

    def test_simulator_equity_tracking(self):
        """Test equity curve is tracked correctly."""
        sim = self._create_simulator(initial_balance=10000)

        # Buy
        sim.apply_decision({"action": "BUY", "size": 0.5}, current_price=100)
        sim.update_equity(100)

        # Price goes up
        sim.update_equity(110)

        # Price goes down
        sim.update_equity(95)

        assert len(sim.equity_curve) >= 3  # Initial + updates
        # Check equity changes with price

    def test_simulator_stop_loss_trigger(self):
        """Test stop loss closes position."""
        sim = self._create_simulator(initial_balance=10000)

        # Buy with stop loss
        sim.apply_decision({"action": "BUY", "size": 0.5, "stop_loss": 95}, current_price=100)

        # Price drops below stop loss
        sim.update_equity(90)

        # Position should be closed
        assert sim.position_size == 0
        assert len(sim.trades) >= 1


class TestCryptoMetrics:
    """Test crypto benchmark metrics calculation."""

    def _create_evaluator(self):
        """Helper to create evaluator with mock messenger."""
        from cio_agent.crypto_benchmark import CryptoTradingEvaluator
        from unittest.mock import Mock

        messenger = Mock()
        return CryptoTradingEvaluator(messenger=messenger, timeout_seconds=60)

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        evaluator = self._create_evaluator()

        # Positive returns
        equity_curve = [10000, 10100, 10200, 10150, 10300]
        metrics = evaluator._compute_metrics(equity_curve, [], timeframe="1m")

        assert "sharpe" in metrics
        # Sharpe ratio should be a number
        assert isinstance(metrics["sharpe"], (int, float))

    def test_max_drawdown_calculation(self):
        """Test max drawdown calculation."""
        evaluator = self._create_evaluator()

        # Equity curve with 20% drawdown
        equity_curve = [10000, 11000, 12000, 9600, 10000]  # Peak 12000, trough 9600 = 20% DD
        metrics = evaluator._compute_metrics(equity_curve, [], timeframe="1m")

        assert "max_drawdown" in metrics
        assert metrics["max_drawdown"] >= 0.15  # At least 15% drawdown

    def test_win_rate_calculation(self):
        """Test win rate calculation from trades."""
        from cio_agent.crypto_benchmark import TradeRecord

        evaluator = self._create_evaluator()

        trades = [
            TradeRecord(entry_price=100, exit_price=110, size=1, pnl=100, reason="tp"),   # Win
            TradeRecord(entry_price=100, exit_price=95, size=1, pnl=-50, reason="sl"),    # Loss
            TradeRecord(entry_price=100, exit_price=120, size=1, pnl=200, reason="tp"),   # Win
            TradeRecord(entry_price=100, exit_price=105, size=1, pnl=50, reason="tp"),    # Win
        ]

        equity_curve = [10000, 10100, 10050, 10250, 10300]
        metrics = evaluator._compute_metrics(equity_curve, trades, timeframe="1m")

        assert "win_rate" in metrics
        assert metrics["win_rate"] >= 0.5  # Majority wins

    def test_total_return_calculation(self):
        """Test total return calculation."""
        evaluator = self._create_evaluator()

        # 15% gain
        equity_curve = [10000, 10500, 11000, 11500]
        metrics = evaluator._compute_metrics(equity_curve, [], timeframe="1m")

        assert "total_return" in metrics
        assert metrics["total_return"] > 0  # Positive return


class TestCryptoDatasetConfig:
    """Test crypto dataset configuration."""

    def test_crypto_config_defaults(self):
        """Test CryptoDatasetConfig default values."""
        from cio_agent.eval_config import CryptoDatasetConfig

        config = CryptoDatasetConfig(path="data/crypto/test")

        assert config.type == "crypto"
        assert config.pg_enabled is False
        # Check evaluation config exists
        assert config.evaluation is not None

    def test_crypto_config_with_postgres(self):
        """Test CryptoDatasetConfig with PostgreSQL enabled."""
        from cio_agent.eval_config import CryptoDatasetConfig

        config = CryptoDatasetConfig(
            path="data/crypto/test",
            pg_enabled=True,
            pg_host="localhost",
            pg_port=5432,
            pg_dbname="market_data",
            hidden_seed_config="crypto_benchmark_v1",
            window_count=12,
        )

        assert config.pg_enabled is True
        assert config.pg_host == "localhost"
        assert config.hidden_seed_config == "crypto_benchmark_v1"
        assert config.window_count == 12

    def test_crypto_evaluation_config(self):
        """Test CryptoEvaluationConfig values."""
        from cio_agent.eval_config import CryptoEvaluationConfig

        config = CryptoEvaluationConfig(
            initial_balance=20000,
            max_leverage=2.0,
            decision_interval=5,
        )

        assert config.initial_balance == 20000
        assert config.max_leverage == 2.0
        assert config.decision_interval == 5
        assert config.metric_weights["sharpe"] == 0.50  # Default


class TestDecisionParsing:
    """Test parsing agent responses to trading decisions."""

    def test_parse_json_decision(self):
        """Test parsing JSON formatted decision."""
        from cio_agent.crypto_benchmark import _parse_decision

        response = '{"action": "BUY", "size": 0.1, "stop_loss": 48000, "take_profit": 55000}'
        decision = _parse_decision(response, symbol="BTCUSDT")

        assert decision["action"] == "BUY"
        assert decision["size"] == 0.1
        assert decision["stop_loss"] == 48000

    def test_parse_json_in_markdown(self):
        """Test parsing JSON inside markdown code block."""
        from cio_agent.crypto_benchmark import _parse_decision

        response = '''Here's my analysis:
```json
{"action": "SELL", "size": 0.2, "reasoning": "bearish signal"}
```
'''
        decision = _parse_decision(response, symbol="BTCUSDT")

        assert decision["action"] == "SELL"
        assert decision["size"] == 0.2

    def test_parse_invalid_response(self):
        """Test fallback for invalid response."""
        from cio_agent.crypto_benchmark import _parse_decision

        response = "I think you should buy but I'm not sure"
        decision = _parse_decision(response, symbol="BTCUSDT")

        # Should return some valid action (HOLD or extracted from text)
        assert decision["action"] in ["BUY", "SELL", "HOLD", "CLOSE"]
        assert "size" in decision


class TestCryptoIntegration:
    """Integration tests for crypto benchmark (requires test data)."""

    @pytest.mark.skipif(
        not Path("data/crypto/scenarios/sample_btc_window").exists(),
        reason="Test data not available"
    )
    def test_load_sample_scenario(self):
        """Test loading sample BTC scenario."""
        from cio_agent.eval_config import (
            EvaluationConfig, CryptoDatasetConfig,
            SamplingConfig, ConfigurableDatasetLoader
        )

        config = EvaluationConfig(
            datasets=[
                CryptoDatasetConfig(
                    path="data/crypto/scenarios",
                    scenarios=["sample_btc_window"],
                    limit=1,
                )
            ],
            sampling=SamplingConfig(strategy="sequential", total_limit=1),
        )

        loader = ConfigurableDatasetLoader(config)
        examples = loader.load()

        assert len(examples) == 1
        assert examples[0].dataset_type == "crypto"

    @pytest.mark.skipif(
        not Path("data/crypto/hidden").exists() or len(list(Path("data/crypto/hidden").glob("scenario_*"))) == 0,
        reason="Hidden eval data not available"
    )
    def test_load_hidden_scenarios(self):
        """Test loading hidden evaluation scenarios."""
        from cio_agent.eval_config import (
            EvaluationConfig, CryptoDatasetConfig,
            SamplingConfig, ConfigurableDatasetLoader
        )

        config = EvaluationConfig(
            datasets=[
                CryptoDatasetConfig(
                    path="data/crypto/hidden",
                    limit=1,
                )
            ],
            sampling=SamplingConfig(strategy="sequential", total_limit=1),
        )

        loader = ConfigurableDatasetLoader(config)
        examples = loader.load()

        if len(examples) > 0:
            # Scenario ID should be anonymized
            assert "scenario_" in examples[0].example_id
