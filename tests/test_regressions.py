"""
Regression tests for bugs found during end-to-end validation.

Each test here corresponds to a defect that shipped in an earlier commit
and was caught by scripts/e2e_validation.py. They exist so those specific
failures cannot silently return.
"""

import datetime as dt

import pandas as pd
import pytest

from crypto_ai.models.evaluation.criteria import FoldEvaluation, aggregate_walk_forward

START = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)


# ---------------------------------------------------------------------
# FIX-8: naive/aware datetime crash
# ---------------------------------------------------------------------
def test_equity_curve_index_is_timezone_aware(db_session):
    """
    SQLite drops tzinfo on DateTime columns. get_current_risk_state
    compares the equity-curve index against a tz-aware "now", so a naive
    index raised TypeError and killed every paper-trading tick.
    """
    from crypto_ai.paper_trading.portfolio import get_equity_curve, record_snapshot

    for i in range(3):
        record_snapshot(
            db_session, mode="PAPER", timestamp=START + dt.timedelta(minutes=5 * i),
            cash_balance=1000.0, position_qty=0.0, current_price=100.0,
        )
    db_session.commit()

    curve = get_equity_curve(db_session, mode="PAPER")
    assert len(curve) == 3
    assert curve.index.tz is not None, "equity curve index must be tz-aware"
    # The exact comparison that used to raise TypeError:
    _ = curve[curve.index >= dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)]


def test_get_current_risk_state_does_not_raise_on_naive_db_timestamps(db_session):
    from crypto_ai.app.services.live_pipeline import get_current_risk_state
    from crypto_ai.paper_trading.portfolio import record_snapshot

    record_snapshot(
        db_session, mode="PAPER", timestamp=START, cash_balance=1000.0,
        position_qty=0.0, current_price=100.0,
    )
    db_session.commit()

    state = get_current_risk_state(db_session, mode="PAPER")  # used to raise TypeError
    assert state.current_equity == pytest.approx(1000.0)
    assert state.now.tzinfo is not None


def test_as_utc_normalizes_naive_and_preserves_aware():
    from crypto_ai.utils.timeutils import as_utc

    naive = dt.datetime(2024, 1, 1, 12, 0)
    aware = dt.datetime(2024, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    assert as_utc(naive) == aware
    assert as_utc(aware) == aware
    assert as_utc(None) is None


# ---------------------------------------------------------------------
# FIX-9: single-fold walk-forward must not count as a pass
# ---------------------------------------------------------------------
def _passing_fold(i):
    return FoldEvaluation(fold_index=i, passed=True, reasons=[])


def test_single_fold_cannot_produce_an_overall_pass():
    """1/1 folds passed is 100% — but proves nothing about robustness."""
    summary = aggregate_walk_forward([_passing_fold(0)], {"min_folds_passed_pct": 0.8})
    assert summary["pct_passed"] == 1.0
    assert summary["overall_pass"] is False
    assert summary["enough_folds"] is False
    assert "only 1 walk-forward fold" in summary["blocking_reason"]


def test_two_folds_still_insufficient_by_default():
    summary = aggregate_walk_forward([_passing_fold(0), _passing_fold(1)], {"min_folds_passed_pct": 0.8})
    assert summary["overall_pass"] is False


def test_three_passing_folds_do_pass():
    summary = aggregate_walk_forward(
        [_passing_fold(0), _passing_fold(1), _passing_fold(2)], {"min_folds_passed_pct": 0.8}
    )
    assert summary["enough_folds"] is True
    assert summary["overall_pass"] is True
    assert summary["blocking_reason"] is None


def test_min_folds_required_is_configurable():
    summary = aggregate_walk_forward(
        [_passing_fold(0)], {"min_folds_passed_pct": 0.8, "min_folds_required": 1}
    )
    assert summary["overall_pass"] is True


# ---------------------------------------------------------------------
# FIX-1/FIX-2: risk config must actually reach the training backtests
# ---------------------------------------------------------------------
def test_risk_config_is_read_from_risk_yaml_not_settings_yaml():
    """
    settings.yaml's `risk:` block only holds `config_file: risk.yaml`, so
    settings.get("risk.position_sizing...") silently returned the default
    and ignored the user's real configuration.
    """
    from crypto_ai.config.loader import get_risk_config, get_settings

    assert get_settings().get("risk.position_sizing.max_position_percent") is None
    assert get_risk_config()["position_sizing"]["max_position_percent"] == 5.0


def test_backtest_position_sizing_actually_limits_exposure():
    from crypto_ai.backtesting.engine import BacktestEngine

    df = pd.DataFrame({
        "timestamp": [START + dt.timedelta(minutes=5 * i) for i in range(4)],
        "close": [100.0, 100.0, 200.0, 200.0],
        "signal": ["HOLD", "BUY", "HOLD", "SELL"],
    })
    full = BacktestEngine(initial_balance=1000, fee_pct=0, slippage_pct=0, position_size_pct=1.0).run(df)
    capped = BacktestEngine(initial_balance=1000, fee_pct=0, slippage_pct=0, position_size_pct=0.05).run(df)

    # 100% sizing doubles the account on a 2x move; 5% sizing gains ~5%.
    assert full.final_balance == pytest.approx(2000.0, rel=1e-6)
    assert capped.final_balance == pytest.approx(1050.0, rel=1e-6)


# ---------------------------------------------------------------------
# FIX-3: signals must actually be persisted
# ---------------------------------------------------------------------
def test_signal_is_persisted_and_retrievable(db_session):
    from crypto_ai.database.repositories.signals_repo import (
        count_signals_by_type, get_latest_signal, save_signal,
    )
    from crypto_ai.strategies.signal_engine import build_signal

    sig = build_signal(
        symbol="BTC/USDT", timestamp=START, predicted_class="BUY", confidence=0.8,
        risk_score=0.2, model_version="model_001", strategy_version="s1",
    )
    save_signal(db_session, sig)
    db_session.commit()

    latest = get_latest_signal(db_session, "BTC/USDT")
    assert latest is not None
    assert latest.signal == "BUY"
    assert latest.confidence == pytest.approx(0.8)

    counts = count_signals_by_type(db_session, "BTC/USDT", since=START - dt.timedelta(days=1))
    assert counts["BUY"] == 1 and counts["SELL"] == 0 and counts["WAIT"] == 0


def test_risk_override_is_recorded_as_wait_signal(db_session):
    """A model saying BUY but the risk engine vetoing must persist as WAIT."""
    from crypto_ai.database.repositories.signals_repo import get_latest_signal, save_signal
    from crypto_ai.strategies.signal_engine import build_signal

    sig = build_signal(
        symbol="BTC/USDT", timestamp=START, predicted_class="BUY", confidence=0.9,
        risk_score=0.9, model_version="model_001", strategy_version="s1",
        trading_signal_override="WAIT", reason_codes=["daily_loss_limit_reached"],
    )
    save_signal(db_session, sig)
    db_session.commit()

    latest = get_latest_signal(db_session, "BTC/USDT")
    assert latest.signal == "WAIT"
    assert "daily_loss_limit_reached" in latest.reason_codes
    assert "risk_override" in latest.reason_codes


# ---------------------------------------------------------------------
# FIX-7: health check must not mutate the caller's client
# ---------------------------------------------------------------------
def test_check_exchange_does_not_mutate_callers_retry_policy(db_session):
    from crypto_ai.exchange.market_data import MarketDataClient, RetryPolicy
    from crypto_ai.monitoring.health import check_exchange

    class BrokenSource:
        def fetch_ohlcv(self, *a, **k):
            raise ConnectionError("no network")

    client = MarketDataClient(source=BrokenSource(), retry_policy=RetryPolicy(max_attempts=5))
    check_exchange(db_session, client=client)
    assert client.retry_policy.max_attempts == 5, "health check clobbered the caller's retry policy"


def test_retention_bounds_portfolio_snapshots(db_session):
    from crypto_ai.data.processor.retention import apply_retention_policy
    from crypto_ai.database.models.trading import PortfolioSnapshot
    from crypto_ai.paper_trading.portfolio import record_snapshot

    now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    record_snapshot(db_session, "PAPER", now - dt.timedelta(days=400), 1000.0, 0.0, 100.0)
    record_snapshot(db_session, "PAPER", now - dt.timedelta(days=1), 1000.0, 0.0, 100.0)
    db_session.commit()

    apply_retention_policy(db_session, now=now)
    remaining = db_session.query(PortfolioSnapshot).count()
    assert remaining == 1, "old portfolio snapshots should be pruned, recent kept"


# ---------------------------------------------------------------------
# Live collector (FIX-6)
# ---------------------------------------------------------------------
def test_live_collector_ignores_unclosed_candles():
    from crypto_ai.data.collector.live_collector import parse_kline_message

    open_candle = {"k": {"x": False, "t": 1700000000000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "10"}}
    assert parse_kline_message(open_candle) is None


def test_live_collector_parses_closed_candle():
    from crypto_ai.data.collector.live_collector import parse_kline_message

    closed = {"k": {"x": True, "t": 1700000000000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "10"}}
    candle = parse_kline_message(closed)
    assert candle["open"] == 1.0 and candle["high"] == 2.0
    assert candle["close"] == 1.5 and candle["volume"] == 10.0
    assert candle["timestamp"].tzinfo is not None


def test_live_collector_stream_name():
    from crypto_ai.data.collector.live_collector import stream_name

    assert stream_name("BTC/USDT", "5m") == "btcusdt@kline_5m"
    with pytest.raises(ValueError):
        stream_name("BTC/USDT", "7m")
