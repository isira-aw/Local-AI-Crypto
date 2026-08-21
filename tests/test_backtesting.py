import datetime as dt

import numpy as np
import pandas as pd
import pytest

from crypto_ai.backtesting import metrics as m
from crypto_ai.backtesting.engine import BacktestEngine
from crypto_ai.backtesting.reports import format_text_report, save_equity_curve_chart


def _df(prices, signals, start=None):
    start = start or dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    timestamps = [start + dt.timedelta(minutes=5 * i) for i in range(len(prices))]
    return pd.DataFrame({"timestamp": timestamps, "close": prices, "signal": signals})


def test_no_trades_when_always_hold():
    df = _df([100, 101, 102, 103], ["HOLD"] * 4)
    report = BacktestEngine(initial_balance=1000).run(df)
    assert report.metrics["n_trades"] == 0
    assert report.final_balance == pytest.approx(1000.0)


def test_simple_profitable_round_trip():
    prices = [100, 100, 110, 110, 110]
    signals = ["HOLD", "BUY", "HOLD", "SELL", "HOLD"]
    df = _df(prices, signals)
    engine = BacktestEngine(initial_balance=1000, fee_pct=0.0, slippage_pct=0.0)
    report = engine.run(df)
    assert report.metrics["n_trades"] == 1
    assert report.final_balance == pytest.approx(1100.0, rel=1e-6)
    assert report.trades[0].pnl == pytest.approx(100.0, rel=1e-6)


def test_fees_and_slippage_reduce_profit():
    prices = [100, 100, 110, 110]
    signals = ["HOLD", "BUY", "HOLD", "SELL"]
    df = _df(prices, signals)
    no_cost = BacktestEngine(initial_balance=1000, fee_pct=0.0, slippage_pct=0.0).run(df)
    with_cost = BacktestEngine(initial_balance=1000, fee_pct=0.001, slippage_pct=0.0005).run(df)
    assert with_cost.final_balance < no_cost.final_balance
    assert with_cost.fees_paid > 0


def test_stop_loss_triggers_before_signal():
    prices = [100, 100, 80, 80, 80]  # crashes 20% right after entry
    highs = [100, 100, 80, 80, 80]
    lows = [100, 100, 78, 78, 78]
    signals = ["HOLD", "BUY", "HOLD", "HOLD", "SELL"]
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    timestamps = [start + dt.timedelta(minutes=5 * i) for i in range(len(prices))]
    df = pd.DataFrame({"timestamp": timestamps, "close": prices, "high": highs, "low": lows, "signal": signals})

    engine = BacktestEngine(initial_balance=1000, fee_pct=0.0, slippage_pct=0.0, stop_loss_pct=0.05)
    report = engine.run(df)
    assert report.metrics["n_trades"] == 1
    assert report.trades[0].exit_reason == "stop_loss"
    assert report.trades[0].pnl < 0


def test_take_profit_triggers():
    prices = [100, 100, 120, 120, 120]
    highs = [100, 100, 121, 121, 121]
    lows = [100, 100, 119, 119, 119]
    signals = ["HOLD", "BUY", "HOLD", "HOLD", "HOLD"]
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    timestamps = [start + dt.timedelta(minutes=5 * i) for i in range(len(prices))]
    df = pd.DataFrame({"timestamp": timestamps, "close": prices, "high": highs, "low": lows, "signal": signals})

    engine = BacktestEngine(initial_balance=1000, fee_pct=0.0, slippage_pct=0.0, take_profit_pct=0.10)
    report = engine.run(df)
    assert report.trades[0].exit_reason == "take_profit"
    assert report.trades[0].pnl > 0


def test_open_position_force_closed_at_end():
    prices = [100, 100, 105]
    signals = ["HOLD", "BUY", "HOLD"]
    df = _df(prices, signals)
    report = BacktestEngine(initial_balance=1000, fee_pct=0.0, slippage_pct=0.0).run(df)
    assert report.metrics["n_trades"] == 1
    assert report.trades[0].exit_reason == "end_of_data"


def test_no_lookahead_backtest_uses_only_current_and_past_signal():
    """Changing a FUTURE signal must not change past trade decisions/equity."""
    prices = [100, 101, 102, 103, 104, 105]
    signals_a = ["HOLD", "BUY", "HOLD", "HOLD", "SELL", "HOLD"]
    df_a = _df(prices, signals_a)
    report_a = BacktestEngine(initial_balance=1000).run(df_a)

    signals_b = list(signals_a)
    signals_b[-1] = "BUY"  # change only the very last, future signal
    df_b = _df(prices, signals_b)
    report_b = BacktestEngine(initial_balance=1000).run(df_b)

    # Equity up to (and including) the SELL bar must be identical.
    idx = 4
    pd.testing.assert_series_equal(
        report_a.equity_curve.iloc[: idx + 1], report_b.equity_curve.iloc[: idx + 1]
    )


def test_max_drawdown_and_sharpe_basic():
    equity = pd.Series([100, 110, 90, 95, 120])
    dd = m.max_drawdown_pct(equity)
    assert dd == pytest.approx((110 - 90) / 110 * 100, rel=1e-6)

    flat = pd.Series([100, 100, 100, 100])
    assert m.sharpe_ratio(flat) == 0.0


def test_deflated_sharpe_penalizes_many_trials():
    dsr_few = m.deflated_sharpe_ratio(observed_sharpe=0.5, n_trials=1, n_observations=100)
    dsr_many = m.deflated_sharpe_ratio(observed_sharpe=0.5, n_trials=200, n_observations=100)
    assert dsr_many < dsr_few


def test_suspicious_result_flags_catches_too_good():
    metrics = {"sharpe_ratio": 8.0, "win_rate_pct": 95.0, "total_return_pct": 500.0}
    flags = m.suspicious_result_flags(metrics, n_trades=5)
    assert len(flags) >= 3


def test_format_text_report_contains_disclaimer():
    df = _df([100, 100, 110, 110], ["HOLD", "BUY", "HOLD", "SELL"])
    report = BacktestEngine(initial_balance=1000).run(df)
    text = format_text_report(report)
    assert "does not guarantee future results" in text


def test_save_equity_curve_chart(tmp_path):
    df = _df([100, 101, 102, 103, 104], ["HOLD", "BUY", "HOLD", "HOLD", "SELL"])
    report = BacktestEngine(initial_balance=1000).run(df)
    out = save_equity_curve_chart(report, tmp_path / "chart.png")
    assert out.exists()
    assert out.stat().st_size > 0
