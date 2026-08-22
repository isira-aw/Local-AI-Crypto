"""
Tests for the paper-trading epoch reset and the evidence-based
live-trading gate.

The point of both: the live-trading checklist must be satisfiable only by
evidence the system can verify, not by a caller asserting things are fine.
"""

import datetime as dt

import pytest

from crypto_ai.app.services.paper_reset import (
    get_epoch_start,
    reset_paper_trading,
    summarize_since_epoch,
)
from crypto_ai.database.models.trading import PaperTrade, PortfolioSnapshot

NOW = dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)


def _add_trade(session, entry: dt.datetime, pnl: float = 5.0):
    session.add(PaperTrade(
        symbol="BTC/USDT", side="BUY", entry_time=entry, entry_price=100.0,
        exit_time=entry + dt.timedelta(minutes=30), exit_price=105.0,
        quantity=1.0, pnl=pnl, status="CLOSED", strategy_version="s1",
    ))


# ---------------------------------------------------------------------
# Epoch reset
# ---------------------------------------------------------------------
def test_without_a_reset_days_count_from_first_trade(db_session):
    _add_trade(db_session, NOW - dt.timedelta(days=10))
    db_session.commit()

    summary = summarize_since_epoch(db_session, now=NOW)
    assert summary["epoch_start"] is None
    assert summary["closed_trades_since_epoch"] == 1
    assert summary["days_since_epoch"] == pytest.approx(10.0, abs=0.1)
    assert "no reset recorded" in summary["counting_from"]


def test_reset_zeroes_the_clock_and_discards_prior_trades(db_session):
    for i in range(5):
        _add_trade(db_session, NOW - dt.timedelta(days=20 + i))
    db_session.add(PortfolioSnapshot(
        mode="PAPER", timestamp=NOW - dt.timedelta(days=20),
        cash_balance=1000.0, position_qty=0.0, position_value=0.0, total_equity=1000.0,
    ))
    db_session.commit()
    assert db_session.query(PaperTrade).count() == 5

    marker = reset_paper_trading(
        db_session, reason="risk cap was disabled before this point",
        create_backup=False, now=NOW,
    )

    assert marker["started_at"] == NOW.isoformat()
    assert marker["discarded"]["paper_trades"] == 5
    assert db_session.query(PaperTrade).count() == 0
    assert db_session.query(PortfolioSnapshot).count() == 0

    summary = summarize_since_epoch(db_session, now=NOW)
    assert summary["days_since_epoch"] == 0.0
    assert summary["closed_trades_since_epoch"] == 0
    assert summary["counting_from"] == "epoch marker"


def test_trades_before_the_epoch_never_count_even_if_kept(db_session):
    """--keep-history retains rows for the record but excludes them."""
    _add_trade(db_session, NOW - dt.timedelta(days=30))
    db_session.commit()

    reset_paper_trading(db_session, reason="test", purge_history=False, create_backup=False, now=NOW)

    assert db_session.query(PaperTrade).count() == 1, "history should be retained"
    summary = summarize_since_epoch(db_session, now=NOW)
    assert summary["closed_trades_since_epoch"] == 0, "pre-epoch trades must not count"


def test_trades_after_the_epoch_do_count(db_session):
    reset_paper_trading(db_session, reason="test", create_backup=False, now=NOW)
    _add_trade(db_session, NOW + dt.timedelta(days=1))
    db_session.commit()

    summary = summarize_since_epoch(db_session, now=NOW + dt.timedelta(days=2))
    assert summary["closed_trades_since_epoch"] == 1
    assert summary["days_since_epoch"] == pytest.approx(2.0, abs=0.1)


def test_reset_is_recorded_as_a_warning_event(db_session):
    from crypto_ai.database.models.monitoring import SystemEvent

    reset_paper_trading(db_session, reason="because I said so", create_backup=False, now=NOW)
    event = db_session.query(SystemEvent).filter_by(event="paper_trading_reset").one()
    assert event.severity == "WARNING"
    assert "because I said so" in event.message


def test_latest_reset_wins_and_history_is_kept(db_session):
    reset_paper_trading(db_session, reason="first", create_backup=False, now=NOW)
    later = NOW + dt.timedelta(days=5)
    reset_paper_trading(db_session, reason="second", create_backup=False, now=later)

    assert get_epoch_start(db_session) == later

    from crypto_ai.database.models.monitoring import PerformanceMetric
    markers = db_session.query(PerformanceMetric).filter_by(scope="paper_trading_epoch").count()
    assert markers == 2, "every reset should leave an audit trail"


# ---------------------------------------------------------------------
# exchange_connection_stable cannot be satisfied by a simulator
# ---------------------------------------------------------------------
def test_simulated_exchange_checks_never_prove_connectivity(db_session):
    from crypto_ai.exchange.market_data import MarketDataClient
    from crypto_ai.monitoring.health import check_exchange, has_verified_real_exchange_connectivity

    class FakeSource:
        def fetch_ohlcv(self, *a, **k):
            return [[int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000), 1, 2, 0.5, 1.5, 10]]

    # 50 perfectly healthy checks — but against an injected source.
    for _ in range(50):
        check_exchange(db_session, client=MarketDataClient(source=FakeSource()))
    db_session.commit()

    ok, reason = has_verified_real_exchange_connectivity(db_session)
    assert ok is False, "a simulated client must never satisfy this gate item"
    assert "simulated" in reason or "no health checks against the real" in reason


def test_real_source_checks_are_tagged_as_real(db_session, monkeypatch):
    from crypto_ai.monitoring import health

    # Simulate what a genuine CcxtBinanceSource probe records, without
    # actually hitting the network.
    from crypto_ai.exchange.market_data import CcxtBinanceSource, MarketDataClient

    class FakeCcxt(CcxtBinanceSource):
        def __init__(self):  # skip real ccxt construction
            pass

        def fetch_ohlcv(self, *a, **k):
            return [[int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000), 1, 2, 0.5, 1.5, 10]]

    for _ in range(25):
        health.check_exchange(db_session, client=MarketDataClient(source=FakeCcxt()))
    db_session.commit()

    ok, reason = health.has_verified_real_exchange_connectivity(db_session)
    assert ok is True, reason
    assert "real Binance API" in reason


def test_gate_from_db_fails_closed_on_a_fresh_system(db_session):
    from crypto_ai.risk.safety_rules import evaluate_live_trading_gate_from_db

    result = evaluate_live_trading_gate_from_db(
        db_session,
        emergency_stop_tested=True, recovery_tested=True, backup_restore_tested=True,
        api_withdrawal_disabled=True, regulatory_check_acknowledged=True,
        user_explicitly_enabled_live=True,
    )
    assert result.allowed is False
    # The three evidence-based items must all be blocking on a fresh system.
    for item in ("paper_trading_long_enough", "enough_paper_trades", "exchange_connection_stable"):
        assert item in result.blocking_reasons, f"{item} should block but did not"
    assert "paper_trading_epoch" in result.evidence
    assert "exchange_connectivity" in result.evidence


def test_gate_uses_epoch_not_raw_table_totals(db_session):
    """Old trades must not push the gate past its minimums after a reset."""
    from crypto_ai.risk.safety_rules import evaluate_live_trading_gate_from_db

    for i in range(500):
        _add_trade(db_session, NOW - dt.timedelta(days=200 + i % 50))
    db_session.commit()

    reset_paper_trading(db_session, reason="pre-fix data", create_backup=False, now=NOW)

    result = evaluate_live_trading_gate_from_db(
        db_session,
        emergency_stop_tested=True, recovery_tested=True, backup_restore_tested=True,
        api_withdrawal_disabled=True, regulatory_check_acknowledged=True,
        user_explicitly_enabled_live=True,
    )
    assert result.evidence["paper_trading_epoch"]["closed_trades_since_epoch"] == 0
    assert "enough_paper_trades" in result.blocking_reasons


# ---------------------------------------------------------------------
# Items 1, 2, 6, 8: no more self-declared or vacuous passes
# ---------------------------------------------------------------------
def _gate_kwargs(**overrides):
    kw = dict(
        paper_trading_days=99, paper_trade_count=500, paper_net_return_pct=10.0,
        paper_max_drawdown_pct=5.0, walk_forward_passed=True, days_since_critical_error=99,
        regulatory_check_acknowledged=True, exchange_connection_stable=True,
        emergency_stop_tested=True, recovery_tested=True, backup_restore_tested=True,
        api_withdrawal_disabled=True, user_explicitly_enabled_live=True,
        historical_data_validated=True, backtest_completed=True,
        days_of_operating_history=90.0,
    )
    kw.update(overrides)
    return kw


def test_item1_and_2_are_no_longer_hardcoded_true():
    """
    Both were literally `True` in the checklist dict. Omitting the evidence
    must now block, not silently pass.
    """
    from crypto_ai.risk.safety_rules import evaluate_live_trading_gate

    r = evaluate_live_trading_gate(**_gate_kwargs(
        historical_data_validated=False, backtest_completed=False))
    assert "historical_data_validated" in r.blocking_reasons
    assert "backtest_completed" in r.blocking_reasons


def test_item1_and_2_default_to_blocking_when_not_supplied():
    """A caller that forgets to pass evidence must fail closed."""
    from crypto_ai.risk.safety_rules import evaluate_live_trading_gate

    kw = _gate_kwargs()
    kw.pop("historical_data_validated")
    kw.pop("backtest_completed")
    r = evaluate_live_trading_gate(**kw)
    assert "historical_data_validated" in r.blocking_reasons
    assert "backtest_completed" in r.blocking_reasons


def test_item6_zero_trades_is_unknown_not_pass():
    """0.0% return from an account that never traded is absent, not passing."""
    from crypto_ai.risk.safety_rules import evaluate_live_trading_gate

    r = evaluate_live_trading_gate(**_gate_kwargs(
        paper_trade_count=0, paper_net_return_pct=0.0))
    assert "paper_net_return_acceptable" in r.blocking_reasons

    r_none = evaluate_live_trading_gate(**_gate_kwargs(paper_net_return_pct=None))
    assert "paper_net_return_acceptable" in r_none.blocking_reasons


def test_item7_none_drawdown_still_fails_closed():
    from crypto_ai.risk.safety_rules import evaluate_live_trading_gate

    r = evaluate_live_trading_gate(**_gate_kwargs(paper_max_drawdown_pct=None))
    assert "paper_drawdown_acceptable" in r.blocking_reasons


def test_item8_requires_observation_not_just_absence_of_errors():
    """
    A fresh database has no errors because it has no history. 14 clean days
    requires 14 days of watching.
    """
    from crypto_ai.risk.safety_rules import evaluate_live_trading_gate

    r = evaluate_live_trading_gate(**_gate_kwargs(
        days_since_critical_error=10_000, days_of_operating_history=0.7))
    assert "no_recent_critical_errors" in r.blocking_reasons

    ok = evaluate_live_trading_gate(**_gate_kwargs(
        days_since_critical_error=30, days_of_operating_history=30.0))
    assert "no_recent_critical_errors" not in ok.blocking_reasons


def test_historical_data_check_blocks_on_empty_database(db_session):
    from crypto_ai.risk.safety_rules import check_historical_data_validated

    ok, reason = check_historical_data_validated(db_session, "BTC/USDT", "5m")
    assert ok is False
    assert "no BTC/USDT 5m market data" in reason


def test_historical_data_check_blocks_on_too_little_data(db_session):
    import datetime as dt
    from crypto_ai.database.repositories.market_data_repo import upsert_candles
    from crypto_ai.risk.safety_rules import check_historical_data_validated

    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    upsert_candles(db_session, "BTC/USDT", "5m", [
        {"timestamp": start + dt.timedelta(minutes=5 * i), "open": 100.0, "high": 101.0,
         "low": 99.0, "close": 100.0, "volume": 1.0} for i in range(50)])
    db_session.commit()

    ok, reason = check_historical_data_validated(db_session, "BTC/USDT", "5m")
    assert ok is False
    assert "need >=" in reason


def test_backtest_check_blocks_until_a_run_is_recorded(db_session):
    import datetime as dt
    from crypto_ai.database.models.trading import BacktestRun
    from crypto_ai.risk.safety_rules import check_backtest_completed

    ok, reason = check_backtest_completed(db_session, "BTC/USDT")
    assert ok is False and "no backtest run recorded" in reason

    now = dt.datetime.now(dt.timezone.utc)
    db_session.add(BacktestRun(
        strategy_version="s1", symbol="BTC/USDT", timeframe="5m",
        start_time=now - dt.timedelta(days=1), end_time=now,
        initial_balance=1000.0, final_balance=1010.0,
        metrics={"n_trades": 42, "total_return_pct": 1.0},
    ))
    db_session.commit()

    ok2, reason2 = check_backtest_completed(db_session, "BTC/USDT")
    assert ok2 is True and "42 trades" in reason2


def test_days_of_operating_history_is_zero_on_fresh_database(db_session):
    from crypto_ai.risk.safety_rules import days_of_operating_history

    assert days_of_operating_history(db_session) == 0.0
