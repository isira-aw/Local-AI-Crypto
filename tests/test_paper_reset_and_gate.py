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
