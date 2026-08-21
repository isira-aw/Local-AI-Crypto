import datetime as dt

import pytest

from crypto_ai.app.services.reporting import format_text_report, generate_daily_report
from crypto_ai.app.services.recovery import reconcile_exchange_state, run_startup_sequence
from crypto_ai.exchange.live import LiveTradingNotAllowedError, place_order
from crypto_ai.risk.safety_rules import LiveTradingGateResult
from crypto_ai.scheduler.jobs import guarded


def test_guarded_decorator_swallows_exceptions_and_logs(db_session, monkeypatch):
    import crypto_ai.scheduler.jobs as jobs_module

    # Patch session_scope used inside the decorator to reuse our test session
    # (a context manager wrapping the shared db_session, no separate commit).
    from contextlib import contextmanager

    @contextmanager
    def fake_session_scope():
        yield db_session

    monkeypatch.setattr(jobs_module, "session_scope", fake_session_scope)

    @guarded("exploding_job")
    def exploding_job():
        raise RuntimeError("boom")

    exploding_job()  # must not raise

    from crypto_ai.database.models.monitoring import SystemEvent
    event = db_session.query(SystemEvent).filter_by(event="job_failed").one()
    assert "boom" in event.message
    assert event.context["job"] == "exploding_job"


def test_guarded_decorator_allows_successful_job_through():
    calls = []

    @guarded("ok_job")
    def ok_job():
        calls.append(1)

    ok_job()
    assert calls == [1]


def test_generate_daily_report_has_expected_shape(db_session):
    report = generate_daily_report(db_session, symbol="BTC/USDT")
    assert report["mode"] == "RESEARCH"
    assert "predictions" in report
    assert "paper_trading" in report
    assert "model" in report
    assert "recommendation" in report
    assert "does not guarantee" in report["disclaimer"]


def test_generate_daily_report_recommends_continue_research_with_no_trades(db_session):
    report = generate_daily_report(db_session, symbol="BTC/USDT")
    assert report["recommendation"] == "CONTINUE RESEARCH"


def test_format_text_report_is_readable(db_session):
    report = generate_daily_report(db_session, symbol="BTC/USDT")
    text = format_text_report(report)
    assert "CRYPTO AI DAILY REPORT" in text
    assert "Recommendation:" in text


def test_reconcile_exchange_state_fails_safe(db_session):
    result = reconcile_exchange_state(db_session)
    assert result is False  # must never claim reconciliation succeeded when unimplemented

    from crypto_ai.database.models.monitoring import SystemEvent
    event = db_session.query(SystemEvent).filter_by(event="exchange_reconciliation_not_implemented").one()
    assert event.severity == "WARNING"


def test_run_startup_sequence_research_mode_does_not_require_exchange_reconciliation():
    from crypto_ai.database.base import reset_engine_for_tests, Base
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

    import crypto_ai.database.base as db_base
    db_base._engine = engine
    db_base._SessionFactory = SessionLocal
    try:
        result = run_startup_sequence(SessionLocal)
        assert result["mode"] == "RESEARCH"
        assert result["exchange_reconciled"] is True
        assert "database" in result["health"]
    finally:
        reset_engine_for_tests()


def test_place_order_refused_without_gate_result():
    with pytest.raises(LiveTradingNotAllowedError):
        place_order()


def test_place_order_refused_with_blocked_gate():
    gate = LiveTradingGateResult(allowed=False, checklist={}, blocking_reasons=["mode_is_testnet_or_live"])
    with pytest.raises(LiveTradingNotAllowedError):
        place_order(gate_result=gate)


def test_place_order_still_unimplemented_even_with_allowed_gate():
    gate = LiveTradingGateResult(allowed=True, checklist={}, blocking_reasons=[])
    with pytest.raises(NotImplementedError):
        place_order(gate_result=gate)
