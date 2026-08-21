import datetime as dt

import numpy as np
import pandas as pd
import pytest

from crypto_ai.database.models.monitoring import HealthCheck, SystemEvent
from crypto_ai.monitoring import alerts, health
from crypto_ai.monitoring.drift import (
    detect_accuracy_drift,
    detect_feature_drift,
    population_stability_index,
    run_drift_check,
)
from crypto_ai.monitoring.metrics import get_hardware_profile, get_resource_usage


def test_check_database_ok(db_session):
    result = health.check_database(db_session)
    assert result.status == health.STATUS_OK
    assert result.latency_ms is not None


def test_check_llm_disabled_is_degraded_not_down(db_session):
    result = health.check_llm(db_session)
    # LLM_ENABLED=false in tests -> "DISABLED" maps to DEGRADED (not a failure).
    assert result.status == health.STATUS_DEGRADED


def test_check_exchange_handles_unreachable_source_gracefully(db_session):
    class BrokenSource:
        def fetch_ohlcv(self, *a, **k):
            raise ConnectionError("no network")

    from crypto_ai.exchange.market_data import MarketDataClient

    client = MarketDataClient(source=BrokenSource())
    result = health.check_exchange(db_session, client=client)
    assert result.status == health.STATUS_DOWN


def test_run_all_health_checks_never_raises(db_session):
    class BrokenSource:
        def fetch_ohlcv(self, *a, **k):
            raise ConnectionError("no network")

    from crypto_ai.exchange.market_data import MarketDataClient

    summary = health.run_all_health_checks(db_session, exchange_client=MarketDataClient(source=BrokenSource()))
    assert "database" in summary
    assert "llm" in summary
    assert "exchange" in summary


def test_get_resource_usage_returns_sane_values():
    usage = get_resource_usage()
    assert 0 <= usage["cpu_percent"] <= 100
    assert 0 <= usage["ram_percent"] <= 100
    assert usage["cpu_count"] >= 1


def test_get_hardware_profile_recommends_settings():
    profile = get_hardware_profile()
    assert profile["recommended_max_workers"] >= 1
    assert isinstance(profile["llm_recommendation"], str)


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(1)
    data = pd.Series(rng.normal(0, 1, 2000))
    psi = population_stability_index(data, data)
    assert psi == pytest.approx(0.0, abs=0.05)


def test_psi_high_for_shifted_distribution():
    rng = np.random.default_rng(1)
    expected = pd.Series(rng.normal(0, 1, 2000))
    actual = pd.Series(rng.normal(5, 1, 2000))  # big mean shift
    psi = population_stability_index(expected, actual)
    assert psi > 0.25


def test_detect_feature_drift_flags_shifted_feature():
    rng = np.random.default_rng(2)
    training_df = pd.DataFrame({"volatility": rng.normal(0, 1, 2000), "rsi": rng.uniform(30, 70, 2000)})
    recent_df = pd.DataFrame({"volatility": rng.normal(6, 1, 500), "rsi": rng.uniform(30, 70, 500)})

    psi_by_feature, flagged = detect_feature_drift(training_df, recent_df, ["volatility", "rsi"])
    assert flagged is True
    assert psi_by_feature["volatility"] > psi_by_feature["rsi"]


def test_detect_accuracy_drift():
    assert detect_accuracy_drift(recent_accuracy_pct=40.0, expected_accuracy_pct=55.0) is True
    assert detect_accuracy_drift(recent_accuracy_pct=53.0, expected_accuracy_pct=55.0) is False
    assert detect_accuracy_drift(None, 55.0) is False


def test_run_drift_check_flags_for_review_but_returns_report_only():
    rng = np.random.default_rng(3)
    training_df = pd.DataFrame({"volatility": rng.normal(0, 1, 2000)})
    recent_df = pd.DataFrame({"volatility": rng.normal(8, 1, 500)})

    report = run_drift_check(training_df, recent_df, ["volatility"], recent_accuracy_pct=40.0, expected_accuracy_pct=55.0)
    assert report.flagged_for_review is True
    assert report.feature_drift_flagged is True
    assert report.accuracy_drift_flagged is True
    assert len(report.reasons) == 2


def test_notify_writes_system_event(db_session):
    alerts.notify(db_session, alerts.EVENT_DRIFT_DETECTED, "drift found", severity="WARNING", context={"psi": 0.4})
    event = db_session.query(SystemEvent).filter_by(event=alerts.EVENT_DRIFT_DETECTED).one()
    assert event.severity == "WARNING"
    assert event.context["psi"] == 0.4


def test_notify_never_raises_when_telegram_misconfigured(db_session, monkeypatch):
    monkeypatch.setenv("NOTIFY_TELEGRAM_ENABLED", "true")
    monkeypatch.delenv("NOTIFY_TELEGRAM_BOT_TOKEN", raising=False)
    alerts.notify(db_session, alerts.EVENT_SYSTEM_STARTED, "started")  # must not raise
