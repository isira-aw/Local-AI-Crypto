"""
Tests for the scheduled maintenance operations added after end-to-end
validation found them missing: the drift check job and the weekly
retrain (Sections 23 and 44).
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from crypto_ai.app.services.maintenance import (
    get_latest_drift_status,
    run_drift_check_job,
    run_scheduled_retrain,
)
from crypto_ai.database.repositories.market_data_repo import upsert_candles

START = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)


def _seed(session, n=1200, regime_shift_at=None, seed=3):
    """Seed n candles; optionally make the tail a wildly different regime."""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.002, n))
    if regime_shift_at is not None:
        # Violent, obviously-different regime for the tail.
        tail = n - regime_shift_at
        close[regime_shift_at:] = close[regime_shift_at] * np.cumprod(1 + rng.normal(0, 0.08, tail))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    rows = [
        {
            "timestamp": START + dt.timedelta(minutes=5 * i),
            "open": float(open_[i]), "high": float(high[i]), "low": float(low[i]),
            "close": float(close[i]), "volume": float(abs(rng.normal(10, 3))),
        }
        for i in range(n)
    ]
    upsert_candles(session, "BTC/USDT", "5m", rows)
    session.commit()


def test_drift_check_reports_insufficient_data_when_empty(db_session):
    result = run_drift_check_job(db_session, symbol="BTC/USDT", timeframe="5m")
    assert result["status"] == "insufficient_data"


def test_drift_check_runs_and_persists_status(db_session):
    _seed(db_session, n=1200)
    result = run_drift_check_job(db_session, symbol="BTC/USDT", timeframe="5m")

    assert result["status"] == "ok"
    assert "max_psi" in result
    assert isinstance(result["flagged_for_review"], bool)

    stored = get_latest_drift_status(db_session)
    assert stored is not None
    assert stored["max_psi"] == result["max_psi"]


def test_drift_check_flags_an_obvious_regime_change(db_session):
    _seed(db_session, n=1200, regime_shift_at=850)
    result = run_drift_check_job(db_session, symbol="BTC/USDT", timeframe="5m")
    assert result["status"] == "ok"
    assert result["feature_drift_flagged"] is True, f"expected drift flag, got {result}"
    assert result["reasons"], "a flagged drift must explain itself"


def test_drift_check_emits_a_warning_event_when_flagged(db_session):
    from crypto_ai.database.models.monitoring import SystemEvent

    _seed(db_session, n=1200, regime_shift_at=850)
    run_drift_check_job(db_session, symbol="BTC/USDT", timeframe="5m")

    events = db_session.query(SystemEvent).filter_by(event="drift_detected").all()
    assert len(events) >= 1
    assert events[0].severity == "WARNING"


def test_drift_never_triggers_automatic_promotion(db_session):
    """
    Section 23: drift flags for review, it must NEVER retrain-and-promote
    on its own. A drift check on a system with no champion must leave it
    with no champion.
    """
    from crypto_ai.models.registry import registry

    _seed(db_session, n=1200, regime_shift_at=850)
    run_drift_check_job(db_session, symbol="BTC/USDT", timeframe="5m")
    assert registry.get_champion(db_session, "btc_direction_classifier") is None


def test_scheduled_retrain_skips_when_data_is_insufficient(db_session):
    _seed(db_session, n=300)
    result = run_scheduled_retrain(db_session, min_bars=1000)
    assert result["status"] == "insufficient_data"
    assert result["bars_needed"] == 1000


def test_scheduled_retrain_runs_pipeline_when_data_is_available(db_session):
    _seed(db_session, n=1500)
    result = run_scheduled_retrain(db_session, min_bars=500)
    assert result["status"] == "ok"
    assert "results" in result
    assert "promotion" in result
    # Every attempt must be registered, pass or fail (Section 24).
    for r in result["results"]:
        assert "algorithm" in r


def test_scheduled_retrain_logs_completion_event(db_session):
    from crypto_ai.database.models.monitoring import SystemEvent

    _seed(db_session, n=1500)
    run_scheduled_retrain(db_session, min_bars=500)
    events = db_session.query(SystemEvent).filter_by(event="training_completed").all()
    assert len(events) >= 1
