import datetime as dt

import numpy as np
import pandas as pd
import pytest

from crypto_ai.app.services.live_pipeline import run_prediction_and_paper_trade_tick
from crypto_ai.database.models.trading import Prediction
from crypto_ai.database.repositories.market_data_repo import upsert_candles
from crypto_ai.features.feature_pipeline import compute_features
from crypto_ai.features.labels import compute_labels
from crypto_ai.models.registry import registry
from crypto_ai.models.training.pipeline import run_training_pipeline


def _sine_wave_ohlcv(n=3000, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    t = np.arange(n)
    close = 100 + 8 * np.sin(2 * np.pi * t / 60) + rng.normal(0, 0.05, n)
    high = close + np.abs(rng.normal(0, 0.05, n))
    low = close - np.abs(rng.normal(0, 0.05, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.uniform(1, 10, n)
    timestamps = [start + dt.timedelta(minutes=5 * i) for i in range(n)]
    return pd.DataFrame(
        {"timestamp": timestamps, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


_LENIENT_WF = {"n_folds": 3, "min_train_bars": 800, "validation_bars": 300, "embargo_bars": 5, "final_test_bars": 300}
_LENIENT_LABELING = {"horizon_bars": 5, "positive_threshold": 0.01, "negative_threshold": -0.01, "assumed_fee_pct": 0.0005, "assumed_slippage_pct": 0.0002}
_LENIENT_CRITERIA = {"min_folds_passed_pct": 0.5, "min_precision": 0.3, "max_drawdown_pct": 0.95, "min_sharpe": -10, "min_trades_per_fold": 1}


def test_tick_reports_insufficient_data_with_no_candles(db_session):
    result = run_prediction_and_paper_trade_tick(db_session, symbol="BTC/USDT", timeframe="5m")
    assert result["status"] == "insufficient_data"


def test_tick_reports_no_champion_model_once_enough_data_exists(db_session):
    df = _sine_wave_ohlcv(n=400)
    upsert_candles(db_session, "BTC/USDT", "5m", df.to_dict("records"))
    db_session.commit()

    result = run_prediction_and_paper_trade_tick(db_session, symbol="BTC/USDT", timeframe="5m")
    assert result["status"] == "no_champion_model"


def test_tick_refuses_on_feature_version_mismatch(db_session):
    df = _sine_wave_ohlcv(n=400)
    upsert_candles(db_session, "BTC/USDT", "5m", df.to_dict("records"))
    db_session.commit()

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    fake_model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    fake_model.fit(np.random.rand(10, 2), ["BUY"] * 5 + ["SELL"] * 5)
    version = registry.register_model_version(
        db_session, "btc_direction_classifier", "BTC/USDT", "5m", "logistic_regression",
        feature_version="some_old_version", strategy_version="s1", estimator=fake_model,
        hyperparameters={}, metrics={}, walk_forward_results={}, status=registry.STATUS_CANDIDATE,
    )
    registry.promote_to_champion(db_session, version)
    db_session.commit()

    result = run_prediction_and_paper_trade_tick(db_session, symbol="BTC/USDT", timeframe="5m")
    assert result["status"] == "feature_version_mismatch"


def test_full_tick_produces_a_prediction_and_status_ok(db_session):
    df = _sine_wave_ohlcv(n=3000)
    upsert_candles(db_session, "BTC/USDT", "5m", df.to_dict("records"))
    db_session.commit()

    features = compute_features(df)
    labels = compute_labels(
        df, horizon_bars=_LENIENT_LABELING["horizon_bars"],
        positive_threshold=_LENIENT_LABELING["positive_threshold"],
        negative_threshold=_LENIENT_LABELING["negative_threshold"],
        fee_pct=_LENIENT_LABELING["assumed_fee_pct"], slippage_pct=_LENIENT_LABELING["assumed_slippage_pct"],
    )
    run_training_pipeline(
        db_session, "BTC/USDT", "5m", features, labels, model_name="btc_direction_classifier",
        algorithms=["random_forest"], walk_forward_cfg=_LENIENT_WF,
        labeling_cfg=_LENIENT_LABELING, promotion_criteria=_LENIENT_CRITERIA,
    )
    db_session.commit()

    champion = registry.get_champion(db_session, "btc_direction_classifier")
    if champion is None:
        pytest.skip("No candidate cleared even lenient criteria on this synthetic dataset/seed.")

    result = run_prediction_and_paper_trade_tick(db_session, symbol="BTC/USDT", timeframe="5m")
    assert result["status"] == "ok"
    assert result["predicted_class"] in ("BUY", "HOLD", "SELL")
    assert result["final_signal"] in ("BUY", "HOLD", "SELL", "WAIT")
    assert 0 <= result["confidence"] <= 1
    assert 0 <= result["risk_score"] <= 1

    n_predictions = db_session.query(Prediction).count()
    assert n_predictions == 1


def test_tick_never_bypasses_risk_manager_when_emergency_stop_active(db_session):
    from crypto_ai.risk.emergency import activate_emergency_stop, reset_emergency_stop

    df = _sine_wave_ohlcv(n=3000)
    upsert_candles(db_session, "BTC/USDT", "5m", df.to_dict("records"))
    db_session.commit()

    features = compute_features(df)
    labels = compute_labels(
        df, horizon_bars=_LENIENT_LABELING["horizon_bars"],
        positive_threshold=_LENIENT_LABELING["positive_threshold"],
        negative_threshold=_LENIENT_LABELING["negative_threshold"],
        fee_pct=_LENIENT_LABELING["assumed_fee_pct"], slippage_pct=_LENIENT_LABELING["assumed_slippage_pct"],
    )
    run_training_pipeline(
        db_session, "BTC/USDT", "5m", features, labels, model_name="btc_direction_classifier",
        algorithms=["random_forest"], walk_forward_cfg=_LENIENT_WF,
        labeling_cfg=_LENIENT_LABELING, promotion_criteria=_LENIENT_CRITERIA,
    )
    db_session.commit()
    champion = registry.get_champion(db_session, "btc_direction_classifier")
    if champion is None:
        pytest.skip("No candidate cleared even lenient criteria on this synthetic dataset/seed.")

    try:
        activate_emergency_stop("test")
        result = run_prediction_and_paper_trade_tick(db_session, symbol="BTC/USDT", timeframe="5m")
        assert result["status"] == "ok"
        assert result["final_signal"] == "WAIT"
        assert "emergency_stop_active" in result["reason_codes"]
        assert result["action"]["action"] == "NONE"
    finally:
        reset_emergency_stop()
