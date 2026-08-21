import datetime as dt

import pytest

from crypto_ai.database.repositories.market_data_repo import upsert_candles
from crypto_ai.features.labels import BUY, HOLD, SELL
from crypto_ai.models.evaluation.prediction_evaluator import (
    evaluate_due_predictions,
    record_prediction,
    summarize_prediction_accuracy,
)

START = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)


def _seed_market_data(session, prices: dict[int, float]):
    rows = [
        {"timestamp": START + dt.timedelta(minutes=5 * i), "open": p, "high": p, "low": p, "close": p, "volume": 1.0}
        for i, p in prices.items()
    ]
    upsert_candles(session, "BTC/USDT", "5m", rows)
    session.commit()


def test_prediction_not_evaluated_before_horizon_elapses(db_session):
    _seed_market_data(db_session, {0: 100.0})
    record_prediction(
        db_session, "BTC/USDT", "5m", START, horizon_bars=12,
        data_version="d1", feature_version="f1", model_version="model_001", strategy_version="s1",
        predicted_class=BUY, confidence=0.7, probabilities={}, price_at_prediction=100.0,
    )
    db_session.commit()

    n = evaluate_due_predictions(db_session, now=START + dt.timedelta(minutes=10))
    assert n == 0


def test_prediction_evaluated_correctly_when_buy_was_right(db_session):
    _seed_market_data(db_session, {0: 100.0, 12: 110.0})
    record_prediction(
        db_session, "BTC/USDT", "5m", START, horizon_bars=12,
        data_version="d1", feature_version="f1", model_version="model_001", strategy_version="s1",
        predicted_class=BUY, confidence=0.7, probabilities={}, price_at_prediction=100.0,
    )
    db_session.commit()

    n = evaluate_due_predictions(db_session, now=START + dt.timedelta(minutes=65))
    db_session.commit()
    assert n == 1

    from crypto_ai.database.models.trading import Prediction
    pred = db_session.query(Prediction).one()
    assert pred.actual_class == BUY
    assert pred.was_correct is True
    assert pred.actual_return == pytest.approx(0.10, rel=1e-6)


def test_prediction_marked_incorrect_when_sell_predicted_but_price_rose(db_session):
    _seed_market_data(db_session, {0: 100.0, 12: 110.0})
    record_prediction(
        db_session, "BTC/USDT", "5m", START, horizon_bars=12,
        data_version="d1", feature_version="f1", model_version="model_001", strategy_version="s1",
        predicted_class=SELL, confidence=0.9, probabilities={}, price_at_prediction=100.0,
    )
    db_session.commit()
    evaluate_due_predictions(db_session, now=START + dt.timedelta(minutes=65))
    db_session.commit()

    from crypto_ai.database.models.trading import Prediction
    pred = db_session.query(Prediction).one()
    assert pred.was_correct is False


def test_evaluate_skips_when_future_price_not_yet_available(db_session):
    _seed_market_data(db_session, {0: 100.0})  # no bar 12 minutes later yet
    record_prediction(
        db_session, "BTC/USDT", "5m", START, horizon_bars=12,
        data_version="d1", feature_version="f1", model_version="model_001", strategy_version="s1",
        predicted_class=BUY, confidence=0.7, probabilities={}, price_at_prediction=100.0,
    )
    db_session.commit()
    n = evaluate_due_predictions(db_session, now=START + dt.timedelta(minutes=65))
    assert n == 0


def test_summarize_prediction_accuracy(db_session):
    _seed_market_data(db_session, {0: 100.0, 12: 110.0, 24: 90.0, 36: 100.0})
    # Correct BUY.
    record_prediction(
        db_session, "BTC/USDT", "5m", START, horizon_bars=12,
        data_version="d1", feature_version="f1", model_version="model_001", strategy_version="s1",
        predicted_class=BUY, confidence=0.7, probabilities={}, price_at_prediction=100.0,
    )
    # Correct SELL (from 110 -> 90).
    record_prediction(
        db_session, "BTC/USDT", "5m", START + dt.timedelta(minutes=60), horizon_bars=12,
        data_version="d1", feature_version="f1", model_version="model_001", strategy_version="s1",
        predicted_class=SELL, confidence=0.8, probabilities={}, price_at_prediction=110.0,
    )
    db_session.commit()

    evaluate_due_predictions(db_session, now=START + dt.timedelta(minutes=180))
    db_session.commit()

    summary = summarize_prediction_accuracy(db_session, symbol="BTC/USDT")
    assert summary["n_predictions"] == 2
    assert summary["accuracy_pct"] == 100.0
    assert summary["directional_accuracy_pct"] == 100.0
    assert summary["win_rate_pct"] == 100.0


def test_summarize_with_no_evaluated_predictions_returns_none_fields(db_session):
    summary = summarize_prediction_accuracy(db_session)
    assert summary["n_predictions"] == 0
    assert summary["accuracy_pct"] is None
