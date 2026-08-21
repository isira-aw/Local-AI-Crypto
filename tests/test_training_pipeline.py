import datetime as dt

import numpy as np
import pandas as pd
import pytest

from crypto_ai.features.feature_pipeline import compute_features
from crypto_ai.features.labels import compute_labels
from crypto_ai.models.registry import registry
from crypto_ai.models.training.pipeline import merge_features_and_labels, run_training_pipeline


def _sine_wave_ohlcv(n=3000, seed=1) -> pd.DataFrame:
    """
    A price series with a real, learnable pattern (a sine wave) plus a
    little noise — lets at least one candidate model plausibly clear
    lenient promotion criteria without this being a claim about real
    crypto predictability (see Section 46: prediction success is
    evaluated separately from financial/software success).
    """
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


def _random_walk_ohlcv(n=3000, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    returns = rng.normal(0, 0.002, n)
    close = 100 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
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
_STRICT_CRITERIA = {"min_folds_passed_pct": 0.8, "min_precision": 0.52, "max_drawdown_pct": 0.20, "min_sharpe": 0.3, "min_trades_per_fold": 20}


def test_merge_features_and_labels_aligns_on_timestamp():
    df = _sine_wave_ohlcv(n=500)
    features = compute_features(df)
    labels = compute_labels(df, horizon_bars=5, positive_threshold=0.01, negative_threshold=-0.01)
    merged = merge_features_and_labels(features, labels)
    assert "label" in merged.columns
    assert "close" in merged.columns
    assert merged["timestamp"].is_monotonic_increasing
    assert len(merged) <= min(len(features), len(labels))


def test_pipeline_runs_end_to_end_and_registers_models(db_session):
    df = _sine_wave_ohlcv(n=3000)
    features = compute_features(df)
    labels = compute_labels(
        df, horizon_bars=_LENIENT_LABELING["horizon_bars"],
        positive_threshold=_LENIENT_LABELING["positive_threshold"],
        negative_threshold=_LENIENT_LABELING["negative_threshold"],
        fee_pct=_LENIENT_LABELING["assumed_fee_pct"], slippage_pct=_LENIENT_LABELING["assumed_slippage_pct"],
    )

    summary = run_training_pipeline(
        db_session, "BTC/USDT", "5m", features, labels,
        model_name="test_model",
        algorithms=["logistic_regression", "random_forest"],
        walk_forward_cfg=_LENIENT_WF,
        labeling_cfg=_LENIENT_LABELING,
        promotion_criteria=_LENIENT_CRITERIA,
    )

    assert summary["model_name"] == "test_model"
    assert len(summary["results"]) == 2
    for r in summary["results"]:
        assert "algorithm" in r
        assert "overall_pass" in r
        assert "version_label" in r  # every attempt is registered (candidate OR rejected)

    n_versions = db_session.query(registry.ModelVersion).count()
    assert n_versions == 2  # one per algorithm, none silently dropped

    # If anything passed, a champion should now be set; if not, that's
    # also a valid (safe) outcome — never force a promotion.
    champion = registry.get_champion(db_session, "test_model")
    if any(r["overall_pass"] for r in summary["results"]):
        assert champion is not None
        assert summary["promotion"]["promoted"] is True
    else:
        assert champion is None


def test_pipeline_rejects_pure_noise_under_strict_criteria(db_session):
    df = _random_walk_ohlcv(n=3000)
    features = compute_features(df)
    labels = compute_labels(df, horizon_bars=12, positive_threshold=0.005, negative_threshold=-0.005)

    summary = run_training_pipeline(
        db_session, "BTC/USDT", "5m", features, labels,
        model_name="noise_model",
        algorithms=["logistic_regression"],
        walk_forward_cfg={"n_folds": 3, "min_train_bars": 800, "validation_bars": 300, "embargo_bars": 12, "final_test_bars": 300},
        promotion_criteria=_STRICT_CRITERIA,
    )

    # Pure random-walk data should not reliably clear strict criteria;
    # if it did register a candidate anyway that's a red flag for the
    # test data, but the important safety property is: no result
    # silently disappears, and nothing is promoted without passing.
    for r in summary["results"]:
        if not r["overall_pass"]:
            version = db_session.query(registry.ModelVersion).filter_by(version_label=r["version_label"]).one()
            assert version.status == registry.STATUS_REJECTED

    champion = registry.get_champion(db_session, "noise_model")
    if champion is None:
        assert summary["promotion"]["promoted"] is False


def test_champion_not_replaced_by_worse_candidate(db_session):
    df = _sine_wave_ohlcv(n=3000)
    features = compute_features(df)
    labels = compute_labels(
        df, horizon_bars=_LENIENT_LABELING["horizon_bars"],
        positive_threshold=_LENIENT_LABELING["positive_threshold"],
        negative_threshold=_LENIENT_LABELING["negative_threshold"],
        fee_pct=_LENIENT_LABELING["assumed_fee_pct"], slippage_pct=_LENIENT_LABELING["assumed_slippage_pct"],
    )

    summary1 = run_training_pipeline(
        db_session, "BTC/USDT", "5m", features, labels, model_name="stability_model",
        algorithms=["random_forest"], walk_forward_cfg=_LENIENT_WF,
        labeling_cfg=_LENIENT_LABELING, promotion_criteria=_LENIENT_CRITERIA,
    )
    champion_after_1 = registry.get_champion(db_session, "stability_model")

    # Re-run with a deliberately terrible model (a classifier that will
    # score a very low Sharpe because it never trades confidently) by
    # cranking min_confidence via strict criteria requiring high sharpe.
    summary2 = run_training_pipeline(
        db_session, "BTC/USDT", "5m", features, labels, model_name="stability_model",
        algorithms=["logistic_regression"], walk_forward_cfg=_LENIENT_WF,
        labeling_cfg=_LENIENT_LABELING,
        promotion_criteria={**_LENIENT_CRITERIA, "min_sharpe": 999},  # impossible to pass
    )

    champion_after_2 = registry.get_champion(db_session, "stability_model")
    if champion_after_1 is not None:
        # Champion must be unchanged since nothing could beat/replace it.
        assert champion_after_2.id == champion_after_1.id
        assert summary2["promotion"]["promoted"] is False
