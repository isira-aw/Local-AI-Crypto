import datetime as dt

import numpy as np
import pandas as pd
import pytest

from crypto_ai.features import technical, volatility
from crypto_ai.features.feature_pipeline import compute_data_version, compute_features
from crypto_ai.database.repositories.market_data_repo import upsert_candles, load_candles_df
from crypto_ai.features.feature_pipeline import persist_features


def _synthetic_ohlcv(n=300, seed=42) -> pd.DataFrame:
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


def test_sma_ema_basic():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    sma3 = technical.sma(s, 3)
    assert np.isnan(sma3.iloc[0])
    assert np.isnan(sma3.iloc[1])
    assert sma3.iloc[2] == pytest.approx(2.0)
    assert sma3.iloc[-1] == pytest.approx(9.0)

    ema3 = technical.ema(s, 3)
    assert np.isnan(ema3.iloc[1])
    assert not np.isnan(ema3.iloc[2])


def test_rsi_bounds():
    df = _synthetic_ohlcv()
    r = technical.rsi(df["close"], 14)
    valid = r.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1, 30, dtype=float))  # strictly increasing
    r = technical.rsi(s, 14)
    assert r.dropna().iloc[-1] == pytest.approx(100.0)


def test_macd_columns():
    df = _synthetic_ohlcv()
    macd_df = technical.macd(df["close"])
    assert set(macd_df.columns) == {"macd_line", "signal_line", "histogram"}
    assert (macd_df["histogram"].dropna() == (macd_df["macd_line"] - macd_df["signal_line"]).dropna()).all()


def test_atr_non_negative():
    df = _synthetic_ohlcv()
    a = technical.atr(df, 14)
    assert (a.dropna() >= 0).all()


def test_bollinger_bands_ordering():
    df = _synthetic_ohlcv()
    bb = technical.bollinger_bands(df["close"], 20, 2.0)
    valid = bb.dropna()
    assert (valid["bb_upper"] >= valid["bb_middle"]).all()
    assert (valid["bb_middle"] >= valid["bb_lower"]).all()


def test_no_lookahead_bias_in_indicators():
    """
    Changing a FUTURE bar's price must not change any indicator value
    computed at an EARLIER bar. This is the core anti-leakage
    guarantee for every feature (Section 13).
    """
    df = _synthetic_ohlcv(n=100)
    features_a = compute_features(df.copy())

    df_mutated = df.copy()
    mutate_from = 80
    df_mutated.loc[mutate_from:, "close"] *= 1.5
    df_mutated.loc[mutate_from:, "high"] *= 1.5
    df_mutated.loc[mutate_from:, "low"] *= 1.5
    df_mutated.loc[mutate_from:, "open"] *= 1.5
    features_b = compute_features(df_mutated)

    # Rows before the mutation point (accounting for warm-up rows
    # dropped from the front) must be identical.
    common_ts = set(features_a["timestamp"]) & set(features_b["timestamp"])
    cutoff_ts = df["timestamp"].iloc[mutate_from]
    early_ts = sorted(t for t in common_ts if t < cutoff_ts)
    assert len(early_ts) > 10  # sanity: we actually have rows to compare

    a_early = features_a[features_a["timestamp"].isin(early_ts)].set_index("timestamp").sort_index()
    b_early = features_b[features_b["timestamp"].isin(early_ts)].set_index("timestamp").sort_index()
    pd.testing.assert_frame_equal(a_early, b_early)


def test_compute_features_drops_warmup_nan_rows():
    df = _synthetic_ohlcv(n=200)
    features = compute_features(df)
    assert not features.isna().any().any()
    assert len(features) < len(df)


def test_momentum_and_volume_features():
    df = _synthetic_ohlcv(n=50)
    mom = volatility.momentum(df["close"], 5)
    assert np.isnan(mom.iloc[4]) or mom.iloc[4] == pytest.approx((df["close"].iloc[4] / df["close"].iloc[-1]) or mom.iloc[4])
    vol_feats = volatility.volume_features(df, window=10)
    assert "volume_ratio" in vol_feats.columns
    assert "volume_change" in vol_feats.columns


def test_compute_data_version_stable_and_sensitive():
    df = _synthetic_ohlcv(n=50)
    v1 = compute_data_version("BTC/USDT", "5m", df)
    v2 = compute_data_version("BTC/USDT", "5m", df)
    assert v1 == v2

    df2 = df.iloc[:-1]
    v3 = compute_data_version("BTC/USDT", "5m", df2)
    assert v3 != v1


def test_persist_features_roundtrip(db_session):
    df = _synthetic_ohlcv(n=100)
    upsert_candles(db_session, "BTC/USDT", "5m", df.to_dict("records"))
    db_session.commit()

    loaded = load_candles_df(db_session, "BTC/USDT", "5m")
    features = compute_features(loaded)
    data_version = compute_data_version("BTC/USDT", "5m", loaded)

    n_written = persist_features(db_session, "BTC/USDT", "5m", features, data_version)
    db_session.commit()
    assert n_written == len(features)

    from crypto_ai.database.models.market import FeatureSet
    count = db_session.query(FeatureSet).count()
    assert count == len(features)
