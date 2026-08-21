"""
Feature pipeline (Phase 6).

What is it?
    Turns raw OHLCV candles into a feature matrix ready for model
    training/inference, and (optionally) persists it to the
    `features` table for traceability.

Why is it needed?
    So "which indicators, with which settings, were used to produce
    this prediction" is always answerable — see Section 11's
    requirement that every prediction trace back to a feature_version.

How does it work?
    1. Compute every configured indicator (technical.py, volatility.py).
    2. Drop the warm-up rows where a rolling window hasn't filled yet
       (they contain NaN and must never be fed to a model).
    3. Tag the output with FEATURE_VERSION and a data_version hash so
       a later change to this file (e.g. adding an indicator) produces
       a new, distinguishable version instead of silently mixing old
       and new feature sets in the database.

Data-leakage note: every function this pipeline calls is strictly
causal (see technical.py's docstring). Nothing here ever looks at
future rows.
"""

from __future__ import annotations

import hashlib

import pandas as pd
from sqlalchemy.orm import Session

from crypto_ai.config.loader import get_settings
from crypto_ai.database.models.market import FeatureSet
from crypto_ai.features import technical, volatility

# Bump this whenever the feature computation logic changes in a way
# that changes historical values (new indicator, changed window, bug
# fix). Old feature rows stay in the DB tagged with their old version.
FEATURE_VERSION = "features_v1"


def compute_data_version(symbol: str, timeframe: str, df: pd.DataFrame) -> str:
    """
    A short, stable fingerprint of the input data used to build
    features — lets us tell whether a model was trained on data that
    has since changed (e.g. a gap got backfilled).
    """
    if df.empty:
        basis = f"{symbol}|{timeframe}|empty"
    else:
        basis = (
            f"{symbol}|{timeframe}|{len(df)}|"
            f"{df['timestamp'].iloc[0]}|{df['timestamp'].iloc[-1]}"
        )
    return hashlib.sha256(basis.encode()).hexdigest()[:16]


def compute_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """
    df: OHLCV DataFrame with columns [timestamp, open, high, low, close, volume],
        sorted ascending by timestamp, no duplicates or gaps ideally
        (run the Phase 5 validators first).

    Returns a new DataFrame: [timestamp, <feature columns...>], with
    warm-up rows (containing any NaN) dropped.
    """
    cfg = config or get_settings().get("features", {})
    close = df["close"]
    out = pd.DataFrame({"timestamp": df["timestamp"]})

    for w in cfg.get("sma_windows", [10, 20, 50]):
        out[f"sma_{w}"] = technical.sma(close, w)
        # Distance from price to the moving average, normalized — more
        # comparable across price regimes than the raw SMA value.
        out[f"dist_sma_{w}"] = (close - out[f"sma_{w}"]) / out[f"sma_{w}"]

    for w in cfg.get("ema_windows", [10, 20, 50]):
        out[f"ema_{w}"] = technical.ema(close, w)
        out[f"dist_ema_{w}"] = (close - out[f"ema_{w}"]) / out[f"ema_{w}"]

    out["rsi"] = technical.rsi(close, cfg.get("rsi_window", 14))

    macd_df = technical.macd(
        close,
        fast=cfg.get("macd_fast", 12),
        slow=cfg.get("macd_slow", 26),
        signal=cfg.get("macd_signal", 9),
    )
    out = pd.concat([out, macd_df.add_prefix("macd_")], axis=1)

    out["atr"] = technical.atr(df, cfg.get("atr_window", 14))
    out["atr_pct"] = out["atr"] / close  # ATR relative to price, comparable across time

    bb_df = technical.bollinger_bands(
        close,
        window=cfg.get("bollinger_window", 20),
        num_std=cfg.get("bollinger_std", 2.0),
    )
    out = pd.concat([out, bb_df], axis=1)

    for w in cfg.get("volatility_windows", [12, 48]):
        out[f"volatility_{w}"] = volatility.rolling_volatility(close, w)

    for w in cfg.get("momentum_windows", [3, 6, 12]):
        out[f"momentum_{w}"] = volatility.momentum(close, w)

    vol_df = volatility.volume_features(df)
    out = pd.concat([out, vol_df], axis=1)

    # Rename the duplicated macd_ prefix columns cleanly.
    out = out.rename(columns={c: c.replace("macd_macd_", "macd_") for c in out.columns})

    feature_cols = [c for c in out.columns if c != "timestamp"]
    out = out.dropna(subset=feature_cols).reset_index(drop=True)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "timestamp"]


def persist_features(
    session: Session,
    symbol: str,
    timeframe: str,
    feature_df: pd.DataFrame,
    data_version: str,
    feature_version: str = FEATURE_VERSION,
) -> int:
    """Upsert computed feature rows into the `features` table."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    if feature_df.empty:
        return 0

    dialect = session.bind.dialect.name
    insert_fn = pg_insert if dialect == "postgresql" else sqlite_insert

    cols = feature_columns(feature_df)
    rows = [
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": row["timestamp"],
            "feature_version": feature_version,
            "data_version": data_version,
            "values": {c: (None if pd.isna(row[c]) else float(row[c])) for c in cols},
        }
        for _, row in feature_df.iterrows()
    ]

    stmt = insert_fn(FeatureSet).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol", "timeframe", "timestamp", "feature_version"],
        # NOTE: "values" collides with ColumnCollection.values() on the
        # `excluded` accessor, so it must be looked up by subscript.
        set_={"values": stmt.excluded["values"], "data_version": stmt.excluded.data_version},
    )
    session.execute(stmt)
    return len(rows)
