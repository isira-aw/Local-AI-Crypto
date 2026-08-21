"""
Volatility, momentum and volume-based features.

These complement technical.py's trend/oscillator indicators with
features that describe *how much* and *how fast* price is moving,
which matters a lot for position sizing and for the risk engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def rolling_volatility(close: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation of log returns — a standard volatility proxy."""
    return log_returns(close).rolling(window=window, min_periods=window).std()


def momentum(close: pd.Series, window: int) -> pd.Series:
    """Percent price change over the last `window` bars."""
    return close.pct_change(periods=window)


def volume_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    volume_ratio: current volume relative to its rolling average
        (>1 means more activity than usual).
    volume_change: percent change in volume vs. the previous bar.
    """
    avg_volume = df["volume"].rolling(window=window, min_periods=window).mean()
    return pd.DataFrame(
        {
            "volume_ratio": df["volume"] / avg_volume.replace(0, np.nan),
            "volume_change": df["volume"].pct_change(),
        }
    )
