"""
Standard technical indicators.

Every function here is strictly causal: the value at row i is computed
only from rows <= i. This is what makes them safe to use as ML
features — see Section 13 (prevent data leakage). Do not "center" any
rolling window or use `.shift(-n)` in this file.

Beginner note: these are just different ways of summarizing recent
price action into a single number (e.g. "is the price trending up?",
"is it overbought?"). feature_pipeline.py decides which of these are
actually used as model inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average — the plain average of the last `window` closes."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average — like SMA but weights recent bars more."""
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (0-100). Above ~70 is traditionally
    considered "overbought", below ~30 "oversold" — but treat this as
    one signal among many, not a rule.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    # Where avg_loss is 0 and avg_gain > 0, RSI is 100 by definition.
    result = result.where(avg_loss != 0, 100.0)
    return result


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """
    Moving Average Convergence/Divergence. Returns a DataFrame with
    columns: macd_line, signal_line, histogram.
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd_line": macd_line, "signal_line": signal_line, "histogram": histogram}
    )


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range — a measure of how much the price is moving
    around, in price units. Used both as a feature and by the risk
    engine for position sizing / stop-loss distance.
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands. Returns columns: bb_middle, bb_upper, bb_lower,
    bb_percent_b (0=at lower band, 1=at upper band), bb_bandwidth
    (how wide the bands are relative to price — a volatility proxy).
    """
    middle = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    percent_b = (series - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = (upper - lower) / middle.replace(0, np.nan)
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_percent_b": percent_b,
            "bb_bandwidth": bandwidth,
        }
    )
