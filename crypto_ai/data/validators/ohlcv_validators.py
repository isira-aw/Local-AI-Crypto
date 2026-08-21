"""
Data validation for OHLCV candles (Section 10 / Phase 5).

What is it?
    Pure functions that check a DataFrame of candles for the kinds of
    problems that silently ruin ML models: gaps, duplicates, bad
    timestamps, and OHLCV values that don't make physical sense.

Why is it needed?
    Garbage in, garbage out. A single stretch of missing 5m bars can
    quietly shift every feature computed after it if not caught.

What goes in / comes out?
    In: a pandas DataFrame with columns
        [timestamp, open, high, low, close, volume], sorted or not.
    Out: a ValidationReport describing what (if anything) is wrong.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

_TIMEFRAME_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400,
}


@dataclass
class Gap:
    start: dt.datetime
    end: dt.datetime
    missing_bars: int


@dataclass
class ValidationReport:
    n_rows: int
    n_duplicates: int
    gaps: list[Gap] = field(default_factory=list)
    bad_ohlc_rows: list[int] = field(default_factory=list)
    non_monotonic: bool = False

    @property
    def is_clean(self) -> bool:
        return (
            self.n_duplicates == 0
            and not self.gaps
            and not self.bad_ohlc_rows
            and not self.non_monotonic
        )

    def summary(self) -> str:
        if self.is_clean:
            return f"OK: {self.n_rows} rows, no issues found."
        parts = [f"{self.n_rows} rows"]
        if self.n_duplicates:
            parts.append(f"{self.n_duplicates} duplicate timestamps")
        if self.gaps:
            total_missing = sum(g.missing_bars for g in self.gaps)
            parts.append(f"{len(self.gaps)} gaps ({total_missing} missing bars)")
        if self.bad_ohlc_rows:
            parts.append(f"{len(self.bad_ohlc_rows)} rows with invalid OHLC values")
        if self.non_monotonic:
            parts.append("timestamps not monotonically increasing")
        return "ISSUES FOUND: " + "; ".join(parts)


def detect_duplicates(df: pd.DataFrame) -> int:
    return int(df["timestamp"].duplicated().sum())


def detect_non_monotonic(df: pd.DataFrame) -> bool:
    ts = df["timestamp"]
    return bool((ts.diff().dropna() <= pd.Timedelta(0)).any())


def detect_gaps(df: pd.DataFrame, timeframe: str) -> list[Gap]:
    """Find stretches where consecutive bars are more than one interval apart."""
    if timeframe not in _TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    interval = pd.Timedelta(seconds=_TIMEFRAME_SECONDS[timeframe])

    ordered = df.sort_values("timestamp").drop_duplicates(subset="timestamp")
    if len(ordered) < 2:
        return []

    ts = ordered["timestamp"].reset_index(drop=True)
    diffs = ts.diff()

    gaps: list[Gap] = []
    for i in range(1, len(ts)):
        gap_size = diffs.iloc[i]
        if gap_size > interval:
            missing = int(gap_size / interval) - 1
            gaps.append(Gap(start=ts.iloc[i - 1], end=ts.iloc[i], missing_bars=missing))
    return gaps


def detect_bad_ohlc(df: pd.DataFrame) -> list[int]:
    """
    Rows that violate basic OHLCV invariants:
      - high must be >= max(open, close, low)
      - low must be <= min(open, close, high)
      - all prices/volume must be positive and finite
    """
    bad_mask = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (df["volume"] < 0)
        | (~df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1))
    )
    return df.index[bad_mask].tolist()


def validate_ohlcv(df: pd.DataFrame, timeframe: str) -> ValidationReport:
    if df.empty:
        return ValidationReport(n_rows=0, n_duplicates=0)

    return ValidationReport(
        n_rows=len(df),
        n_duplicates=detect_duplicates(df),
        gaps=detect_gaps(df, timeframe),
        bad_ohlc_rows=detect_bad_ohlc(df),
        non_monotonic=detect_non_monotonic(df),
    )
