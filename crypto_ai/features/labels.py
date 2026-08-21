"""
Automatic label generation (Phase 7 / Section 12).

What is it?
    Turns future price movement into a training label — BUY, HOLD, or
    SELL — automatically. The user never labels a single row by hand.

Why is it needed?
    Supervised ML needs a target. "Will the price be meaningfully
    higher/lower `horizon_bars` from now, after accounting for trading
    costs?" is a target we can compute for every historical bar.

How does it work?
    future_return = close[t + horizon] / close[t] - 1

    net_return = future_return - round_trip_cost
        where round_trip_cost = 2 * (fee_pct + slippage_pct)
        (paid once to enter, once to exit)

    label =
        BUY  if net_return > positive_threshold
        SELL if net_return < negative_threshold
        HOLD otherwise

    Thresholds and costs are all configurable in settings.yaml under
    `labeling` — nothing here is hard-coded.

Data-leakage note: labels are DELIBERATELY built from future prices —
that is the point of a training target. What must never happen is a
FEATURE (features/feature_pipeline.py) using future data. The training
pipeline (Phase 9) is responsible for aligning features at time t with
labels computed from t+horizon, and for the walk-forward embargo gap
that keeps validation labels from overlapping training data.
"""

from __future__ import annotations

import pandas as pd

from crypto_ai.config.loader import get_settings

LABEL_VERSION = "labels_v1"

BUY = "BUY"
HOLD = "HOLD"
SELL = "SELL"


def compute_labels(
    df: pd.DataFrame,
    horizon_bars: int | None = None,
    positive_threshold: float | None = None,
    negative_threshold: float | None = None,
    fee_pct: float | None = None,
    slippage_pct: float | None = None,
) -> pd.DataFrame:
    """
    df: OHLCV DataFrame with columns [timestamp, close, ...], sorted
        ascending, no gaps/duplicates.

    Returns [timestamp, close, future_return, net_return, label].
    The last `horizon_bars` rows are dropped since their future price
    is unknown (label is undefined, not leaked).
    """
    cfg = get_settings().get("labeling", {})
    horizon_bars = horizon_bars if horizon_bars is not None else cfg.get("horizon_bars", 12)
    positive_threshold = (
        positive_threshold if positive_threshold is not None else cfg.get("positive_threshold", 0.005)
    )
    negative_threshold = (
        negative_threshold if negative_threshold is not None else cfg.get("negative_threshold", -0.005)
    )
    fee_pct = fee_pct if fee_pct is not None else cfg.get("assumed_fee_pct", 0.001)
    slippage_pct = slippage_pct if slippage_pct is not None else cfg.get("assumed_slippage_pct", 0.0005)

    if horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1")

    close = df["close"]
    future_close = close.shift(-horizon_bars)
    future_return = future_close / close - 1.0

    round_trip_cost = 2 * (fee_pct + slippage_pct)
    net_return = future_return - round_trip_cost

    label = pd.Series(HOLD, index=df.index)
    label = label.mask(net_return > positive_threshold, BUY)
    label = label.mask(net_return < negative_threshold, SELL)

    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "close": close,
            "future_return": future_return,
            "net_return": net_return,
            "label": label,
        }
    )
    # Drop rows where the future is unknown (last horizon_bars rows).
    out = out.iloc[: len(out) - horizon_bars if horizon_bars > 0 else len(out)]
    out = out.reset_index(drop=True)
    return out


def label_distribution(labels: pd.Series) -> dict:
    counts = labels.value_counts()
    total = len(labels)
    return {cls: {"count": int(counts.get(cls, 0)), "pct": round(counts.get(cls, 0) / total * 100, 2)} for cls in (BUY, HOLD, SELL)} if total else {}
