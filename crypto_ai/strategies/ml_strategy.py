"""
ML-driven strategy (Phase 9): turns model output into BUY/SELL/HOLD.

This is deliberately a thin, auditable layer: the model produces a
class + confidence, and this just applies a minimum-confidence filter.
All the interesting decisions (position sizing, stop-loss, risk
limits) live in risk/ and backtesting/engine.py, not here.
"""

from __future__ import annotations

import pandas as pd

STRATEGY_VERSION = "ml_strategy_v1"


def predictions_to_signals(
    predicted_class: pd.Series,
    confidence: pd.Series,
    min_confidence_to_trade: float = 0.55,
) -> pd.Series:
    """
    predicted_class: Series of "BUY"/"HOLD"/"SELL" per bar.
    confidence: Series of the model's confidence in that class (0-1).

    A BUY/SELL prediction below `min_confidence_to_trade` is treated as
    HOLD — a model that isn't confident shouldn't trade just because it
    picked a class.
    """
    signals = predicted_class.copy()
    low_confidence = confidence < min_confidence_to_trade
    signals = signals.mask(low_confidence, "HOLD")
    return signals
