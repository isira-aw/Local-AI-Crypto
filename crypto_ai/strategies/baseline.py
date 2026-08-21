"""
Baseline strategies (Section 6/16): buy-and-hold and random.

Why is it needed?
    An ML strategy is only interesting if it beats these. Every
    training/backtest run compares against both — see Section 6:
    "Never claim success based only on prediction accuracy."
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold_signals(df: pd.DataFrame) -> pd.Series:
    """BUY on the first bar, hold forever, never sell."""
    signals = pd.Series("HOLD", index=df.index)
    if len(signals) > 0:
        signals.iloc[0] = "BUY"
    return signals


def random_signals(df: pd.DataFrame, seed: int = 42, buy_prob: float = 0.05, sell_prob: float = 0.05) -> pd.Series:
    """
    Randomly emits BUY/SELL/HOLD signals at the given probabilities.
    Used as a sanity-check baseline: any real strategy must beat this
    on average, or it isn't adding value.
    """
    rng = np.random.default_rng(seed)
    draws = rng.random(len(df))
    signals = pd.Series("HOLD", index=df.index)
    signals = signals.mask(draws < buy_prob, "BUY")
    signals = signals.mask((draws >= buy_prob) & (draws < buy_prob + sell_prob), "SELL")
    return signals
