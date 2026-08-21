"""
Position sizing (Section 20).

What is it?
    Decides what FRACTION of portfolio equity to risk on a trade,
    given the model's confidence and current market volatility.

Why is it needed?
    A fixed position size ignores two things that matter a lot: how
    sure the model actually is, and how much the market is currently
    moving. Scaling down in both low-confidence and high-volatility
    conditions is a simple, well-understood way to reduce the chance
    of one bad trade doing outsized damage.
"""

from __future__ import annotations


def calculate_position_size_pct(
    confidence: float,
    volatility_pct: float,
    max_position_percent: float,
    min_confidence_to_trade: float,
    reference_volatility_pct: float = 1.0,
) -> float:
    """
    Returns a position size as a PERCENT of equity (0-max_position_percent).

    confidence: model's confidence in the current signal (0-1).
    volatility_pct: recent volatility (e.g. ATR% or rolling std of
        returns, in percent) — higher means smaller position.
    reference_volatility_pct: the volatility level at which no
        volatility scaling is applied (1.0 = "normal"); above this,
        size shrinks proportionally.
    """
    if confidence < min_confidence_to_trade:
        return 0.0

    # Linearly scale from 0 at min_confidence_to_trade up to 1 at
    # confidence=1.0, so being barely above the threshold still means
    # a small size, not the full allocation.
    confidence_range = max(1e-9, 1.0 - min_confidence_to_trade)
    confidence_scale = min(1.0, (confidence - min_confidence_to_trade) / confidence_range)

    volatility_pct = max(volatility_pct, 1e-9)
    volatility_scale = min(1.0, reference_volatility_pct / volatility_pct)

    size = max_position_percent * confidence_scale * volatility_scale
    return max(0.0, min(size, max_position_percent))


def position_size_pct_to_fraction(position_size_pct: float) -> float:
    """max_position_percent is expressed like '5.0' meaning 5%; the
    backtester/paper engine want a 0-1 fraction of equity."""
    return position_size_pct / 100.0
