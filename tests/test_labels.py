import datetime as dt

import pandas as pd
import pytest

from crypto_ai.features.labels import BUY, HOLD, SELL, compute_labels, label_distribution


def _flat_then_move_df():
    """
    Construct a close series where we know exactly what the future
    return at each point will be, to test label thresholds precisely.
    """
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    closes = [100, 100, 100, 110, 100, 90, 100, 100, 100, 100]  # 10 bars
    timestamps = [start + dt.timedelta(minutes=5 * i) for i in range(len(closes))]
    return pd.DataFrame({"timestamp": timestamps, "close": closes})


def test_labels_buy_sell_hold_thresholds():
    df = _flat_then_move_df()
    # horizon=1: bar i's label depends on close[i+1].
    out = compute_labels(
        df, horizon_bars=1, positive_threshold=0.02, negative_threshold=-0.02,
        fee_pct=0.0, slippage_pct=0.0,
    )
    # close[2]=100 -> close[3]=110 => +10% => BUY
    assert out.loc[2, "label"] == BUY
    # close[3]=110 -> close[4]=100 => -9.09% => SELL
    assert out.loc[3, "label"] == SELL
    # close[0]=100 -> close[1]=100 => 0% => HOLD
    assert out.loc[0, "label"] == HOLD


def test_labels_drop_last_horizon_rows_no_leakage():
    df = _flat_then_move_df()
    horizon = 3
    out = compute_labels(df, horizon_bars=horizon, positive_threshold=0.005, negative_threshold=-0.005)
    assert len(out) == len(df) - horizon
    # Every remaining row must have a defined (non-null) future_return.
    assert out["future_return"].notna().all()


def test_labels_fees_reduce_effective_move():
    df = _flat_then_move_df()
    # Without fees, +10% move clears a 5% threshold -> BUY.
    out_no_fee = compute_labels(
        df, horizon_bars=1, positive_threshold=0.05, negative_threshold=-0.05,
        fee_pct=0.0, slippage_pct=0.0,
    )
    assert out_no_fee.loc[2, "label"] == BUY

    # With a round-trip cost that eats the whole move, net return lands
    # inside the HOLD band instead of clearing the BUY threshold.
    out_high_fee = compute_labels(
        df, horizon_bars=1, positive_threshold=0.05, negative_threshold=-0.05,
        fee_pct=0.05, slippage_pct=0.0,
    )
    assert out_high_fee.loc[2, "label"] == HOLD


def test_horizon_must_be_positive():
    df = _flat_then_move_df()
    with pytest.raises(ValueError):
        compute_labels(df, horizon_bars=0)


def test_label_distribution_sums_to_100():
    df = _flat_then_move_df()
    out = compute_labels(df, horizon_bars=1, positive_threshold=0.02, negative_threshold=-0.02, fee_pct=0, slippage_pct=0)
    dist = label_distribution(out["label"])
    total_pct = sum(v["pct"] for v in dist.values())
    assert total_pct == pytest.approx(100.0, abs=0.5)
