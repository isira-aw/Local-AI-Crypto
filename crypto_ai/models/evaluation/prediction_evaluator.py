"""
Prediction evaluation (Phase 11 / Section 21).

    AI prediction at 10:00
            v
    Wait 1 hour (the prediction's horizon)
            v
    Get actual BTC price
            v
    Calculate return
            v
    Was prediction correct?
            v
    Update metrics

What is it?
    Closes the loop on every prediction the system makes: once enough
    time has passed for its horizon to elapse, look up what actually
    happened and record it on the same row.

Why is it needed?
    Section 21: "Do not call the AI successful merely because it has
    high classification accuracy." This is what lets the daily/weekly
    reports (and the dashboard) show real, ongoing accuracy — not a
    one-time backtest number that might go stale as the market
    changes.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from crypto_ai.config.loader import get_settings
from crypto_ai.database.models.trading import Prediction
from crypto_ai.database.repositories.market_data_repo import get_price_at_or_after
from crypto_ai.features.labels import BUY, HOLD, SELL

_TIMEFRAME_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


def _as_utc(ts: dt.datetime) -> dt.datetime:
    """
    Postgres round-trips timezone-aware datetimes faithfully; SQLite
    (used only in tests/dev) drops tzinfo. Everywhere in this app
    "naive" is assumed to mean UTC, so normalize before comparing.
    """
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=dt.timezone.utc)


def record_prediction(
    session: Session,
    symbol: str,
    timeframe: str,
    timestamp: dt.datetime,
    horizon_bars: int,
    data_version: str,
    feature_version: str,
    model_version: str,
    strategy_version: str,
    predicted_class: str,
    confidence: float,
    probabilities: dict,
    price_at_prediction: float,
) -> Prediction:
    row = Prediction(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=timestamp,
        horizon_bars=horizon_bars,
        data_version=data_version,
        feature_version=feature_version,
        model_version=model_version,
        strategy_version=strategy_version,
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
        price_at_prediction=price_at_prediction,
    )
    session.add(row)
    session.flush()
    return row


def _classify_return(net_return: float, positive_threshold: float, negative_threshold: float) -> str:
    if net_return > positive_threshold:
        return BUY
    if net_return < negative_threshold:
        return SELL
    return HOLD


def evaluate_due_predictions(session: Session, now: dt.datetime | None = None) -> int:
    """
    Finds every un-evaluated prediction whose horizon has elapsed,
    looks up the actual price at that point, and fills in the
    actual_return / actual_class / was_correct columns.

    Returns the number of predictions evaluated.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    settings = get_settings()
    label_cfg = settings.get("labeling", {})
    positive_threshold = label_cfg.get("positive_threshold", 0.005)
    negative_threshold = label_cfg.get("negative_threshold", -0.005)
    fee_pct = label_cfg.get("assumed_fee_pct", 0.001)
    slippage_pct = label_cfg.get("assumed_slippage_pct", 0.0005)
    round_trip_cost = 2 * (fee_pct + slippage_pct)

    pending = session.execute(
        select(Prediction).where(Prediction.evaluated_at.is_(None))
    ).scalars().all()

    n_evaluated = 0
    for pred in pending:
        interval = _TIMEFRAME_SECONDS.get(pred.timeframe, 300)
        due_at = _as_utc(pred.timestamp) + dt.timedelta(seconds=interval * pred.horizon_bars)
        if due_at > _as_utc(now):
            continue  # horizon hasn't elapsed yet

        result = get_price_at_or_after(session, pred.symbol, pred.timeframe, due_at)
        if result is None:
            continue  # we don't have market data that far forward yet

        _, actual_price = result
        actual_return = actual_price / pred.price_at_prediction - 1
        net_return = actual_return - round_trip_cost
        actual_class = _classify_return(net_return, positive_threshold, negative_threshold)

        pred.actual_future_price = actual_price
        pred.actual_return = actual_return
        pred.actual_class = actual_class
        pred.was_correct = actual_class == pred.predicted_class
        pred.evaluated_at = now
        n_evaluated += 1

    session.flush()
    return n_evaluated


def summarize_prediction_accuracy(
    session: Session, symbol: str | None = None, model_version: str | None = None, since: dt.datetime | None = None
) -> dict:
    """
    Section 21's report fields: prediction accuracy, directional
    accuracy, expected vs. actual return, win rate.
    """
    stmt = select(Prediction).where(Prediction.evaluated_at.is_not(None))
    if symbol:
        stmt = stmt.where(Prediction.symbol == symbol)
    if model_version:
        stmt = stmt.where(Prediction.model_version == model_version)
    if since:
        stmt = stmt.where(Prediction.timestamp >= since)

    rows = session.execute(stmt).scalars().all()
    if not rows:
        return {
            "n_predictions": 0, "accuracy_pct": None, "directional_accuracy_pct": None,
            "avg_actual_return_pct": None, "win_rate_pct": None,
        }

    n = len(rows)
    n_correct = sum(1 for r in rows if r.was_correct)

    # "Directional accuracy" ignores HOLD predictions/outcomes and asks:
    # when the model called a direction (BUY/SELL), did price actually
    # move that way?
    directional_rows = [r for r in rows if r.predicted_class in (BUY, SELL)]
    n_directional_correct = sum(
        1 for r in directional_rows
        if (r.predicted_class == BUY and r.actual_return > 0)
        or (r.predicted_class == SELL and r.actual_return < 0)
    )

    trade_rows = [r for r in rows if r.predicted_class in (BUY, SELL)]
    wins = sum(
        1 for r in trade_rows
        if (r.predicted_class == BUY and r.actual_return > 0)
        or (r.predicted_class == SELL and r.actual_return < 0)
    )

    return {
        "n_predictions": n,
        "accuracy_pct": round(n_correct / n * 100, 2),
        "directional_accuracy_pct": (
            round(n_directional_correct / len(directional_rows) * 100, 2) if directional_rows else None
        ),
        "avg_actual_return_pct": round(sum(r.actual_return for r in rows) / n * 100, 4),
        "win_rate_pct": round(wins / len(trade_rows) * 100, 2) if trade_rows else None,
        "n_buy_sell_predictions": len(trade_rows),
    }
