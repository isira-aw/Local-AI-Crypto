"""
Scheduled maintenance operations: drift checks and controlled retraining
(Sections 23 and 44).

These live here rather than inside scheduler/jobs.py so that the CLI, the
API, and the scheduler all invoke exactly the same logic — a drift check
run by hand must mean the same thing as one run by the scheduler.

Critically (Section 23): drift being detected NEVER triggers an automatic
retrain-and-promote. It raises a flag and records it. Retraining, when it
happens, still goes through the same candidate-vs-champion comparison and
promotion criteria as any other training run.
"""

from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from crypto_ai.config.loader import get_settings
from crypto_ai.database.models.monitoring import PerformanceMetric
from crypto_ai.database.repositories.market_data_repo import load_candles_df
from crypto_ai.features.feature_pipeline import compute_features
from crypto_ai.features.labels import compute_labels
from crypto_ai.models.evaluation.prediction_evaluator import summarize_prediction_accuracy
from crypto_ai.models.registry import registry
from crypto_ai.monitoring.alerts import EVENT_DRIFT_DETECTED, EVENT_NEW_CANDIDATE_MODEL, EVENT_TRAINING_COMPLETED, notify
from crypto_ai.monitoring.drift import run_drift_check

logger = logging.getLogger(__name__)

DRIFT_SCOPE = "drift"

# How much of the available history is treated as the "training era"
# baseline that recent data is compared against.
_TRAINING_FRACTION = 0.7


def run_drift_check_job(
    session: Session, symbol: str | None = None, timeframe: str | None = None,
    model_name: str = "btc_direction_classifier",
) -> dict:
    """
    Compares the feature distribution of the recent window against the
    period the champion was trained on, and live prediction accuracy
    against the champion's own held-out test expectation.

    Returns a serializable dict (also persisted to performance_metrics so
    the dashboard and daily report can show a drift status).
    """
    settings = get_settings()
    symbol = symbol or settings.get("exchange.symbol", "BTC/USDT")
    timeframe = timeframe or settings.get("data.primary_timeframe", "5m")

    df = load_candles_df(session, symbol, timeframe)
    if len(df) < 500:
        return {"status": "insufficient_data", "bars": len(df)}

    features = compute_features(df)
    if len(features) < 200:
        return {"status": "insufficient_features", "rows": len(features)}

    feature_cols = [c for c in features.columns if c != "timestamp"]
    split = int(len(features) * _TRAINING_FRACTION)
    training_window = features.iloc[:split]
    recent_window = features.iloc[split:]

    # Expected accuracy: what the champion actually achieved on its
    # never-seen final test set. Comparing live accuracy against a number
    # from the *training* period would be far too optimistic a baseline.
    expected_accuracy_pct = None
    champion = registry.get_champion(session, model_name)
    if champion is not None:
        precision = (champion.metrics or {}).get("classification", {}).get("precision_macro")
        if precision is not None:
            expected_accuracy_pct = float(precision) * 100

    live = summarize_prediction_accuracy(session, symbol=symbol, model_version=champion.version_label if champion else None)
    recent_accuracy_pct = live.get("accuracy_pct")

    report = run_drift_check(
        training_window, recent_window, feature_cols,
        recent_accuracy_pct=recent_accuracy_pct,
        expected_accuracy_pct=expected_accuracy_pct,
    )

    result = {
        "status": "ok",
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "max_psi": round(report.max_psi, 4),
        "feature_drift_flagged": report.feature_drift_flagged,
        "accuracy_drift_flagged": report.accuracy_drift_flagged,
        "flagged_for_review": report.flagged_for_review,
        "recent_accuracy_pct": recent_accuracy_pct,
        "expected_accuracy_pct": expected_accuracy_pct,
        "reasons": report.reasons,
        # Only the worst few features, so the stored blob stays small.
        "top_drifting_features": dict(
            sorted(report.feature_psi.items(), key=lambda kv: kv[1], reverse=True)[:5]
        ),
    }

    session.add(PerformanceMetric(
        scope=DRIFT_SCOPE, timestamp=dt.datetime.now(dt.timezone.utc), metrics=result,
    ))
    session.commit()

    if report.flagged_for_review:
        notify(
            session, EVENT_DRIFT_DETECTED,
            "Drift detected — flagged for review (no automatic retrain/promotion). "
            + "; ".join(report.reasons),
            severity="WARNING", context={"max_psi": result["max_psi"]},
        )

    return result


def get_latest_drift_status(session: Session) -> dict | None:
    row = session.execute(
        select(PerformanceMetric)
        .where(PerformanceMetric.scope == DRIFT_SCOPE)
        .order_by(PerformanceMetric.timestamp.desc())
        .limit(1)
    ).scalar_one_or_none()
    return row.metrics if row else None


def run_scheduled_retrain(
    session: Session, symbol: str | None = None, timeframe: str | None = None,
    model_name: str = "btc_direction_classifier", min_bars: int = 1000,
) -> dict:
    """
    The weekly retrain (Section 44): rebuild features/labels from whatever
    market data currently exists and run the full training pipeline.

    Promotion still happens only through maybe_promote_champion(), i.e. a
    new model replaces the champion only if it beats it on held-out data
    and passes every promotion criterion.
    """
    from crypto_ai.models.training.pipeline import run_training_pipeline

    settings = get_settings()
    symbol = symbol or settings.get("exchange.symbol", "BTC/USDT")
    timeframe = timeframe or settings.get("data.primary_timeframe", "5m")

    df = load_candles_df(session, symbol, timeframe)
    if len(df) < min_bars:
        logger.info("Skipping scheduled retrain: only %s bars available (need %s).", len(df), min_bars)
        return {"status": "insufficient_data", "bars": len(df), "bars_needed": min_bars}

    features = compute_features(df)
    labels = compute_labels(df)

    summary = run_training_pipeline(session, symbol, timeframe, features, labels, model_name=model_name)

    n_passed = sum(1 for r in summary["results"] if r.get("overall_pass"))
    notify(
        session, EVENT_TRAINING_COMPLETED,
        f"Scheduled retrain complete: {n_passed}/{len(summary['results'])} algorithms passed walk-forward validation.",
        context={"promotion": summary["promotion"]},
    )
    if summary["promotion"].get("promoted"):
        notify(
            session, EVENT_NEW_CANDIDATE_MODEL,
            f"New champion promoted: {summary['promotion'].get('version_label')}",
            context=summary["promotion"],
        )

    return {"status": "ok", **summary}
