"""
Scheduler (Phase 16 / Section 25/44).

What is it?
    Wires up the periodic jobs described in Section 44 (collect data,
    generate predictions, paper trade, daily/weekly reports, health +
    drift checks) using APScheduler, with every job wrapped so a
    failure in one job is logged and skipped rather than ever crashing
    the whole process.

Why is it needed?
    Section 25: "The application must not crash permanently because...
    one API request fails." A scheduler is exactly the kind of
    long-running process where an unhandled exception in one job must
    not take down every other job.

How do I start it?
    python run.py scheduler
    (or it runs inside `python run.py start`, alongside the API)

All intervals are read from settings.yaml `scheduler:` and are fully
configurable — nothing here is hard-coded (Section 44).
"""

from __future__ import annotations

import datetime as dt
import logging

from apscheduler.schedulers.background import BackgroundScheduler

from crypto_ai.config.loader import get_settings
from crypto_ai.database.base import session_scope
from crypto_ai.database.repositories.events_repo import log_event
from crypto_ai.monitoring import health
from crypto_ai.monitoring.alerts import notify

logger = logging.getLogger(__name__)


def guarded(job_name: str):
    """
    Decorator: catches and logs any exception from a scheduled job
    instead of letting APScheduler's executor silently swallow it (or
    worse, letting a repeated failure kill the scheduler thread).
    """

    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                logger.info("Starting scheduled job: %s", job_name)
                fn(*args, **kwargs)
                logger.info("Finished scheduled job: %s", job_name)
            except Exception as exc:  # noqa: BLE001 - a job must never take the scheduler down
                logger.exception("Scheduled job '%s' failed", job_name)
                try:
                    with session_scope() as session:
                        log_event(
                            session, component="scheduler", event="job_failed", severity="ERROR",
                            message=f"{job_name}: {exc}", context={"job": job_name},
                        )
                except Exception:  # noqa: BLE001 - even logging the failure must not raise
                    logger.exception("Additionally failed to record job failure for '%s'", job_name)

        wrapper.__name__ = f"guarded_{job_name}"
        return wrapper

    return decorator


@guarded("collect_data")
def collect_data_job() -> None:
    from crypto_ai.data.downloader.historical import backfill_all

    with session_scope() as session:
        backfill_all(session)


@guarded("health_check")
def health_check_job() -> None:
    with session_scope() as session:
        health.run_all_health_checks(session)


@guarded("evaluate_predictions")
def evaluate_predictions_job() -> None:
    from crypto_ai.models.evaluation.prediction_evaluator import evaluate_due_predictions

    with session_scope() as session:
        evaluate_due_predictions(session)


@guarded("predict_and_paper_trade")
def predict_and_paper_trade_job() -> None:
    from crypto_ai.app.services.live_pipeline import run_prediction_and_paper_trade_tick

    with session_scope() as session:
        result = run_prediction_and_paper_trade_tick(session)
        if result["status"] != "ok":
            logger.info("predict_and_paper_trade tick skipped: %s", result["status"])


@guarded("apply_retention_policy")
def apply_retention_policy_job() -> None:
    from crypto_ai.data.processor.retention import apply_retention_policy

    with session_scope() as session:
        apply_retention_policy(session)


@guarded("daily_report")
def daily_report_job() -> None:
    from crypto_ai.app.services.reporting import generate_daily_report

    with session_scope() as session:
        report = generate_daily_report(session)
        notify(session, "daily_report", "Daily report generated", context={"date": report.get("date")})


@guarded("drift_check")
def drift_check_job() -> None:
    from crypto_ai.app.services.maintenance import run_drift_check_job

    with session_scope() as session:
        result = run_drift_check_job(session)
        if result.get("flagged_for_review"):
            logger.warning("Drift flagged for review: %s", result.get("reasons"))


@guarded("weekly_retrain")
def weekly_retrain_job() -> None:
    from crypto_ai.app.services.maintenance import run_scheduled_retrain

    with session_scope() as session:
        result = run_scheduled_retrain(session)
        logger.info("Scheduled retrain finished with status=%s", result.get("status"))


def build_scheduler() -> BackgroundScheduler:
    settings = get_settings()
    cfg = settings.get("scheduler", {})
    scheduler = BackgroundScheduler(timezone="UTC")

    scheduler.add_job(
        collect_data_job, "interval", minutes=cfg.get("collect_data_minutes", 5), id="collect_data", replace_existing=True,
    )
    scheduler.add_job(
        evaluate_predictions_job, "interval", minutes=cfg.get("prediction_interval_minutes", 5),
        id="evaluate_predictions", replace_existing=True,
    )
    scheduler.add_job(
        predict_and_paper_trade_job, "interval", minutes=cfg.get("prediction_interval_minutes", 5),
        id="predict_and_paper_trade", replace_existing=True,
    )
    scheduler.add_job(
        health_check_job, "interval", minutes=cfg.get("health_check_minutes", 1), id="health_check", replace_existing=True,
    )
    scheduler.add_job(
        drift_check_job, "interval", minutes=cfg.get("drift_check_minutes", 60),
        id="drift_check", replace_existing=True,
    )
    scheduler.add_job(
        daily_report_job, "cron", hour=cfg.get("daily_report_hour_utc", 0), minute=0, id="daily_report", replace_existing=True,
    )
    scheduler.add_job(
        apply_retention_policy_job, "cron", hour=cfg.get("daily_report_hour_utc", 0), minute=15,
        id="apply_retention_policy", replace_existing=True,
    )
    weekday_map = {"monday": "mon", "tuesday": "tue", "wednesday": "wed", "thursday": "thu", "friday": "fri", "saturday": "sat", "sunday": "sun"}
    scheduler.add_job(
        weekly_retrain_job, "cron",
        day_of_week=weekday_map.get(cfg.get("weekly_retrain_day", "sunday"), "sun"),
        hour=cfg.get("weekly_retrain_hour_utc", 1), minute=0, id="weekly_retrain", replace_existing=True,
    )
    return scheduler
