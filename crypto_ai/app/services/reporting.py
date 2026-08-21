"""
Daily/weekly report generation (Section 22/45).

Pulls together prediction accuracy, paper-trading performance, model
registry status, and system health into the single report structure
described in Section 45 — used by both the scheduler's daily job and
`run.py report`.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from crypto_ai.config.loader import get_settings
from crypto_ai.database.models.ml import MLModel, ModelVersion
from crypto_ai.database.repositories.market_data_repo import get_latest_timestamp, load_candles_df
from crypto_ai.database.repositories.signals_repo import count_signals_by_type
from crypto_ai.models.evaluation.prediction_evaluator import summarize_prediction_accuracy
from crypto_ai.models.registry import registry
from crypto_ai.monitoring import health
from crypto_ai.paper_trading.simulator import summarize_paper_trading

RECOMMEND_CONTINUE = "CONTINUE RESEARCH"
RECOMMEND_NOT_READY = "NOT READY FOR REAL TRADING"
RECOMMEND_TESTNET_CANDIDATE = "CANDIDATE FOR TESTNET"


def _model_status(session: Session, model_name: str) -> dict:
    model = session.execute(select(MLModel).where(MLModel.name == model_name)).scalar_one_or_none()
    if model is None:
        return {"champion": None, "candidate": None}
    champion = registry.get_champion(session, model_name)
    candidate = (
        session.query(ModelVersion)
        .filter(ModelVersion.model_id == model.id, ModelVersion.status == registry.STATUS_CANDIDATE)
        .order_by(ModelVersion.created_at.desc())
        .first()
    )
    return {
        "champion": champion.version_label if champion else None,
        "candidate": candidate.version_label if candidate else None,
    }


def _recommendation(paper_summary: dict, prediction_summary: dict) -> str:
    n_trades = paper_summary.get("n_trades") or 0
    n_predictions = prediction_summary.get("n_predictions") or 0

    # Not enough evidence yet, on either axis: keep researching.
    if n_trades < 20 or n_predictions < 100:
        return RECOMMEND_CONTINUE

    # Section 46: prediction quality and financial results are separate
    # kinds of success. A strategy whose directional calls are no better
    # than a coin flip is not testnet-ready even if P/L happens to be up.
    directional = prediction_summary.get("directional_accuracy_pct")
    if directional is not None and directional <= 50:
        return RECOMMEND_NOT_READY
    if (paper_summary.get("total_return_pct") or -1) <= 0:
        return RECOMMEND_NOT_READY
    if (paper_summary.get("max_drawdown_pct") or 100) > 20:
        return RECOMMEND_NOT_READY
    if paper_summary.get("suspicious_flags"):
        return RECOMMEND_NOT_READY
    return RECOMMEND_TESTNET_CANDIDATE


def generate_daily_report(session: Session, symbol: str | None = None, model_name: str = "btc_direction_classifier") -> dict:
    settings = get_settings()
    symbol = symbol or settings.get("exchange.symbol", "BTC/USDT")
    timeframe = settings.get("data.primary_timeframe", "5m")
    now = dt.datetime.now(dt.timezone.utc)
    since_yesterday = now - dt.timedelta(days=1)

    latest_ts = get_latest_timestamp(session, symbol, timeframe)
    current_price = None
    if latest_ts is not None:
        df = load_candles_df(session, symbol, timeframe, start=latest_ts)
        if not df.empty:
            current_price = float(df.iloc[-1]["close"])

    prediction_summary = summarize_prediction_accuracy(session, symbol=symbol, since=since_yesterday)
    paper_summary = summarize_paper_trading(session, symbol, timeframe)
    model_status = _model_status(session, model_name)
    system_health = health.run_all_health_checks(session)
    signal_counts = count_signals_by_type(session, symbol, since=since_yesterday)

    from crypto_ai.app.services.maintenance import get_latest_drift_status
    drift_status = get_latest_drift_status(session)

    report = {
        "date": now.date().isoformat(),
        "mode": settings.mode,
        "symbol": symbol,
        "current_price": current_price,
        "predictions": prediction_summary,
        "signal_counts": signal_counts,
        "drift_status": drift_status,
        "paper_trading": paper_summary,
        "model": model_status,
        "system_health": system_health,
        "recommendation": _recommendation(paper_summary, prediction_summary),
        "disclaimer": "Past performance does not guarantee future results.",
    }
    return report


def _format_drift(drift: dict | None) -> str:
    if not drift:
        return "not yet checked"
    if drift.get("status") != "ok":
        return str(drift.get("status"))
    if drift.get("flagged_for_review"):
        return f"FLAGGED — {'; '.join(drift.get('reasons', []))}"
    return f"stable (max PSI {drift.get('max_psi')})"


def format_text_report(report: dict) -> str:
    p = report["paper_trading"]
    pred = report["predictions"]
    lines = [
        "CRYPTO AI DAILY REPORT",
        "",
        f"Date: {report['date']}",
        f"Mode: {report['mode']}",
        "",
        f"{report['symbol']}:",
        f"  Current price: {report['current_price']}",
        "",
        "Predictions:",
        f"  BUY: {report.get('signal_counts', {}).get('BUY', 0)}   "
        f"SELL: {report.get('signal_counts', {}).get('SELL', 0)}   "
        f"WAIT: {report.get('signal_counts', {}).get('WAIT', 0)}",
        f"  Evaluated: {pred.get('n_predictions', 0)}",
        f"  Accuracy: {pred.get('accuracy_pct')}",
        f"  Directional accuracy: {pred.get('directional_accuracy_pct')}",
        "",
        "Paper trading:",
        f"  Starting balance: {p.get('starting_balance')}",
        f"  Current equity: {p.get('current_equity')}",
        f"  Total return: {p.get('total_return_pct')}%",
        f"  Max drawdown: {p.get('max_drawdown_pct')}%",
        f"  Trades: {p.get('n_trades')}",
        "",
        "Model:",
        f"  Champion: {report['model']['champion']}",
        f"  Candidate: {report['model']['candidate']}",
        f"  Drift status: {_format_drift(report.get('drift_status'))}",
        "",
        "System:",
        f"  Database: {report['system_health'].get('database', {}).get('status')}",
        f"  Exchange: {report['system_health'].get('exchange', {}).get('status')}",
        f"  LLM: {report['system_health'].get('llm', {}).get('status')}",
        "",
        f"Recommendation: {report['recommendation']}",
        "",
        report["disclaimer"],
    ]
    return "\n".join(lines)
