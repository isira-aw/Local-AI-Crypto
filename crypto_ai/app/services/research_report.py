"""
Weekly and Month-1 research reports (Section 22).

What is it?
    A longer-horizon companion to the daily report. Where the daily
    report answers "what happened today", this answers "is this system
    actually showing evidence of anything, after N weeks of running".

Why is it needed?
    Section 22 defines a MONTH_1_RESEARCH_MODE whose end-of-month output
    has specific sections: model performance, strategy performance,
    paper-trading performance, risk metrics, model stability, data
    quality, and a recommendation.

    The recommendation is deliberately conservative and can only be one
    of three values. "Profit is positive" alone never produces
    CANDIDATE FOR TESTNET — Section 22: "Never automatically recommend
    real trading merely because profit is positive."

How do I start it?
    python run.py research-report            (defaults to 30 days)
    python run.py research-report --days 7   (weekly)
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from crypto_ai.app.services.maintenance import get_latest_drift_status
from crypto_ai.config.loader import get_settings
from crypto_ai.data.validators.ohlcv_validators import validate_ohlcv
from crypto_ai.database.models.market import MarketData
from crypto_ai.database.models.ml import MLModel, ModelVersion
from crypto_ai.database.models.monitoring import SystemEvent
from crypto_ai.database.repositories.market_data_repo import load_candles_df
from crypto_ai.database.repositories.signals_repo import count_signals_by_type
from crypto_ai.models.evaluation.prediction_evaluator import summarize_prediction_accuracy
from crypto_ai.models.registry import registry
from crypto_ai.paper_trading.simulator import summarize_paper_trading
from crypto_ai.utils.timeutils import as_utc, utc_now

RECOMMEND_CONTINUE = "CONTINUE RESEARCH"
RECOMMEND_NOT_READY = "NOT READY FOR REAL TRADING"
RECOMMEND_TESTNET_CANDIDATE = "CANDIDATE FOR TESTNET"


def _model_performance(session: Session, model_name: str) -> dict:
    model = session.execute(select(MLModel).where(MLModel.name == model_name)).scalar_one_or_none()
    if model is None:
        return {"champion": None, "n_versions": 0, "versions": []}

    versions = (
        session.query(ModelVersion)
        .filter(ModelVersion.model_id == model.id)
        .order_by(ModelVersion.created_at.asc())
        .all()
    )
    champion = registry.get_champion(session, model_name)
    return {
        "champion": champion.version_label if champion else None,
        "champion_metrics": (champion.metrics or {}) if champion else None,
        "n_versions": len(versions),
        "n_candidates": sum(1 for v in versions if v.status == registry.STATUS_CANDIDATE),
        "n_rejected": sum(1 for v in versions if v.status == registry.STATUS_REJECTED),
        "versions": [
            {
                "version": v.version_label, "algorithm": v.algorithm, "status": v.status,
                "folds_passed": (v.walk_forward_results or {}).get("n_passed"),
                "folds_total": (v.walk_forward_results or {}).get("n_folds"),
            }
            for v in versions
        ],
    }


def _model_stability(model_perf: dict) -> dict:
    """
    Stability here means: did the champion pass consistently across
    walk-forward folds, and how often has the champion changed? A system
    that promotes a new champion every week is not stable, however good
    each individual model looked.
    """
    versions = model_perf.get("versions", [])
    champions_over_time = [v for v in versions if v["status"] in ("CHAMPION", "RETIRED")]

    fold_consistency = None
    for v in versions:
        if v["version"] == model_perf.get("champion") and v.get("folds_total"):
            fold_consistency = f"{v['folds_passed']}/{v['folds_total']} folds passed"

    return {
        "champion_changes": max(0, len(champions_over_time) - 1),
        "champion_fold_consistency": fold_consistency,
        "n_rejected_attempts": model_perf.get("n_rejected", 0),
        "assessment": (
            "no champion yet" if not model_perf.get("champion")
            else "stable" if len(champions_over_time) <= 1
            else f"champion changed {len(champions_over_time) - 1} time(s)"
        ),
    }


def _data_quality(session: Session, symbol: str, timeframe: str, since: dt.datetime) -> dict:
    df = load_candles_df(session, symbol, timeframe, start=since)
    if df.empty:
        return {"bars": 0, "status": "NO DATA", "issues": ["no market data in the reporting window"]}

    report = validate_ohlcv(df, timeframe)
    return {
        "bars": len(df),
        "status": "OK" if report.is_clean else "ISSUES",
        "n_gaps": len(report.gaps),
        "n_duplicates": report.n_duplicates,
        "n_bad_bars": len(report.bad_ohlc_rows),
        "summary": report.summary(),
    }


def _risk_metrics(session: Session, paper: dict, since: dt.datetime) -> dict:
    critical = (
        session.query(SystemEvent)
        .filter(SystemEvent.severity.in_(["ERROR", "CRITICAL"]), SystemEvent.timestamp >= since)
        .count()
    )
    return {
        "max_drawdown_pct": paper.get("max_drawdown_pct"),
        "sharpe_ratio": paper.get("sharpe_ratio"),
        "profit_factor": paper.get("profit_factor"),
        "n_error_events": critical,
        "suspicious_flags": paper.get("suspicious_flags", []),
    }


def _recommendation(paper: dict, predictions: dict, stability: dict, data_quality: dict, risk: dict) -> tuple[str, list[str]]:
    """
    Returns (recommendation, reasons). Section 22: never recommend real
    trading just because profit is positive.
    """
    reasons: list[str] = []

    n_trades = paper.get("n_trades") or 0
    n_predictions = predictions.get("n_predictions") or 0

    if n_trades < 100:
        reasons.append(f"only {n_trades} paper trades so far (want >= 100 before drawing conclusions)")
    if n_predictions < 200:
        reasons.append(f"only {n_predictions} evaluated predictions (want >= 200)")
    if not reasons:
        # Enough evidence exists — now judge its quality.
        total_return = paper.get("total_return_pct")
        if total_return is None or total_return <= 0:
            reasons.append(f"paper-trading net return is {total_return}% (needs to be positive)")
        drawdown = paper.get("max_drawdown_pct")
        if drawdown is not None and drawdown > 20:
            reasons.append(f"max drawdown {drawdown:.1f}% exceeds the 20% limit")
        directional = predictions.get("directional_accuracy_pct")
        if directional is not None and directional <= 50:
            reasons.append(f"directional accuracy {directional:.1f}% is no better than a coin flip")
        if risk.get("suspicious_flags"):
            reasons.append("results carry too-good-to-be-true warnings: " + "; ".join(risk["suspicious_flags"]))
        if risk.get("n_error_events", 0) > 0:
            reasons.append(f"{risk['n_error_events']} error/critical system event(s) in the window")
        if data_quality.get("status") not in ("OK",):
            reasons.append(f"data quality: {data_quality.get('summary', data_quality.get('status'))}")
        if stability.get("champion_changes", 0) > 2:
            reasons.append(f"champion changed {stability['champion_changes']} times — model is not stable")
        if not stability.get("champion"):
            pass  # covered by model section

    if n_trades < 100 or n_predictions < 200:
        return RECOMMEND_CONTINUE, reasons
    if reasons:
        return RECOMMEND_NOT_READY, reasons
    return RECOMMEND_TESTNET_CANDIDATE, ["all configured research criteria met — see docs/REAL_TRADING.md for the separate live-trading gate"]


def generate_research_report(
    session: Session, days: int = 30, symbol: str | None = None,
    model_name: str = "btc_direction_classifier",
) -> dict:
    settings = get_settings()
    symbol = symbol or settings.get("exchange.symbol", "BTC/USDT")
    timeframe = settings.get("data.primary_timeframe", "5m")
    now = utc_now()
    since = now - dt.timedelta(days=days)

    paper = summarize_paper_trading(session, symbol, timeframe)
    predictions = summarize_prediction_accuracy(session, symbol=symbol, since=since)
    model_perf = _model_performance(session, model_name)
    stability = _model_stability(model_perf)
    data_quality = _data_quality(session, symbol, timeframe, since)
    risk = _risk_metrics(session, paper, since)
    signals = count_signals_by_type(session, symbol, since=since)
    drift = get_latest_drift_status(session)

    recommendation, reasons = _recommendation(paper, predictions, stability, data_quality, risk)

    return {
        "period_days": days,
        "generated_at": now.isoformat(),
        "mode": settings.mode,
        "symbol": symbol,
        "model_performance": model_perf,
        "strategy_performance": {"signal_counts": signals, "predictions": predictions},
        "paper_trading_performance": paper,
        "risk_metrics": risk,
        "model_stability": stability,
        "data_quality": data_quality,
        "drift_status": drift,
        "recommendation": recommendation,
        "recommendation_reasons": reasons,
        "disclaimer": "Past performance does not guarantee future results. This is an experimental research system.",
    }


def format_research_report(report: dict) -> str:
    p = report["paper_trading_performance"]
    pred = report["strategy_performance"]["predictions"]
    sig = report["strategy_performance"]["signal_counts"]
    m = report["model_performance"]
    st = report["model_stability"]
    dq = report["data_quality"]
    r = report["risk_metrics"]

    def num(v, suffix="", fmt="{:.2f}"):
        return "n/a" if v is None else (fmt.format(v) + suffix)

    lines = [
        f"RESEARCH REPORT — last {report['period_days']} days",
        "=" * 60,
        f"Generated: {report['generated_at']}",
        f"Mode: {report['mode']}   Symbol: {report['symbol']}",
        "",
        "MODEL PERFORMANCE",
        f"  Champion: {m['champion'] or 'none yet'}",
        f"  Versions trained: {m['n_versions']} (candidates: {m.get('n_candidates', 0)}, rejected: {m.get('n_rejected', 0)})",
        "",
        "STRATEGY PERFORMANCE",
        f"  Signals: BUY={sig.get('BUY', 0)}  SELL={sig.get('SELL', 0)}  WAIT={sig.get('WAIT', 0)}",
        f"  Predictions evaluated: {pred.get('n_predictions', 0)}",
        f"  Prediction accuracy: {num(pred.get('accuracy_pct'), '%')}",
        f"  Directional accuracy: {num(pred.get('directional_accuracy_pct'), '%')}",
        "",
        "PAPER TRADING PERFORMANCE",
        f"  Trades: {p.get('n_trades', 0)}",
        f"  Current equity: {num(p.get('current_equity'))}",
        f"  Total return: {num(p.get('total_return_pct'), '%')}",
        f"  Win rate: {num(p.get('win_rate_pct'), '%')}",
        f"  Profit factor: {num(p.get('profit_factor'))}",
        "",
        "RISK METRICS",
        f"  Max drawdown: {num(r.get('max_drawdown_pct'), '%')}",
        f"  Sharpe ratio: {num(r.get('sharpe_ratio'))}",
        f"  Error/critical events: {r.get('n_error_events', 0)}",
        "",
        "MODEL STABILITY",
        f"  Assessment: {st['assessment']}",
        f"  Champion fold consistency: {st.get('champion_fold_consistency') or 'n/a'}",
        f"  Rejected attempts: {st.get('n_rejected_attempts', 0)}",
        "",
        "DATA QUALITY",
        f"  Bars in window: {dq.get('bars', 0)}",
        f"  Status: {dq.get('status')}",
        f"  {dq.get('summary', '')}",
        "",
        f"RECOMMENDATION: {report['recommendation']}",
    ]
    for reason in report["recommendation_reasons"]:
        lines.append(f"  - {reason}")

    if r.get("suspicious_flags"):
        lines += ["", "WARNINGS:"] + [f"  - {f}" for f in r["suspicious_flags"]]

    lines += ["", report["disclaimer"]]
    return "\n".join(lines)
