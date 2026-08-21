"""
Dashboard API routes (Phase 14 / Section 28).

One route per dashboard page described in Section 28. Every route
returns plain JSON — no secrets are ever included (Section 30: "Never
expose API keys to the frontend").
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from crypto_ai.app.services.reporting import generate_daily_report
from crypto_ai.config.loader import get_settings
from crypto_ai.database.base import get_session_factory
from crypto_ai.database.models.trading import BacktestRun
from crypto_ai.database.repositories.market_data_repo import get_latest_timestamp, load_candles_df
from crypto_ai.exchange.paper import get_balance, get_position
from crypto_ai.features.feature_pipeline import compute_features
from crypto_ai.models.evaluation.prediction_evaluator import summarize_prediction_accuracy
from crypto_ai.llm.client import check_llm_health
from crypto_ai.models.registry import registry
from crypto_ai.monitoring import health
from crypto_ai.app.services.maintenance import get_latest_drift_status
from crypto_ai.database.repositories.signals_repo import get_latest_signal, get_recent_signals
from crypto_ai.monitoring.metrics import get_resource_usage
from crypto_ai.paper_trading.portfolio import get_closed_trades
from crypto_ai.paper_trading.simulator import summarize_paper_trading
from crypto_ai.risk.emergency import activate_emergency_stop, emergency_stop_details, is_emergency_stop_active, reset_emergency_stop

router = APIRouter(prefix="/api")


def get_db() -> Session:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


@router.get("/overview")
def overview(db: Session = Depends(get_db)):
    settings = get_settings()
    symbol = settings.get("exchange.symbol", "BTC/USDT")
    timeframe = settings.get("data.primary_timeframe", "5m")

    latest_ts = get_latest_timestamp(db, symbol, timeframe)
    current_price = None
    if latest_ts is not None:
        df = load_candles_df(db, symbol, timeframe, start=latest_ts)
        if not df.empty:
            current_price = float(df.iloc[-1]["close"])

    paper_summary = summarize_paper_trading(db, symbol, timeframe)
    position = get_position(db, symbol)
    champion = registry.get_champion(db, "btc_direction_classifier")
    latest_signal = get_latest_signal(db, symbol)
    drift = get_latest_drift_status(db)

    return {
        "mode": settings.mode,
        "symbol": symbol,
        "current_price": current_price,
        "paper_balance": get_balance(db),
        "current_position": position,
        "paper_trading_summary": paper_summary,
        "champion_model": champion.version_label if champion else None,
        "current_signal": (
            {
                "signal": latest_signal.signal,
                "confidence": latest_signal.confidence,
                "risk_score": latest_signal.risk_score,
                "reason_codes": latest_signal.reason_codes,
                "timestamp": latest_signal.timestamp.isoformat(),
            }
            if latest_signal else None
        ),
        "drift_status": drift,
        "trading_enabled": settings.safety.trading_enabled,
        "emergency_stop": is_emergency_stop_active(),
        "warning": "Past performance does not guarantee future results.",
    }


@router.get("/market")
def market(
    timeframe: str = Query(default=None),
    limit: int = Query(default=300, le=2000),
    db: Session = Depends(get_db),
):
    settings = get_settings()
    symbol = settings.get("exchange.symbol", "BTC/USDT")
    timeframe = timeframe or settings.get("data.primary_timeframe", "5m")

    df = load_candles_df(db, symbol, timeframe)
    if df.empty:
        return {"symbol": symbol, "timeframe": timeframe, "candles": [], "indicators": []}

    df = df.tail(limit).reset_index(drop=True)
    features = compute_features(df)

    candles = [
        {"timestamp": row.timestamp.isoformat(), "open": row.open, "high": row.high, "low": row.low, "close": row.close, "volume": row.volume}
        for row in df.itertuples()
    ]
    indicators = features.tail(limit).assign(timestamp=lambda d: d["timestamp"].apply(lambda t: t.isoformat())).to_dict("records")

    return {"symbol": symbol, "timeframe": timeframe, "candles": candles, "indicators": indicators}


@router.get("/ai")
def ai_page(db: Session = Depends(get_db)):
    settings = get_settings()
    symbol = settings.get("exchange.symbol", "BTC/USDT")
    accuracy = summarize_prediction_accuracy(db, symbol=symbol)
    champion = registry.get_champion(db, "btc_direction_classifier")

    return {
        "champion_model": champion.version_label if champion else None,
        "champion_metrics": champion.metrics if champion else None,
        "prediction_accuracy": accuracy,
        "llm_status": check_llm_health(),
        "drift_status": get_latest_drift_status(db),
        "recent_signals": [
            {
                "timestamp": s_.timestamp.isoformat(), "signal": s_.signal,
                "confidence": s_.confidence, "risk_score": s_.risk_score,
                "reason_codes": s_.reason_codes, "model_version": s_.model_version,
                "llm_explanation": s_.llm_explanation,
            }
            for s_ in get_recent_signals(db, symbol, limit=25)
        ],
    }


@router.get("/backtests")
def backtests(limit: int = Query(default=20, le=100), db: Session = Depends(get_db)):
    rows = db.query(BacktestRun).order_by(BacktestRun.created_at.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "strategy_version": r.strategy_version,
            "model_version": r.model_version,
            "symbol": r.symbol,
            "timeframe": r.timeframe,
            "start_time": r.start_time.isoformat(),
            "end_time": r.end_time.isoformat(),
            "initial_balance": r.initial_balance,
            "final_balance": r.final_balance,
            "metrics": r.metrics,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


@router.get("/paper-trading")
def paper_trading(limit: int = Query(default=50, le=500), db: Session = Depends(get_db)):
    settings = get_settings()
    symbol = settings.get("exchange.symbol", "BTC/USDT")
    summary = summarize_paper_trading(db, symbol)
    trades = get_closed_trades(db, symbol)[-limit:]
    position = get_position(db, symbol)

    return {
        "summary": summary,
        "current_position": position,
        "recent_trades": [
            {
                "id": t.id, "side": t.side, "entry_time": t.entry_time.isoformat(),
                "entry_price": t.entry_price,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "exit_price": t.exit_price, "pnl": t.pnl, "reason": t.reason,
            }
            for t in trades
        ],
    }


@router.get("/system")
def system_page(db: Session = Depends(get_db)):
    settings = get_settings()
    return {
        "mode": settings.mode,
        "trading_enabled": settings.safety.trading_enabled,
        "real_trading_enabled": settings.safety.real_trading_enabled,
        "emergency_stop": emergency_stop_details() or {"active": False},
        "resource_usage": get_resource_usage(),
        "health": health.run_all_health_checks(db),
        "drift_status": get_latest_drift_status(db),
    }


@router.get("/report/daily")
def daily_report(db: Session = Depends(get_db)):
    return generate_daily_report(db)


@router.post("/emergency-stop/activate")
def emergency_stop_activate(reason: str = "Activated from dashboard"):
    activate_emergency_stop(reason)
    return {"active": True, "reason": reason}


@router.post("/emergency-stop/reset")
def emergency_stop_reset():
    reset_emergency_stop()
    return {"active": False}
