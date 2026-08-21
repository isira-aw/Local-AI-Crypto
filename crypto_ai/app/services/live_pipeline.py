"""
Live prediction + paper-trading tick (Phase 16 — ties everything
together).

What is it?
    The one function that runs the whole loop for a single new bar:
    load recent data -> compute features -> ask the champion model for
    a prediction -> pass it through the risk engine -> act on the
    result in the paper-trading engine -> (optionally) ask the local
    LLM to explain it. This is what the scheduler calls every
    `prediction_interval_minutes` (Section 44), and what
    `python run.py paper` calls once for a manual/ad-hoc tick.

Why is it needed?
    Every other module in this system (features, labels, training,
    backtesting, risk, paper trading, LLM) is independently testable
    and composable — this is the one place that wires them together
    into the actual "observe -> decide -> act" loop described in
    Section 2. Keeping that wiring in one small, readable function
    (rather than scattered across the scheduler and the CLI) makes it
    possible to audit exactly what happens on every tick.

Safety notes:
    - Refuses to trade if the champion's feature_version doesn't match
      the CURRENT feature pipeline version — silently mismatched
      features would make predictions meaningless.
    - Every decision still goes through RiskManager.evaluate() — this
      function does not bypass any risk limit.
"""

from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from crypto_ai.config.loader import get_risk_config, get_settings
from crypto_ai.database.models.trading import PaperTrade
from crypto_ai.database.repositories.market_data_repo import load_candles_df
from crypto_ai.features.feature_pipeline import FEATURE_VERSION, compute_data_version, compute_features
from crypto_ai.llm.analyst import explain_signal
from crypto_ai.models.inference.predict import predict_single
from crypto_ai.models.registry import registry
from crypto_ai.paper_trading.engine import PaperTradingEngine
from crypto_ai.paper_trading.portfolio import get_equity_curve
from crypto_ai.risk.manager import RiskManager, RiskState
from crypto_ai.utils.timeutils import as_utc, utc_now
from crypto_ai.database.repositories.signals_repo import save_signal
from crypto_ai.strategies.ml_strategy import STRATEGY_VERSION
from crypto_ai.strategies.signal_engine import build_signal

logger = logging.getLogger(__name__)

MIN_BARS_FOR_FEATURES = 250  # comfortably more than the longest indicator window

# How much history a single live tick loads. Only the LAST row's features
# are used, so loading the whole table (which grows without bound) on every
# tick is wasted work — this window is generous enough to warm up every
# indicator while keeping each tick's cost constant as history accumulates.
LIVE_INFERENCE_WINDOW_BARS = 1000


def get_current_risk_state(session: Session, mode: str = "PAPER", now: dt.datetime | None = None) -> RiskState:
    now = as_utc(now) or utc_now()
    equity_curve = get_equity_curve(session, mode=mode)

    if equity_curve.empty:
        starting_equity = get_settings().get("paper_trading.starting_balance_usdt", 1000.0)
        return RiskState(
            current_equity=starting_equity, starting_equity_today=starting_equity,
            peak_equity=starting_equity, consecutive_losses=0, open_positions=0, now=now,
        )

    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    todays_curve = equity_curve[equity_curve.index >= today_start]
    starting_equity_today = float(todays_curve.iloc[0]) if len(todays_curve) else float(equity_curve.iloc[0])
    current_equity = float(equity_curve.iloc[-1])
    peak_equity = float(equity_curve.cummax().iloc[-1])

    recent_trades = (
        session.query(PaperTrade)
        .filter(PaperTrade.status == "CLOSED")
        .order_by(PaperTrade.exit_time.desc())
        .limit(10)
        .all()
    )
    consecutive_losses = 0
    last_loss_at = None
    for t in recent_trades:
        if t.pnl is not None and t.pnl < 0:
            consecutive_losses += 1
            if last_loss_at is None:
                last_loss_at = as_utc(t.exit_time)
        else:
            break

    open_positions = session.query(PaperTrade).filter(PaperTrade.status == "OPEN").count()

    return RiskState(
        current_equity=current_equity, starting_equity_today=starting_equity_today,
        peak_equity=peak_equity, consecutive_losses=consecutive_losses,
        open_positions=open_positions, last_loss_at=last_loss_at, now=now,
    )


def run_prediction_and_paper_trade_tick(
    session: Session, symbol: str | None = None, timeframe: str | None = None,
    model_name: str = "btc_direction_classifier", explain: bool = False,
) -> dict:
    settings = get_settings()
    risk_cfg = get_risk_config()
    symbol = symbol or settings.get("exchange.symbol", "BTC/USDT")
    timeframe = timeframe or settings.get("data.primary_timeframe", "5m")

    df = load_candles_df(session, symbol, timeframe, limit=LIVE_INFERENCE_WINDOW_BARS)
    if len(df) < MIN_BARS_FOR_FEATURES:
        return {"status": "insufficient_data", "bars_available": len(df), "bars_needed": MIN_BARS_FOR_FEATURES}

    champion = registry.get_champion(session, model_name)
    if champion is None:
        return {"status": "no_champion_model"}

    if champion.feature_version != FEATURE_VERSION:
        return {
            "status": "feature_version_mismatch",
            "champion_feature_version": champion.feature_version,
            "current_feature_version": FEATURE_VERSION,
        }

    features = compute_features(df)
    if features.empty:
        return {"status": "insufficient_data_for_features"}

    latest = features.iloc[[-1]]
    feature_cols = [c for c in features.columns if c != "timestamp"]
    latest_timestamp = df["timestamp"].iloc[-1]
    latest_close = float(df["close"].iloc[-1])

    estimator = registry.load_estimator(champion)
    predicted_class, confidence, probabilities = predict_single(estimator, latest[feature_cols])

    volatility_col = "volatility_12" if "volatility_12" in latest.columns else None
    volatility_pct = float(latest[volatility_col].iloc[0]) * 100 if volatility_col else 1.0

    state = get_current_risk_state(session, mode="PAPER")
    risk_manager = RiskManager(risk_cfg)
    decision = risk_manager.evaluate(predicted_class, confidence, state, volatility_pct)

    label_cfg = settings.get("labeling", {})
    data_version = compute_data_version(symbol, timeframe, df)
    from crypto_ai.models.evaluation.prediction_evaluator import record_prediction

    prediction = record_prediction(
        session, symbol, timeframe, latest_timestamp, label_cfg.get("horizon_bars", 12),
        data_version, champion.feature_version, champion.version_label, STRATEGY_VERSION,
        predicted_class, confidence, probabilities, latest_close,
    )

    # Build the structured Section 19 signal. The risk engine's verdict —
    # not the model's raw class — is what actually gets acted on, and the
    # reason codes record why if they differ.
    signal_obj = build_signal(
        symbol=symbol,
        timestamp=latest_timestamp,
        predicted_class=predicted_class,
        confidence=confidence,
        risk_score=decision.risk_score,
        model_version=champion.version_label,
        strategy_version=STRATEGY_VERSION,
        trading_signal_override="WAIT" if decision.allowed_signal not in ("BUY", "SELL") else None,
        reason_codes=decision.reason_codes,
    )

    explanation = None
    if explain:
        explanation = explain_signal({
            "symbol": symbol, "price": latest_close, "signal": signal_obj.signal,
            "confidence": confidence, "risk_score": decision.risk_score,
        })
        signal_obj.llm_explanation = explanation

    save_signal(session, signal_obj, prediction_id=prediction.id)

    # decision.position_size_pct is a PERCENT (e.g. 5.0); the engine wants a
    # 0-1 fraction of equity. A 0.0 size means "do not open" — never fall
    # back to 1.0 (100% of the account), which is what the previous
    # `or 1.0` did.
    position_fraction = decision.position_size_pct / 100.0
    engine = PaperTradingEngine(
        symbol=symbol,
        position_size_pct=position_fraction,
        stop_loss_pct=(risk_cfg.get("orders", {}).get("stop_loss_percent") or 0) / 100 or None,
        take_profit_pct=(risk_cfg.get("orders", {}).get("take_profit_percent") or 0) / 100 or None,
    )
    action = engine.process_bar(
        session, timestamp=latest_timestamp, close=latest_close, signal=decision.allowed_signal,
        strategy_version=STRATEGY_VERSION, model_version=champion.version_label,
        reason=",".join(decision.reason_codes),
    )
    session.commit()

    return {
        "status": "ok",
        "symbol": symbol,
        "timestamp": latest_timestamp.isoformat(),
        "price": latest_close,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "final_signal": decision.allowed_signal,
        "signal": signal_obj.signal,
        "risk_score": decision.risk_score,
        "reason_codes": decision.reason_codes,
        "action": action,
        "model_version": champion.version_label,
        "explanation": explanation,
    }
