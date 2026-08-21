"""
LLM analyst layer (Section 18).

What is it?
    Turns already-computed numbers (signal, indicators, performance)
    into a plain-language explanation. This is the ONLY thing the LLM
    is used for in this system — never for the underlying trading
    decision itself.

Why is it needed?
    A beginner user should be able to understand *why* the system did
    something, not just see a number. Section 18: "The LLM output must
    NOT directly execute trades."

What goes in / comes out?
    In: a Signal (strategies/signal_engine.py) plus market/portfolio
        context (all already computed elsewhere).
    Out: a short natural-language string, or a template-based
        fallback string if the LLM is disabled/unreachable — the
        caller never has to check which one it got.
"""

from __future__ import annotations

from crypto_ai.llm.client import generate_safe
from crypto_ai.llm.prompts import daily_report, signal_explanation


def explain_signal(context: dict) -> str:
    """
    context keys: symbol, price, signal, confidence, risk_score, rsi,
    ema_trend, macd_histogram, volatility, current_position,
    recent_performance. Missing keys fall back to "n/a".
    """
    safe_context = {**_signal_context_defaults(), **context}
    prompt = signal_explanation.build_prompt(safe_context)
    result = generate_safe(prompt, system=signal_explanation.SYSTEM_PROMPT)
    if result is not None:
        return result
    return _fallback_signal_explanation(safe_context)


def _signal_context_defaults() -> dict:
    return {
        "symbol": "BTC/USDT", "price": "n/a", "signal": "WAIT", "confidence": 0.0,
        "risk_score": 0.0, "rsi": "n/a", "ema_trend": "n/a", "macd_histogram": "n/a",
        "volatility": "n/a", "current_position": "none", "recent_performance": "n/a",
    }


def _fallback_signal_explanation(context: dict) -> str:
    """Used when the LLM is disabled or unreachable — still useful, just less fluent."""
    return (
        f"Signal: {context['signal']} (confidence {context['confidence']:.0%}, "
        f"risk score {context['risk_score']:.2f}). "
        f"[Local LLM unavailable — this is a template summary, not an AI-generated "
        f"explanation. Enable an LLM in .env for plain-language reasoning.]"
    )


def summarize_daily_report(context: dict) -> str:
    safe_context = {**_daily_report_context_defaults(), **context}
    prompt = daily_report.build_prompt(safe_context)
    result = generate_safe(prompt, system=daily_report.SYSTEM_PROMPT)
    if result is not None:
        return result
    return (
        "[Local LLM unavailable — see the structured numbers above for "
        "today's summary. Enable an LLM in .env for a plain-language recap.]"
    )


def _daily_report_context_defaults() -> dict:
    return {
        "date": "n/a", "mode": "RESEARCH", "btc_price": "n/a", "n_buy": 0, "n_sell": 0, "n_wait": 0,
        "accuracy": "n/a", "starting_balance": "n/a", "current_equity": "n/a", "daily_pnl": "n/a",
        "total_pnl": "n/a", "max_drawdown": "n/a", "current_exposure": "n/a",
        "champion_model": "none", "candidate_model": "none", "drift_status": "n/a",
    }
