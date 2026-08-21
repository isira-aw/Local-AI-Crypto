"""
Structured signal engine (Section 19).

What is it?
    Builds the single structured object that everything downstream
    (dashboard, LLM explainer, paper trading, risk engine) consumes.
    Internally, models/labels use BUY/HOLD/SELL; the externally facing
    signal uses BUY/SELL/WAIT (WAIT == "don't act right now", matching
    the spec's example JSON) — never parsed from free-form LLM text.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

_CLASS_TO_SIGNAL = {"BUY": "BUY", "SELL": "SELL", "HOLD": "WAIT"}


@dataclass
class Signal:
    symbol: str
    signal: str  # BUY | SELL | WAIT
    confidence: float
    risk_score: float
    model_version: str
    strategy_version: str
    timestamp: dt.datetime
    reason_codes: list[str] = field(default_factory=list)
    llm_explanation: str | None = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "signal": self.signal,
            "confidence": round(self.confidence, 4),
            "risk_score": round(self.risk_score, 4),
            "model_version": self.model_version,
            "strategy_version": self.strategy_version,
            "timestamp": self.timestamp.isoformat(),
            "reason_codes": self.reason_codes,
            "llm_explanation": self.llm_explanation,
        }


def build_signal(
    symbol: str,
    timestamp: dt.datetime,
    predicted_class: str,
    confidence: float,
    risk_score: float,
    model_version: str,
    strategy_version: str,
    trading_signal_override: str | None = None,
    reason_codes: list[str] | None = None,
) -> Signal:
    """
    predicted_class: raw model output, "BUY"/"HOLD"/"SELL".
    trading_signal_override: if the risk engine vetoes a trade (e.g.
        daily loss limit hit), pass "WAIT" here to force the final
        signal regardless of what the model said, and add a reason
        code explaining why.
    """
    signal_value = trading_signal_override or _CLASS_TO_SIGNAL.get(predicted_class, "WAIT")
    codes = list(reason_codes or [])
    if trading_signal_override and trading_signal_override != _CLASS_TO_SIGNAL.get(predicted_class, "WAIT"):
        codes.append("risk_override")

    return Signal(
        symbol=symbol,
        signal=signal_value,
        confidence=confidence,
        risk_score=risk_score,
        model_version=model_version,
        strategy_version=strategy_version,
        timestamp=timestamp,
        reason_codes=codes,
    )
