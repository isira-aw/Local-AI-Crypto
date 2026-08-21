"""
Risk manager (Phase 13 / Section 20).

What is it?
    The single gate every proposed trading signal passes through
    before it is allowed to open or close a position — in PAPER,
    TESTNET, and LIVE mode alike, so paper trading is a faithful
    preview of how real trading would actually behave.

Why is it needed?
    A model can be confident and still be wrong. This is what enforces
    the limits that don't depend on the model being right: max
    position size, daily loss limit, drawdown limit, a cooldown after
    consecutive losses, and the global emergency stop / trading-enabled
    switches.

How does it work?
    evaluate() takes the model's raw proposal (signal + confidence)
    plus the account's current risk state (today's P/L, drawdown,
    consecutive losses, open positions) and returns a RiskDecision:
    either the proposal is allowed through as-is, or it's overridden
    to WAIT with reason codes explaining why.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from crypto_ai.config.loader import get_risk_config, get_settings
from crypto_ai.risk.emergency import is_emergency_stop_active
from crypto_ai.risk.position_sizing import calculate_position_size_pct


@dataclass
class RiskState:
    """Current account state the risk manager needs to make a decision."""

    current_equity: float
    starting_equity_today: float
    peak_equity: float
    consecutive_losses: int = 0
    open_positions: int = 0
    last_loss_at: "object" = None  # datetime | None; kept loosely typed to avoid import cycles
    now: "object" = None  # datetime | None


@dataclass
class RiskDecision:
    allowed_signal: str  # BUY | SELL | WAIT
    position_size_pct: float
    risk_score: float
    reason_codes: list[str] = field(default_factory=list)
    blocked: bool = False


class RiskManager:
    def __init__(self, risk_config: dict | None = None):
        self.cfg = risk_config or get_risk_config()

    def _daily_loss_pct(self, state: RiskState) -> float:
        if state.starting_equity_today <= 0:
            return 0.0
        return max(0.0, (state.starting_equity_today - state.current_equity) / state.starting_equity_today * 100)

    def _drawdown_pct(self, state: RiskState) -> float:
        if state.peak_equity <= 0:
            return 0.0
        return max(0.0, (state.peak_equity - state.current_equity) / state.peak_equity * 100)

    def _in_cooldown(self, state: RiskState) -> bool:
        limit = self.cfg.get("loss_limits", {}).get("consecutive_loss_limit", 3)
        if state.consecutive_losses < limit:
            return False
        cooldown_minutes = self.cfg.get("loss_limits", {}).get("cooldown_minutes", 120)
        if state.last_loss_at is None or state.now is None:
            return True  # unknown timing -> fail safe, stay in cooldown
        elapsed = (state.now - state.last_loss_at).total_seconds() / 60
        return elapsed < cooldown_minutes

    def compute_risk_score(self, state: RiskState, volatility_pct: float) -> float:
        """0 (low risk) to 1 (high risk) — surfaced on the Signal (Section 19)."""
        drawdown_component = min(1.0, self._drawdown_pct(state) / max(1.0, self.cfg.get("loss_limits", {}).get("max_total_drawdown_percent", 20.0)))
        daily_loss_component = min(1.0, self._daily_loss_pct(state) / max(1.0, self.cfg.get("loss_limits", {}).get("max_daily_loss_percent", 2.0)))
        volatility_component = min(1.0, volatility_pct / 5.0)  # 5%+ recent volatility treated as "high"
        loss_streak_component = min(1.0, state.consecutive_losses / max(1, self.cfg.get("loss_limits", {}).get("consecutive_loss_limit", 3)))
        return round((drawdown_component + daily_loss_component + volatility_component + loss_streak_component) / 4, 4)

    def evaluate(
        self,
        proposed_signal: str,
        confidence: float,
        state: RiskState,
        volatility_pct: float = 1.0,
    ) -> RiskDecision:
        settings = get_settings()
        reasons: list[str] = []
        signal = proposed_signal

        # --- Hard stops: checked first and override everything else. ---
        if self.cfg.get("safety", {}).get("emergency_stop", False) or is_emergency_stop_active():
            return RiskDecision("WAIT", 0.0, 1.0, ["emergency_stop_active"], blocked=True)

        if signal == "BUY":
            if not settings.safety.trading_enabled and settings.mode in ("TESTNET", "LIVE"):
                return RiskDecision("WAIT", 0.0, self.compute_risk_score(state, volatility_pct), ["trading_disabled"], blocked=True)

            max_open = self.cfg.get("exposure", {}).get("max_open_positions", 1)
            if state.open_positions >= max_open:
                reasons.append("max_open_positions_reached")
                signal = "WAIT"

            if self._daily_loss_pct(state) >= self.cfg.get("loss_limits", {}).get("max_daily_loss_percent", 2.0):
                reasons.append("daily_loss_limit_reached")
                signal = "WAIT"

            if self._drawdown_pct(state) >= self.cfg.get("loss_limits", {}).get("max_total_drawdown_percent", 20.0):
                reasons.append("max_drawdown_reached")
                signal = "WAIT"

            if self._in_cooldown(state):
                reasons.append("cooldown_after_consecutive_losses")
                signal = "WAIT"

            min_confidence = self.cfg.get("position_sizing", {}).get("min_confidence_to_trade", 0.55)
            if confidence < min_confidence:
                reasons.append("confidence_below_minimum")
                signal = "WAIT"

        risk_score = self.compute_risk_score(state, volatility_pct)

        if signal != "BUY":
            position_size_pct = 0.0
        else:
            position_size_pct = calculate_position_size_pct(
                confidence=confidence,
                volatility_pct=volatility_pct,
                max_position_percent=self.cfg.get("position_sizing", {}).get("max_position_percent", 5.0),
                min_confidence_to_trade=self.cfg.get("position_sizing", {}).get("min_confidence_to_trade", 0.55),
            )

        return RiskDecision(
            allowed_signal=signal,
            position_size_pct=position_size_pct,
            risk_score=risk_score,
            reason_codes=reasons,
            blocked=bool(reasons),
        )
