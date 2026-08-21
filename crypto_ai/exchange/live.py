"""
Real order placement (Phase 17/18 — TESTNET and LIVE modes).

This module is intentionally a GATED STUB, not a working order-execution
client. Sections 4, 31 and 59 all require, before any real (or even
testnet) order is placed:

  - A regulatory/access check for the user's jurisdiction and the
    exchange's current availability there (Section 4) — this cannot be
    verified automatically by this codebase and must be done by the
    user, then explicitly acknowledged (see risk.yaml
    `live_trading_gate.require_regulatory_check_ack` and
    risk/safety_rules.py).
  - The full Section 59 checklist (walk-forward validation, sufficient
    paper trading, drawdown/risk limits, emergency stop tested, backup
    restore tested, withdrawal permission disabled on the API key,
    etc.) — see risk/safety_rules.py:evaluate_live_trading_gate().
  - MODE=TESTNET or MODE=LIVE AND REAL_TRADING_ENABLED=true, both set
    deliberately by the user (never automatically).

Implementing real order placement (ccxt private endpoints: create
order, cancel order, fetch open orders/positions, fetch balance) is
the next step once all of the above is in place for your specific
situation — see docs/REAL_TRADING.md for the full checklist and
implementation notes. Until then, every entry point here refuses.
"""

from __future__ import annotations

from crypto_ai.risk.safety_rules import LiveTradingGateResult


class LiveTradingNotAllowedError(Exception):
    """Raised whenever anything in exchange/live.py is called before the
    full safety gate (risk/safety_rules.py) passes."""


def place_order(*args, gate_result: LiveTradingGateResult | None = None, **kwargs):
    if gate_result is None or not gate_result.allowed:
        blocking = gate_result.blocking_reasons if gate_result else ["gate_not_evaluated"]
        raise LiveTradingNotAllowedError(
            f"Real order placement refused. Blocking reasons: {blocking}. "
            f"See docs/REAL_TRADING.md and risk/safety_rules.py."
        )
    # Deliberately unimplemented beyond this point — see module docstring.
    raise NotImplementedError(
        "Real order execution is not implemented in this repository. "
        "This is a safety boundary, not a bug: implement ccxt private-endpoint "
        "order placement here only after completing docs/REAL_TRADING.md."
    )


def get_open_orders(*args, **kwargs):
    raise NotImplementedError("See exchange/live.py module docstring.")


def get_account_positions(*args, **kwargs):
    raise NotImplementedError("See exchange/live.py module docstring.")
