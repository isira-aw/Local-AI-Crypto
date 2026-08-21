"""
Performance metrics for backtests, paper trading and prediction
evaluation (Sections 15/16/46).

These are pure functions over an equity curve and/or trade list so
they can be reused by the backtester, the paper-trading evaluator, and
the daily/weekly reports without duplicating logic.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_PERIODS_PER_YEAR = {
    "1m": 365 * 24 * 60,
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "1h": 365 * 24,
    "4h": 365 * 6,
    "1d": 365,
}


def periodic_returns(equity_curve: pd.Series) -> pd.Series:
    return equity_curve.pct_change().dropna()


def total_return_pct(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2 or equity_curve.iloc[0] == 0:
        return 0.0
    return (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100


def max_drawdown_pct(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max.replace(0, np.nan)
    return abs(float(drawdown.min(skipna=True) or 0.0)) * 100


def sharpe_ratio(equity_curve: pd.Series, timeframe: str = "5m", risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio computed from per-bar returns."""
    rets = periodic_returns(equity_curve)
    if len(rets) < 2 or rets.std(ddof=1) == 0:
        return 0.0
    periods_per_year = TRADING_PERIODS_PER_YEAR.get(timeframe, 365)
    excess = rets - risk_free_rate / periods_per_year
    return float(excess.mean() / rets.std(ddof=1) * math.sqrt(periods_per_year))


def sortino_ratio(equity_curve: pd.Series, timeframe: str = "5m", risk_free_rate: float = 0.0) -> float:
    rets = periodic_returns(equity_curve)
    if len(rets) < 2:
        return 0.0
    periods_per_year = TRADING_PERIODS_PER_YEAR.get(timeframe, 365)
    downside = rets[rets < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if not downside_std:
        return 0.0
    excess = rets - risk_free_rate / periods_per_year
    return float(excess.mean() / downside_std * math.sqrt(periods_per_year))


def win_rate_pct(trade_pnls: list[float]) -> float:
    if not trade_pnls:
        return 0.0
    wins = sum(1 for p in trade_pnls if p > 0)
    return wins / len(trade_pnls) * 100


def profit_factor(trade_pnls: list[float]) -> float:
    gross_win = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p < 0))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def average_win_loss(trade_pnls: list[float]) -> tuple[float, float]:
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return avg_win, avg_loss


def deflated_sharpe_ratio(observed_sharpe: float, n_trials: int, n_observations: int, skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """
    Approximate Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Why this exists (Section 15 / 47): if you test many model/feature/
    threshold variants and report the best Sharpe ratio, that number
    is inflated purely by how many things you tried — this discounts
    for that so "best of 50 variants" isn't mistaken for real edge.

    Returns the probability (0-1) that the observed Sharpe ratio is
    genuinely positive after correcting for the number of trials.
    Treat < 0.95 as "not statistically convincing" for anything you're
    about to promote to champion.
    """
    if n_trials <= 1 or n_observations <= 1 or observed_sharpe == 0:
        # No multiple-testing correction needed/possible; fall back to
        # a simple one-sample test against zero.
        n_trials = max(n_trials, 1)

    # Expected maximum Sharpe ratio under the null (all trials are
    # noise) via the Euler-Mascheroni approximation. With only one
    # trial there is nothing to correct for, so the expectation is 0.
    euler_mascheroni = 0.5772156649
    if n_trials > 1:
        expected_max_sr = (1 - euler_mascheroni) * _inv_norm_cdf(1 - 1 / n_trials) + \
            euler_mascheroni * _inv_norm_cdf(1 - 1 / (n_trials * math.e))
        # Variance of Sharpe ratio estimates across trials is unknown in
        # practice; assume unit variance of the SR estimator itself as a
        # conservative simplification widely used for this quick check.
        expected_max_sr = max(expected_max_sr, 0.0) / math.sqrt(n_observations)
    else:
        expected_max_sr = 0.0

    sr_std = math.sqrt(
        max(1e-9, (1 - skew * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe ** 2) / (n_observations - 1))
    )
    z = (observed_sharpe - expected_max_sr) / sr_std
    return float(_norm_cdf(z))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _inv_norm_cdf(p: float) -> float:
    """Acklam's approximation of the inverse normal CDF (no scipy dependency)."""
    if p <= 0:
        return -8.0
    if p >= 1:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def suspicious_result_flags(metrics: dict, n_trades: int, train_return_pct: float | None = None, test_return_pct: float | None = None) -> list[str]:
    """
    Too-good-to-be-true detector (Section 47). Returns a list of
    human-readable warnings; empty list means nothing looked obviously
    off (this is NOT a guarantee the result is real).
    """
    flags = []
    if n_trades < 20:
        flags.append(f"Too few trades ({n_trades}) to draw a reliable conclusion.")
    if metrics.get("sharpe_ratio", 0) > 5:
        flags.append(f"Unrealistically high Sharpe ratio ({metrics['sharpe_ratio']:.2f}).")
    if metrics.get("win_rate_pct", 0) > 90:
        flags.append(f"Very high win rate ({metrics['win_rate_pct']:.1f}%) — check for leakage.")
    if metrics.get("total_return_pct", 0) > 300:
        flags.append(f"Extreme backtest return ({metrics['total_return_pct']:.1f}%).")
    if train_return_pct is not None and test_return_pct is not None:
        if train_return_pct > 0 and test_return_pct < train_return_pct * 0.2:
            flags.append("Large train/test performance gap — possible overfitting.")
    return flags
