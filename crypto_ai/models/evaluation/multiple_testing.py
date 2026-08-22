"""
Multiple-testing correction for model promotion (Section 15).

The problem
-----------
This pipeline trains three algorithms and picks the best. Report only the
winner's Sharpe ratio and that number is biased upward purely because
several variants were tried — the maximum of N noisy estimates exceeds the
true value even when every variant is worthless. The more you try, the
better your "best" looks, with no additional skill involved.

Section 15 requires correcting for this. `backtesting/metrics.py` had a
`deflated_sharpe_ratio()` implementation but NOTHING CALLED IT, so no
correction was actually being applied to any promotion decision. This
module closes that gap and is invoked from the promotion path.

The correction
--------------
Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014). Under the null that
every trial has zero true Sharpe, the expected maximum observed Sharpe
across N trials is

    E[max SR] ~= sigma_SR * [ (1-g) * Z(1 - 1/N) + g * Z(1 - 1/(N*e)) ]

where g is the Euler-Mascheroni constant, Z is the inverse normal CDF, and
sigma_SR is the spread of Sharpe estimates across trials. The DSR is then
the probability that the observed Sharpe genuinely exceeds that benchmark:

    DSR = Phi( (SR - E[max SR]) / sigma(SR_hat) )

    sigma(SR_hat) = sqrt( (1 - skew*SR + (kurtosis-1)/4 * SR^2) / (n - 1) )

Units matter. That standard error is defined for a PER-PERIOD Sharpe
ratio, while everything else in this project reports an annualized one, so
the annualized value is converted back before the test and the result is
reported in both units. Getting this wrong inflates DSR toward 1.0 and
would quietly defeat the purpose of the check.

Reading the result: DSR is a probability. Below ~0.95, the observed Sharpe
is not distinguishable from what N random trials would produce anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from crypto_ai.backtesting.metrics import TRADING_PERIODS_PER_YEAR, _inv_norm_cdf, _norm_cdf

EULER_MASCHERONI = 0.5772156649

# A candidate must clear this probability to be promoted. 0.95 is the
# conventional bar for "not plausibly explained by having tried N things".
DEFAULT_MIN_DSR = 0.95

# Minimum number of trials before the cross-sectional standard deviation of
# their Sharpe ratios is a usable estimate of sigma_SR.
#
# With very few trials that estimate is dominated by the winner itself,
# which is circular: a strong result inflates sigma_SR, which raises the
# expected-maximum bar, which then penalises that same strong result. With
# three trials of [15.0, 0.2, -0.1] annual Sharpe, the winner alone drives
# sigma_SR and deflates its own DSR from 0.98 to 0.93. Below this count we
# fall back to the standard assumption that trials differ only by
# estimation noise (sigma_SR = 1/sqrt(n)).
MIN_TRIALS_FOR_EMPIRICAL_SIGMA = 4


@dataclass
class MultipleTestingResult:
    observed_sharpe_annual: float
    observed_sharpe_per_period: float
    n_trials: int
    n_observations: int
    expected_max_sharpe_per_period: float
    deflated_sharpe_probability: float
    threshold: float
    passed: bool
    reason: str = ""
    trial_sharpes: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "observed_sharpe_annual": round(self.observed_sharpe_annual, 4),
            "observed_sharpe_per_period": round(self.observed_sharpe_per_period, 6),
            "n_trials": self.n_trials,
            "n_observations": self.n_observations,
            "expected_max_sharpe_per_period": round(self.expected_max_sharpe_per_period, 6),
            "deflated_sharpe_probability": round(self.deflated_sharpe_probability, 4),
            "threshold": self.threshold,
            "passed": self.passed,
            "reason": self.reason,
        }


def expected_max_sharpe(n_trials: int, sigma_sharpe: float) -> float:
    """Expected maximum per-period Sharpe across `n_trials` worthless trials."""
    if n_trials <= 1:
        return 0.0  # nothing to correct for when only one thing was tried
    z1 = _inv_norm_cdf(1 - 1 / n_trials)
    z2 = _inv_norm_cdf(1 - 1 / (n_trials * math.e))
    return sigma_sharpe * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def deflated_sharpe(
    observed_sharpe_annual: float,
    n_trials: int,
    n_observations: int,
    timeframe: str = "5m",
    trial_sharpes_annual: list[float] | None = None,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    threshold: float = DEFAULT_MIN_DSR,
) -> MultipleTestingResult:
    """
    `trial_sharpes_annual`, when supplied, gives the ACTUAL spread of Sharpe
    estimates across the variants tried, which is a far better estimate of
    sigma_SR than assuming pure estimation noise.
    """
    periods_per_year = TRADING_PERIODS_PER_YEAR.get(timeframe, 365)
    ann = math.sqrt(periods_per_year)

    sr = observed_sharpe_annual / ann  # per-period
    trials = [s / ann for s in (trial_sharpes_annual or [])]

    if n_observations < 2:
        return MultipleTestingResult(
            observed_sharpe_annual, sr, n_trials, n_observations, 0.0, 0.0, threshold,
            passed=False, reason="too few observations to test", trial_sharpes=trials,
        )

    noise_sigma = 1.0 / math.sqrt(n_observations)
    if len(trials) >= MIN_TRIALS_FOR_EMPIRICAL_SIGMA:
        sigma_sr = float(np.std(trials, ddof=1)) or noise_sigma
    else:
        sigma_sr = noise_sigma

    e_max = expected_max_sharpe(n_trials, sigma_sr)

    variance = (1 - skew * sr + (kurtosis - 1) / 4 * sr**2) / (n_observations - 1)
    sigma_hat = math.sqrt(max(variance, 1e-12))

    dsr = float(_norm_cdf((sr - e_max) / sigma_hat))
    passed = dsr >= threshold

    if n_trials <= 1:
        reason = "only one variant tried — no multiple-testing correction needed"
    elif passed:
        reason = (
            f"survives correction for {n_trials} variants "
            f"(DSR {dsr:.3f} >= {threshold})"
        )
    else:
        reason = (
            f"does NOT survive correction for {n_trials} variants tried: "
            f"DSR {dsr:.3f} < {threshold}. The observed Sharpe "
            f"({observed_sharpe_annual:.2f} annual) is not distinguishable from "
            f"the best of {n_trials} worthless trials."
        )

    return MultipleTestingResult(
        observed_sharpe_annual=observed_sharpe_annual,
        observed_sharpe_per_period=sr,
        n_trials=n_trials,
        n_observations=n_observations,
        expected_max_sharpe_per_period=e_max,
        deflated_sharpe_probability=dsr,
        threshold=threshold,
        passed=passed,
        reason=reason,
        trial_sharpes=trials,
    )


def count_trials(training_results: list[dict], prior_variants: int = 0) -> int:
    """
    How many variants were effectively tried.

    Counts every algorithm evaluated in this run, plus any variants
    previously registered for the same model. That cumulative view is the
    honest one: if you have trained twenty models over six weeks and are
    promoting the best, twenty is the multiple-testing surface, not three.
    """
    return max(1, len(training_results) + prior_variants)


def collect_trial_sharpes(training_results: list[dict]) -> list[float]:
    """Final-test Sharpe of every algorithm evaluated in this run."""
    out = []
    for r in training_results:
        sharpe = (
            r.get("final_test_metrics", {}).get("backtest", {}).get("sharpe_ratio")
        )
        if sharpe is None:
            # A rejected algorithm has no final-test metrics; fall back to the
            # mean of its fold Sharpes so its attempt still counts as a trial.
            folds = r.get("walk_forward", {}).get("folds", [])
            fold_sharpes = [
                f.get("backtest_metrics", {}).get("sharpe_ratio")
                for f in folds
                if f.get("backtest_metrics", {}).get("sharpe_ratio") is not None
            ]
            sharpe = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0
        out.append(float(sharpe))
    return out
