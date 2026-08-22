"""
Multiple-testing correction tests (Section 15).

The correction existed as `deflated_sharpe_ratio()` in metrics.py but had
ZERO callers, so no promotion decision was ever corrected for the fact that
three algorithms were tried. These tests pin down both the maths and — more
importantly — that the promotion path actually enforces it.
"""

import datetime as dt

import numpy as np
import pytest

from crypto_ai.models.evaluation.multiple_testing import (
    DEFAULT_MIN_DSR,
    collect_trial_sharpes,
    count_trials,
    deflated_sharpe,
    expected_max_sharpe,
)

START = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)


# ---------------------------------------------------------------------
# The maths
# ---------------------------------------------------------------------
def test_one_trial_needs_no_correction():
    assert expected_max_sharpe(n_trials=1, sigma_sharpe=0.05) == 0.0
    r = deflated_sharpe(2.0, n_trials=1, n_observations=4000)
    assert "no multiple-testing correction needed" in r.reason


def test_expected_max_grows_with_more_trials():
    """Trying more things raises the bar a result must clear."""
    e3 = expected_max_sharpe(n_trials=3, sigma_sharpe=0.05)
    e50 = expected_max_sharpe(n_trials=50, sigma_sharpe=0.05)
    e500 = expected_max_sharpe(n_trials=500, sigma_sharpe=0.05)
    assert 0 < e3 < e50 < e500


def test_same_sharpe_deflates_further_as_trials_increase():
    kw = dict(n_observations=4000, timeframe="5m", trial_sharpes_annual=[1.0, 0.8, 0.9])
    few = deflated_sharpe(1.0, n_trials=3, **kw).deflated_sharpe_probability
    many = deflated_sharpe(1.0, n_trials=200, **kw).deflated_sharpe_probability
    assert many < few, "more trials must reduce confidence in the same Sharpe"


def test_annualized_sharpe_is_deannualized_before_testing():
    """
    The standard error formula is for a per-period Sharpe. Treating an
    annualized value as per-period inflates DSR toward 1.0 and would
    silently defeat the check.
    """
    r = deflated_sharpe(2.0, n_trials=3, n_observations=4000, timeframe="5m")
    periods = 365 * 24 * 12
    assert r.observed_sharpe_per_period == pytest.approx(2.0 / np.sqrt(periods), rel=1e-6)
    assert r.observed_sharpe_per_period < r.observed_sharpe_annual


def test_a_wildly_high_sharpe_still_passes():
    """The correction must not reject everything — a real edge survives."""
    r = deflated_sharpe(
        observed_sharpe_annual=12.0, n_trials=3, n_observations=8000,
        timeframe="5m", trial_sharpes_annual=[12.0, 0.1, -0.2],
    )
    assert r.passed is True, r.reason


def test_a_marginal_sharpe_from_many_trials_fails():
    r = deflated_sharpe(
        observed_sharpe_annual=0.4, n_trials=60, n_observations=4000,
        timeframe="5m", trial_sharpes_annual=[0.4, 0.35, -0.1, 0.2],
    )
    assert r.passed is False
    assert "does NOT survive correction" in r.reason


def test_too_few_observations_fails_closed():
    r = deflated_sharpe(5.0, n_trials=3, n_observations=1)
    assert r.passed is False
    assert "too few observations" in r.reason


# ---------------------------------------------------------------------
# Trial counting
# ---------------------------------------------------------------------
def test_count_trials_includes_prior_variants():
    results = [{"algorithm": "a"}, {"algorithm": "b"}, {"algorithm": "c"}]
    assert count_trials(results) == 3
    # Twenty models trained over previous weeks are part of the surface.
    assert count_trials(results, prior_variants=17) == 20


def test_count_trials_never_below_one():
    assert count_trials([]) == 1


def test_collect_trial_sharpes_falls_back_to_fold_mean_for_rejected():
    results = [
        {"algorithm": "passed", "final_test_metrics": {"backtest": {"sharpe_ratio": 1.5}}},
        {
            "algorithm": "rejected",
            "walk_forward": {"folds": [
                {"backtest_metrics": {"sharpe_ratio": -1.0}},
                {"backtest_metrics": {"sharpe_ratio": 1.0}},
            ]},
        },
    ]
    sharpes = collect_trial_sharpes(results)
    assert sharpes[0] == 1.5
    assert sharpes[1] == pytest.approx(0.0)  # mean of -1 and 1


# ---------------------------------------------------------------------
# The promotion path actually enforces it
# ---------------------------------------------------------------------
def _candidate(label, sharpe):
    return {
        "algorithm": "random_forest", "overall_pass": True, "version_label": label,
        "final_test_metrics": {"backtest": {"sharpe_ratio": sharpe}},
    }


def test_promotion_is_blocked_when_correction_fails(db_session, monkeypatch):
    """
    A candidate whose raw Sharpe would win must still be refused if it
    cannot survive the correction for how many variants were tried.
    """
    from crypto_ai.models.training.pipeline import maybe_promote_champion

    result = maybe_promote_champion(
        db_session, "btc_direction_classifier", [_candidate("model_001", 0.5)],
        n_trials=80, trial_sharpes=[0.5, 0.4, -0.3, 0.1], timeframe="5m",
    )
    assert result["promoted"] is False
    assert "does NOT survive correction" in result["reason"]
    assert result["multiple_testing"]["n_trials"] == 80


def test_blocked_promotion_is_logged_for_audit(db_session):
    from crypto_ai.database.models.monitoring import SystemEvent
    from crypto_ai.models.training.pipeline import maybe_promote_champion

    maybe_promote_champion(
        db_session, "btc_direction_classifier", [_candidate("model_001", 0.5)],
        n_trials=80, trial_sharpes=[0.5, 0.4, -0.3], timeframe="5m",
    )
    db_session.commit()
    events = db_session.query(SystemEvent).filter_by(
        event="promotion_blocked_by_multiple_testing"
    ).all()
    assert len(events) == 1
    assert "DSR" in events[0].message


def test_no_candidates_still_reports_cleanly(db_session):
    from crypto_ai.models.training.pipeline import maybe_promote_champion

    result = maybe_promote_champion(db_session, "btc_direction_classifier", [], n_trials=3)
    assert result["promoted"] is False
    assert "no candidates" in result["reason"]


def test_strong_candidate_survives_and_is_promoted(db_session):
    """
    End to end: a genuinely strong result clears the correction, gets
    promoted, and has the correction recorded on the version.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from crypto_ai.models.registry import registry
    from crypto_ai.models.training.pipeline import maybe_promote_champion

    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    model.fit(np.random.rand(20, 3), ["BUY"] * 10 + ["SELL"] * 10)
    registry.register_model_version(
        db_session, "btc_direction_classifier", "BTC/USDT", "5m", "random_forest",
        feature_version="features_v1", strategy_version="s1", estimator=model,
        hyperparameters={}, metrics={}, walk_forward_results={},
        status=registry.STATUS_CANDIDATE,
    )
    db_session.commit()

    result = maybe_promote_champion(
        db_session, "btc_direction_classifier", [_candidate("model_001", 15.0)],
        n_trials=3, trial_sharpes=[15.0, 0.2, -0.1], timeframe="5m",
    )
    assert result["promoted"] is True, result
    assert result["multiple_testing"]["passed"] is True

    champion = registry.get_champion(db_session, "btc_direction_classifier")
    assert champion is not None
    assert "multiple_testing" in champion.metrics, "correction must be recorded on the version"
