"""
Weekly / Month-1 research report tests (Section 22).

The recommendation logic is the important part: Section 22 says the system
must "never automatically recommend real trading merely because profit is
positive", so these tests pin that behaviour down.
"""

import datetime as dt

import pytest

from crypto_ai.app.services.research_report import (
    RECOMMEND_CONTINUE,
    RECOMMEND_NOT_READY,
    RECOMMEND_TESTNET_CANDIDATE,
    _recommendation,
    format_research_report,
    generate_research_report,
)

_GOOD_PAPER = {
    "n_trades": 150, "total_return_pct": 12.0, "max_drawdown_pct": 8.0,
    "sharpe_ratio": 1.1, "profit_factor": 1.6, "win_rate_pct": 55.0,
    "current_equity": 1120.0, "suspicious_flags": [],
}
_GOOD_PREDICTIONS = {"n_predictions": 500, "accuracy_pct": 58.0, "directional_accuracy_pct": 56.0}
_GOOD_STABILITY = {"champion_changes": 0, "assessment": "stable", "champion": "model_001"}
_GOOD_DATA = {"status": "OK", "bars": 5000, "summary": "OK"}
_GOOD_RISK = {"n_error_events": 0, "suspicious_flags": [], "max_drawdown_pct": 8.0}


def test_too_little_evidence_recommends_continue_research():
    rec, reasons = _recommendation(
        {"n_trades": 5, "total_return_pct": 900.0, "max_drawdown_pct": 0.1, "suspicious_flags": []},
        {"n_predictions": 10, "directional_accuracy_pct": 99.0},
        _GOOD_STABILITY, _GOOD_DATA, _GOOD_RISK,
    )
    assert rec == RECOMMEND_CONTINUE
    assert any("paper trades" in r for r in reasons)


def test_positive_profit_alone_does_not_recommend_testnet():
    """Section 22's central rule."""
    paper = {**_GOOD_PAPER, "max_drawdown_pct": 45.0}  # profitable but reckless
    rec, reasons = _recommendation(paper, _GOOD_PREDICTIONS, _GOOD_STABILITY, _GOOD_DATA,
                                   {**_GOOD_RISK, "max_drawdown_pct": 45.0})
    assert rec == RECOMMEND_NOT_READY
    assert any("drawdown" in r for r in reasons)


def test_coin_flip_directional_accuracy_blocks_testnet():
    rec, reasons = _recommendation(
        _GOOD_PAPER, {**_GOOD_PREDICTIONS, "directional_accuracy_pct": 49.0},
        _GOOD_STABILITY, _GOOD_DATA, _GOOD_RISK,
    )
    assert rec == RECOMMEND_NOT_READY
    assert any("coin flip" in r for r in reasons)


def test_suspicious_flags_block_testnet():
    rec, reasons = _recommendation(
        _GOOD_PAPER, _GOOD_PREDICTIONS, _GOOD_STABILITY, _GOOD_DATA,
        {**_GOOD_RISK, "suspicious_flags": ["Unrealistically high Sharpe ratio (9.1)."]},
    )
    assert rec == RECOMMEND_NOT_READY
    assert any("too-good-to-be-true" in r for r in reasons)


def test_critical_errors_block_testnet():
    rec, reasons = _recommendation(
        _GOOD_PAPER, _GOOD_PREDICTIONS, _GOOD_STABILITY, _GOOD_DATA,
        {**_GOOD_RISK, "n_error_events": 3},
    )
    assert rec == RECOMMEND_NOT_READY
    assert any("error/critical" in r for r in reasons)


def test_unstable_champion_blocks_testnet():
    rec, reasons = _recommendation(
        _GOOD_PAPER, _GOOD_PREDICTIONS, {"champion_changes": 5, "assessment": "unstable", "champion": "model_009"},
        _GOOD_DATA, _GOOD_RISK,
    )
    assert rec == RECOMMEND_NOT_READY
    assert any("not stable" in r for r in reasons)


def test_bad_data_quality_blocks_testnet():
    rec, reasons = _recommendation(
        _GOOD_PAPER, _GOOD_PREDICTIONS, _GOOD_STABILITY,
        {"status": "ISSUES", "summary": "ISSUES FOUND: 12 gaps", "bars": 100}, _GOOD_RISK,
    )
    assert rec == RECOMMEND_NOT_READY
    assert any("data quality" in r for r in reasons)


def test_everything_good_can_reach_testnet_candidate():
    rec, reasons = _recommendation(_GOOD_PAPER, _GOOD_PREDICTIONS, _GOOD_STABILITY, _GOOD_DATA, _GOOD_RISK)
    assert rec == RECOMMEND_TESTNET_CANDIDATE
    # Even then it points at the separate live-trading gate.
    assert any("live-trading gate" in r for r in reasons)


def test_report_generates_on_empty_database(db_session):
    report = generate_research_report(db_session, days=30)
    assert report["recommendation"] == RECOMMEND_CONTINUE
    for section in ("model_performance", "strategy_performance", "paper_trading_performance",
                    "risk_metrics", "model_stability", "data_quality"):
        assert section in report, f"missing Section 22 section: {section}"


def test_formatted_report_contains_all_section_headings(db_session):
    report = generate_research_report(db_session, days=30)
    text = format_research_report(report)
    for heading in ("MODEL PERFORMANCE", "STRATEGY PERFORMANCE", "PAPER TRADING PERFORMANCE",
                    "RISK METRICS", "MODEL STABILITY", "DATA QUALITY", "RECOMMENDATION"):
        assert heading in text, f"report text missing {heading}"
    assert "does not guarantee future results" in text
