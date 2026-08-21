"""Prompt template for the daily-report LLM summary (Section 18/45)."""

SYSTEM_PROMPT = (
    "You summarize one day of an experimental automated crypto research "
    "system's activity for a beginner. Be factual and concise (4-6 "
    "sentences). Never claim guaranteed profit. If performance looks "
    "unusually good or the sample size is small, say so plainly."
)

USER_TEMPLATE = """\
Date: {date}
Mode: {mode}

BTC price: {btc_price}
Predictions today: BUY={n_buy}, SELL={n_sell}, WAIT={n_wait}
Prediction accuracy (evaluated so far): {accuracy}

Paper trading:
  Starting balance: {starting_balance}
  Current equity: {current_equity}
  Today's P/L: {daily_pnl}
  Total P/L: {total_pnl}

Risk:
  Max drawdown: {max_drawdown}
  Current exposure: {current_exposure}

Model: champion={champion_model}, candidate={candidate_model}
Drift status: {drift_status}

Summarize today's activity for the user in plain language.
"""


def build_prompt(context: dict) -> str:
    return USER_TEMPLATE.format(**context)
