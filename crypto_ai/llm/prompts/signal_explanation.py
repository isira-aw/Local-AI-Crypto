"""
Prompt template for explaining a single trading signal (Section 18).

The LLM only ever receives numbers that have ALREADY been computed by
the ML model, feature pipeline and risk engine — it never sees raw
account/API data, and it never decides the signal itself (Section 19:
"do not parse critical trading decisions from free-form LLM text").
"""

SYSTEM_PROMPT = (
    "You are a plain-language trading assistant for a beginner investor. "
    "You explain WHY an automated system produced a signal, using only the "
    "numbers you are given. You never invent numbers. You never tell the "
    "user to buy or sell — the system already decided that; you only "
    "explain the reasoning. Keep it to 2-4 short sentences. Never claim "
    "guaranteed profit or say something is risk-free."
)

USER_TEMPLATE = """\
Symbol: {symbol}
Current price: {price}
Signal: {signal}
Model confidence: {confidence:.0%}
Risk score (0=low, 1=high): {risk_score:.2f}

Technical context:
  RSI (14): {rsi}
  EMA(20) vs price: {ema_trend}
  MACD histogram: {macd_histogram}
  Recent volatility: {volatility}

Current position: {current_position}
Recent performance: {recent_performance}

Explain in plain language why the signal is "{signal}" given the numbers
above. Do not use the words "guaranteed" or "risk-free".
"""


def build_prompt(context: dict) -> str:
    return USER_TEMPLATE.format(**context)
