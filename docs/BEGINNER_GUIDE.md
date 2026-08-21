# Beginner Guide — What This Project Actually Is

If you read nothing else, read this page.

## The one-sentence version

An automated local research system that collects Bitcoin price data, trains
and honestly evaluates machine-learning models, backtests and paper-trades
strategies, uses a local AI model to explain what it's doing in plain
English, and — only after weeks of evidence and your explicit, repeated
approval — could eventually be allowed to trade a small, capped amount of
real money.

## What this project is **not**

- It is **not** "an AI that predicts Bitcoin and makes you money." Nobody can
  promise that, and anyone who does is not being honest with you.
- It is **not** going to trade real money by default. Real trading starts
  **disabled** and stays disabled until you pass a long checklist and
  explicitly turn it on (`docs/REAL_TRADING.md`).
- It is **not** financial advice.

## The safety progression

Every strategy this system produces walks through these stages, in order,
before it's allowed anywhere near real money:

```
DEVELOPMENT           (writing/testing the code itself)
    v
HISTORICAL DATA       (download years of real BTC/USDT prices)
    v
AUTOMATIC TRAINING    (walk-forward validated ML models)
    v
BACKTESTING           (simulate the strategy on historical data)
    v
LIVE OBSERVATION      (watch the model's live predictions, no trading)
    v
PAPER TRADING         (simulate trading with fake money on live prices)
    v
PERFORMANCE VALIDATION (weeks of proof, honestly measured)
    v
TESTNET / DEMO        (exchange's practice-money sandbox, needs a regulatory check)
    v
STRICT SAFETY CHECKS  (Section 59 checklist — see docs/REAL_TRADING.md)
    v
OPTIONAL, VERY SMALL REAL TRADING (capped, reversible, never automatic)
```

Nothing skips a stage. A model doing well in backtesting does not
automatically get to paper trade with different rules; passing paper trading
does not automatically enable real trading. Every arrow in that diagram is a
deliberate decision, and most of them require you personally to act (run a
command, edit a config value, or click a dashboard button).

## The three kinds of "success" — don't confuse them

1. **Software success**: does the system run reliably, collect data, train
   models, and recover from things like internet drops without you having to
   babysit it? This is what most of the code in this repo is about.
2. **Prediction success**: does the model's prediction actually correlate
   with what happens next, more than you'd expect by chance — even after
   accounting for the fact that many models/settings were tried (see
   "multiple-testing correction" in `docs/MODEL_TRAINING.md`)?
3. **Financial success**: after real trading costs (fees, slippage), does
   the strategy make money on a risk-adjusted basis, compared to just buying
   and holding BTC?

A system can have (1) without (2) or (3). Never assume good software means
good predictions, and never assume good predictions mean good financial
results after costs. The dashboard and reports keep these separate on
purpose.

## What you'll actually do, week by week

**Week 0 (setup):** Install dependencies, start the database, run
`python run.py setup`, then `python run.py download-data`. No account, no
API key, no money involved.

**Week 0-1 (first model):** `python run.py train` — automatically builds
features, labels, and walk-forward-validated models from the data you
downloaded. `python run.py backtest` to see how the best model would have
performed historically.

**Ongoing (paper trading):** Run `python run.py start` to bring up the
scheduler + dashboard. The system will keep making predictions and
paper-trading them automatically. Check the dashboard whenever you like.

**After weeks of paper trading:** Read the daily/weekly reports. If — and
only if — performance looks genuinely good (not just "the number went up
once"), you can start thinking about testnet. That's a much longer
conversation covered in `docs/REAL_TRADING.md` and `docs/REGULATORY_NOTES.md`.

## Where to go next

- Never touched crypto before? `docs/CRYPTO_BASICS.md`
- Ready to install? `docs/INSTALLATION.md`
- Curious how data becomes a prediction? `docs/DATA_PIPELINE.md` then
  `docs/MODEL_TRAINING.md`
- Something broke? `docs/TROUBLESHOOTING.md`
