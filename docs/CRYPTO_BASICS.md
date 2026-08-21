# Crypto Basics (Start Here If You're New)

This project assumes you can program, but does **not** assume you know anything
about cryptocurrency trading. This page explains the vocabulary used everywhere
else in the docs.

## What is cryptocurrency?

A cryptocurrency is a digital asset that exists as entries in a shared,
public ledger (a "blockchain") instead of in a bank's private database.
Nobody needs a bank's permission to send or receive it. Its price is set by
supply and demand on exchanges, 24/7 — there's no closing bell.

## What is Bitcoin (BTC)?

The first and largest cryptocurrency by value. Think of it as "digital gold":
scarce (only 21 million will ever exist), decentralized, and widely traded.
This project focuses on BTC because it's the most liquid (easiest to buy/sell
without moving the price) and best-documented crypto asset.

## What is USDT (Tether)?

A "stablecoin" — a cryptocurrency designed to always be worth $1 USD. Traders
use USDT to move value between crypto assets without cashing out to a bank
account each time. When this project says "your paper balance is 1000 USDT,"
that's roughly the same as saying "$1000."

## What does "BTC/USDT" mean?

A **trading pair**. "BTC/USDT" means "how much USDT does one BTC cost."
If BTC/USDT = 65,000, one Bitcoin costs 65,000 USDT (~$65,000).

## What is Spot trading?

The simplest kind of trading: you own the actual asset. Buy BTC with USDT →
you now hold real BTC. Sell it later → you get USDT back (hopefully more than
you started with, but not guaranteed). This is different from:

- **Margin trading**: borrowing money to trade with more than you have.
- **Futures/leverage**: betting on price direction with borrowed exposure,
  which can lose you more than your initial deposit.
- **Short selling**: profiting when price goes *down*.

This project **only** does spot trading, and never uses leverage. That's a
deliberate, permanent safety choice for the "small real money" phase — see
Section 5 of the design and `docs/REAL_TRADING.md`.

## What is Binance?

The exchange (a website/API where buyers and sellers meet) this project
connects to. It's one of the largest crypto exchanges by trading volume,
which matters because more volume = prices that are harder for one trader to
distort = a fairer testing ground for a strategy.

## What is an API, and what is an API key?

An API (Application Programming Interface) is how a program — not a human
clicking buttons — talks to an exchange. **Public** API endpoints (like "what
is the current price of BTC") need no credentials; this project uses them
for all data collection, training, backtesting, and paper trading.

**Private** endpoints (like "buy 0.01 BTC with my account") need an **API
key** — a password-like credential proving the request comes from your
account. An API key is normally paired with an **API secret** (used to sign
requests cryptographically).

### Why must an API key be protected?

Anyone who has your API key + secret can do anything that key is allowed to
do — potentially trade or withdraw funds. Treat it like a password:

- Never put it in code that gets committed to Git.
- Only grant it the permissions it needs (trading — never withdrawals).
- Keep testnet and live keys separate.

See `docs/REAL_TRADING.md` and `docs/INSTALLATION.md` for exactly how this
project stores and protects keys.

## What is paper trading?

"Trading" with fake money against real (or replayed) prices. You get a
realistic preview of how a strategy behaves — including realistic fees and
slippage — without risking anything. See `docs/PAPER_TRADING.md`.

## What is backtesting?

Simulating a strategy against *historical* data to see how it would have
performed. Backtesting is faster than paper trading (you can test years of
data in seconds) but is more prone to fooling you (see "overfitting" below).
See `docs/BACKTESTING.md`.

## What does the ML model do?

It looks at recent price/volume patterns (encoded as "features" — see
`docs/DATA_PIPELINE.md`) and predicts whether the price is likely to go up,
down, or stay flat over the next hour (or whatever horizon is configured).
It is a **statistical guess**, not a crystal ball.

## What does the local LLM do?

Nothing to do with the actual trading decision. It takes numbers that are
*already computed* (the model's prediction, current risk level, recent
performance) and explains them in plain English, e.g. "the model sees weak
upward momentum but volatility is high, so no trade is being made." See
`docs/MODEL_TRAINING.md` and Section 18 of the design document.

## Why isn't profit guaranteed?

Because nothing in trading is. Markets change constantly; a pattern that
worked in the training data may stop working. This project is built to be
honest about that: it reports many different metrics (not just "accuracy"),
flags suspiciously good results, and requires weeks of paper-trading proof
before it will even consider letting you enable real trading with a small,
capped amount of money. Read that as a feature, not a limitation — a system
that promises guaranteed profit is a red flag, not a good sign.

## Why check local regulations before enabling testnet/live trading?

Whether personal crypto trading is legal, restricted, or requires
registration/reporting varies by country and changes over time. Whether
Binance itself is available in your country also varies and changes. This is
**your** responsibility to check before you connect a real (or even testnet)
account — see `docs/REGULATORY_NOTES.md`. None of this blocks research,
backtesting, or paper trading, which use no account and no real money.

## Next steps

- New to the project? Continue to `docs/INSTALLATION.md`.
- Want the big picture first? Read `docs/BEGINNER_GUIDE.md`.
