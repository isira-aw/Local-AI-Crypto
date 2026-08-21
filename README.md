# Local AI Crypto

A local, automated cryptocurrency **research and paper-trading** system for
BTC/USDT spot trading — with a strict, checklist-gated path toward
(optional, capped, never-automatic) small real trading.

> **This is not "an AI that predicts Bitcoin and makes you money."**
> It is an automated local quantitative research system that collects
> market data, trains and evaluates models using leakage-safe walk-forward
> validation, performs realistic backtesting and paper trading, uses a
> local LLM for plain-language analysis, continuously measures its own
> performance and drift, and only permits carefully controlled real Spot
> trading after sufficient evidence and an explicit regulatory check.
>
> Past performance does not guarantee future results. Nothing in this
> project is financial advice, and no profit is promised or implied.

New to crypto? Start at **[docs/CRYPTO_BASICS.md](docs/CRYPTO_BASICS.md)**
and **[docs/BEGINNER_GUIDE.md](docs/BEGINNER_GUIDE.md)**. Ready to install?
**[docs/INSTALLATION.md](docs/INSTALLATION.md)**.

## Quickstart

```bash
git clone <this-repo> && cd Local-AI-Crypto
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
docker compose up -d postgres      # or install Postgres directly — see docs/INSTALLATION.md
python run.py setup
python run.py system-check
python run.py download-data
python run.py train
python run.py backtest
python run.py start                # dashboard at http://localhost:8000
```

No API key or exchange account is needed for any of the above — all of it
runs on Binance's public market-data endpoints and your own machine.

## The safety progression

```
DEVELOPMENT -> HISTORICAL DATA -> AUTOMATIC TRAINING -> BACKTESTING
   -> LIVE OBSERVATION -> PAPER TRADING -> PERFORMANCE VALIDATION
   -> TESTNET/DEMO -> STRICT SAFETY CHECKS -> optional, very small real trading
```

Real trading is **disabled by default** (`REAL_TRADING_ENABLED=false`,
`TRADING_ENABLED=false`, `MODE=RESEARCH`) and stays disabled until you
explicitly pass every item on the checklist in
[docs/REAL_TRADING.md](docs/REAL_TRADING.md) — good backtest/paper-trading
performance never auto-promotes to real trading. See
[docs/REGULATORY_NOTES.md](docs/REGULATORY_NOTES.md) before you even
consider testnet or live trading.

## Project layout

```
crypto_ai/
├── app/            FastAPI app, dashboard, cross-cutting services (reporting, recovery, live tick)
├── config/         settings.yaml / risk.yaml + the loader that merges them with .env
├── data/           historical downloader, live collector, validators, retention
├── database/       SQLAlchemy models, Alembic migrations, repositories
├── features/       technical/volatility indicators, feature pipeline, label generation
├── models/         training pipeline (walk-forward), evaluation/criteria, registry, inference
├── strategies/      baseline (buy-and-hold/random), ML strategy, structured signal engine
├── backtesting/    engine, metrics (incl. deflated Sharpe), reports/charts
├── paper_trading/  portfolio, engine, simulator
├── risk/           position sizing, RiskManager gate, live-trading checklist, emergency stop
├── llm/            local LLM client (Ollama/LM Studio), analyst (explanations only), prompts
├── exchange/       public market-data client, paper adapter, gated live-trading stub
├── monitoring/     health checks, drift detection, resource metrics, alerts
├── scheduler/      APScheduler jobs (all guarded against crashing the process)
└── tests/
docs/               every document referenced throughout this README
run.py              unified CLI — see `python run.py --help`
docker-compose.yml  Postgres (+ optional backend/scheduler containers), host-mounted volumes
```

## What's implemented vs. what's a documented next step

Implemented and tested (187 tests, `pytest`, plus an end-to-end validation
harness at `scripts/e2e_validation.py` that runs the whole workflow against a
simulated exchange): the full research loop —
historical data download/validation, feature/label generation, walk-forward
training with a real promotion gate, backtesting, paper trading with crash
-safe state, prediction accuracy tracking, drift detection, the risk engine,
local LLM integration, a working dashboard, retention cleanup, and the
scheduler/recovery wiring that ties them together
(`app/services/live_pipeline.py`).

Deliberately **not** implemented: real order execution
(`exchange/live.py` is a gated stub that refuses to run — see
[docs/REAL_TRADING.md](docs/REAL_TRADING.md)). This is a safety boundary,
not an oversight — building it out is the right next step only once you've
personally cleared the checklist for your own situation.

## Documentation index

| Doc | Covers |
|---|---|
| [CRYPTO_BASICS.md](docs/CRYPTO_BASICS.md) | Crypto/BTC/USDT/Spot/Binance/API vocabulary |
| [BEGINNER_GUIDE.md](docs/BEGINNER_GUIDE.md) | The big picture, week by week |
| [INSTALLATION.md](docs/INSTALLATION.md) | Step-by-step setup (Docker and direct) |
| [DATA_PIPELINE.md](docs/DATA_PIPELINE.md) | Downloader, validation, features, labels |
| [MODEL_TRAINING.md](docs/MODEL_TRAINING.md) | Walk-forward validation, promotion criteria, registry |
| [BACKTESTING.md](docs/BACKTESTING.md) | Reading a backtest report honestly |
| [PAPER_TRADING.md](docs/PAPER_TRADING.md) | Live-data simulation, prediction evaluation |
| [REAL_TRADING.md](docs/REAL_TRADING.md) | The Section 59 checklist, API key safety, emergency stop |
| [REGULATORY_NOTES.md](docs/REGULATORY_NOTES.md) | What to check before testnet/live, and when |
| [POWER_FAILURE_RECOVERY.md](docs/POWER_FAILURE_RECOVERY.md) | 24/7 operation, Windows auto-start, recovery testing |
| [BACKUP_RESTORE.md](docs/BACKUP_RESTORE.md) | What's backed up, what's excluded, and how to verify a restore |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common problems and fixes |

## Commands

```
python run.py setup           # first-run checks + migrations
python run.py system-check    # hardware-based recommendations
python run.py download-data   # resumable historical backfill
python run.py collect-live    # stream live closed candles over WebSocket
python run.py train           # walk-forward training pipeline
python run.py backtest        # backtest the current champion
python run.py evaluate        # score elapsed predictions
python run.py drift-check     # feature-distribution + accuracy drift check
python run.py paper           # one manual paper-trading tick
python run.py report          # daily report
python run.py research-report # weekly / Month-1 report (--days 7 or 30)
python run.py backup          # backup database + config + models
python run.py restore <file>  # restore (dry run by default; --apply to commit)
python run.py verify-backup <file>
python run.py list-backups
python run.py start           # dashboard + scheduler (or --api-only)
python run.py scheduler       # scheduler only (used by the Docker "scheduler" service)
python run.py stop            # stop a `start`ed process
```

## Running the tests

```bash
pytest
```

Tests run against an isolated in-memory SQLite database — no external
services required.

## License

Add a license of your choosing before distributing this project.
