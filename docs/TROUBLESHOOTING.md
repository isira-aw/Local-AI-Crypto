# Troubleshooting

## Database

**"Could not connect to the database" / "connection refused"**
- Docker: is Postgres running? `docker compose ps` — if not,
  `docker compose up -d postgres`.
- Direct install: is the Postgres service running on your machine?
- Check `DATABASE_URL` in `.env` — `localhost` if running Python directly,
  `postgres` (the Docker service name) if running inside Docker.
- `python run.py system-check` reports current database status.

**"relation does not exist" / missing tables**
- Migrations haven't been applied. Run `python run.py setup`, or directly:
  `alembic upgrade head`.

## Exchange / internet

**"Market data call failed" / exchange unreachable**
- Check your internet connection.
- Check whether your network/firewall/VPN can reach `api.binance.com` —
  some networks and regions block it.
- This is expected to fail if you're offline; it does not block training
  or backtesting on data you've already downloaded.
- Retries happen automatically with backoff — a single transient failure
  usually resolves itself.

**Downloads are slow / seem stuck**
- The downloader respects the exchange's rate limits by design — large
  historical backfills (years of 1-minute data) can legitimately take a
  while. Check the logs (dashboard's System page, or `data_store/logs/`)
  for progress; it writes to the database incrementally, so you can check
  how far it's gotten via the dashboard's Market page even mid-download.

## Local LLM

**LLM status shows "DOWN" on the dashboard**
- Is Ollama/LM Studio actually running? (`ollama list` should show your
  pulled models for Ollama.)
- Check `LLM_BASE_URL` in `.env`. From inside Docker on Windows/Mac, this
  must be `http://host.docker.internal:11434`, not `http://localhost:11434`
  — `localhost` inside a container refers to the container itself, not
  your host machine. (On Linux, you may need your host's Docker bridge IP
  instead — see Docker's networking docs for your platform.)
- Check `LLM_MODEL` matches a model you've actually pulled/loaded.
- Remember: this is optional. Set `LLM_ENABLED=false` and everything else
  keeps working — a "DOWN" LLM never blocks training, backtesting, or
  paper trading.

**LLM status shows "DISABLED"**
- Expected if `LLM_ENABLED=false` — not an error.

## Training

**"No model passed walk-forward validation"**
- See `docs/MODEL_TRAINING.md` — usually means not enough historical data
  yet, or (correctly) that no tested model showed a real, robust edge at
  the current settings. Not a bug.

**Training is very slow / uses too much CPU**
- Lower `resource_limits.max_workers` in `settings.yaml` (or `MAX_WORKERS`
  in `.env`).
- `python run.py system-check` gives a hardware-based recommendation.
- Don't run training and the local LLM at full load simultaneously on
  modest hardware (Section 53) — they compete for the same CPU/RAM.

## Dashboard / API

**Dashboard loads but shows "n/a" everywhere**
- Normal on a fresh install with no data yet. Run `python run.py
  download-data`, then `python run.py train`.

**Port 8000 already in use**
- Another process is using it. `python run.py start --port 8001` (or stop
  whatever else is using 8000).

## General

**"It looks too good" — huge returns, very high accuracy, huge Sharpe**
- This is very likely overfitting or a data leak, not a real edge. See the
  "too-good-to-be-true" warnings in `docs/MODEL_TRAINING.md` and
  `docs/BACKTESTING.md` — the system tries to flag these automatically,
  but always sanity-check anything that looks implausibly good yourself.

**Something else is broken**
- Check `data_store/logs/` and the dashboard's System page for recent
  `system_events` — every important event is logged with a component,
  severity, and message (never secrets).
- Check `python run.py system-check` for an overall picture of what's
  reachable and what isn't.
