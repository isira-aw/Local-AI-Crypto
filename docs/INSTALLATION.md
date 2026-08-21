# Installation Guide

This guide assumes you have never set up a Python project before. Every
command is explained. Windows instructions are given first (with notes for
macOS/Linux); adjust if you're on a different OS.

You have two installation paths:

- **Docker method** (recommended if you're not sure): one command starts the
  database and app in isolated containers. Less to install on your machine.
- **Direct Python method**: install Python and Postgres yourself. More
  control, slightly more setup steps.

Both are documented below. Pick one.

## 0. What you need before starting

| Tool | Why | Required? |
|---|---|---|
| Python 3.11+ | Runs the application | Always |
| Git | Downloads (clones) this repository | Always |
| Docker + Docker Compose | Runs the database (and optionally the whole app) in containers | Recommended |
| Node.js | Only needed if you build a fancier frontend later — the built-in dashboard needs nothing extra | Optional |
| Ollama or LM Studio | Runs the local AI explanation layer | Optional |

Check what you already have (Windows PowerShell — use a regular terminal on
macOS/Linux, the commands are the same):

```powershell
python --version
git --version
docker --version
docker compose version
```

If `python --version` fails, install Python from https://python.org
(check "Add Python to PATH" during setup on Windows). If `git --version`
fails, install Git from https://git-scm.com. If you plan to use Docker,
install Docker Desktop from https://docker.com and make sure it's running
before continuing.

## 1. Get the code

```powershell
git clone <this-repository-url>
cd Local-AI-Crypto
```

`git clone` downloads a copy of the project. `cd` ("change directory") moves
your terminal into the project folder — every command below assumes you're
standing in this folder.

## 2. Create a Python virtual environment

A **virtual environment** is an isolated folder of Python packages just for
this project, so it doesn't conflict with anything else on your machine.

```powershell
python -m venv .venv
```

Activate it (you'll need to do this every time you open a new terminal to
work on this project):

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Your terminal prompt should now start with `(.venv)`.

## 3. Install Python dependencies

```powershell
pip install -r requirements.txt
```

This reads `requirements.txt` and installs everything the project needs
(web framework, database driver, machine learning libraries, etc.) into the
virtual environment you just activated.

## 4. Configure your `.env` file

```powershell
copy .env.example .env      # Windows
cp .env.example .env        # macOS / Linux
```

Open `.env` in a text editor. For research/paper trading, **you don't need
to change anything** — the defaults work with no API keys and no account.
Read the comments in the file; every setting explains itself.

## 5. Set up the database

### Docker method (recommended)

```powershell
docker compose up -d postgres
```

This starts a Postgres database in a container. `-d` means "detached" (runs
in the background). Data is stored on your machine under
`./docker/postgres-data`, not inside the container, so it survives container
restarts/recreation and can be backed up or moved to another machine (see
`docs/POWER_FAILURE_RECOVERY.md`).

### Direct method (no Docker)

Install PostgreSQL 16 from https://www.postgresql.org/download/, create a
database and user matching your `.env`'s `DATABASE_URL`
(default: user `crypto_ai`, password `crypto_ai`, database `crypto_ai`), and
change `localhost` in `DATABASE_URL` if needed.

### Apply the database schema

Either way, run:

```powershell
python run.py setup
```

This checks your configuration, waits for the database to be reachable, and
applies the schema (via Alembic migrations). You should see
`OK Database migrations applied`.

## 6. (Optional) Set up a local LLM

The system works perfectly well with `LLM_ENABLED=false` in `.env` — data
collection, training, backtesting and paper trading don't need it. The LLM
only adds plain-language explanations.

**Recommended: Ollama** (simplest to install)

1. Install from https://ollama.com
2. Pull a model sized for your RAM (`python run.py system-check` will tell
   you what it recommends once installed — see below):
   ```powershell
   ollama pull llama3.1:8b-instruct-q4_K_M
   ```
3. Make sure `.env` has `LLM_ENABLED=true`, `LLM_PROVIDER=ollama`,
   `LLM_MODEL=llama3.1:8b-instruct-q4_K_M`.
4. If running the app via Docker on Windows/Mac, `LLM_BASE_URL` should be
   `http://host.docker.internal:11434` (already the Docker Compose default)
   so the container can reach Ollama running on your host machine.

**Alternative: LM Studio** — load a model, start its local server (OpenAI-
compatible API), set `LLM_PROVIDER=lmstudio` and `LLM_BASE_URL` to match.

## 7. Check your system

```powershell
python run.py system-check
```

This inspects your actual CPU, RAM, and disk, and prints concrete
recommendations (worker count, batch size, which size of LLM model to use)
instead of generic advice — see Section 52 of the design document.

## 8. Download historical data and train a first model

```powershell
python run.py download-data
python run.py train
python run.py backtest
```

See `docs/DATA_PIPELINE.md`, `docs/MODEL_TRAINING.md`, and
`docs/BACKTESTING.md` for what each of these actually does.

## 9. Start the full system (dashboard + scheduler)

```powershell
python run.py start
```

Then open http://localhost:8000 in a browser.

### Docker method (whole app in containers)

```powershell
docker compose up -d
```

Starts Postgres, the backend/dashboard, and the scheduler, all in
containers. Open http://localhost:8000.

## Troubleshooting

See `docs/TROUBLESHOOTING.md`. Common first issues: database not reachable
(is Postgres running?), exchange not reachable (check your internet/firewall
— this is expected to fail if you're offline, and is not required for
training/backtesting on already-downloaded data), LLM not reachable (is
Ollama/LM Studio actually running?).
