#!/usr/bin/env python3
"""
Local AI Crypto — unified command-line entry point.

Run `python run.py --help` to see every command, or `python run.py
<command> --help` for details on one. Every command is documented in
docs/ — this file is deliberately thin; the real logic lives in
crypto_ai/.

Beginner note: if you've never used a CLI tool structured this way,
each "command" below (setup, train, backtest, ...) is a separate
mini-program. You run them one at a time, in the order shown in
docs/INSTALLATION.md.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger("run")

app = typer.Typer(add_completion=False, help="Local AI Crypto — research, paper-trading and (eventually) live trading system.")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent
PID_FILE = PROJECT_ROOT / "data_store" / "run.pid"


def _ensure_directories() -> None:
    for sub in ("raw", "processed", "models", "backups", "logs"):
        (PROJECT_ROOT / "data_store" / sub).mkdir(parents=True, exist_ok=True)


@app.command()
def setup():
    """First-run setup: checks dependencies, creates directories, sets up the database, checks the LLM/exchange."""
    console.rule("Local AI Crypto — Setup")
    _ensure_directories()
    console.print("[green]OK[/green] data_store/ directories created")

    if not (PROJECT_ROOT / ".env").exists():
        console.print(
            "[yellow]No .env file found.[/yellow] Copy .env.example to .env before continuing:\n"
            "  cp .env.example .env"
        )

    from crypto_ai.config.loader import get_settings

    try:
        settings = get_settings()
        console.print(f"[green]OK[/green] Configuration loaded (mode={settings.mode})")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]FAILED[/red] Could not load configuration: {exc}")
        raise typer.Exit(1)

    from crypto_ai.database.base import wait_for_database

    console.print("Checking database connection...")
    if wait_for_database(max_attempts=3, base_delay_seconds=1.0):
        console.print("[green]OK[/green] Database reachable")
        console.print("Running migrations (alembic upgrade head)...")
        _run_alembic_upgrade()
    else:
        console.print(
            "[red]FAILED[/red] Could not reach the database.\n"
            "  If using Docker: docker compose up -d postgres\n"
            "  Otherwise, check DATABASE_URL in .env\n"
            "  See docs/TROUBLESHOOTING.md"
        )

    if settings.llm.enabled:
        from crypto_ai.llm.client import check_llm_health

        result = check_llm_health()
        color = "green" if result["status"] == "OK" else "yellow"
        console.print(f"[{color}]{result['status']}[/{color}] Local LLM ({settings.llm.provider} @ {settings.llm.base_url})")
    else:
        console.print("[dim]LLM disabled (LLM_ENABLED=false) — skipping[/dim]")

    console.print("Checking exchange public API connectivity...")
    try:
        from crypto_ai.exchange.market_data import MarketDataClient, RetryPolicy

        client = MarketDataClient(retry_policy=RetryPolicy(max_attempts=1))
        client.fetch_ohlcv_page("BTC/USDT", "5m", dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=30), limit=1)
        console.print("[green]OK[/green] Exchange public API reachable")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]WARNING[/yellow] Exchange not reachable right now: {exc}")
        console.print("  This is fine if you're offline — retry later, or see docs/TROUBLESHOOTING.md")

    console.rule("Setup complete")
    console.print("Next: [bold]python run.py system-check[/bold], then [bold]python run.py download-data[/bold]")


def _run_alembic_upgrade() -> None:
    try:
        from alembic import command
        from alembic.config import Config

        cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
        command.upgrade(cfg, "head")
        console.print("[green]OK[/green] Database migrations applied")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]FAILED[/red] Migration failed: {exc}")


@app.command("system-check")
def system_check():
    """Inspect this machine's hardware and recommend concrete settings (Section 52)."""
    from crypto_ai.monitoring.metrics import get_hardware_profile, get_resource_usage

    console.rule("System Check")

    table = Table(show_header=False)
    table.add_row("Operating system", f"{platform.system()} {platform.release()}")
    table.add_row("Python version", platform.python_version())
    table.add_row("Docker", _tool_version("docker"))
    table.add_row("Docker Compose", _tool_version("docker", "compose", "version"))
    table.add_row("Node.js", _tool_version("node"))
    console.print(table)

    usage = get_resource_usage()
    profile = get_hardware_profile()

    hw_table = Table(title="Hardware")
    hw_table.add_row("CPU cores", str(usage["cpu_count"]))
    hw_table.add_row("RAM total", f"{profile['total_ram_gb']} GB")
    hw_table.add_row("RAM available", f"{usage['ram_available_gb']} GB")
    hw_table.add_row("Disk free", f"{usage['disk_free_gb']} GB")
    hw_table.add_row("GPU", "Not checked — this system is CPU-first by design (Section 9)")
    console.print(hw_table)

    console.rule("Recommended settings")
    console.print(f"MAX_WORKERS={profile['recommended_max_workers']}")
    console.print(f"BATCH_SIZE={profile['recommended_batch_size']}")
    console.print(f"LLM: {profile['llm_recommendation']}")

    console.rule("Connectivity")
    _print_status("Internet", _check_internet())
    from crypto_ai.database.base import wait_for_database

    _print_status("Database", wait_for_database(max_attempts=1))
    from crypto_ai.config.loader import get_settings

    if get_settings().llm.enabled:
        from crypto_ai.llm.client import check_llm_health

        result = check_llm_health()
        _print_status("Local LLM", result["status"] == "OK")
    else:
        console.print("Local LLM: [dim]disabled[/dim]")


def _tool_version(*cmd: str) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return result.stdout.strip().splitlines()[0] if result.returncode == 0 else "not found"
    except Exception:  # noqa: BLE001
        return "not found"


def _check_internet() -> bool:
    try:
        import httpx

        httpx.get("https://www.google.com", timeout=5)
        return True
    except Exception:  # noqa: BLE001
        return False


def _print_status(label: str, ok: bool) -> None:
    color, text = ("green", "OK") if ok else ("yellow", "unavailable")
    console.print(f"{label}: [{color}]{text}[/{color}]")


@app.command("download-data")
def download_data():
    """Automatically download/resume historical BTC/USDT OHLCV data for all configured timeframes."""
    from crypto_ai.database.base import session_scope
    from crypto_ai.data.downloader.historical import backfill_all

    console.rule("Downloading historical data")
    with session_scope() as session:
        results = backfill_all(session)

    table = Table()
    for col in ("Symbol", "Timeframe", "New rows", "Notes"):
        table.add_column(col)
    for r in results:
        table.add_row(
            r.get("symbol", ""), r.get("timeframe", ""), str(r.get("new_rows", "-")),
            r.get("error") or r.get("validation", ""),
        )
    console.print(table)


@app.command("collect-live")
def collect_live(
    symbol: str = typer.Option(None, help="Defaults to config"),
    timeframe: str = typer.Option(None, help="Defaults to config"),
):
    """Stream live closed candles from Binance's public WebSocket into the database."""
    from crypto_ai.data.collector.live_collector import run_collector

    console.rule("Live market-data collector")
    console.print("Streaming closed candles. Press Ctrl+C to stop.")
    run_collector(symbol, timeframe)


@app.command()
def backup():
    """Create a backup archive of the database, config and trained models (Section 33)."""
    from crypto_ai.app.services.backup import create_backup, verify_backup

    console.rule("Creating backup")
    path = create_backup()
    result = verify_backup(path)
    console.print(f"[green]OK[/green] Backup written: {path}")
    console.print(f"  size: {result['size_bytes'] / 1024 / 1024:.2f} MB, entries: {result['n_entries']}")
    console.print("  [dim]Secrets (.env, keys) are deliberately excluded.[/dim]")
    console.print("\nTest it before you rely on it:")
    console.print(f"  [bold]python run.py restore {path}[/bold]   (dry run, touches nothing)")


@app.command("list-backups")
def list_backups_cmd():
    """List backup archives created so far."""
    from crypto_ai.app.services.backup import list_backups

    rows = list_backups()
    if not rows:
        console.print("No backups yet. Create one with: python run.py backup")
        return
    table = Table()
    for col in ("Archive", "Size (MB)", "Created (UTC)"):
        table.add_column(col)
    for r in rows:
        table.add_row(Path(r["path"]).name, str(r["size_mb"]), r["modified"])
    console.print(table)


@app.command("verify-backup")
def verify_backup_cmd(archive: str = typer.Argument(..., help="Path to a .tar.gz backup archive")):
    """Check a backup archive is complete and contains no secrets."""
    from crypto_ai.app.services.backup import verify_backup

    result = verify_backup(Path(archive))
    if result["ok"]:
        console.print(f"[green]OK[/green] {archive} looks valid ({result['n_entries']} entries)")
        console.print(result["manifest"])
    else:
        console.print(f"[red]PROBLEMS FOUND[/red] in {archive}:")
        for p_ in result["problems"]:
            console.print(f"  - {p_}")
        raise typer.Exit(1)


@app.command()
def restore(
    archive: str = typer.Argument(..., help="Path to a .tar.gz backup archive"),
    apply: bool = typer.Option(False, "--apply", help="Actually overwrite live data (default is a dry run)"),
):
    """Restore a backup. Defaults to a DRY RUN that touches nothing (Section 33)."""
    from crypto_ai.app.services.backup import restore_backup

    console.rule("Restore" + ("" if apply else " (dry run)"))
    result = restore_backup(Path(archive), apply_to_live=apply)
    console.print(f"Extracted to: {result['extracted_to']}")
    console.print(f"Manifest: {result['manifest']}")
    if result["applied_to_live"]:
        console.print("[green]Live data restored.[/green] A copy of the previous database was kept alongside it.")
    if result["next_step"]:
        console.print(result["next_step"])


@app.command("drift-check")
def drift_check():
    """Run a drift check now (feature distribution + live accuracy degradation)."""
    from crypto_ai.app.services.maintenance import run_drift_check_job
    from crypto_ai.database.base import session_scope

    with session_scope() as session:
        result = run_drift_check_job(session)
    if result.get("status") != "ok":
        console.print(f"[yellow]Drift check skipped: {result.get('status')}[/yellow]")
        return
    color = "red" if result["flagged_for_review"] else "green"
    console.print(f"[{color}]flagged_for_review={result['flagged_for_review']}[/{color}]")
    console.print(f"max PSI: {result['max_psi']}")
    if result["reasons"]:
        for r in result["reasons"]:
            console.print(f"  - {r}")
    console.print(f"top drifting features: {result['top_drifting_features']}")


@app.command()
def train(
    symbol: str = typer.Option(None, help="Defaults to config"),
    timeframe: str = typer.Option(None, help="Defaults to config"),
    model_name: str = typer.Option("btc_direction_classifier"),
):
    """Run the automatic walk-forward training pipeline (Section 14)."""
    from crypto_ai.config.loader import get_settings
    from crypto_ai.database.base import session_scope
    from crypto_ai.database.repositories.market_data_repo import load_candles_df
    from crypto_ai.features.feature_pipeline import compute_features
    from crypto_ai.features.labels import compute_labels
    from crypto_ai.models.training.pipeline import run_training_pipeline

    settings = get_settings()
    symbol = symbol or settings.get("exchange.symbol", "BTC/USDT")
    timeframe = timeframe or settings.get("data.primary_timeframe", "5m")

    console.rule(f"Training on {symbol} {timeframe}")
    with session_scope() as session:
        df = load_candles_df(session, symbol, timeframe)
        if len(df) < 1000:
            console.print(f"[red]Not enough data ({len(df)} bars).[/red] Run: python run.py download-data")
            raise typer.Exit(1)

        console.print(f"Computing features and labels over {len(df)} bars...")
        features = compute_features(df)
        labels = compute_labels(df)

        summary = run_training_pipeline(session, symbol, timeframe, features, labels, model_name=model_name)

    table = Table()
    for col in ("Algorithm", "Passed walk-forward", "Version", "Notes"):
        table.add_column(col)
    for r in summary["results"]:
        notes = r.get("error") or r.get("reason") or ""
        if not notes and not r.get("overall_pass"):
            wf = r.get("walk_forward", {})
            notes = f"{wf.get('n_passed', 0)}/{wf.get('n_folds', 0)} folds passed"
        table.add_row(r["algorithm"], "YES" if r.get("overall_pass") else "no", r.get("version_label", "-"), notes)
    console.print(table)
    console.print(f"Promotion: {summary['promotion']}")
    if not any(r.get("overall_pass") for r in summary["results"]):
        console.print(
            "\n[yellow]No model passed walk-forward validation.[/yellow] This is common with a "
            "small amount of history — see docs/MODEL_TRAINING.md for what to try next "
            "(more data, looser criteria for experimentation, or accept that this symbol/"
            "timeframe may not have a learnable edge at these settings)."
        )


@app.command()
def backtest(
    symbol: str = typer.Option(None), timeframe: str = typer.Option(None),
    model_name: str = typer.Option("btc_direction_classifier"),
):
    """Backtest the current champion model against recent history and compare to buy-and-hold."""
    from crypto_ai.backtesting.engine import BacktestEngine
    from crypto_ai.backtesting.reports import format_text_report, save_equity_curve_chart
    from crypto_ai.config.loader import get_settings
    from crypto_ai.database.base import session_scope
    from crypto_ai.database.repositories.backtest_repo import save_backtest_run
    from crypto_ai.database.repositories.market_data_repo import load_candles_df
    from crypto_ai.features.feature_pipeline import compute_features
    from crypto_ai.models.inference.predict import predict_single
    from crypto_ai.models.registry import registry
    from crypto_ai.strategies.ml_strategy import STRATEGY_VERSION, predictions_to_signals

    settings = get_settings()
    symbol = symbol or settings.get("exchange.symbol", "BTC/USDT")
    timeframe = timeframe or settings.get("data.primary_timeframe", "5m")

    with session_scope() as session:
        champion = registry.get_champion(session, model_name)
        if champion is None:
            console.print("[red]No champion model yet.[/red] Run: python run.py train")
            raise typer.Exit(1)

        df = load_candles_df(session, symbol, timeframe)
        features = compute_features(df)
        feature_cols = [c for c in features.columns if c != "timestamp"]
        estimator = registry.load_estimator(champion)

        import pandas as pd

        proba = estimator.predict_proba(features[feature_cols])
        classes = estimator.classes_
        predicted_class = pd.Series([classes[i] for i in proba.argmax(axis=1)])
        confidence = pd.Series(proba[range(len(proba)), proba.argmax(axis=1)])
        signals = predictions_to_signals(predicted_class, confidence)

        merged = features[["timestamp"]].copy()
        merged["close"] = df.set_index("timestamp").loc[features["timestamp"], "close"].values
        merged["signal"] = signals.values

        engine = BacktestEngine(
            initial_balance=settings.get("paper_trading.starting_balance_usdt", 1000.0), timeframe=timeframe,
        )
        report = engine.run(merged, symbol=symbol)
        save_backtest_run(
            session, report, STRATEGY_VERSION,
            start_time=merged["timestamp"].iloc[0], end_time=merged["timestamp"].iloc[-1],
            model_version=champion.version_label,
        )

        chart_path = PROJECT_ROOT / "data_store" / "reports" / f"backtest_{champion.version_label}.png"
        save_equity_curve_chart(report, chart_path)

    console.print(format_text_report(report))
    console.print(f"\nEquity curve chart saved to: {chart_path}")


@app.command()
def evaluate():
    """Evaluate all predictions whose horizon has elapsed and print accuracy stats (Section 21)."""
    from crypto_ai.database.base import session_scope
    from crypto_ai.models.evaluation.prediction_evaluator import evaluate_due_predictions, summarize_prediction_accuracy

    with session_scope() as session:
        n = evaluate_due_predictions(session)
        summary = summarize_prediction_accuracy(session)

    console.print(f"Evaluated {n} new predictions.")
    console.print(summary)


@app.command()
def paper(
    symbol: str = typer.Option(None), timeframe: str = typer.Option(None),
    model_name: str = typer.Option("btc_direction_classifier"),
    explain: bool = typer.Option(False, help="Ask the local LLM to explain the resulting signal"),
):
    """Run one prediction + paper-trading tick using the current champion model."""
    from crypto_ai.app.services.live_pipeline import run_prediction_and_paper_trade_tick
    from crypto_ai.database.base import session_scope

    with session_scope() as session:
        result = run_prediction_and_paper_trade_tick(session, symbol, timeframe, model_name, explain=explain)

    console.print(result)
    if result["status"] != "ok":
        console.print("[yellow]Tick skipped — see status above.[/yellow]")


@app.command()
def report():
    """Generate and print the daily report (Section 45)."""
    from crypto_ai.app.services.reporting import format_text_report, generate_daily_report
    from crypto_ai.database.base import session_scope

    with session_scope() as session:
        rep = generate_daily_report(session)
    console.print(format_text_report(rep))


@app.command()
def scheduler():
    """Run the background scheduler only (used by the `scheduler` Docker service)."""
    from crypto_ai.scheduler.jobs import build_scheduler
    import time

    console.rule("Starting scheduler")
    sched = build_scheduler()
    sched.start()
    console.print("Scheduler running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sched.shutdown()


@app.command()
def start(
    api_only: bool = typer.Option(False, "--api-only", help="Run only the dashboard/API, not the scheduler"),
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
):
    """Run the full application: startup recovery, the scheduler, and the dashboard/API."""
    import uvicorn

    from crypto_ai.app.services.recovery import run_startup_sequence
    from crypto_ai.database.base import get_session_factory

    console.rule("Starting Local AI Crypto")
    try:
        result = run_startup_sequence(get_session_factory())
        console.print(f"Startup recovery complete: mode={result['mode']}, health={result['health']}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Startup recovery failed: {exc}[/red]")
        raise typer.Exit(1)

    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))

    sched = None
    if not api_only:
        from crypto_ai.scheduler.jobs import build_scheduler

        sched = build_scheduler()
        sched.start()
        console.print("[green]Scheduler started[/green]")

    console.print(f"Dashboard: http://localhost:{port}  (API: http://localhost:{port}/api)")
    try:
        uvicorn.run("crypto_ai.app.main:app", host=host, port=port, log_level="info")
    finally:
        if sched:
            sched.shutdown()
        PID_FILE.unlink(missing_ok=True)


@app.command()
def stop():
    """Stop a `python run.py start` process running in the background (reads data_store/run.pid)."""
    if not PID_FILE.exists():
        console.print("No run.pid file found — is the app running via `python run.py start`?")
        raise typer.Exit(1)
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        console.print(f"Sent stop signal to process {pid}.")
    except ProcessLookupError:
        console.print("Process not found — it may have already stopped.")
    PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    app()
