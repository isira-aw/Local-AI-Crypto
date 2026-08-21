# Power Failure Recovery

Relevant if you plan to run this system 24/7 on a machine that might lose
power (common in some regions, irrelevant in others — you know your own
situation). This document is about **paper trading / research** recovery,
which is safe to test freely. Do **not** test power-failure recovery with
real trading enabled — see the warning at the bottom.

## The recovery sequence

```
POWER FAILURE
    v
COMPUTER OFF
    v
POWER RETURNS
    v
COMPUTER AUTO BOOTS
    v
APPLICATION AUTO STARTS
    v
CONNECT TO DATABASE (retries with backoff — crypto_ai/database/base.py:wait_for_database)
    v
HEALTH CHECKS (database/exchange/LLM)
    v
[TESTNET/LIVE ONLY] RECONCILE EXCHANGE STATE
    v
CONTINUE
```

This is implemented in `crypto_ai/app/services/recovery.py:run_startup_sequence`,
called automatically by `python run.py start`.

**Section 26's core rule: never assume the local database is correct after a
crash.** For paper trading, the database *is* the source of truth (there's
no external exchange account to double-check against), and every read in
`paper_trading/portfolio.py` goes straight to the database — nothing is
cached in memory across restarts, so paper-trading state is safe by
construction. For testnet/live trading, the exchange's own account state
must be treated as authoritative instead — see the warning below.

## Setting up auto-recovery on Windows

### 1. BIOS/UEFI: automatic power-on after AC recovery

Restart your PC, enter BIOS/UEFI setup (usually `Del` or `F2` at boot), and
look for a setting like "Restore on AC Power Loss" / "After Power Failure"
— set it to **On** or **Last State**. Exact wording varies by motherboard.

### 2. Windows: automatic startup

Add a shortcut to your startup folder (`Win+R`, type `shell:startup`,
Enter) that runs `docker compose up -d` in this project's folder — or use
Task Scheduler for more control (trigger: "At startup", action: run the
command).

### 3. Docker: auto-start containers

`docker-compose.yml` already sets `restart: unless-stopped` on every
service — once Docker Desktop itself starts, your containers restart
automatically. Make sure Docker Desktop is configured to start with
Windows (Docker Desktop Settings → General → "Start Docker Desktop when you
log in").

### 4. Application auto-start

If using Docker, step 3 covers this. If running directly with Python
(no Docker), use Task Scheduler to run
`python run.py start` at startup, or run it as a Windows service using a
tool like NSSM.

### 5. Database recovery

Postgres running in Docker with the bind-mounted volume
(`./docker/postgres-data`) recovers automatically on container restart — no
manual steps needed. If Postgres was killed mid-write, it runs its own
crash-recovery on startup (standard Postgres behavior); `wait_for_database`
retries with backoff specifically to ride out this brief recovery window.

## Testing it safely

1. With `MODE=RESEARCH` or `MODE=PAPER` (never `LIVE`), start the system
   normally and let it paper trade for a while.
2. Simulate a crash: `docker compose kill` (or just close the terminal
   running `python run.py start`) — this is more abrupt than a graceful
   stop, closer to what a real power loss looks like.
3. Restart: `docker compose up -d` (or `python run.py start` again).
4. Check the dashboard's System page and the daily report — paper-trading
   balance/position should be exactly where they were, and the health
   checks should all pass once the database is back up.

## Warning: do not test this with real trading enabled

If you ever reach the live-trading phase (`docs/REAL_TRADING.md`), do not
use power-loss testing as a way to "see what happens" with real money on
the line. Test recovery thoroughly in PAPER mode first — this is one of the
required checklist items in Section 59 (`recovery_tested`) precisely so it
gets validated before real money is involved, not after.

## Manual, hardware-level fallback (live-trading phase only)

This is intentionally low-tech and independent of whether any of the
software above is working correctly: if you are ever unsure whether the bot
recovered correctly after an outage while real money was involved,
physically power off the machine and check your positions directly from
the exchange's app or website on your phone. This does not depend on this
codebase, your database, or your network being in any particular state —
it's the fallback of last resort.
