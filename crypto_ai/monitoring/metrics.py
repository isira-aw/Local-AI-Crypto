"""
System resource metrics (Section 9/52/53): CPU/RAM/disk usage, used by
`run.py system-check`, the dashboard's System page, and to decide
whether the scheduler should skip a heavy job (e.g. training) because
the machine is already under load.
"""

from __future__ import annotations

from pathlib import Path

import psutil

from crypto_ai.config.loader import get_settings


def get_resource_usage() -> dict:
    disk = psutil.disk_usage(str(get_settings().data_store_dir.parent))
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.2),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_available_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
        "disk_percent": disk.percent,
        "disk_free_gb": round(disk.free / (1024 ** 3), 2),
        "cpu_count": psutil.cpu_count(logical=True) or 1,
    }


def get_hardware_profile() -> dict:
    """Used by `run.py system-check` (Section 9/52) to recommend concrete settings."""
    usage = get_resource_usage()
    total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    cpu_count = usage["cpu_count"]

    if total_ram_gb < 8:
        llm_recommendation = "Skip the local LLM (LLM_ENABLED=false) or use a small (<=3B) quantized model."
        max_workers = 1
    elif total_ram_gb < 16:
        llm_recommendation = "A quantized 7-8B model (e.g. llama3.1:8b-instruct-q4_K_M) should fit alongside the app."
        max_workers = min(2, max(1, cpu_count - 1))
    else:
        llm_recommendation = "A quantized 7-8B (or larger, if you have >24GB RAM) model should run comfortably."
        max_workers = min(4, max(1, cpu_count - 1))

    return {
        "total_ram_gb": total_ram_gb,
        "cpu_count": cpu_count,
        "disk_free_gb": usage["disk_free_gb"],
        "recommended_max_workers": max_workers,
        "recommended_batch_size": 128 if total_ram_gb < 8 else 256,
        "llm_recommendation": llm_recommendation,
    }
