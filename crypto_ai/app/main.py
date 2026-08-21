"""
FastAPI application entry point (Phase 14).

What is it?
    Serves the JSON API (crypto_ai/app/api/routes.py) and the static
    dashboard (crypto_ai/app/dashboard/static/index.html) from one
    process, so a beginner only has to run one thing to see the
    dashboard in a browser.

How do I start it?
    python run.py start        (full app: API + scheduler)
    python run.py start --api-only   (just the dashboard/API)
    Then open http://localhost:8000 in a browser.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from crypto_ai.app.api.routes import router as api_router

STATIC_DIR = Path(__file__).resolve().parent / "dashboard" / "static"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Local AI Crypto — Research & Paper Trading Dashboard",
        description=(
            "Experimental automated crypto research system. Not financial "
            "advice. Past performance does not guarantee future results."
        ),
    )

    # The dashboard is served from the same origin in normal use; CORS is
    # only relevant for local frontend development (e.g. a separate Vite
    # dev server), and is restricted to localhost.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="dashboard")

    return app


app = create_app()
