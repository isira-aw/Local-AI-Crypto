import datetime as dt

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from crypto_ai.app.api import routes as routes_module
from crypto_ai.app.main import create_app
from crypto_ai.database.models import Base
from crypto_ai.database.repositories.market_data_repo import upsert_candles
from crypto_ai.risk.emergency import is_emergency_stop_active, reset_emergency_stop

START = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)


@pytest.fixture()
def client():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=None)
    # Use StaticPool so the SAME in-memory DB is shared across the
    # multiple connections the TestClient's requests will open.
    from sqlalchemy.pool import StaticPool

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    TestSessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

    def override_get_db():
        session = TestSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app = create_app()
    app.dependency_overrides[routes_module.get_db] = override_get_db

    with TestClient(app) as c:
        c._test_sessionmaker = TestSessionLocal  # stash for seeding in tests
        yield c

    reset_emergency_stop()


def _seed_candles(client, n=50):
    session = client._test_sessionmaker()
    rows = [
        {"timestamp": START + dt.timedelta(minutes=5 * i), "open": 100 + i * 0.1, "high": 101 + i * 0.1,
         "low": 99 + i * 0.1, "close": 100 + i * 0.1, "volume": 5.0}
        for i in range(n)
    ]
    upsert_candles(session, "BTC/USDT", "5m", rows)
    session.commit()
    session.close()


def test_overview_endpoint_returns_expected_shape(client):
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "RESEARCH"
    assert "warning" in data
    assert "does not guarantee" in data["warning"]


def test_overview_with_no_data_does_not_error(client):
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    assert resp.json()["current_price"] is None


def test_market_endpoint_returns_candles_and_indicators(client):
    _seed_candles(client, n=100)
    resp = client.get("/api/market?timeframe=5m&limit=100")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "BTC/USDT"
    assert len(data["candles"]) == 100
    assert len(data["indicators"]) > 0  # some warm-up rows dropped, but not all


def test_market_endpoint_empty_when_no_data(client):
    resp = client.get("/api/market?timeframe=5m")
    assert resp.status_code == 200
    assert resp.json()["candles"] == []


def test_ai_endpoint_returns_llm_status_and_no_champion_initially(client):
    resp = client.get("/api/ai")
    assert resp.status_code == 200
    data = resp.json()
    assert data["champion_model"] is None
    assert data["llm_status"]["status"] == "DISABLED"


def test_backtests_endpoint_empty_list_initially(client):
    resp = client.get("/api/backtests")
    assert resp.status_code == 200
    assert resp.json() == []


def test_paper_trading_endpoint_shape(client):
    resp = client.get("/api/paper-trading")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "current_position" in data
    assert "recent_trades" in data


def test_system_endpoint_reports_health_and_flags(client):
    resp = client.get("/api/system")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "RESEARCH"
    assert data["emergency_stop"]["active"] is False
    assert "resource_usage" in data
    assert "health" in data


def test_daily_report_endpoint(client):
    resp = client.get("/api/report/daily")
    assert resp.status_code == 200
    assert "recommendation" in resp.json()


def test_emergency_stop_activate_and_reset_roundtrip(client):
    assert is_emergency_stop_active() is False

    resp = client.post("/api/emergency-stop/activate", params={"reason": "test"})
    assert resp.status_code == 200
    assert resp.json()["active"] is True
    assert is_emergency_stop_active() is True

    resp2 = client.get("/api/system")
    assert resp2.json()["emergency_stop"]["active"] is True

    resp3 = client.post("/api/emergency-stop/reset")
    assert resp3.json()["active"] is False
    assert is_emergency_stop_active() is False


def test_overview_exposes_signal_and_drift_fields(client):
    """Section 28: the Overview page must show current signal, confidence
    and risk — these were absent until signals were actually persisted."""
    resp = client.get("/api/overview")
    data = resp.json()
    for key in ("current_signal", "drift_status", "champion_model", "paper_trading_summary"):
        assert key in data, f"overview missing {key}"
    # With no signals recorded yet both are None rather than missing/erroring.
    assert data["current_signal"] is None
    assert data["drift_status"] is None


def test_ai_page_exposes_recent_signals_and_drift(client):
    data = client.get("/api/ai").json()
    assert "recent_signals" in data
    assert "drift_status" in data
    assert data["recent_signals"] == []


def test_system_page_exposes_drift_status(client):
    assert "drift_status" in client.get("/api/system").json()


def test_overview_shows_persisted_signal(client):
    import datetime as dt
    from crypto_ai.database.repositories.signals_repo import save_signal
    from crypto_ai.strategies.signal_engine import build_signal

    session = client._test_sessionmaker()
    sig = build_signal(
        symbol="BTC/USDT", timestamp=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
        predicted_class="BUY", confidence=0.77, risk_score=0.15,
        model_version="model_001", strategy_version="s1",
    )
    save_signal(session, sig)
    session.commit()
    session.close()

    data = client.get("/api/overview").json()
    assert data["current_signal"]["signal"] == "BUY"
    assert data["current_signal"]["confidence"] == 0.77

    ai = client.get("/api/ai").json()
    assert len(ai["recent_signals"]) == 1
