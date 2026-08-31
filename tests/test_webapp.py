from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from trading_agent.agent import AutonomousTradingAgent
from trading_agent.config import Settings, apply_settings_update
from trading_agent.events import EventBus
from trading_agent.webapp.server import create_app
from trading_agent.webapp.supervisor import AgentSupervisor

TOKEN = "test-token-abc"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def settings(tmp_path, monkeypatch):
    # overlay se uklada relativne k CWD - v testech smerovat do tmp_path
    monkeypatch.chdir(tmp_path)
    return Settings(
        ALPACA_API_KEY="key", ALPACA_SECRET_KEY="secret", ALPACA_PAPER=True, SYMBOLS="AAPL,MSFT",
        MODEL_PATH=str(tmp_path / "model.pkl"), BANDIT_PATH=str(tmp_path / "bandit.json"),
        DB_PATH=str(tmp_path / "db.sqlite"), LOG_DIR=str(tmp_path / "logs"),
        KILL_SWITCH_FILE=str(tmp_path / "STOP"), WEB_API_TOKEN=TOKEN,
    )


@pytest.fixture
def supervisor(settings):
    settings.ensure_directories()
    sup = AgentSupervisor(settings)
    sup.broker.get_account = lambda: SimpleNamespace(
        equity="100000", cash="40000", buying_power="80000", portfolio_value="100000",
        pattern_day_trader=False, daytrade_count=0,
    )
    sup.broker.get_positions = lambda: []
    yield sup
    sup.shutdown()


@pytest.fixture
def client(supervisor):
    return TestClient(create_app(supervisor))


# --------------------------------------------------------------- autentizace
def test_api_requires_token(client):
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/status", headers={"Authorization": "Bearer spatny"}).status_code == 401


def test_api_accepts_header_and_query_token(client):
    assert client.get("/api/status", headers=AUTH).status_code == 200
    assert client.get(f"/api/status?token={TOKEN}").status_code == 200


def test_websocket_rejects_bad_token(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/ws?token=spatny") as ws:
            ws.receive_json()


# ------------------------------------------------------------------- status
def test_status_reports_mode_and_account(client):
    data = client.get("/api/status", headers=AUTH).json()
    assert data["running"] is False
    assert data["mode"] == "PAPER"
    assert data["symbols"] == ["AAPL", "MSFT"]
    assert data["equity"] == pytest.approx(100000.0)


def test_status_survives_broker_failure(client, supervisor):
    def boom():
        raise RuntimeError("Alpaca nedostupna")

    supervisor.broker.get_account = boom
    data = client.get("/api/status", headers=AUTH).json()
    assert "Alpaca nedostupna" in data["account_error"]
    assert data["running"] is False  # endpoint stale odpovida


# ----------------------------------------------------------------- nastaveni
def test_settings_masks_secrets(client):
    data = client.get("/api/settings", headers=AUTH).json()
    assert data["alpaca_secret_key"] != "secret"
    assert data["web_api_token"] != TOKEN
    assert data["alpaca_api_key"].endswith("key")  # posledni 4 znaky zustavaji


def test_settings_schema_covers_known_fields(client):
    schema = client.get("/api/settings/schema", headers=AUTH).json()
    names = {f["name"] for group in schema for f in group["fields"]}
    assert {"risk_per_trade", "min_confidence", "symbols_raw", "dry_run"} <= names
    assert all(f["name"] in Settings.model_fields for group in schema for f in group["fields"])


def test_settings_update_applies_and_persists(client, supervisor):
    response = client.post("/api/settings", headers=AUTH, json={"risk_per_trade": 0.025, "dry_run": True})
    assert response.status_code == 200
    assert supervisor.settings.risk_per_trade == pytest.approx(0.025)
    assert supervisor.settings.dry_run is True
    assert response.json()["risk_per_trade"] == pytest.approx(0.025)


def test_settings_update_rejects_invalid_value(client, supervisor):
    response = client.post("/api/settings", headers=AUTH, json={"risk_per_trade": 5.0})
    assert response.status_code == 422
    assert response.json()["detail"][0]["field"] == "RISK_PER_TRADE"
    assert supervisor.settings.risk_per_trade == pytest.approx(0.01)  # zustalo puvodni


def test_settings_update_rejects_unknown_field(client):
    response = client.post("/api/settings", headers=AUTH, json={"neexistujici": 1})
    assert response.status_code == 400


def test_apply_settings_update_keeps_object_identity():
    """Agent drzi referenci na Settings - identita objektu se nesmi zmenit,
    jinak by se 'hot' zmeny do bezicicho agenta nepropsaly."""
    settings = Settings()
    before = id(settings)
    apply_settings_update(settings, {"min_confidence": 0.75})
    assert id(settings) == before
    assert settings.min_confidence == pytest.approx(0.75)


# -------------------------------------------------------------------- data
def test_data_endpoints_return_lists(client):
    for path in ("/api/positions", "/api/trades", "/api/signals", "/api/equity_curve"):
        response = client.get(path, headers=AUTH)
        assert response.status_code == 200, path
        assert isinstance(response.json(), list), path


def test_trades_endpoint_returns_recorded_trade(client, supervisor):
    trade_id = supervisor.storage.open_trade("AAPL", "buy", 10, 100.0, 95.0, 110.0, "ml_model", 0.7)
    supervisor.storage.close_trade(trade_id, 105.0, "take_profit")
    trades = client.get("/api/trades", headers=AUTH).json()
    assert trades[0]["symbol"] == "AAPL"
    assert trades[0]["pnl"] == pytest.approx(50.0)


def test_model_stats_without_running_agent(client):
    assert client.get("/api/model_stats", headers=AUTH).json() == {"available": False}


# ----------------------------------------------------------------- ovladani
def test_kill_switch_toggle(client, supervisor):
    from pathlib import Path

    on = client.post("/api/control/kill_switch", headers=AUTH, json={"active": True}).json()
    assert on["kill_switch_active"] is True
    assert Path(supervisor.settings.kill_switch_file).exists()

    off = client.post("/api/control/kill_switch", headers=AUTH, json={"active": False}).json()
    assert off["kill_switch_active"] is False
    assert not Path(supervisor.settings.kill_switch_file).exists()


def test_close_endpoints_require_running_agent(client):
    assert client.post("/api/control/close_all", headers=AUTH).status_code == 409
    assert client.post("/api/control/close_position/AAPL", headers=AUTH).status_code == 409


def test_start_refuses_without_credentials(supervisor):
    supervisor.settings.alpaca_api_key = ""
    ok, message = supervisor.start()
    assert ok is False
    assert "ALPACA_API_KEY" in message


def test_start_refuses_unconfirmed_live_trading(supervisor):
    supervisor.settings.alpaca_paper = False
    supervisor.settings.i_understand_live_trading_risk = False
    ok, message = supervisor.start()
    assert ok is False
    assert "I_UNDERSTAND_LIVE_TRADING_RISK" in message


def test_supervisor_lifecycle(supervisor, monkeypatch):
    """start/stop/restart bez sitovych volani - agent.run() jen ceka na stop."""
    monkeypatch.setattr(
        AutonomousTradingAgent, "run",
        lambda self, install_signal_handlers=True: self._stop_event.wait(),
    )
    ok, _ = supervisor.start()
    assert ok and supervisor.is_running()

    assert supervisor.start()[0] is False  # dvakrat spustit nejde

    supervisor.settings.min_confidence = 0.8
    assert supervisor.agent.settings.min_confidence == pytest.approx(0.8)  # sdilena reference

    ok, _ = supervisor.stop()
    assert ok and not supervisor.is_running()

    ok, _ = supervisor.restart()
    assert ok and supervisor.is_running()


# --------------------------------------------------------------- event bus
def test_event_bus_delivers_to_subscribers():
    bus = EventBus()
    q1, q2 = bus.subscribe(), bus.subscribe()
    bus.publish({"type": "signal", "symbol": "AAPL"})

    for q in (q1, q2):
        event = q.get_nowait()
        assert event["symbol"] == "AAPL"
        assert "ts" in event  # razitko doplnuje bus

    bus.unsubscribe(q1)
    bus.publish({"type": "heartbeat"})
    assert q1.empty()
    assert q2.get_nowait()["type"] == "heartbeat"


def test_event_bus_history_replays_for_late_subscriber():
    bus = EventBus()
    bus.publish({"type": "agent_status", "status": "started"})
    assert [e["type"] for e in bus.history()] == ["agent_status"]


def test_event_bus_drops_events_for_full_queue():
    """Pomaly/odpojeny klient nesmi zablokovat agenta."""
    bus = EventBus()
    q = bus.subscribe()
    for i in range(q.maxsize + 50):
        bus.publish({"type": "signal", "i": i})
    assert q.full()


def test_websocket_streams_published_events(client, supervisor):
    with client.websocket_connect(f"/ws?token={TOKEN}") as ws:
        supervisor.events.publish({"type": "trade_opened", "symbol": "MSFT"})
        event = ws.receive_json()
        assert event["type"] == "trade_opened"
        assert event["symbol"] == "MSFT"


# ---------------------------------------------------------------- dashboard
def test_dashboard_is_served_without_token(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Alpaca Trading Agent" in response.text
