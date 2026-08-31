"""FastAPI aplikace: REST API + WebSocket pro webovy dashboard agenta.

Cely API povrch (krome staticky servirovaneho dashboardu samotneho) je za
bearer-token autentizaci (`Settings.web_api_token`) - toto rozhrani umi menit
risk limity a zavirat pozice na skutecnem uctu, takze neautentizovany pristup
by byl vazne bezpecnostni riziko, zejmena pokud by nekdo spustil dashboard na
neco jineho nez 127.0.0.1.
"""

from __future__ import annotations

import asyncio
import logging
import queue
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from trading_agent import __version__
from trading_agent.webapp.settings_api import SETTINGS_SCHEMA
from trading_agent.webapp.supervisor import AgentSupervisor

logger = logging.getLogger("trading_agent.webapp.server")

STATIC_DIR = Path(__file__).parent / "static"


def create_app(supervisor: AgentSupervisor) -> FastAPI:
    app = FastAPI(title="Alpaca Trading Agent Dashboard")

    def require_token(authorization: Optional[str] = Header(default=None), token: Optional[str] = None) -> None:
        expected = supervisor.settings.web_api_token
        provided = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else token
        if not expected or provided != expected:
            raise HTTPException(status_code=401, detail="Neplatny nebo chybejici pristupovy token.")

    router = APIRouter(prefix="/api", dependencies=[Depends(require_token)])

    # ------------------------------------------------------------- status
    @router.get("/status")
    def get_status() -> dict[str, Any]:
        data = supervisor.status()
        try:
            account = supervisor.broker.get_account()
            data.update(
                equity=float(account.equity),
                cash=float(account.cash),
                buying_power=float(account.buying_power),
                portfolio_value=float(account.portfolio_value),
                pattern_day_trader=bool(account.pattern_day_trader),
                daytrade_count=account.daytrade_count,
            )
        except Exception as exc:
            data["account_error"] = str(exc)
        agent = supervisor.agent
        if agent is not None:
            data["model_auc"] = round(agent.model.rolling_score(), 4)
            data["model_samples"] = agent.model.n_samples
        return data

    @router.get("/positions")
    def get_positions() -> list[dict[str, Any]]:
        try:
            positions = supervisor.broker.get_positions()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Alpaca API chyba: {exc}") from exc
        open_trades = {t.symbol: t for t in supervisor.storage.get_all_open_trades()}
        result = []
        for p in positions:
            trade = open_trades.get(p.symbol)
            result.append(
                {
                    "symbol": p.symbol,
                    "side": getattr(p.side, "value", str(p.side)),
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price) if p.current_price is not None else None,
                    "unrealized_pl": float(p.unrealized_pl) if p.unrealized_pl is not None else None,
                    "unrealized_plpc": float(p.unrealized_plpc) if p.unrealized_plpc is not None else None,
                    "market_value": float(p.market_value) if p.market_value is not None else None,
                    "stop_price": trade.stop_price if trade else None,
                    "take_profit_price": trade.take_profit_price if trade else None,
                    "strategy": trade.strategy_name if trade else None,
                }
            )
        return result

    @router.get("/trades")
    def get_trades(limit: int = 100) -> list[dict[str, Any]]:
        trades = supervisor.storage.get_recent_closed_trades(limit=min(max(limit, 1), 500))
        return [asdict(t) for t in trades]

    @router.get("/signals")
    def get_signals(limit: int = 100) -> list[dict[str, Any]]:
        return supervisor.storage.get_recent_signals(limit=min(max(limit, 1), 500))

    @router.get("/equity_curve")
    def get_equity_curve(limit: int = 500) -> list[dict[str, Any]]:
        return supervisor.storage.get_equity_curve(limit=min(max(limit, 1), 5000))

    @router.get("/model_stats")
    def get_model_stats() -> dict[str, Any]:
        agent = supervisor.agent
        if agent is None:
            return {"available": False}
        return {
            "available": True,
            "model_auc": round(agent.model.rolling_score(), 4),
            "model_samples": agent.model.n_samples,
            "bandit": agent.bandit.stats(),
        }

    # ------------------------------------------------------------ settings
    @router.get("/settings")
    def get_settings() -> dict[str, Any]:
        return supervisor.settings.to_public_dict()

    @router.get("/settings/schema")
    def get_settings_schema() -> list[dict]:
        return SETTINGS_SCHEMA

    @router.post("/settings")
    def post_settings(updates: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            supervisor.update_settings(updates)
        except ValidationError as exc:
            # exc.errors() obsahuje v "ctx" surovy Exception objekt, ktery neni
            # JSON serializovatelny - poslat jen field+zpravu
            errors = [{"field": e["loc"][-1] if e["loc"] else None, "message": e["msg"]} for e in exc.errors()]
            raise HTTPException(status_code=422, detail=errors) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return supervisor.settings.to_public_dict()

    # ------------------------------------------------------------- control
    @router.post("/control/start")
    def control_start() -> dict[str, Any]:
        ok, message = supervisor.start()
        return {"ok": ok, "message": message}

    @router.post("/control/stop")
    def control_stop() -> dict[str, Any]:
        ok, message = supervisor.stop()
        return {"ok": ok, "message": message}

    @router.post("/control/restart")
    def control_restart() -> dict[str, Any]:
        ok, message = supervisor.restart()
        return {"ok": ok, "message": message}

    @router.post("/control/kill_switch")
    def control_kill_switch(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        active = bool(payload.get("active", True))
        path = Path(supervisor.settings.kill_switch_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if active:
            path.touch()
        else:
            path.unlink(missing_ok=True)
        return {"kill_switch_active": path.exists()}

    @router.post("/control/close_position/{symbol}")
    def control_close_position(symbol: str) -> dict[str, Any]:
        if supervisor.agent is None:
            raise HTTPException(status_code=409, detail="Agent nebezi.")
        ok = supervisor.agent.manual_close_position(symbol.upper())
        if not ok:
            raise HTTPException(status_code=404, detail=f"Zadna otevrena pozice pro {symbol.upper()}.")
        return {"ok": True}

    @router.post("/control/close_all")
    def control_close_all() -> dict[str, Any]:
        if supervisor.agent is None:
            raise HTTPException(status_code=409, detail="Agent nebezi.")
        return {"closed": supervisor.agent.manual_close_all()}

    app.include_router(router)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        """Overeni, ze server bezi - zamerne bez tokenu, aby slo diagnostikovat
        i bez prihlaseni (nevraci zadne udaje o uctu ani nastaveni)."""
        return {
            "status": "ok",
            "version": __version__,
            "agent_running": supervisor.is_running(),
            "dashboard_file": (STATIC_DIR / "index.html").is_file(),
        }

    # ------------------------------------------------------------ websocket
    @app.websocket("/ws")
    async def ws_events(websocket: WebSocket, token: str = "") -> None:
        expected = supervisor.settings.web_api_token
        if not expected or token != expected:
            await websocket.close(code=4401)
            return
        await websocket.accept()
        q = supervisor.events.subscribe()
        # Sleduje odpojeni klienta. Bez toho by `q.get()` v pracovnim vlakne
        # zustalo viset navzdy a kazde odpojeni by unesalo jedno vlakno z
        # threadpoolu (a po vycerpani limitu by prestaly odpovidat i REST routy).
        receive_task = asyncio.create_task(websocket.receive())
        try:
            for event in supervisor.events.history():
                await websocket.send_json(event)
            while not receive_task.done():
                try:
                    event = await asyncio.to_thread(q.get, True, 1.0)
                except queue.Empty:
                    continue
                await websocket.send_json(event)
        except (WebSocketDisconnect, RuntimeError, ConnectionResetError):
            pass
        finally:
            receive_task.cancel()
            supervisor.events.unsubscribe(q)

    # dashboard (staticke soubory) - mountovano az po API routach, aby "/" jako
    # catch-all nezastinil "/api/..." a "/ws"
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="dashboard")

    return app
