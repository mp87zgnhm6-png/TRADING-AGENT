"""Rizeni zivotniho cyklu agenta z weboveho rozhrani.

Agent bezi na pozadi ve vlastnim vlakne; supervisor drzi jeho aktualni
instanci a dovoluje ji z FastAPI endpointu bezpecne start/stop/restart.
`EventBus`, `Storage` a `AlpacaBroker` na urovni supervisoru zustavaji zive
i kdyz je agent zastaveny, takze dashboard muze porad zobrazit historii
obchodu, aktualni pozice na uctu i nastaveni - jen se nezpracovavaji nove
svicky a neotviraji nove obchody.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

from trading_agent.agent import AutonomousTradingAgent
from trading_agent.broker.alpaca_client import AlpacaBroker
from trading_agent.config import Settings, apply_settings_update
from trading_agent.data.storage import Storage
from trading_agent.events import EventBus

logger = logging.getLogger("trading_agent.webapp.supervisor")


class AgentSupervisor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.events = EventBus()
        self.storage = Storage(settings.db_path)
        self.broker = AlpacaBroker(settings)
        self._agent: Optional[AutonomousTradingAgent] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_error: Optional[str] = None

    @property
    def agent(self) -> Optional[AutonomousTradingAgent]:
        return self._agent

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict[str, Any]:
        return {
            "running": self.is_running(),
            "mode": "PAPER" if self.settings.alpaca_paper else "LIVE",
            "dry_run": self.settings.dry_run,
            "symbols": self.settings.symbols,
            "kill_switch_active": Path(self.settings.kill_switch_file).exists(),
            "last_error": self._last_error,
        }

    def start(self) -> tuple[bool, str]:
        with self._lock:
            if self.is_running():
                return False, "Agent uz bezi."
            if not self.settings.alpaca_api_key or not self.settings.alpaca_secret_key:
                return False, "Chybi ALPACA_API_KEY / ALPACA_SECRET_KEY - doplnte je v nastaveni."
            if not self.settings.live_trading_confirmed():
                return False, (
                    "ALPACA_PAPER=false, ale I_UNDERSTAND_LIVE_TRADING_RISK neni potvrzeno - "
                    "obchodovani naostro je zablokovane."
                )
            self._last_error = None
            agent = AutonomousTradingAgent(self.settings, event_bus=self.events)
            thread = threading.Thread(target=self._run, args=(agent,), name="agent-supervisor", daemon=True)
            self._agent = agent
            self._thread = thread
            thread.start()
            return True, "Agent spusten."

    def _run(self, agent: AutonomousTradingAgent) -> None:
        try:
            agent.run(install_signal_handlers=False)
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("Agent skoncil chybou.")
            self.events.publish({"type": "agent_status", "status": "error", "error": str(exc)})

    def stop(self, timeout: float = 20.0) -> tuple[bool, str]:
        with self._lock:
            agent, thread = self._agent, self._thread
            if agent is None or thread is None or not thread.is_alive():
                return False, "Agent uz nebezi."
            agent.request_stop()
            thread.join(timeout=timeout)
            still_alive = thread.is_alive()
            self._agent = None
            self._thread = None
            if still_alive:
                return False, "Agent neodpovedel v casovem limitu, pokracuje na pozadi."
            return True, "Agent zastaven."

    def restart(self) -> tuple[bool, str]:
        self.stop()
        # cold nastaveni (API klice, symboly, timeframe, data feed, ...) mohla byt
        # mezitim zmenena - broker se rekonstruuje s aktualnimi hodnotami
        self.broker = AlpacaBroker(self.settings)
        return self.start()

    def update_settings(self, updates: dict[str, Any]) -> Settings:
        return apply_settings_update(self.settings, updates)

    def shutdown(self) -> None:
        self.stop()
        self.storage.close()
