"""Jednoduchy in-process pub/sub pro zivy prenos udalosti agenta do dashboardu.

Agent bezi ve vlastnich vlaknech (viz `trading_agent/agent.py`), dashboard bezi
na FastAPI/asyncio ve stejnem procesu. `EventBus` je most mezi nimi: agent
vola `publish()` (synchronni, volatelne z libovolneho vlakna), websocket
handler na strane FastAPI odebira udalosti pres `subscribe()` a posila je
pripojenym prohlizecum.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any


class EventBus:
    def __init__(self, history_size: int = 200):
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._history: deque[dict[str, Any]] = deque(maxlen=history_size)

    def publish(self, event: dict[str, Any]) -> None:
        payload = {"ts": datetime.now(timezone.utc).isoformat(), **event}
        with self._lock:
            self._history.append(payload)
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass  # pomaly/odpojeny klient - radsi zahodit nez blokovat agenta

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)
