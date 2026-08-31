"""SQLite uloziste - historie obchodu, signalu, equity krivky a metrik modelu.

Je to jediny trvaly "pamet" agenta krome souboru s ulozenym modelem/banditem.
Vse beží v jednom souboru (db_path), pristup je serializovan zamkem, protoze
k nemu pristupuje jak streamovaci vlakno, tak hlavni smycka.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    stop_price REAL,
    take_profit_price REAL,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    pnl REAL,
    pnl_pct REAL,
    exit_reason TEXT,
    strategy_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    broker_order_id TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    direction INTEGER NOT NULL,
    confidence REAL NOT NULL,
    acted INTEGER NOT NULL DEFAULT 0,
    features_json TEXT
);

CREATE TABLE IF NOT EXISTS equity_curve (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    equity REAL NOT NULL,
    cash REAL,
    buying_power REAL
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    rolling_metric REAL,
    n_samples INTEGER
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_status ON trades(symbol, status);
CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON equity_curve(timestamp);
"""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TradeRecord:
    id: int
    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: Optional[float]
    stop_price: Optional[float]
    take_profit_price: Optional[float]
    entry_time: str
    exit_time: Optional[str]
    pnl: Optional[float]
    pnl_pct: Optional[float]
    exit_reason: Optional[str]
    strategy_name: str
    confidence: float
    status: str
    broker_order_id: Optional[str] = None
    extra: dict = field(default_factory=dict)


class Storage:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    # ------------------------------------------------------------- trades
    def open_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        stop_price: float,
        take_profit_price: float,
        strategy_name: str,
        confidence: float,
        entry_time: Optional[str] = None,
        broker_order_id: Optional[str] = None,
    ) -> int:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO trades
                   (symbol, side, qty, entry_price, stop_price, take_profit_price,
                    entry_time, strategy_name, confidence, status, broker_order_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)""",
                (
                    symbol,
                    side,
                    qty,
                    entry_price,
                    stop_price,
                    take_profit_price,
                    entry_time or _utcnow_iso(),
                    strategy_name,
                    confidence,
                    broker_order_id,
                ),
            )
            return int(cur.lastrowid)

    def get_open_trade(self, symbol: str) -> Optional[TradeRecord]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM trades WHERE symbol = ? AND status = 'open' ORDER BY id DESC LIMIT 1",
                (symbol,),
            )
            row = cur.fetchone()
            return self._row_to_trade(row) if row else None

    def get_all_open_trades(self) -> list[TradeRecord]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM trades WHERE status = 'open'")
            return [self._row_to_trade(r) for r in cur.fetchall()]

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str,
        exit_time: Optional[str] = None,
    ) -> Optional[TradeRecord]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
            row = cur.fetchone()
            if row is None:
                return None
            entry_price = row["entry_price"]
            qty = row["qty"]
            side = row["side"]
            direction = 1 if side == "buy" else -1
            pnl = (exit_price - entry_price) * qty * direction
            pnl_pct = ((exit_price - entry_price) / entry_price) * direction if entry_price else 0.0
            cur.execute(
                """UPDATE trades SET exit_price=?, exit_time=?, pnl=?, pnl_pct=?,
                   exit_reason=?, status='closed' WHERE id=?""",
                (exit_price, exit_time or _utcnow_iso(), pnl, pnl_pct, exit_reason, trade_id),
            )
            cur.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
            return self._row_to_trade(cur.fetchone())

    def get_recent_closed_trades(self, limit: int = 100) -> list[TradeRecord]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM trades WHERE status='closed' ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [self._row_to_trade(r) for r in cur.fetchall()]

    @staticmethod
    def _row_to_trade(row: sqlite3.Row) -> TradeRecord:
        return TradeRecord(
            id=row["id"],
            symbol=row["symbol"],
            side=row["side"],
            qty=row["qty"],
            entry_price=row["entry_price"],
            exit_price=row["exit_price"],
            stop_price=row["stop_price"],
            take_profit_price=row["take_profit_price"],
            entry_time=row["entry_time"],
            exit_time=row["exit_time"],
            pnl=row["pnl"],
            pnl_pct=row["pnl_pct"],
            exit_reason=row["exit_reason"],
            strategy_name=row["strategy_name"],
            confidence=row["confidence"],
            status=row["status"],
            broker_order_id=row["broker_order_id"] if "broker_order_id" in row.keys() else None,
        )

    # ------------------------------------------------------------ signaly
    def record_signal(
        self,
        symbol: str,
        strategy_name: str,
        direction: int,
        confidence: float,
        acted: bool,
        features: Optional[dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO signals (timestamp, symbol, strategy_name, direction,
                   confidence, acted, features_json) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    timestamp or _utcnow_iso(),
                    symbol,
                    strategy_name,
                    direction,
                    confidence,
                    int(acted),
                    json.dumps(features) if features else None,
                ),
            )

    def get_recent_signals(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
        return [
            {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "symbol": r["symbol"],
                "strategy_name": r["strategy_name"],
                "direction": r["direction"],
                "confidence": r["confidence"],
                "acted": bool(r["acted"]),
            }
            for r in rows
        ]

    # -------------------------------------------------------------- equity
    def record_equity(self, equity: float, cash: float, buying_power: float) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO equity_curve (timestamp, equity, cash, buying_power) VALUES (?, ?, ?, ?)",
                (_utcnow_iso(), equity, cash, buying_power),
            )

    def get_peak_equity(self) -> Optional[float]:
        with self._cursor() as cur:
            cur.execute("SELECT MAX(equity) AS peak FROM equity_curve")
            row = cur.fetchone()
            return row["peak"] if row and row["peak"] is not None else None

    def get_latest_equity(self) -> Optional[float]:
        with self._cursor() as cur:
            cur.execute("SELECT equity FROM equity_curve ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            return row["equity"] if row else None

    def get_equity_at_or_before(self, iso_timestamp: str) -> Optional[float]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT equity FROM equity_curve WHERE timestamp <= ? ORDER BY timestamp DESC LIMIT 1",
                (iso_timestamp,),
            )
            row = cur.fetchone()
            return row["equity"] if row else None

    def get_equity_curve(self, limit: int = 500) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT timestamp, equity, cash, buying_power FROM equity_curve ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in reversed(rows)]  # chronologicky (nejstarsi prvni)

    def get_daily_pnl(self, day_start_iso: str) -> float:
        """Realizovany PnL z obchodu uzavrenych od zacatku dnesniho obchodniho dne."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT COALESCE(SUM(pnl), 0) AS s FROM trades WHERE status='closed' AND exit_time >= ?",
                (day_start_iso,),
            )
            row = cur.fetchone()
            return float(row["s"]) if row else 0.0

    # -------------------------------------------------------- model metriky
    def record_model_metric(self, rolling_metric: float, n_samples: int) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO model_metrics (timestamp, rolling_metric, n_samples) VALUES (?, ?, ?)",
                (_utcnow_iso(), rolling_metric, n_samples),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
