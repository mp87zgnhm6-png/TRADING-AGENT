"""Rolling buffer poslednich N svicek pro kazdy sledovany symbol."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

import pandas as pd


class SymbolBarBuffer:
    """Drzi posledních `maxlen` OHLCV svicek jednoho symbolu, serazenych podle casu."""

    def __init__(self, symbol: str, maxlen: int):
        self.symbol = symbol
        self.maxlen = maxlen
        self._rows: deque[dict] = deque(maxlen=maxlen)
        self._last_ts: Optional[pd.Timestamp] = None

    def seed(self, df: pd.DataFrame) -> None:
        """Naplni buffer historickymi daty (DataFrame indexovany timestampem, sloupce OHLCV)."""
        self._rows.clear()
        for ts, row in df.tail(self.maxlen).iterrows():
            self._rows.append(
                {
                    "timestamp": pd.Timestamp(ts),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )
        if self._rows:
            self._last_ts = self._rows[-1]["timestamp"]

    def append(self, row: dict) -> bool:
        """Prida novou svicku. Vraci False pokud jde o duplicitu/starsi svicku (ignorovano)."""
        ts = pd.Timestamp(row["timestamp"])
        if self._last_ts is not None and ts <= self._last_ts:
            return False
        self._rows.append(
            {
                "timestamp": ts,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )
        self._last_ts = ts
        return True

    def to_dataframe(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(list(self._rows)).set_index("timestamp")
        return df

    @property
    def last_close(self) -> Optional[float]:
        return self._rows[-1]["close"] if self._rows else None

    def __len__(self) -> int:
        return len(self._rows)


class MultiSymbolBarStore:
    """Spravuje jeden SymbolBarBuffer pro kazdy symbol z watchlistu."""

    def __init__(self, symbols: Iterable[str], maxlen: int):
        self._buffers = {s: SymbolBarBuffer(s, maxlen) for s in symbols}

    def seed_from(self, historical: dict[str, pd.DataFrame]) -> None:
        for symbol, df in historical.items():
            if symbol in self._buffers and not df.empty:
                self._buffers[symbol].seed(df)

    def append(self, symbol: str, row: dict) -> bool:
        if symbol not in self._buffers:
            return False
        return self._buffers[symbol].append(row)

    def get(self, symbol: str) -> SymbolBarBuffer:
        return self._buffers[symbol]

    def dataframe(self, symbol: str) -> pd.DataFrame:
        return self._buffers[symbol].to_dataframe()

    def ready(self, symbol: str, min_bars: int) -> bool:
        return len(self._buffers.get(symbol, [])) >= min_bars

    def symbols(self) -> list[str]:
        return list(self._buffers.keys())
