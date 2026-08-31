"""Risk management: velikost pozice, stop/take-profit, denni a drawdown pojistky.

Toto je jedina cast agenta, ktera smi rict "ne" - zadny signal, byt sebejistejsi,
neobejde limity nastavene tady. Vse je zalozene na ATR (volatilite), aby se
velikost pozice i vzdalenost stopu automaticky prizpusobovaly aktualnimu
kolisani daneho symbolu.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from trading_agent.config import Settings
from trading_agent.data.storage import Storage
from trading_agent.logging_setup import get_logger

logger = get_logger("risk")

_EASTERN = ZoneInfo("America/New_York")


def trading_day_start_utc_iso(now: datetime | None = None) -> str:
    """Zacatek dnesniho obchodniho dne (pulnoc v New Yorku) jako UTC ISO retezec."""
    now = now or datetime.now(timezone.utc)
    eastern_now = now.astimezone(_EASTERN)
    day_start_eastern = eastern_now.replace(hour=0, minute=0, second=0, microsecond=0)
    return day_start_eastern.astimezone(timezone.utc).isoformat()


@dataclass
class RiskDecision:
    allowed: bool
    reason: str = ""


class RiskManager:
    def __init__(self, settings: Settings, storage: Storage):
        self.settings = settings
        self.storage = storage

    # ------------------------------------------------------------- sizing
    def position_size(self, equity: float, price: float, atr: float) -> int:
        """Fixed-fractional sizing: risknout `risk_per_trade` % equity na stop_distance,
        omezeno shora `max_position_pct` podilem equity v jednom symbolu."""
        if price <= 0 or atr <= 0 or equity <= 0:
            return 0
        risk_amount = equity * self.settings.risk_per_trade
        stop_distance = atr * self.settings.atr_stop_multiplier
        if stop_distance <= 0:
            return 0
        qty_by_risk = risk_amount / stop_distance
        qty_by_cap = (equity * self.settings.max_position_pct) / price
        return max(math.floor(min(qty_by_risk, qty_by_cap)), 0)

    def stop_and_take_profit(self, entry_price: float, atr: float, direction: int) -> tuple[float, float]:
        stop_distance = atr * self.settings.atr_stop_multiplier
        tp_distance = atr * self.settings.atr_take_profit_multiplier
        if direction > 0:
            return entry_price - stop_distance, entry_price + tp_distance
        return entry_price + stop_distance, entry_price - tp_distance

    # -------------------------------------------------------------- guardy
    def kill_switch_active(self) -> bool:
        return Path(self.settings.kill_switch_file).exists()

    def check_drawdown_circuit_breaker(self, current_equity: float) -> RiskDecision:
        peak = self.storage.get_peak_equity()
        if peak is None or peak <= 0:
            return RiskDecision(True)
        drawdown = (peak - current_equity) / peak
        if drawdown >= self.settings.max_drawdown_pct:
            return RiskDecision(
                False, f"drawdown {drawdown:.2%} >= limit {self.settings.max_drawdown_pct:.2%} (circuit breaker)"
            )
        return RiskDecision(True)

    def check_daily_loss_limit(self, current_equity: float) -> RiskDecision:
        if current_equity <= 0:
            return RiskDecision(True)
        day_start = trading_day_start_utc_iso()
        daily_pnl = self.storage.get_daily_pnl(day_start)
        loss_pct = -daily_pnl / current_equity
        if loss_pct >= self.settings.daily_loss_limit_pct:
            return RiskDecision(
                False, f"denni ztrata {loss_pct:.2%} >= limit {self.settings.daily_loss_limit_pct:.2%}"
            )
        return RiskDecision(True)

    def check_new_entry_allowed(
        self,
        current_equity: float,
        open_positions_count: int,
        market_open: bool,
    ) -> RiskDecision:
        """Souhrnna kontrola pred otevrenim JAKEKOLIV nove pozice."""
        if self.kill_switch_active():
            return RiskDecision(False, f"kill-switch soubor '{self.settings.kill_switch_file}' existuje")
        if not market_open:
            return RiskDecision(False, "trh je aktualne zavreny")
        if open_positions_count >= self.settings.max_open_positions:
            return RiskDecision(False, f"dosazen limit max_open_positions={self.settings.max_open_positions}")

        drawdown_check = self.check_drawdown_circuit_breaker(current_equity)
        if not drawdown_check.allowed:
            return drawdown_check

        daily_check = self.check_daily_loss_limit(current_equity)
        if not daily_check.allowed:
            return daily_check

        return RiskDecision(True)
