"""Provadeni obchodu: otevirani/zavirani pozic pres Alpaca a udrzovani lokalni evidence.

Kazdy vstup se posila jako bracket order (trh + stop-loss + take-profit v jedne
zadosti), takze ochranu pozice zajistuje primo burza, i kdyby agent docasne
vypadl. V DRY_RUN rezimu se nic neposila na Alpaca, jen se simuluje a loguje -
dalsi bezpecnostni sit navic k paper uctu.
"""

from __future__ import annotations

from typing import Optional

from alpaca.trading.enums import OrderSide

from trading_agent.broker.alpaca_client import AlpacaBroker
from trading_agent.config import Settings
from trading_agent.data.storage import Storage
from trading_agent.logging_setup import get_logger

logger = get_logger("execution")


class OrderManager:
    def __init__(self, broker: AlpacaBroker, storage: Storage, settings: Settings):
        self.broker = broker
        self.storage = storage
        self.settings = settings

    @property
    def dry_run(self) -> bool:
        # cteno primo z (sdileneho) Settings objektu, aby prepnuti DRY_RUN
        # z webove dashboardu zabralo okamzite, bez restartu agenta
        return self.settings.dry_run

    def enter_position(
        self,
        symbol: str,
        direction: int,
        qty: int,
        entry_price: float,
        stop_price: float,
        take_profit_price: float,
        strategy_name: str,
        confidence: float,
    ) -> bool:
        if qty <= 0 or direction == 0:
            return False
        side_str = "buy" if direction > 0 else "sell"

        if self.dry_run:
            logger.info(
                "[DRY_RUN] %s %s x%d @ ~%.2f (SL=%.2f TP=%.2f, strat=%s conf=%.2f)",
                side_str, symbol, qty, entry_price, stop_price, take_profit_price, strategy_name, confidence,
            )
            self.storage.open_trade(
                symbol, side_str, qty, entry_price, stop_price, take_profit_price, strategy_name, confidence
            )
            return True

        side = OrderSide.BUY if direction > 0 else OrderSide.SELL
        order = self.broker.submit_bracket_order(symbol, qty, side, stop_price, take_profit_price)
        if order is None:
            return False

        fill_price = float(order.filled_avg_price) if order.filled_avg_price else entry_price
        self.storage.open_trade(
            symbol,
            side_str,
            qty,
            fill_price,
            stop_price,
            take_profit_price,
            strategy_name,
            confidence,
            broker_order_id=str(order.id),
        )
        logger.info(
            "Otevrena pozice %s %s x%d @ %.2f (strat=%s, conf=%.2f)",
            side_str, symbol, qty, fill_price, strategy_name, confidence,
        )
        return True

    def exit_position(self, symbol: str, exit_price: float, reason: str) -> Optional[tuple[str, float]]:
        """Uzavre otevreny obchod (na burze i v uloziste). Vraci (strategy_name,
        realized_pnl_pct) pro zpetnou vazbu banditu, nebo None pokud nic nebylo otevrene."""
        open_trade = self.storage.get_open_trade(symbol)
        if open_trade is None:
            return None

        if not self.dry_run:
            self.broker.close_position(symbol)

        closed = self.storage.close_trade(open_trade.id, exit_price, reason)
        if closed is None:
            return None
        logger.info(
            "Uzavrena pozice %s (%s): PnL=%.2f (%.2f%%), duvod=%s",
            symbol, closed.strategy_name, closed.pnl, (closed.pnl_pct or 0.0) * 100, reason,
        )
        return closed.strategy_name, closed.pnl_pct or 0.0

    # -------------------------------------------------------- reconciliace
    def fill_price_from_order_id(self, order_id: Optional[str]) -> Optional[float]:
        if not order_id:
            return None
        try:
            order = self.broker.trading.get_order_by_id(order_id)
        except Exception as exc:
            # Sit navic: jakakoli chyba (spatny/API vyprsely order, vypadek site, ...)
            # nesmi shodit reconciliaci ostatnich pozic v tehle iteraci - jen se
            # spolehneme na fallback cenu u volajiciho.
            logger.warning("Nepodarilo se dohledat fill order %s: %s", order_id, exc)
            return None
        if order.filled_avg_price:
            return float(order.filled_avg_price)
        for leg in order.legs or []:
            if leg.filled_avg_price:
                return float(leg.filled_avg_price)
        return None

    def sync_open_positions(self) -> None:
        """Po (re)startu srovna skutecny stav uctu na Alpaca s lokalni databazi."""
        alpaca_positions = {p.symbol: p for p in self.broker.get_positions()}
        local_open = {t.symbol: t for t in self.storage.get_all_open_trades()}

        for symbol, pos in alpaca_positions.items():
            if symbol not in local_open:
                logger.warning(
                    "Pozice %s existuje na Alpaca, ale ne v lokalni databazi - dodatecne ji zaznamenavam "
                    "(bez zname puvodni strategie).",
                    symbol,
                )
                qty = abs(float(pos.qty))
                price = float(pos.avg_entry_price)
                side = "buy" if float(pos.qty) > 0 else "sell"
                self.storage.open_trade(
                    symbol, side, qty, price, price * 0.95, price * 1.05, "reconciled", 0.5
                )

        for symbol, trade in local_open.items():
            if symbol not in alpaca_positions:
                exit_price = (
                    self.fill_price_from_order_id(trade.broker_order_id)
                    or trade.take_profit_price
                    or trade.entry_price
                )
                logger.info(
                    "Pozice %s uz na Alpaca neni otevrena (pravdepodobne splnen SL/TP behem odstavky) - "
                    "uzaviram v db za odhadovanou cenu %.2f.",
                    symbol, exit_price,
                )
                self.storage.close_trade(trade.id, exit_price, "reconciled_on_startup")
