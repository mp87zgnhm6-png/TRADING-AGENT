"""Tenka vrstva nad alpaca-py - jedine misto, kde agent mluvi s burzou Alpaca.

Zamerne obaluje REST i streaming klienty jedne knihovny (alpaca-py), aby zbytek
agenta nikdy nepracoval s Alpaca SDK primo. Diky tomu je cely system napojeny
vyhradne na Alpaca a pripadna zmena/aktualizace SDK se resi na jednom miste.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.live.stock import StockDataStream
from alpaca.data.models.bars import Bar
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.models import Clock, Order, Position, TradeAccount
from alpaca.trading.requests import (
    ClosePositionRequest,
    GetOrdersRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from trading_agent.config import RUNTIME_OVERLAY_PATH, Settings
from trading_agent.logging_setup import get_logger

logger = get_logger("broker")

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _feed(settings: Settings) -> DataFeed:
    try:
        return DataFeed(settings.data_feed.lower())
    except ValueError:
        logger.warning("Neznamy DATA_FEED=%s, pouzivam IEX.", settings.data_feed)
        return DataFeed.IEX


def _mask_key(key: str) -> str:
    if not key:
        return "(prazdny)"
    return f"{key[:4]}...{key[-4:]} (delka {len(key)})" if len(key) > 10 else f"(delka {len(key)})"


def credentials_error_hint(exc: APIError, settings: Settings) -> str:
    """Prelozi odmitnuti od Alpaca API na konkretni navod, co zkontrolovat.

    401/403 na /v2/account neni vypadek site ani docasna chyba - jsou to spatne
    pristupove udaje, a nejcastejsi pricinou je zamena PAPER a LIVE klicu.
    """
    status = getattr(exc, "status_code", None)
    if status not in (401, 403):
        return f"Alpaca API odmitla overeni uctu: {exc}"

    account_type = "paper" if settings.alpaca_paper else "live"
    endpoint = "paper-api.alpaca.markets" if settings.alpaca_paper else "api.alpaca.markets"
    lines = [
        f"Alpaca odmitla pristupove udaje (HTTP {status}) na {endpoint}.",
        f"Pouzity API key: {_mask_key(settings.alpaca_api_key)}, "
        f"secret: {'vyplnen' if settings.alpaca_secret_key else 'PRAZDNY'}.",
        "Co zkontrolovat:",
        f"  1) ALPACA_PAPER={str(settings.alpaca_paper).lower()}, takze musite pouzit klice vygenerovane "
        f"pro {account_type.upper()} ucet - paper a live ucet maji ODLISNE klice "
        "(paper klice zpravidla zacinaji 'PK', live 'AK').",
        "  2) Key i secret jsou zkopirovane cele, bez mezer a uvozovek (secret Alpaca zobrazi jen jednou).",
        "  3) Klice nebyly na app.alpaca.markets regenerovane nebo smazane.",
        f"  4) Hodnoty ulozene z dashboardu ({RUNTIME_OVERLAY_PATH}) maji prednost pred .env - "
        "pokud jste klice zadavali tam, opravte je tamtez, nebo soubor smazte.",
    ]
    return "\n".join(lines)


class AlpacaBroker:
    """Obaluje TradingClient (ordery/ucet) a StockHistoricalDataClient/StockDataStream (data)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._feed = _feed(settings)
        self.trading = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=settings.alpaca_paper,
        )
        self.data = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )
        self.timeframe = TimeFrame(settings.timeframe_minutes, TimeFrameUnit.Minute)

    # ------------------------------------------------------------------ ucet
    def get_account(self) -> TradeAccount:
        return self.trading.get_account()

    def get_clock(self) -> Clock:
        return self.trading.get_clock()

    def is_market_open(self) -> bool:
        try:
            return bool(self.get_clock().is_open)
        except APIError as exc:
            logger.error("Nepodarilo se zjistit stav trhu: %s", exc)
            return False

    # -------------------------------------------------------------- pozice
    def get_positions(self) -> list[Position]:
        return list(self.trading.get_all_positions())

    def get_position(self, symbol: str) -> Optional[Position]:
        try:
            return self.trading.get_open_position(symbol)
        except APIError:
            return None

    def close_position(self, symbol: str, qty: Optional[float] = None) -> Optional[Order]:
        opts = ClosePositionRequest(qty=str(qty)) if qty else None
        try:
            return self.trading.close_position(symbol, close_options=opts)
        except APIError as exc:
            logger.error("Zavreni pozice %s selhalo: %s", symbol, exc)
            return None

    def close_all_positions(self, cancel_orders: bool = True) -> None:
        try:
            self.trading.close_all_positions(cancel_orders=cancel_orders)
        except APIError as exc:
            logger.error("Hromadne zavreni pozic selhalo: %s", exc)

    # --------------------------------------------------------------- ordery
    def get_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        filt = GetOrdersRequest(status="open", symbols=[symbol] if symbol else None)
        return list(self.trading.get_orders(filt))

    def get_recent_orders(self, limit: int = 50) -> list[Order]:
        filt = GetOrdersRequest(status="all", limit=limit, direction="desc")
        return list(self.trading.get_orders(filt))

    def cancel_all_orders(self) -> None:
        try:
            self.trading.cancel_orders()
        except APIError as exc:
            logger.error("Zruseni vsech orderu selhalo: %s", exc)

    def submit_bracket_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        stop_loss_price: float,
        take_profit_price: float,
    ) -> Optional[Order]:
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            extended_hours=False,
            take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 2)),
            stop_loss=StopLossRequest(stop_price=round(stop_loss_price, 2)),
        )
        try:
            return self.trading.submit_order(request)
        except APIError as exc:
            logger.error("Bracket order %s %s x%s selhal: %s", side, symbol, qty, exc)
            return None

    def submit_market_order(self, symbol: str, qty: float, side: OrderSide) -> Optional[Order]:
        request = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY)
        try:
            return self.trading.submit_order(request)
        except APIError as exc:
            logger.error("Market order %s %s x%s selhal: %s", side, symbol, qty, exc)
            return None

    # ------------------------------------------------------------------ data
    def get_historical_bars(
        self,
        symbols: list[str],
        lookback_days: int = 30,
        end: Optional[datetime] = None,
    ) -> dict[str, pd.DataFrame]:
        """Stahne historicke svicky pro dany seznam symbolu. Vraci {symbol: DataFrame(OHLCV)}."""

        end_dt = end or datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=lookback_days)
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=self.timeframe,
            start=start_dt,
            end=end_dt,
            feed=self._feed,
            adjustment="raw",
        )
        try:
            barset = self.data.get_stock_bars(request)
        except APIError as exc:
            logger.error("Stahovani historickych dat selhalo: %s", exc)
            return {}

        df_all = barset.df
        if df_all is None or df_all.empty:
            return {}

        result: dict[str, pd.DataFrame] = {}
        if isinstance(df_all.index, pd.MultiIndex):
            available = df_all.index.get_level_values(0).unique()
            for symbol in symbols:
                if symbol in available:
                    sdf = df_all.xs(symbol, level=0).copy()
                    sdf.index.name = "timestamp"
                    result[symbol] = sdf[OHLCV_COLUMNS]
        else:
            # jediny symbol vraci plochy DataFrame
            result[symbols[0]] = df_all[OHLCV_COLUMNS].copy()
        return result

    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        """Posledni obchodovana cena pro kazdy symbol - pro rychlou kontrolu,
        co si lze za dostupny kapital vubec dovolit."""
        try:
            request = StockLatestTradeRequest(symbol_or_symbols=symbols, feed=self._feed)
            trades = self.data.get_stock_latest_trade(request)
        except APIError as exc:
            logger.error("Nepodarilo se ziskat aktualni ceny: %s", exc)
            return {}
        return {symbol: float(trade.price) for symbol, trade in trades.items() if trade is not None}

    # -------------------------------------------------------------- stream
    def create_stream(self) -> StockDataStream:
        return StockDataStream(
            api_key=self.settings.alpaca_api_key,
            secret_key=self.settings.alpaca_secret_key,
            feed=self._feed,
        )

    @staticmethod
    def bar_to_row(bar: Bar) -> dict:
        return {
            "timestamp": pd.Timestamp(bar.timestamp),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
        }
