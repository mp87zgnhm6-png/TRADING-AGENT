"""Hlavni orchestrator: spoji broker, data, model, strategii, risk a exekuci
do jednoho nepretrzite bezicicho, samoucicho se agenta.

Beh vlaken
----------
`StockDataStream.run()` z alpaca-py je blokujici volani, ktere si uvnitr
zaklada vlastni asyncio smycku (`asyncio.run`) - nelze ho tedy spustit jako
task uvnitr jine bezici smycky. Proto bezi ve vlastnim vlakne (`_stream_thread`)
a kazda prichozi svicka spousti `_on_bar` primo v tomto vlakne. Hlavni vlakno
procesu mezitim bezi periodickou udrzbovou smycku (`_maintenance_loop`):
kontroluje casove exity, sesynchronizuje pozice uzavrene brokerem (SL/TP) a
periodicky uklada stav modelu/banditu na disk. Obe vlakna sdileji stav pres
`self._lock`.
"""

from __future__ import annotations

import signal
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from alpaca.data.models.bars import Bar

from trading_agent.broker.alpaca_client import AlpacaBroker
from trading_agent.config import Settings
from trading_agent.data.bar_buffer import MultiSymbolBarStore
from trading_agent.data.storage import Storage
from trading_agent.execution.order_manager import OrderManager
from trading_agent.features.indicators import build_feature_row
from trading_agent.logging_setup import get_logger
from trading_agent.model.online_model import LabelQueue, OnlineDirectionModel
from trading_agent.risk.risk_manager import RiskManager
from trading_agent.strategy.bandit import EpsilonGreedyBandit
from trading_agent.strategy.signal_engine import ALL_ARMS, SignalEngine

logger = get_logger("agent")

HEARTBEAT_INTERVAL_SECONDS = 300


class AutonomousTradingAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.broker = AlpacaBroker(settings)
        self.storage = Storage(settings.db_path)
        self.bar_store = MultiSymbolBarStore(settings.symbols, maxlen=settings.bar_lookback)
        self.model = OnlineDirectionModel.load_or_create(settings.model_path)
        self.bandit = EpsilonGreedyBandit.load_or_create(settings.bandit_path, ALL_ARMS)
        self.signal_engine = SignalEngine(self.model, self.bandit)
        self.risk = RiskManager(settings, self.storage)
        self.orders = OrderManager(self.broker, self.storage, dry_run=settings.dry_run)
        self.label_queue = LabelQueue(settings.prediction_horizon_bars)

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._stream = None
        self._stream_thread: Optional[threading.Thread] = None
        self._last_persist = 0.0
        self._last_heartbeat = 0.0

    # ------------------------------------------------------------- setup
    def _log_account(self) -> None:
        account = self.broker.get_account()
        logger.info(
            "Ucet pripojen (%s): equity=%.2f cash=%.2f buying_power=%.2f PDT=%s daytrades=%s",
            "PAPER" if self.settings.alpaca_paper else "LIVE",
            float(account.equity), float(account.cash), float(account.buying_power),
            account.pattern_day_trader, account.daytrade_count,
        )
        self.storage.record_equity(float(account.equity), float(account.cash), float(account.buying_power))

    def _seed_history(self) -> None:
        logger.info("Stahuji historicka data pro %d symbolu (%s)...", len(self.settings.symbols), self.settings.symbols)
        lookback_days = max(10, (self.settings.bar_lookback * self.settings.timeframe_minutes) // 360 + 5)
        historical = self.broker.get_historical_bars(self.settings.symbols, lookback_days=lookback_days)
        self.bar_store.seed_from(historical)
        for symbol in self.settings.symbols:
            logger.info("  %s: %d svicek nacteno", symbol, len(self.bar_store.get(symbol)))

    def _last_known_price(self, symbol: str) -> Optional[float]:
        if symbol in self.bar_store.symbols():
            return self.bar_store.get(symbol).last_close
        return None

    # --------------------------------------------------------- live data
    async def _on_bar_async(self, bar: Bar) -> None:
        # subscribe_bars vyzaduje async handler; telo je synchronni a rychle,
        # takze nevadi, ze uvnitr nic neni `await`ovano.
        self._on_bar(bar)

    def _on_bar(self, bar: Bar) -> None:
        symbol = bar.symbol
        row = AlpacaBroker.bar_to_row(bar)
        with self._lock:
            if not self.bar_store.append(symbol, row):
                return  # duplicitni/starsi svicka
            try:
                self._process_new_bar(symbol, row["close"])
            except Exception:
                logger.exception("Chyba pri zpracovani nove svicky %s.", symbol)

    def _process_new_bar(self, symbol: str, close_price: float) -> None:
        feat, feat_row = build_feature_row(self.bar_store.dataframe(symbol))
        if feat is None:
            return  # jeste neni dost historie na spolehlive indikatory

        # --- samoucici krok: dolozit "label" vzorkum, kterym prave uplynul horizont ---
        for old_features, label in self.label_queue.advance_and_resolve(symbol, close_price):
            self.model.learn(old_features, label)
        self.label_queue.push(symbol, feat, close_price)

        signal = self.signal_engine.generate(symbol, feat)

        acted = False
        if (
            self.storage.get_open_trade(symbol) is None
            and signal.direction != 0
            and signal.confidence >= self.settings.min_confidence
            and (signal.direction > 0 or self.settings.allow_short)
        ):
            acted = self._try_enter(symbol, signal, close_price, feat_row)

        self.storage.record_signal(
            symbol, signal.strategy_used, signal.direction, signal.confidence, acted=acted, features=feat
        )

    def _try_enter(self, symbol: str, signal, close_price: float, feat_row) -> bool:
        try:
            account = self.broker.get_account()
            equity = float(account.equity)
        except Exception as exc:
            logger.error("Nepodarilo se ziskat equity uctu, vstup preskocen: %s", exc)
            return False

        open_positions_count = len(self.storage.get_all_open_trades())
        decision = self.risk.check_new_entry_allowed(equity, open_positions_count, self.broker.is_market_open())
        if not decision.allowed:
            logger.debug("Vstup do %s zamitnut risk managerem: %s", symbol, decision.reason)
            return False

        atr_value = float(feat_row["atr"])
        qty = self.risk.position_size(equity, close_price, atr_value)
        if qty <= 0:
            return False

        stop_price, take_profit_price = self.risk.stop_and_take_profit(close_price, atr_value, signal.direction)
        return self.orders.enter_position(
            symbol, signal.direction, qty, close_price, stop_price, take_profit_price,
            signal.strategy_used, signal.confidence,
        )

    # --------------------------------------------------------- udrzba
    def _check_time_based_exits(self) -> None:
        if self.settings.max_hold_bars <= 0:
            return
        max_hold_seconds = self.settings.max_hold_bars * self.settings.timeframe_minutes * 60
        now = datetime.now(timezone.utc)
        for trade in self.storage.get_all_open_trades():
            entry_time = datetime.fromisoformat(trade.entry_time)
            if (now - entry_time).total_seconds() < max_hold_seconds:
                continue
            exit_price = self._last_known_price(trade.symbol) or trade.entry_price
            result = self.orders.exit_position(trade.symbol, exit_price, "max_hold_time")
            if result:
                self.bandit.update(result[0], result[1])

    def _reconcile_closed_positions(self) -> None:
        """Detekuje pozice, ktere mezitim uzavrel bracket order primo na Alpaca
        (SL/TP), a promita jejich vysledek do lokalni evidence i do banditu -
        toto je hlavni cesta, kterou se bandit dozvi o vetsine vysledku obchodu."""
        open_trades = self.storage.get_all_open_trades()
        if not open_trades:
            return
        alpaca_symbols = {p.symbol for p in self.broker.get_positions()}
        for trade in open_trades:
            if trade.symbol in alpaca_symbols:
                continue
            exit_price = (
                self.orders.fill_price_from_order_id(trade.broker_order_id)
                or self._last_known_price(trade.symbol)
                or trade.take_profit_price
                or trade.entry_price
            )
            closed = self.storage.close_trade(trade.id, exit_price, "closed_by_broker")
            if closed:
                self.bandit.update(closed.strategy_name, closed.pnl_pct or 0.0)
                logger.info(
                    "Reconciliace: %s uzavren brokerem (strat=%s), PnL=%.2f (%.2f%%)",
                    trade.symbol, closed.strategy_name, closed.pnl, (closed.pnl_pct or 0.0) * 100,
                )

    def _maybe_persist(self) -> None:
        if time.monotonic() - self._last_persist < self.settings.persist_interval_minutes * 60:
            return
        self.model.save(self.settings.model_path)
        self.bandit.save(self.settings.bandit_path)
        self.storage.record_model_metric(self.model.rolling_score(), self.model.n_samples)
        self._last_persist = time.monotonic()
        logger.info("Stav modelu a banditu ulozen na disk.")

    def _heartbeat(self) -> None:
        try:
            account = self.broker.get_account()
            equity = float(account.equity)
            self.storage.record_equity(equity, float(account.cash), float(account.buying_power))
            logger.info(
                "Heartbeat: equity=%.2f otevrene=%d model_auc=%.3f (n=%d) bandit=%s",
                equity, len(self.storage.get_all_open_trades()), self.model.rolling_score(),
                self.model.n_samples, self.bandit.stats(),
            )
        except Exception as exc:
            logger.error("Heartbeat selhal: %s", exc)

    def _maintenance_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    self._check_time_based_exits()
                    self._reconcile_closed_positions()
                    self._maybe_persist()
                if time.monotonic() - self._last_heartbeat > HEARTBEAT_INTERVAL_SECONDS:
                    self._heartbeat()
                    self._last_heartbeat = time.monotonic()
            except Exception:
                logger.exception("Chyba v udrzbove smycce, pokracuji dal.")
            self._stop_event.wait(self.settings.loop_interval_seconds)

    # --------------------------------------------------------------- run
    def _run_stream(self) -> None:
        try:
            self._stream.run()
        except Exception:
            logger.exception("Streamovaci vlakno neocekavane spadlo.")
            self._stop_event.set()

    def _handle_signal(self, signum, frame) -> None:
        logger.info("Prijat signal %s, zahajuji bezpecne zastaveni...", signum)
        self._stop_event.set()

    def run(self) -> None:
        self.settings.ensure_directories()
        if not self.settings.live_trading_confirmed():
            raise RuntimeError(
                "ALPACA_PAPER=false, ale I_UNDERSTAND_LIVE_TRADING_RISK neni nastaveno na 'true'. "
                "Obchodovani naostro odmitnuto - nastavte promennou explicitne v .env, pokud tohle opravdu chcete."
            )

        logger.info(
            "Spoustim autonomniho tradingbota (%s, dry_run=%s, symboly=%s)",
            "PAPER" if self.settings.alpaca_paper else "LIVE", self.settings.dry_run, self.settings.symbols,
        )
        self._log_account()
        self.orders.sync_open_positions()
        self._seed_history()

        self._stream = self.broker.create_stream()
        self._stream.subscribe_bars(self._on_bar_async, *self.settings.symbols)
        self._stream_thread = threading.Thread(target=self._run_stream, name="alpaca-stream", daemon=True)
        self._stream_thread.start()

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        logger.info("Agent bezi. Pro bezpecne zastaveni stisknete Ctrl+C.")
        try:
            self._maintenance_loop()
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        logger.info("Ukoncuji agenta...")
        if self.settings.close_positions_on_shutdown:
            logger.info("CLOSE_POSITIONS_ON_SHUTDOWN=true - zaviram vsechny otevrene pozice na burze.")
            self.broker.close_all_positions(cancel_orders=True)
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                logger.debug("Chyba pri zastavovani streamu (pravdepodobne uz byl zastaveny).", exc_info=True)
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=5)
        with self._lock:
            self.model.save(self.settings.model_path)
            self.bandit.save(self.settings.bandit_path)
        self.storage.close()
        logger.info("Agent bezpecne zastaven, stav modelu/banditu ulozen.")
