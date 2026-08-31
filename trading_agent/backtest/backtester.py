"""Event-driven, walk-forward backtest nad historickymi daty z Alpaca.

Pouziva PRESNE stejnou logiku jako zivy agent (indikatory, online model,
bandit, pravidlove strategie, ATR risk sizing) a bar po baru ji prehrava
chronologicky - vcetne toho, ze se online model bezi ucit uz behem backtestu.
Vysledek tak realisticky ukazuje, jak by se samoucici agent choval, kdyby
tehdy bezel naostro (nejde o zpetne "nafitovani" na cela data najednou).

Zjednoduseni oproti realnemu obchodovani: bez poplatku/skluzu (Alpaca akcie
jsou bez provizi) a SL/TP se vyhodnocuji proti high/low kazde svicky
(pesimisticky - nejdriv se kontroluje stop-loss, pak az take-profit).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_agent.config import Settings
from trading_agent.data.bar_buffer import MultiSymbolBarStore
from trading_agent.features.indicators import build_feature_row
from trading_agent.model.online_model import LabelQueue, OnlineDirectionModel
from trading_agent.strategy.bandit import EpsilonGreedyBandit
from trading_agent.strategy.signal_engine import ALL_ARMS, SignalEngine

BARS_PER_YEAR_6H30 = 252 * 6.5 * 60  # obchodnich minut za rok (US akciovy trh)


@dataclass
class BacktestTrade:
    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    pnl: float
    pnl_pct: float
    strategy: str
    exit_reason: str


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[BacktestTrade]
    stats: dict


def compute_stats(
    equity: pd.Series, trades: list[BacktestTrade], initial_equity: float, timeframe_minutes: int
) -> dict:
    if equity.empty:
        return {}

    final_equity = float(equity.iloc[-1])
    total_return_pct = (final_equity / initial_equity - 1.0) * 100.0

    bars_per_year = BARS_PER_YEAR_6H30 / max(timeframe_minutes, 1)
    returns = equity.pct_change().dropna()
    sharpe = float(returns.mean() / returns.std() * np.sqrt(bars_per_year)) if returns.std() > 0 else 0.0

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown_pct = float(drawdown.min()) * 100.0

    years = len(equity) / bars_per_year if bars_per_year > 0 else 0.0
    cagr_pct = ((final_equity / initial_equity) ** (1.0 / years) - 1.0) * 100.0 if years > 0 and final_equity > 0 else 0.0

    n_trades = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate_pct = (len(wins) / n_trades * 100.0) if n_trades else 0.0
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    return {
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "cagr_pct": cagr_pct,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "num_trades": n_trades,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
    }


class Backtester:
    def __init__(self, settings: Settings, initial_equity: float = 100_000.0, seed: int = 42):
        self.settings = settings
        self.initial_equity = initial_equity
        self.model = OnlineDirectionModel(seed=seed)
        self.bandit = EpsilonGreedyBandit(ALL_ARMS, seed=seed)
        self.signal_engine = SignalEngine(self.model, self.bandit)
        self.label_queue = LabelQueue(settings.prediction_horizon_bars)

    def run(self, historical: dict[str, pd.DataFrame]) -> BacktestResult:
        settings = self.settings
        frames = {s: df.sort_index() for s, df in historical.items() if len(df) >= 40}
        if not frames:
            raise ValueError("Nedostatek historickych dat pro backtest (min. 40 svicek na symbol).")

        bar_store = MultiSymbolBarStore(frames.keys(), maxlen=settings.bar_lookback)
        all_ts = sorted(set().union(*(set(df.index) for df in frames.values())))

        realized_pnl = 0.0
        open_positions: dict[str, dict] = {}
        trades: list[BacktestTrade] = []
        equity_points: list[tuple[pd.Timestamp, float]] = []
        latest_close: dict[str, float] = {}

        def unrealized_pnl() -> float:
            total = 0.0
            for symbol, pos in open_positions.items():
                price = latest_close.get(symbol, pos["entry_price"])
                direction = 1 if pos["side"] == "buy" else -1
                total += (price - pos["entry_price"]) * pos["qty"] * direction
            return total

        for ts in all_ts:
            for symbol, df in frames.items():
                if ts not in df.index:
                    continue
                row = df.loc[ts]
                bar_store.append(symbol, {"timestamp": ts, **{c: float(row[c]) for c in ("open", "high", "low", "close", "volume")}})

                bar_close = float(row["close"])
                bar_high = float(row["high"])
                bar_low = float(row["low"])
                latest_close[symbol] = bar_close

                # 1) sprava existujici pozice - kontrola SL/TP proti high/low teto svicky
                if symbol in open_positions:
                    pos = open_positions[symbol]
                    exit_price, reason = None, None
                    if pos["side"] == "buy":
                        if bar_low <= pos["stop"]:
                            exit_price, reason = pos["stop"], "stop_loss"
                        elif bar_high >= pos["tp"]:
                            exit_price, reason = pos["tp"], "take_profit"
                    else:
                        if bar_high >= pos["stop"]:
                            exit_price, reason = pos["stop"], "stop_loss"
                        elif bar_low <= pos["tp"]:
                            exit_price, reason = pos["tp"], "take_profit"

                    if exit_price is not None:
                        direction = 1 if pos["side"] == "buy" else -1
                        pnl = (exit_price - pos["entry_price"]) * pos["qty"] * direction
                        pnl_pct = ((exit_price - pos["entry_price"]) / pos["entry_price"]) * direction
                        realized_pnl += pnl
                        trades.append(
                            BacktestTrade(
                                symbol=symbol, side=pos["side"], qty=pos["qty"], entry_price=pos["entry_price"],
                                exit_price=exit_price, entry_time=pos["entry_time"], exit_time=ts,
                                pnl=pnl, pnl_pct=pnl_pct, strategy=pos["strategy"], exit_reason=reason,
                            )
                        )
                        self.bandit.update(pos["strategy"], pnl_pct)
                        del open_positions[symbol]

                # 2) features + samoucici krok (label pro vzorek stary `horizon` svicek)
                feat, feat_row = build_feature_row(bar_store.dataframe(symbol))
                if feat is None:
                    continue

                for old_features, label in self.label_queue.advance_and_resolve(symbol, bar_close):
                    self.model.learn(old_features, label)
                self.label_queue.push(symbol, feat, bar_close)

                # 3) novy vstup, pokud v symbolu nic neni otevrene a je misto v portfoliu
                if symbol in open_positions or len(open_positions) >= settings.max_open_positions:
                    continue

                signal = self.signal_engine.generate(symbol, feat)
                if signal.direction == 0 or signal.confidence < settings.min_confidence:
                    continue
                if signal.direction < 0 and not settings.allow_short:
                    continue

                atr_value = float(feat_row["atr"])
                if atr_value <= 0:
                    continue

                equity_now = self.initial_equity + realized_pnl + unrealized_pnl()
                stop_distance = atr_value * settings.atr_stop_multiplier
                qty_by_risk = (equity_now * settings.risk_per_trade) / stop_distance
                qty_by_cap = (equity_now * settings.max_position_pct) / bar_close
                qty = int(min(qty_by_risk, qty_by_cap))
                if qty <= 0:
                    continue

                if signal.direction > 0:
                    side, stop, tp = "buy", bar_close - stop_distance, bar_close + atr_value * settings.atr_take_profit_multiplier
                else:
                    side, stop, tp = "sell", bar_close + stop_distance, bar_close - atr_value * settings.atr_take_profit_multiplier

                open_positions[symbol] = {
                    "side": side, "qty": qty, "entry_price": bar_close, "stop": stop, "tp": tp,
                    "strategy": signal.strategy_used, "entry_time": ts,
                }

            equity_points.append((ts, self.initial_equity + realized_pnl + unrealized_pnl()))

        # dozavrit pozice otevrene na konci backtestu (mark-to-market posledni znamou cenou)
        for symbol, pos in list(open_positions.items()):
            exit_price = latest_close.get(symbol, pos["entry_price"])
            direction = 1 if pos["side"] == "buy" else -1
            pnl = (exit_price - pos["entry_price"]) * pos["qty"] * direction
            pnl_pct = ((exit_price - pos["entry_price"]) / pos["entry_price"]) * direction
            realized_pnl += pnl
            trades.append(
                BacktestTrade(
                    symbol=symbol, side=pos["side"], qty=pos["qty"], entry_price=pos["entry_price"],
                    exit_price=exit_price, entry_time=pos["entry_time"], exit_time=all_ts[-1],
                    pnl=pnl, pnl_pct=pnl_pct, strategy=pos["strategy"], exit_reason="end_of_backtest",
                )
            )
            self.bandit.update(pos["strategy"], pnl_pct)

        equity_series = pd.Series(dict(equity_points)).sort_index()
        stats = compute_stats(equity_series, trades, self.initial_equity, settings.timeframe_minutes)
        return BacktestResult(equity_curve=equity_series, trades=trades, stats=stats)
