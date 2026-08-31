#!/usr/bin/env python3
"""Backtest samoucicho se agenta nad historickymi daty z Alpaca.

Pouziti:
    python scripts/run_backtest.py --days 90
    python scripts/run_backtest.py --days 30 --symbols AAPL,MSFT --out data/backtests/run1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from trading_agent.backtest.backtester import Backtester  # noqa: E402
from trading_agent.broker.alpaca_client import AlpacaBroker  # noqa: E402
from trading_agent.config import load_settings  # noqa: E402
from trading_agent.logging_setup import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest samoucicho se Alpaca tradingbota.")
    parser.add_argument("--days", type=int, default=60, help="Kolik dni historie pouzit (vychozi 60).")
    parser.add_argument("--symbols", type=str, default=None, help="Carkou oddeleny seznam symbolu (jinak SYMBOLS z .env).")
    parser.add_argument("--equity", type=float, default=100_000.0, help="Pocatecni kapital backtestu.")
    parser.add_argument("--out", type=str, default=None, help="Slozka pro ulozeni equity_curve.csv a trades.csv.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    logger = setup_logging(settings)

    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        logger.error("ALPACA_API_KEY / ALPACA_SECRET_KEY nejsou nastaveny (potrebne i pro historicka data).")
        return 1

    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else settings.symbols

    broker = AlpacaBroker(settings)
    logger.info("Stahuji %d dni historie pro %s...", args.days, symbols)
    historical = broker.get_historical_bars(symbols, lookback_days=args.days)
    missing = [s for s in symbols if historical.get(s) is None or historical[s].empty]
    if missing:
        logger.warning("Zadna data pro: %s (vynechavam z backtestu)", missing)
    if not historical:
        logger.error("Nepodarilo se stahnout zadna pouzitelna historicka data.")
        return 1

    backtester = Backtester(settings, initial_equity=args.equity)
    result = backtester.run(historical)
    stats = result.stats

    print("\n" + "=" * 60)
    print(" VYSLEDEK BACKTESTU")
    print("=" * 60)
    print(f" Obdobi:             {args.days} dni, symboly: {', '.join(historical.keys())}")
    print(f" Pocatecni kapital:  {stats['initial_equity']:,.2f}")
    print(f" Konecny kapital:    {stats['final_equity']:,.2f}")
    print(f" Celkovy vynos:      {stats['total_return_pct']:.2f} %")
    print(f" CAGR (odhad):       {stats['cagr_pct']:.2f} %")
    print(f" Sharpe ratio:       {stats['sharpe_ratio']:.2f}")
    print(f" Max drawdown:       {stats['max_drawdown_pct']:.2f} %")
    print(f" Pocet obchodu:      {stats['num_trades']}")
    print(f" Uspesnost:          {stats['win_rate_pct']:.1f} %")
    print(f" Profit factor:      {stats['profit_factor']:.2f}")
    print("=" * 60)
    print(f" Vahy banditu po backtestu: {backtester.bandit.stats()}")
    print(f" Rolling AUC modelu:        {backtester.model.rolling_score():.3f} (n={backtester.model.n_samples})")
    print("=" * 60)
    print(" Pozor: vysledky backtestu nejsou zarukou budouciho vykonu naostro.")

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        result.equity_curve.to_csv(out_dir / "equity_curve.csv", header=["equity"])
        trades_df = pd.DataFrame([t.__dict__ for t in result.trades])
        trades_df.to_csv(out_dir / "trades.csv", index=False)
        logger.info("Vysledky ulozeny do %s", out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
