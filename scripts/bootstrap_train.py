#!/usr/bin/env python3
"""'Zahrivaci' beh: predtrenuje samoucici se model a bandit na historickych datech
PRED prvnim ostrym spustenim agenta, aby nezacinal uplne od nuly.

Interne pousti stejny walk-forward backtest jako scripts/run_backtest.py, ale
vysledny stav modelu/banditu navic ulozi na disk (MODEL_PATH / BANDIT_PATH),
odkud si ho pri startu nacte `scripts/run_live.py`.

Pouziti:
    python scripts/bootstrap_train.py --days 90
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_agent.backtest.backtester import Backtester  # noqa: E402
from trading_agent.broker.alpaca_client import AlpacaBroker  # noqa: E402
from trading_agent.config import load_settings  # noqa: E402
from trading_agent.logging_setup import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predtrenuje model/bandit na historickych datech.")
    parser.add_argument("--days", type=int, default=60, help="Kolik dni historie pouzit k zahrati (vychozi 60).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    logger = setup_logging(settings)

    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        logger.error("ALPACA_API_KEY / ALPACA_SECRET_KEY nejsou nastaveny.")
        return 1

    broker = AlpacaBroker(settings)
    logger.info("Stahuji %d dni historie pro %s...", args.days, settings.symbols)
    historical = broker.get_historical_bars(settings.symbols, lookback_days=args.days)
    if not historical:
        logger.error("Nepodarilo se stahnout zadna historicka data - zahriti preskoceno.")
        return 1

    backtester = Backtester(settings, seed=42)
    result = backtester.run(historical)

    backtester.model.save(settings.model_path)
    backtester.bandit.save(settings.bandit_path)

    logger.info(
        "Zahrivaci beh hotovy: %d obchodu, final equity=%.2f, rolling_auc=%.3f (n=%d vzorku)",
        len(result.trades), result.stats.get("final_equity", 0.0),
        backtester.model.rolling_score(), backtester.model.n_samples,
    )
    logger.info("Model ulozen do %s, bandit do %s.", settings.model_path, settings.bandit_path)
    logger.info("Ted uz muzete spustit: python scripts/run_live.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
