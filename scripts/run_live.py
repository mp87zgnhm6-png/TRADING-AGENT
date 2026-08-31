#!/usr/bin/env python3
"""Spusti autonomniho, samoucicho se tradingbota napojeneho na Alpaca.

Pouziti:
    python scripts/run_live.py

Rezim (paper/live) a vsechna dalsi nastaveni se ctou z .env (viz .env.example).
Na paper uctu (ALPACA_PAPER=true, vychozi) staci mit v .env platne API klice.
Na ostry provoz (ALPACA_PAPER=false) je navic nutne v .env explicitne nastavit
I_UNDERSTAND_LIVE_TRADING_RISK=true - jinak skript odmitne start.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_agent.agent import AutonomousTradingAgent  # noqa: E402
from trading_agent.config import RUNTIME_OVERLAY_PATH, active_overlay_keys, load_settings  # noqa: E402
from trading_agent.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    settings = load_settings()
    logger = setup_logging(settings)

    overridden = active_overlay_keys()
    if overridden:
        logger.info(
            "Nastaveni ulozena z dashboardu (%s) prebijeji .env: %s",
            RUNTIME_OVERLAY_PATH, ", ".join(overridden),
        )

    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        logger.error(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY nejsou nastaveny. "
            "Zkopirujte .env.example do .env a vyplnte pristupove udaje z app.alpaca.markets."
        )
        return 1

    mode = "PAPER (virtualni penize)" if settings.alpaca_paper else "!!! LIVE - SKUTECNE PENIZE !!!"
    print("=" * 72)
    print(f"  Alpaca autonomni samoucici se trading bot")
    print(f"  Rezim:    {mode}")
    print(f"  Symboly:  {', '.join(settings.symbols)}")
    print(f"  DRY_RUN:  {settings.dry_run}")
    print("=" * 72)

    if not settings.live_trading_confirmed():
        logger.error(
            "ALPACA_PAPER=false, ale I_UNDERSTAND_LIVE_TRADING_RISK neni 'true'. "
            "Doporucuji nejdriv dukladne otestovat na paper uctu (ALPACA_PAPER=true). "
            "Pokud opravdu chcete obchodovat naostro, nastavte v .env I_UNDERSTAND_LIVE_TRADING_RISK=true."
        )
        return 1

    agent = AutonomousTradingAgent(settings)
    try:
        agent.run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
