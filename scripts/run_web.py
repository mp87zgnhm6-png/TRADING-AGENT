#!/usr/bin/env python3
"""Spusti autonomniho agenta SPOLU s webovym dashboardem.

Agent bezi na pozadi ve vlastnim vlakne, webovy server (uvicorn) v hlavnim
vlakne - oba ve stejnem procesu, takze dashboard vidi presne ten stav, se
kterym agent prave pracuje, a zmeny nastaveni se propisou okamzite.

Pouziti:
    python scripts/run_web.py
    python scripts/run_web.py --port 8080 --no-autostart
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402

from trading_agent.config import RUNTIME_OVERLAY_PATH, active_overlay_keys, load_settings  # noqa: E402
from trading_agent.logging_setup import setup_logging  # noqa: E402
from trading_agent.webapp.server import create_app  # noqa: E402
from trading_agent.webapp.supervisor import AgentSupervisor  # noqa: E402

UVICORN_LOG_LEVELS = {"critical", "error", "warning", "info", "debug", "trace"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webovy dashboard + autonomni Alpaca trading agent.")
    parser.add_argument("--host", default=None, help="Prepise WEB_HOST z .env (vychozi 127.0.0.1).")
    parser.add_argument("--port", type=int, default=None, help="Prepise WEB_PORT z .env (vychozi 8000).")
    parser.add_argument(
        "--no-autostart",
        action="store_true",
        help="Nespousti agenta automaticky - nabehne jen dashboard (agenta pak spustite tlacitkem).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    logger = setup_logging(settings)

    host = args.host or settings.web_host
    port = args.port or settings.web_port

    print("=" * 72)
    print("  Alpaca autonomni samoucici se trading bot - webovy dashboard")
    print(f"  Adresa:  http://{host}:{port}/")
    print(f"  Token:   {settings.web_api_token}")
    print(f"  Rezim:   {'PAPER (virtualni penize)' if settings.alpaca_paper else 'LIVE - SKUTECNE PENIZE'}"
          f"   DRY_RUN={settings.dry_run}")
    print("=" * 72)
    if host not in ("127.0.0.1", "localhost"):
        print("  UPOZORNENI: dashboard neni omezen na localhost. Kdokoliv, kdo zna token,")
        print("  muze menit risk limity i zavirat pozice - postavte pred nej HTTPS reverse")
        print("  proxy a token drzte v tajnosti.")
        print("=" * 72)

    overridden = active_overlay_keys()
    if overridden:
        logger.info(
            "Nastaveni ulozena z dashboardu (%s) prebijeji .env: %s",
            RUNTIME_OVERLAY_PATH, ", ".join(overridden),
        )

    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        logger.warning(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY nejsou vyplnene - dashboard nabehne, "
            "ale agenta pujde spustit az po jejich doplneni v zalozce Nastaveni."
        )

    supervisor = AgentSupervisor(settings)
    if not args.no_autostart:
        ok, message = supervisor.start()
        logger.info("Autostart agenta: %s", message)
        if not ok:
            logger.warning("Agent nebezi - spustte ho z dashboardu po doplneni nastaveni.")

    app = create_app(supervisor)
    log_level = settings.log_level.lower()
    try:
        uvicorn.run(app, host=host, port=port, log_level=log_level if log_level in UVICORN_LOG_LEVELS else "info")
    finally:
        supervisor.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
