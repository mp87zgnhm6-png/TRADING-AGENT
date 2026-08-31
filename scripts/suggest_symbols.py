#!/usr/bin/env python3
"""Najde symboly, ktere se vejdou do kapitalu na uctu.

Bracket ordery na Alpaca neumi zlomkove akcie, takze plati tvrde pravidlo:
**1 cely kus musi stat max `equity * MAX_POSITION_PCT`**. U maleho uctu proto
drahe tituly (SPY, MSFT, ...) neotevrou zadnou pozici a bot by jen mlcel.

Skript stahne AKTUALNI ceny z Alpaca (zadne natvrdo zapsane ceny, ktere by
rychle zastaraly), spocita kolik kusu se vejde, a vypise hotovy radek SYMBOLS=
k vlozeni do .env nebo do dashboardu.

Pouziti:
    python scripts/suggest_symbols.py
    python scripts/suggest_symbols.py --equity 230 --max-symbols 4
    python scripts/suggest_symbols.py --symbols F,SOFI,INTC,T,PFE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_agent.broker.alpaca_client import AlpacaBroker  # noqa: E402
from trading_agent.config import load_settings  # noqa: E402
from trading_agent.logging_setup import setup_logging  # noqa: E402
from trading_agent.risk.risk_manager import max_affordable_shares  # noqa: E402

# Siroky seznam likvidnich US titulu napric cenovymi hladinami. Skript z nej
# vybira podle SKUTECNE ceny, takze nevadi, ze casem nektere zdrazi/zlevni.
CANDIDATE_SYMBOLS = [
    "F", "SOFI", "INTC", "T", "PFE", "VZ", "BAC", "KVUE", "HBAN", "KEY",
    "SIRI", "NIO", "SNAP", "RIVN", "LYFT", "AAL", "CCL", "WBD", "GRAB", "PLUG",
    "CSCO", "KO", "WFC", "MU", "CMCSA", "HOOD", "UBER", "PLTR", "XLF", "GDX",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vybere symboly, ktere se vejdou do kapitalu na uctu.")
    parser.add_argument("--equity", type=float, default=None, help="Kapital v USD (vychozi: skutecna equity z Alpaca).")
    parser.add_argument("--max-position-pct", type=float, default=None, help="Prepise MAX_POSITION_PCT (0-1).")
    parser.add_argument("--symbols", type=str, default=None, help="Vlastni seznam k provereni (carkou oddeleny).")
    parser.add_argument("--max-symbols", type=int, default=5, help="Kolik symbolu navrhnout (vychozi 5).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    logger = setup_logging(settings)

    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        logger.error("ALPACA_API_KEY / ALPACA_SECRET_KEY nejsou nastaveny.")
        return 1

    broker = AlpacaBroker(settings)
    max_position_pct = args.max_position_pct or settings.max_position_pct

    equity = args.equity
    if equity is None:
        try:
            equity = float(broker.get_account().equity)
        except Exception as exc:
            logger.error("Nepodarilo se zjistit equity uctu (%s). Zadejte ji rucne pres --equity.", exc)
            return 1

    candidates = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else CANDIDATE_SYMBOLS
    )
    logger.info("Stahuji aktualni ceny pro %d symbolu...", len(candidates))
    prices = broker.get_latest_prices(candidates)
    if not prices:
        logger.error("Nepodarilo se stahnout zadne ceny.")
        return 1

    cap = equity * max_position_pct
    rows = []
    for symbol, price in prices.items():
        shares = max_affordable_shares(equity, price, max_position_pct)
        rows.append((symbol, price, shares, shares * price))
    rows.sort(key=lambda r: (-r[2], r[1]))  # nejvic kusu prvni, pak nejlevnejsi

    affordable = [r for r in rows if r[2] >= 1]
    unaffordable = [r for r in rows if r[2] < 1]

    print()
    print("=" * 72)
    print(f"  Kapital: {equity:.2f} USD | MAX_POSITION_PCT: {max_position_pct:.0%}"
          f" | strop na 1 pozici: {cap:.2f} USD")
    print("=" * 72)
    print(f"  {'symbol':8} {'cena':>10} {'kusu':>6} {'hodnota pozice':>16}")
    print("  " + "-" * 44)
    for symbol, price, shares, notional in affordable:
        print(f"  {symbol:8} {price:10.2f} {shares:6d} {notional:16.2f}")

    if not affordable:
        print("  Za tento kapital nelze koupit ani 1 cely kus u zadneho z overovanych symbolu.")
        print(f"  Zvyste MAX_POSITION_PCT (ted {max_position_pct:.0%}), nebo pouzijte levnejsi tituly.")
        print("=" * 72)
        return 1

    if unaffordable:
        names = ", ".join(f"{s} ({p:.2f})" for s, p, _, _ in unaffordable[:8])
        print()
        print(f"  Nevejde se ani 1 kus: {names}")

    chosen = [r[0] for r in affordable[: args.max_symbols]]
    print()
    print("  Navrzene nastaveni (vlozte do .env, nebo do dashboardu -> Nastaveni -> Symboly):")
    print()
    print(f"      SYMBOLS={','.join(chosen)}")
    print()
    print("  Po zmene symbolu je potreba restartovat agenta (tlacitko v dashboardu).")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
