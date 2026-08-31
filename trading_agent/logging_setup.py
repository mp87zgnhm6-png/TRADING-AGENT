"""Centralni nastaveni logovani - konzole + rotujici soubor."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from trading_agent.config import Settings

_CONFIGURED = False


def setup_logging(settings: Settings) -> logging.Logger:
    global _CONFIGURED

    root = logging.getLogger("trading_agent")
    if _CONFIGURED:
        return root

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "trading_agent.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    root.propagate = False
    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"trading_agent.{name}")
