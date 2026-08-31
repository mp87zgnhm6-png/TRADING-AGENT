"""Centralni konfigurace agenta.

Nacita se ve dvou vrstvach: zaklad je `.env` (tajne klice + pocatecni
hodnoty), pres nej se prekryji zmeny ulozene webovym rozhranim
(`data/runtime_settings.json`) - diky tomu zmena provedena v dashboardu
prezije i restart procesu, aniz by se sahalo na `.env`.
"""

from __future__ import annotations

import json
import logging
import secrets
from pathlib import Path
from typing import Any, Iterable

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("trading_agent.config")

RUNTIME_OVERLAY_PATH = Path("data/runtime_settings.json")
OVERLAY_VERSION = 2
_VERSION_KEY = "_version"

# Nastaveni, ktera pri poslednim `load_settings()` prebila hodnoty z `.env`.
_ACTIVE_OVERLAY_KEYS: list[str] = []


class Settings(BaseSettings):
    """Vsechna nastavitelna chovani agenta. Vse lze prepsat pres .env, env vars,
    nebo (za behu) pres webove rozhrani."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Alpaca pristup ---
    alpaca_api_key: str = Field(default="", alias="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(default="", alias="ALPACA_SECRET_KEY")
    alpaca_paper: bool = Field(default=True, alias="ALPACA_PAPER")
    i_understand_live_trading_risk: bool = Field(default=False, alias="I_UNDERSTAND_LIVE_TRADING_RISK")

    # --- Sledovany trh ---
    symbols_raw: str = Field(default="AAPL,MSFT,SPY", alias="SYMBOLS")
    data_feed: str = Field(default="iex", alias="DATA_FEED")
    timeframe_minutes: int = Field(default=5, alias="TIMEFRAME_MINUTES")
    bar_lookback: int = Field(default=300, alias="BAR_LOOKBACK")
    prediction_horizon_bars: int = Field(default=3, alias="PREDICTION_HORIZON_BARS")

    # --- Risk management ---
    risk_per_trade: float = Field(default=0.01, alias="RISK_PER_TRADE")
    max_position_pct: float = Field(default=0.2, alias="MAX_POSITION_PCT")
    max_open_positions: int = Field(default=5, alias="MAX_OPEN_POSITIONS")
    daily_loss_limit_pct: float = Field(default=0.03, alias="DAILY_LOSS_LIMIT_PCT")
    max_drawdown_pct: float = Field(default=0.15, alias="MAX_DRAWDOWN_PCT")
    atr_stop_multiplier: float = Field(default=2.0, alias="ATR_STOP_MULTIPLIER")
    atr_take_profit_multiplier: float = Field(default=3.0, alias="ATR_TAKE_PROFIT_MULTIPLIER")
    min_confidence: float = Field(default=0.55, alias="MIN_CONFIDENCE")
    allow_short: bool = Field(default=False, alias="ALLOW_SHORT")
    extended_hours: bool = Field(default=False, alias="EXTENDED_HOURS")
    max_hold_bars: int = Field(default=0, alias="MAX_HOLD_BARS")

    # --- Provoz ---
    loop_interval_seconds: int = Field(default=30, alias="LOOP_INTERVAL_SECONDS")
    persist_interval_minutes: int = Field(default=15, alias="PERSIST_INTERVAL_MINUTES")
    dry_run: bool = Field(default=False, alias="DRY_RUN")
    kill_switch_file: str = Field(default="data/STOP", alias="KILL_SWITCH_FILE")
    close_positions_on_shutdown: bool = Field(default=False, alias="CLOSE_POSITIONS_ON_SHUTDOWN")

    # --- Cesty ---
    model_path: str = Field(default="data/model_state.pkl", alias="MODEL_PATH")
    bandit_path: str = Field(default="data/bandit_state.json", alias="BANDIT_PATH")
    db_path: str = Field(default="data/trading_agent.db", alias="DB_PATH")
    log_dir: str = Field(default="data/logs", alias="LOG_DIR")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # --- Webove rozhrani ---
    web_host: str = Field(default="127.0.0.1", alias="WEB_HOST")
    web_port: int = Field(default=8000, alias="WEB_PORT")
    web_api_token: str = Field(default="", alias="WEB_API_TOKEN")

    @field_validator("risk_per_trade", "max_position_pct", "daily_loss_limit_pct", "max_drawdown_pct", "min_confidence")
    @classmethod
    def _fraction_in_range(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError(f"ocekavana hodnota mezi 0 a 1, dostal jsem {v}")
        return v

    @field_validator("web_port")
    @classmethod
    def _valid_port(cls, v: int) -> int:
        if not 0 < v < 65536:
            raise ValueError(f"port musi byt v rozsahu 1-65535, dostal jsem {v}")
        return v

    @property
    def symbols(self) -> list[str]:
        return [s.strip().upper() for s in self.symbols_raw.split(",") if s.strip()]

    def ensure_directories(self) -> None:
        for path_str in (self.model_path, self.bandit_path, self.db_path, self.log_dir):
            Path(path_str).parent.mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def live_trading_confirmed(self) -> bool:
        """True pokud je bezpecne obchodovat naostro (paper vzdy OK)."""
        return self.alpaca_paper or self.i_understand_live_trading_risk

    def to_public_dict(self) -> dict[str, Any]:
        """Hodnoty pro API/dashboard - citliva pole jsou maskovana."""
        data = self.model_dump(mode="json")
        if data.get("alpaca_secret_key"):
            data["alpaca_secret_key"] = "•" * 12
        if data.get("alpaca_api_key"):
            key = data["alpaca_api_key"]
            data["alpaca_api_key"] = ("•" * max(len(key) - 4, 0)) + key[-4:]
        data["web_api_token"] = "•" * 12 if data.get("web_api_token") else ""
        data["symbols"] = self.symbols
        return data


def _read_overlay() -> dict:
    if not RUNTIME_OVERLAY_PATH.exists():
        return {}
    try:
        data = json.loads(RUNTIME_OVERLAY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Poskozeny %s (%s), ignoruji ulozene zmeny z dashboardu.", RUNTIME_OVERLAY_PATH, exc)
        return {}

    if not isinstance(data, dict):
        logger.warning("Neocekavany obsah %s, ignoruji ho.", RUNTIME_OVERLAY_PATH)
        return {}

    if data.get(_VERSION_KEY) != OVERLAY_VERSION:
        data = _migrate_legacy_overlay(data)

    return {k: v for k, v in data.items() if k != _VERSION_KEY}


def _migrate_legacy_overlay(data: dict) -> dict:
    """Prevede stary format overlaye (kompletni kopie vsech nastaveni) na novy.

    Stary format zmrazil obsah `.env` z okamziku prvniho spusteni a od te chvile
    ho prebijel - typicky to znamenalo, ze oprava ALPACA_API_KEY v `.env` nemela
    zadny efekt a Alpaca vracela 401. Ze snapshotu nelze poznat, co uzivatel
    zamerne zmenil v dashboardu a co je jen kopie `.env`, proto se zachova pouze
    vygenerovany pristupovy token a zbytek se vraci do rezie `.env`.
    """
    kept = {k: v for k, v in data.items() if k == "WEB_API_TOKEN" and v}
    dropped = sorted(k for k in data if k not in kept and k != _VERSION_KEY)
    _write_overlay(kept)
    logger.warning(
        "Soubor %s byl ve starem formatu a prebijel hodnoty z .env (%s). Byl migrovan - "
        "nadale plati .env a jen zmeny, ktere ulozite v dashboardu.",
        RUNTIME_OVERLAY_PATH,
        ", ".join(dropped) or "zadne polozky",
    )
    return kept


FIELD_ALIASES: dict[str, str] = {name: (f.alias or name) for name, f in Settings.model_fields.items()}


def _write_overlay(data: dict) -> None:
    RUNTIME_OVERLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {_VERSION_KEY: OVERLAY_VERSION, **data}
    tmp = RUNTIME_OVERLAY_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(RUNTIME_OVERLAY_PATH)


def persist_settings_overlay(settings: Settings, changed: Iterable[str]) -> None:
    """Ulozi do overlay souboru POUZE explicitne zmenena pole.

    Zamerne se neuklada kompletni snapshot nastaveni: kdyby se ukladalo vsechno,
    prvni spusteni by zmrazilo tehdejsi obsah `.env` (vcetne API klicu) a
    jakakoli pozdejsi oprava `.env` by se uz neprojevila. Overlay tedy drzi jen
    to, co uzivatel opravdu zmenil v dashboardu; vse ostatni zustava v rezii `.env`.

    Soubor je v .gitignore stejne jako .env - muze obsahovat i API klice zadane
    pres webove rozhrani.
    """
    overlay = _read_overlay()
    dumped = settings.model_dump(by_alias=True)
    for name in changed:
        alias = FIELD_ALIASES.get(name, name)
        if alias in dumped:
            overlay[alias] = dumped[alias]
    _write_overlay(overlay)


def active_overlay_keys() -> list[str]:
    """Nastaveni, ktera aktualne prebijeji `.env` (zmeny ulozene z dashboardu)."""
    return sorted(_ACTIVE_OVERLAY_KEYS)


def load_settings() -> Settings:
    """Nacte `.env` a preklopi pres nej zmeny ulozene z weboveho rozhrani."""
    global _ACTIVE_OVERLAY_KEYS

    settings = Settings()
    overlay = _read_overlay()
    _ACTIVE_OVERLAY_KEYS = []
    if overlay:
        merged = settings.model_dump(by_alias=True)
        merged.update(overlay)
        try:
            settings = Settings(**merged)
            _ACTIVE_OVERLAY_KEYS = list(overlay.keys())
        except Exception as exc:
            logger.warning("Ulozene zmeny z dashboardu neprosly validaci (%s), pouzivam jen .env.", exc)

    if not settings.web_api_token:
        settings.web_api_token = secrets.token_urlsafe(24)
        persist_settings_overlay(settings, ["web_api_token"])
        _ACTIVE_OVERLAY_KEYS = sorted({*_ACTIVE_OVERLAY_KEYS, "WEB_API_TOKEN"})

    settings.ensure_directories()
    return settings


def apply_settings_update(current: Settings, updates: dict[str, Any]) -> Settings:
    """Provede zmenu nastaveni: zvaliduje ji (nova, docasna instance Settings
    projde vsemi pydantic validatory), a teprve pak zapise hodnoty NA MISTE do
    `current`. Zachovani identity objektu je dulezite - agent uz na nem drzi
    referenci, takze 'hot' pole (risk limity, min_confidence, dry_run, ...) se
    projevi okamzite bez restartu; 'cold' pole (API klice, symboly, timeframe, ...)
    vyzaduji rucni restart agenta z dashboardu.
    """
    unknown = [k for k in updates if k not in FIELD_ALIASES]
    if unknown:
        raise ValueError(f"Neznama nastaveni: {unknown}")

    merged = current.model_dump(by_alias=True)
    for name, value in updates.items():
        merged[FIELD_ALIASES[name]] = value

    validated = Settings(**merged)
    for name in Settings.model_fields:
        setattr(current, name, getattr(validated, name))

    current.ensure_directories()
    # do overlaye jde jen to, co uzivatel skutecne zmenil - zbytek nadale ridi .env
    persist_settings_overlay(current, updates.keys())
    return current
