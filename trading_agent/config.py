"""Centralni konfigurace agenta, nactena z prostredi (.env)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Vsechna nastavitelna chovani agenta. Vse lze prepsat pres .env nebo env vars."""

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

    @field_validator("risk_per_trade", "max_position_pct", "daily_loss_limit_pct", "max_drawdown_pct", "min_confidence")
    @classmethod
    def _fraction_in_range(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError(f"ocekavana hodnota mezi 0 a 1, dostal jsem {v}")
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


def load_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
