"""Popis nastaveni pro auto-generovany formular v dashboardu.

`hot: True` znamena, ze zmena se v bezicim agentovi projevi okamzite (bez
restartu) - prislusna cast agenta cte hodnotu vzdy znovu primo ze sdileneho
`Settings` objektu (viz `trading_agent/config.py::apply_settings_update`).
`hot: False` pole (API klice, symboly, timeframe, ...) vyzaduji kliknuti na
"Restartovat agenta" v dashboardu - menit je za behu bez znovu-vytvoreni
brokeru/streamu by vedlo k nekonzistentnimu stavu.
"""

from __future__ import annotations

SETTINGS_SCHEMA: list[dict] = [
    {
        "group": "Alpaca pripojeni",
        "fields": [
            {"name": "alpaca_api_key", "label": "API Key", "type": "password", "hot": False},
            {"name": "alpaca_secret_key", "label": "Secret Key", "type": "password", "hot": False},
            {"name": "alpaca_paper", "label": "Paper trading", "type": "bool", "hot": False},
            {"name": "i_understand_live_trading_risk", "label": "Potvrzuji riziko ostreho obchodovani", "type": "bool", "hot": False},
            {"name": "data_feed", "label": "Datovy feed (iex / sip)", "type": "text", "hot": False},
        ],
    },
    {
        "group": "Trh a symboly",
        "fields": [
            {"name": "symbols_raw", "label": "Symboly (carkou oddelene)", "type": "text", "hot": False},
            {"name": "timeframe_minutes", "label": "Timeframe (minuty na svicku)", "type": "int", "hot": False},
            {"name": "bar_lookback", "label": "Velikost bufferu svicek", "type": "int", "hot": False},
            {"name": "prediction_horizon_bars", "label": "Predikcni horizont (pocet svicek)", "type": "int", "hot": False},
            {"name": "extended_hours", "label": "Obchodovat mimo standardni seanci", "type": "bool", "hot": False},
        ],
    },
    {
        "group": "Risk management",
        "fields": [
            {"name": "risk_per_trade", "label": "Risk na obchod (podil equity, 0-1)", "type": "float", "hot": True, "step": 0.001},
            {"name": "max_position_pct", "label": "Max. podil equity na 1 symbol (0-1)", "type": "float", "hot": True, "step": 0.01},
            {"name": "max_open_positions", "label": "Max. otevrenych pozic", "type": "int", "hot": True},
            {"name": "daily_loss_limit_pct", "label": "Denni ztratovy limit (0-1)", "type": "float", "hot": True, "step": 0.01},
            {"name": "max_drawdown_pct", "label": "Max. drawdown - circuit breaker (0-1)", "type": "float", "hot": True, "step": 0.01},
            {"name": "atr_stop_multiplier", "label": "ATR nasobek stop-loss", "type": "float", "hot": True, "step": 0.1},
            {"name": "atr_take_profit_multiplier", "label": "ATR nasobek take-profit", "type": "float", "hot": True, "step": 0.1},
            {"name": "min_confidence", "label": "Min. jistota signalu pro vstup (0-1)", "type": "float", "hot": True, "step": 0.01},
            {"name": "allow_short", "label": "Povolit short prodeje", "type": "bool", "hot": True},
            {"name": "max_hold_bars", "label": "Max. drzeni pozice (pocet svicek, 0 = vypnuto)", "type": "int", "hot": True},
        ],
    },
    {
        "group": "Provoz agenta",
        "fields": [
            {"name": "loop_interval_seconds", "label": "Interval udrzbove smycky (s)", "type": "int", "hot": True},
            {"name": "persist_interval_minutes", "label": "Interval ukladani stavu modelu (min)", "type": "int", "hot": True},
            {"name": "dry_run", "label": "DRY RUN (neodesilat skutecne ordery)", "type": "bool", "hot": True},
            {"name": "close_positions_on_shutdown", "label": "Zavrit vsechny pozice pri vypnuti agenta", "type": "bool", "hot": True},
            {"name": "kill_switch_file", "label": "Cesta ke kill-switch souboru", "type": "text", "hot": True},
        ],
    },
]

FIELD_NAMES: set[str] = {f["name"] for group in SETTINGS_SCHEMA for f in group["fields"]}
