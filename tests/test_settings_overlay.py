"""Testy pro vrstveni nastaveni: `.env` jako zaklad + zmeny ulozene z dashboardu.

Regrese, kterou tyto testy hlidaji: puvodne se do overlay souboru ukladal
kompletni snapshot vsech nastaveni uz pri prvnim startu (kvuli vygenerovanemu
tokenu). Tim se zmrazily tehdejsi hodnoty z `.env` vcetne API klicu a zadna
pozdejsi oprava `.env` uz se neprojevila - Alpaca pak vracela 401 unauthorized.
"""

from __future__ import annotations

import importlib
import json

import pytest

import trading_agent.config as config_module


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Config modul s overlayem i .env presmerovanymi do docasneho adresare."""
    monkeypatch.chdir(tmp_path)
    module = importlib.reload(config_module)
    monkeypatch.setattr(module, "RUNTIME_OVERLAY_PATH", tmp_path / "data" / "runtime_settings.json")
    return module


def _write_env(tmp_path, api_key: str, secret: str = "secret") -> None:
    (tmp_path / ".env").write_text(f"ALPACA_API_KEY={api_key}\nALPACA_SECRET_KEY={secret}\n", encoding="utf-8")


def test_first_run_persists_only_generated_token(isolated_config, tmp_path):
    _write_env(tmp_path, "PKFIRST")
    settings = isolated_config.load_settings()

    overlay = json.loads(isolated_config.RUNTIME_OVERLAY_PATH.read_text(encoding="utf-8"))
    assert settings.web_api_token
    assert set(overlay) == {"_version", "WEB_API_TOKEN"}
    assert "ALPACA_API_KEY" not in overlay


def test_env_change_is_not_shadowed_by_overlay(isolated_config, tmp_path):
    _write_env(tmp_path, "PKFIRST")
    first = isolated_config.load_settings()

    _write_env(tmp_path, "PKFIXED")
    second = isolated_config.load_settings()

    assert second.alpaca_api_key == "PKFIXED"
    assert second.web_api_token == first.web_api_token  # token zustava stabilni


def test_dashboard_change_persists_and_wins_over_env(isolated_config, tmp_path):
    _write_env(tmp_path, "PKFIRST")
    settings = isolated_config.load_settings()

    isolated_config.apply_settings_update(settings, {"risk_per_trade": 0.025})

    overlay = json.loads(isolated_config.RUNTIME_OVERLAY_PATH.read_text(encoding="utf-8"))
    assert overlay["RISK_PER_TRADE"] == 0.025
    assert "ALPACA_SECRET_KEY" not in overlay  # nezmenena pole se neukladaji

    reloaded = isolated_config.load_settings()
    assert reloaded.risk_per_trade == 0.025
    assert reloaded.alpaca_api_key == "PKFIRST"  # klic dal ridi .env


def test_failed_update_does_not_touch_overlay(isolated_config, tmp_path):
    _write_env(tmp_path, "PKFIRST")
    settings = isolated_config.load_settings()
    before = isolated_config.RUNTIME_OVERLAY_PATH.read_text(encoding="utf-8")

    with pytest.raises(Exception):
        isolated_config.apply_settings_update(settings, {"risk_per_trade": 5.0})

    assert isolated_config.RUNTIME_OVERLAY_PATH.read_text(encoding="utf-8") == before
    assert settings.risk_per_trade != 5.0


def test_legacy_full_snapshot_overlay_is_migrated(isolated_config, tmp_path):
    """Stary overlay (kompletni snapshot) uz nesmi prebijet opravene klice v .env."""
    _write_env(tmp_path, "PKCORRECT")
    isolated_config.RUNTIME_OVERLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.RUNTIME_OVERLAY_PATH.write_text(
        json.dumps(
            {
                "ALPACA_API_KEY": "your_alpaca_key_id",
                "ALPACA_SECRET_KEY": "your_alpaca_secret_key",
                "RISK_PER_TRADE": 0.01,
                "WEB_API_TOKEN": "puvodni-token",
            }
        ),
        encoding="utf-8",
    )

    settings = isolated_config.load_settings()

    assert settings.alpaca_api_key == "PKCORRECT"
    assert settings.alpaca_secret_key == "secret"
    assert settings.web_api_token == "puvodni-token"  # token se zachova

    overlay = json.loads(isolated_config.RUNTIME_OVERLAY_PATH.read_text(encoding="utf-8"))
    assert set(overlay) == {"_version", "WEB_API_TOKEN"}


def test_corrupted_overlay_falls_back_to_env(isolated_config, tmp_path):
    _write_env(tmp_path, "PKCORRECT")
    isolated_config.RUNTIME_OVERLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.RUNTIME_OVERLAY_PATH.write_text("{tohle neni platny json", encoding="utf-8")

    settings = isolated_config.load_settings()
    assert settings.alpaca_api_key == "PKCORRECT"


def test_credentials_error_hint_explains_401(isolated_config):
    """401 z Alpacy musi vratit konkretni navod, ne jen syrovou zpravu API."""
    from unittest.mock import Mock

    from alpaca.common.exceptions import APIError

    from trading_agent.broker.alpaca_client import credentials_error_hint

    http_error = Mock()
    http_error.response.status_code = 401
    exc = APIError('{"message": "unauthorized."}', http_error)

    settings = isolated_config.Settings(
        ALPACA_API_KEY="AKLIVEKEY1234567890", ALPACA_SECRET_KEY="tajemstvi", ALPACA_PAPER=True
    )
    hint = credentials_error_hint(exc, settings)

    assert "401" in hint
    assert "paper-api.alpaca.markets" in hint
    assert "PAPER" in hint
    assert "AKLIVEKEY1234567890" not in hint  # cely klic se nesmi objevit v logu


def test_credentials_error_hint_passes_through_other_errors(isolated_config):
    from unittest.mock import Mock

    from alpaca.common.exceptions import APIError

    from trading_agent.broker.alpaca_client import credentials_error_hint

    http_error = Mock()
    http_error.response.status_code = 500
    exc = APIError('{"message": "server error"}', http_error)

    hint = credentials_error_hint(exc, isolated_config.Settings())
    assert "server error" in hint
    assert "PAPER ucet" not in hint
