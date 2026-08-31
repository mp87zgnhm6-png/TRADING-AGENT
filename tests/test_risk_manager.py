import pytest

from trading_agent.config import Settings
from trading_agent.data.storage import Storage
from trading_agent.risk.risk_manager import RiskManager, trading_day_start_utc_iso


@pytest.fixture
def storage(tmp_path):
    s = Storage(str(tmp_path / "test.db"))
    yield s
    s.close()


@pytest.fixture
def settings():
    return Settings()


def test_position_size_respects_risk_and_cap(settings, storage):
    rm = RiskManager(settings, storage)
    # equity=100000, risk_per_trade=1% -> 1000 risk; price=200, atr=4 -> stop_distance=8 -> qty_by_risk=125
    # cap: max_position_pct=0.2 -> 20000 notional -> qty_by_cap=100 (pritejsi limit)
    assert rm.position_size(equity=100_000, price=200, atr=4) == 100


def test_position_size_zero_on_invalid_inputs(settings, storage):
    rm = RiskManager(settings, storage)
    assert rm.position_size(0, 100, 1) == 0
    assert rm.position_size(100_000, 0, 1) == 0
    assert rm.position_size(100_000, 100, 0) == 0


def test_stop_and_take_profit_long_and_short(settings, storage):
    rm = RiskManager(settings, storage)
    sl, tp = rm.stop_and_take_profit(entry_price=200, atr=4, direction=1)
    assert sl == pytest.approx(200 - 4 * settings.atr_stop_multiplier)
    assert tp == pytest.approx(200 + 4 * settings.atr_take_profit_multiplier)

    sl2, tp2 = rm.stop_and_take_profit(entry_price=200, atr=4, direction=-1)
    assert sl2 == pytest.approx(200 + 4 * settings.atr_stop_multiplier)
    assert tp2 == pytest.approx(200 - 4 * settings.atr_take_profit_multiplier)


def test_drawdown_circuit_breaker(settings, storage):
    rm = RiskManager(settings, storage)
    assert rm.check_drawdown_circuit_breaker(100_000).allowed  # bez historie neni co porovnat

    storage.record_equity(100_000, 50_000, 50_000)
    assert rm.check_drawdown_circuit_breaker(90_000).allowed  # 10 % < limit 15 %
    assert not rm.check_drawdown_circuit_breaker(80_000).allowed  # 20 % >= limit 15 %


def test_daily_loss_limit(settings, storage):
    rm = RiskManager(settings, storage)
    day_start = trading_day_start_utc_iso()
    tid = storage.open_trade("AAPL", "buy", 10, 100, 95, 110, "ml_model", 0.6)
    storage.close_trade(tid, 96.5, "stop_loss")  # realizovana ztrata -35

    assert storage.get_daily_pnl(day_start) == pytest.approx(-35.0)
    assert rm.check_daily_loss_limit(90_000).allowed  # 35/90000 << 3 % limit


def test_kill_switch(tmp_path, storage):
    settings = Settings(KILL_SWITCH_FILE=str(tmp_path / "STOP"))
    rm = RiskManager(settings, storage)
    assert not rm.kill_switch_active()
    (tmp_path / "STOP").touch()
    assert rm.kill_switch_active()


def test_check_new_entry_allowed_blocks_when_market_closed(settings, storage):
    rm = RiskManager(settings, storage)
    decision = rm.check_new_entry_allowed(current_equity=100_000, open_positions_count=0, market_open=False)
    assert not decision.allowed
    assert "zavreny" in decision.reason


def test_check_new_entry_allowed_blocks_at_max_positions(settings, storage):
    rm = RiskManager(settings, storage)
    decision = rm.check_new_entry_allowed(
        current_equity=100_000, open_positions_count=settings.max_open_positions, market_open=True
    )
    assert not decision.allowed
