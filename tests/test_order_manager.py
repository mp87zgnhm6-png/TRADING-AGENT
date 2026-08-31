from types import SimpleNamespace

import pytest

from trading_agent.config import Settings
from trading_agent.data.storage import Storage
from trading_agent.execution.order_manager import OrderManager


@pytest.fixture
def storage(tmp_path):
    s = Storage(str(tmp_path / "test.db"))
    yield s
    s.close()


def _settings(dry_run: bool = False) -> Settings:
    return Settings(DRY_RUN="true" if dry_run else "false")


class FakeBroker:
    def __init__(self):
        self.closed = []
        self.bracket_calls = []

    def submit_bracket_order(self, symbol, qty, side, sl, tp):
        self.bracket_calls.append((symbol, qty, side, sl, tp))
        return SimpleNamespace(id="11111111-1111-1111-1111-111111111111", filled_avg_price=None)

    def close_position(self, symbol, qty=None):
        self.closed.append(symbol)

    def get_positions(self):
        return []

    class trading:
        @staticmethod
        def get_order_by_id(order_id):
            return SimpleNamespace(filled_avg_price="150.25", legs=None)


def test_enter_position_live_records_broker_order_id(storage):
    broker = FakeBroker()
    om = OrderManager(broker, storage, _settings(dry_run=False))

    ok = om.enter_position("AAPL", 1, 10, 150.0, 145.0, 160.0, "ml_model", 0.7)
    assert ok
    trade = storage.get_open_trade("AAPL")
    assert trade.broker_order_id == "11111111-1111-1111-1111-111111111111"
    assert broker.bracket_calls[0][0] == "AAPL"


def test_exit_position_closes_and_returns_reward(storage):
    broker = FakeBroker()
    om = OrderManager(broker, storage, _settings(dry_run=False))
    om.enter_position("AAPL", 1, 10, 150.0, 145.0, 160.0, "ml_model", 0.7)

    result = om.exit_position("AAPL", 158.0, "take_profit")
    assert result is not None
    strategy, pnl_pct = result
    assert strategy == "ml_model"
    assert pnl_pct == pytest.approx((158.0 - 150.0) / 150.0)
    assert broker.closed == ["AAPL"]


def test_exit_position_noop_when_nothing_open(storage):
    broker = FakeBroker()
    om = OrderManager(broker, storage, _settings(dry_run=False))
    assert om.exit_position("AAPL", 100.0, "take_profit") is None


def test_dry_run_never_calls_broker(storage):
    broker = FakeBroker()
    om = OrderManager(broker, storage, _settings(dry_run=True))

    om.enter_position("MSFT", -1, 5, 300.0, 310.0, 280.0, "mean_reversion", 0.6)
    assert broker.bracket_calls == []
    result = om.exit_position("MSFT", 285.0, "signal_reversal")
    assert result == ("mean_reversion", pytest.approx(0.05))
    assert broker.closed == []


def test_fill_price_from_order_id_handles_errors_gracefully(storage):
    class BrokenBroker(FakeBroker):
        class trading:
            @staticmethod
            def get_order_by_id(order_id):
                raise ValueError("badly formed hexadecimal UUID string")

    om = OrderManager(BrokenBroker(), storage, _settings(dry_run=False))
    assert om.fill_price_from_order_id("not-a-uuid") is None


def test_dry_run_is_read_live_from_settings(storage):
    broker = FakeBroker()
    settings = _settings(dry_run=False)
    om = OrderManager(broker, storage, settings)

    om.enter_position("AAPL", 1, 10, 150.0, 145.0, 160.0, "ml_model", 0.7)
    assert broker.bracket_calls  # skutecny order byl odeslan

    settings.dry_run = True  # simuluje zmenu z webove dashboardu za behu
    om.exit_position("AAPL", 158.0, "take_profit")
    assert broker.closed == []  # dry_run=True se projevil okamzite, broker nebyl volan


def test_sync_open_positions_closes_trades_no_longer_on_broker(storage):
    broker = FakeBroker()
    om = OrderManager(broker, storage, _settings(dry_run=False))
    om.enter_position("AAPL", 1, 10, 150.0, 145.0, 160.0, "ml_model", 0.7)

    # broker.get_positions() vraci [] -> pozice uz na Alpaca neexistuje (SL/TP splneno)
    om.sync_open_positions()
    assert storage.get_open_trade("AAPL") is None
    closed = storage.get_recent_closed_trades(1)[0]
    assert closed.exit_reason == "reconciled_on_startup"
    assert closed.exit_price == pytest.approx(150.25)
