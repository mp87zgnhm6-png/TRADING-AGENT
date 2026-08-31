import pytest

from trading_agent.data.storage import Storage


@pytest.fixture
def storage(tmp_path):
    s = Storage(str(tmp_path / "test.db"))
    yield s
    s.close()


def test_open_and_close_trade_computes_pnl(storage):
    tid = storage.open_trade("AAPL", "buy", 10, 100.0, 98.0, 106.0, "ml_model", 0.7)
    assert storage.get_open_trade("AAPL") is not None

    closed = storage.close_trade(tid, 103.0, "take_profit")
    assert closed.pnl == pytest.approx(30.0)
    assert closed.pnl_pct == pytest.approx(0.03)
    assert storage.get_open_trade("AAPL") is None


def test_short_trade_pnl_direction(storage):
    tid = storage.open_trade("AAPL", "sell", 10, 100.0, 105.0, 90.0, "mean_reversion", 0.6)
    closed = storage.close_trade(tid, 95.0, "take_profit")
    assert closed.pnl == pytest.approx(50.0)  # short: cena klesla o 5 -> zisk 5*10
    assert closed.pnl_pct == pytest.approx(0.05)


def test_equity_curve_and_peak(storage):
    storage.record_equity(100_000, 50_000, 50_000)
    storage.record_equity(105_000, 50_000, 50_000)
    storage.record_equity(101_000, 50_000, 50_000)
    assert storage.get_peak_equity() == pytest.approx(105_000)
    assert storage.get_latest_equity() == pytest.approx(101_000)


def test_daily_pnl_only_counts_trades_after_cutoff(storage):
    tid1 = storage.open_trade("AAPL", "buy", 10, 100, 95, 110, "ml_model", 0.6)
    storage.close_trade(tid1, 105, "take_profit", exit_time="2020-01-01T00:00:00+00:00")
    tid2 = storage.open_trade("MSFT", "buy", 5, 200, 190, 220, "ml_model", 0.6)
    storage.close_trade(tid2, 210, "take_profit", exit_time="2099-01-01T00:00:00+00:00")

    pnl_recent = storage.get_daily_pnl("2050-01-01T00:00:00+00:00")
    assert pnl_recent == pytest.approx(50.0)  # jen tid2: 5 * (210-200)


def test_get_all_open_trades(storage):
    storage.open_trade("AAPL", "buy", 1, 100, 95, 110, "ml_model", 0.5)
    storage.open_trade("MSFT", "buy", 1, 200, 190, 220, "ml_model", 0.5)
    open_trades = storage.get_all_open_trades()
    assert {t.symbol for t in open_trades} == {"AAPL", "MSFT"}
