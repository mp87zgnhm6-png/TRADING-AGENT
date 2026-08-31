import numpy as np
import pandas as pd

from trading_agent.backtest.backtester import Backtester
from trading_agent.config import Settings


def _make_ohlcv(n, seed, drift=0.0003):
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, 0.002, n)
    close = 100 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.integers(1000, 10000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_backtester_runs_end_to_end_and_learns():
    settings = Settings(MIN_CONFIDENCE="0.3")
    historical = {"AAPL": _make_ohlcv(300, seed=1), "MSFT": _make_ohlcv(300, seed=2, drift=-0.0002)}

    bt = Backtester(settings, initial_equity=100_000.0, seed=7)
    result = bt.run(historical)

    assert len(result.equity_curve) == len(set(historical["AAPL"].index) | set(historical["MSFT"].index))
    assert result.stats["initial_equity"] == 100_000.0
    assert set(result.stats.keys()) >= {
        "final_equity", "total_return_pct", "cagr_pct", "sharpe_ratio",
        "max_drawdown_pct", "num_trades", "win_rate_pct", "profit_factor",
    }
    # samouceni probehlo behem backtestu (model videl vice vzorku nez jen 1 na obchod)
    assert bt.model.n_samples > 0
    # bandit dostal zpetnou vazbu, pokud vubec nejaky obchod probehl
    if result.stats["num_trades"] > 0:
        assert sum(bt.bandit.counts.values()) == result.stats["num_trades"]


def test_backtester_raises_on_insufficient_data():
    settings = Settings()
    bt = Backtester(settings)
    try:
        bt.run({"AAPL": _make_ohlcv(10, seed=1)})
        assert False, "mel vyhodit ValueError na prilis kratka data"
    except ValueError:
        pass
