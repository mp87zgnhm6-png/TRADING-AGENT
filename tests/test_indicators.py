import numpy as np
import pandas as pd

from trading_agent.features.indicators import (
    FEATURE_COLUMNS,
    atr,
    bollinger_bands,
    build_feature_frame,
    build_feature_row,
    ema,
    macd,
    rsi,
    sma,
)


def _make_ohlcv(n=200, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.001, n)
    close = 100 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(rng.normal(0, 0.0005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0005, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.integers(1000, 10000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_sma_matches_manual_mean():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(s, 3)
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == 2.0
    assert result.iloc[4] == 4.0


def test_ema_reacts_faster_than_sma_to_shock():
    s = pd.Series([1.0] * 20 + [10.0] * 20)
    e = ema(s, 5)
    m = sma(s, 5)
    assert e.iloc[22] > m.iloc[22]


def test_rsi_bounds_and_extremes():
    df = _make_ohlcv()
    r = rsi(df["close"], 14)
    assert r.dropna().between(0, 100).all()

    rising = pd.Series(np.linspace(1, 100, 60))
    assert rsi(rising, 14).iloc[-1] > 70

    falling = pd.Series(np.linspace(100, 1, 60))
    assert rsi(falling, 14).iloc[-1] < 30


def test_macd_columns_and_no_lookahead():
    df = _make_ohlcv()
    result = macd(df["close"])
    assert set(result.columns) == {"macd", "macd_signal", "macd_hist"}
    # cisty online vypocet - hodnota v case t nesmi zaviset na budoucich datech
    truncated = macd(df["close"].iloc[:100])
    pd.testing.assert_series_equal(result["macd"].iloc[:100], truncated["macd"], check_names=False)


def test_bollinger_bands_ordering():
    df = _make_ohlcv()
    bb = bollinger_bands(df["close"])
    valid = bb.dropna()
    assert (valid["bb_upper"] >= valid["bb_mid"]).all()
    assert (valid["bb_mid"] >= valid["bb_lower"]).all()


def test_atr_is_positive():
    df = _make_ohlcv()
    a = atr(df, 14)
    assert (a.dropna() > 0).all()


def test_build_feature_row_requires_enough_history():
    df = _make_ohlcv(n=200)
    feat, row = build_feature_row(df.iloc[:10])
    assert feat is None and row is None

    feat, row = build_feature_row(df)
    assert feat is not None
    assert set(feat.keys()) == set(FEATURE_COLUMNS)
    assert all(np.isfinite(v) for v in feat.values())
    assert row["close"] == df["close"].iloc[-1]


def test_build_feature_frame_no_nan_after_warmup():
    df = _make_ohlcv(n=200)
    enriched = build_feature_frame(df)
    tail = enriched[FEATURE_COLUMNS].iloc[50:]
    assert not tail.isna().any().any()
