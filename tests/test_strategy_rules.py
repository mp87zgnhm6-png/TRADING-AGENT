import pytest

from trading_agent.strategy import rules


def _features(**overrides):
    base = {
        "ema_ratio": 0.0,
        "macd_hist": 0.0,
        "sma_ratio": 0.0,
        "rsi_14": 50.0,
        "bb_pct_b": 0.5,
        "zscore_20": 0.0,
    }
    base.update(overrides)
    return base


def test_trend_following_bullish_agreement():
    feat = _features(ema_ratio=0.01, macd_hist=0.001, sma_ratio=0.005, rsi_14=60)
    direction, confidence = rules.trend_following_signal(feat)
    assert direction == 1
    assert confidence == pytest.approx(1.0)


def test_trend_following_penalizes_extreme_rsi():
    feat = _features(ema_ratio=0.01, macd_hist=0.001, sma_ratio=0.005, rsi_14=85)
    direction, confidence = rules.trend_following_signal(feat)
    assert direction == 1
    assert confidence == pytest.approx(0.5)


def test_trend_following_mixed_signals_low_confidence():
    feat = _features(ema_ratio=0.01, macd_hist=-0.001, sma_ratio=0.0, rsi_14=50)
    _, confidence = rules.trend_following_signal(feat)
    assert confidence < 1.0


def test_mean_reversion_oversold_triggers_buy():
    feat = _features(bb_pct_b=-0.1, rsi_14=25, zscore_20=-2.5)
    direction, confidence = rules.mean_reversion_signal(feat)
    assert direction == 1
    assert confidence == pytest.approx(1.0)


def test_mean_reversion_overbought_triggers_sell():
    feat = _features(bb_pct_b=1.1, rsi_14=75, zscore_20=2.5)
    direction, _ = rules.mean_reversion_signal(feat)
    assert direction == -1


def test_mean_reversion_neutral_gives_no_signal():
    feat = _features()
    direction, confidence = rules.mean_reversion_signal(feat)
    assert direction == 0
    assert confidence == 0.0
