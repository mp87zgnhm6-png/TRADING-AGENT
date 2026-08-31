"""Jednoduche, transparentni pravidlove strategie postavene na indikatorech.

Slouzi jako dalsi "paze" pro bandita ve `signal_engine.py` vedle ML modelu -
davaji agentovi rozumne chovani i v obdobich, kdy se online model teprve
rozkoukava, a soucasne fungujici jako nezavisly hlas pro potvrzeni ML signalu.
"""

from __future__ import annotations


def trend_following_signal(features: dict) -> tuple[int, float]:
    """Sleduje smer trhu: EMA/SMA sklon + MACD histogram, tlumeno pri extremnim RSI."""
    votes = [
        1 if features["ema_ratio"] > 0 else (-1 if features["ema_ratio"] < 0 else 0),
        1 if features["macd_hist"] > 0 else (-1 if features["macd_hist"] < 0 else 0),
        1 if features["sma_ratio"] > 0 else (-1 if features["sma_ratio"] < 0 else 0),
    ]
    score = sum(votes)
    direction = 1 if score > 0 else (-1 if score < 0 else 0)

    rsi = features["rsi_14"]
    rsi_penalty = 0.5 if (rsi > 80 or rsi < 20) else 1.0

    confidence = (abs(score) / len(votes)) * rsi_penalty
    return direction, confidence


def mean_reversion_signal(features: dict) -> tuple[int, float]:
    """Ocekava navrat k prumeru: Bollinger %b + RSI + z-score na extremech."""
    pct_b = features["bb_pct_b"]
    rsi = features["rsi_14"]
    z = features["zscore_20"]

    votes = []
    if pct_b > 1.0:
        votes.append(-1)
    elif pct_b < 0.0:
        votes.append(1)

    if rsi > 70:
        votes.append(-1)
    elif rsi < 30:
        votes.append(1)

    if z > 2.0:
        votes.append(-1)
    elif z < -2.0:
        votes.append(1)

    if not votes:
        return 0, 0.0

    total = sum(votes)
    direction = 1 if total > 0 else (-1 if total < 0 else 0)
    confidence = min(1.0, abs(total) / 3.0)
    return direction, confidence
