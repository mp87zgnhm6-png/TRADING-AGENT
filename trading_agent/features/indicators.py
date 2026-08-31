"""Technicke indikatory a feature engineering pocitane z OHLCV svicek.

Vsechny funkce pracuji cistě v pandas/numpy (zadna zavislost na TA-Lib), aby se
daly bez problemu pouzit jak v backtestu (vektorove na cely DataFrame), tak
v live agentovi (na rolling buffer, kde nas zajima jen posledni radek).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Nazvy sloupcu s "engineered" features - presne tyto se posilaji do self-learning
# modelu (river ocekava plochy dict {nazev_featury: hodnota}).
FEATURE_COLUMNS = [
    "ret_1",
    "sma_ratio",
    "ema_ratio",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_pct_b",
    "bb_bandwidth",
    "atr_pct",
    "momentum_10",
    "zscore_20",
    "volume_ratio_20",
]


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilderovo vyhlazeni (klasicky RSI)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 znamena, ze v okne nebyla jedina klesajici svicka - to je
    # maximalne prekoupeno (RSI=100), ne neutralni hodnota; NaN (warmup, kdy
    # avg_loss jeste neni spocitane) zustava beze zmeny diky `!= 0` na NaN -> True.
    result = result.where(avg_loss != 0, 100.0)
    # uplne plocha cena (zadny pohyb v okne vubec) je skutecne neutralni
    result = result.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return result.clip(0.0, 100.0)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def bollinger_bands(series: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, window)
    std = series.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    band_range = (upper - lower).replace(0.0, np.nan)
    pct_b = (series - lower) / band_range
    bandwidth = band_range / mid.replace(0.0, np.nan)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_pct_b": pct_b, "bb_bandwidth": bandwidth}
    )


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def momentum(series: pd.Series, window: int) -> pd.Series:
    return series.pct_change(window)


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    avg = volume.rolling(window, min_periods=window).mean()
    return volume / avg.replace(0.0, np.nan)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Vezme OHLCV DataFrame a vrati ho obohaceny o vsechny indikatory a features.

    Puvodni OHLCV sloupce jsou zachovany (napr. pro risk management potrebujeme
    surove ATR v cene, ne jen normalizovany atr_pct pouzity jako ML feature).
    """
    out = df.copy()
    close = out["close"]

    out["ret_1"] = close.pct_change(1)

    sma_fast = sma(close, 10)
    sma_slow = sma(close, 30)
    out["sma_ratio"] = (sma_fast / sma_slow) - 1.0

    ema_fast = ema(close, 12)
    ema_slow = ema(close, 26)
    out["ema_ratio"] = (ema_fast / ema_slow) - 1.0

    out["rsi_14"] = rsi(close, 14)

    macd_df = macd(close)
    out["macd"] = macd_df["macd"] / close
    out["macd_signal"] = macd_df["macd_signal"] / close
    out["macd_hist"] = macd_df["macd_hist"] / close

    bb = bollinger_bands(close)
    out["bb_pct_b"] = bb["bb_pct_b"]
    out["bb_bandwidth"] = bb["bb_bandwidth"]

    out["atr"] = atr(out, 14)
    out["atr_pct"] = out["atr"] / close

    out["momentum_10"] = momentum(close, 10)
    out["zscore_20"] = zscore(close, 20)
    out["volume_ratio_20"] = volume_ratio(out["volume"], 20)

    return out


def build_feature_row(df: pd.DataFrame) -> tuple[dict | None, pd.Series | None]:
    """Spocita features na celem df a vrati (feature_dict, cely_posledni_radek) pro posledni svicku.

    Vraci (None, None) pokud jeste neni dost historie (NaN v nekterem z features).
    `feature_dict` obsahuje pouze FEATURE_COLUMNS (vstup do ML modelu i pravidel),
    `posledni_radek` navic obsahuje surove OHLCV + atr - pro risk management a logovani.
    """
    if len(df) < 35:
        return None, None
    enriched = build_feature_frame(df)
    last = enriched.iloc[-1]
    if last[FEATURE_COLUMNS].isna().any():
        return None, None
    feature_dict = {col: float(last[col]) for col in FEATURE_COLUMNS}
    return feature_dict, last
