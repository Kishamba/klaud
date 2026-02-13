"""Technical indicators — pure pandas/numpy, no TA-Lib required."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Basic ────────────────────────────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


# ─── Volatility ───────────────────────────────────────────────────────────────

def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def bb_width(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    upper, mid, lower = bollinger_bands(series, period, std_dev)
    return (upper - lower) / mid.replace(0, np.nan)


def bb_pct_b(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    upper, mid, lower = bollinger_bands(series, period, std_dev)
    band_width = (upper - lower).replace(0, np.nan)
    return (series - lower) / band_width


# ─── Momentum ─────────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    low_min = df["low"].rolling(window=k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(window=k_period, min_periods=k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100.0 * (df["close"] - low_min) / denom
    d = sma(k, d_period)
    return k, d


# ─── Trend ────────────────────────────────────────────────────────────────────

def adx(
    df: pd.DataFrame, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (ADX, +DI, -DI)."""
    high, low, close = df["high"], df["low"], df["close"]

    up_move = high.diff()
    down_move = -(low.diff())

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm_s = pd.Series(plus_dm, index=df.index)
    minus_dm_s = pd.Series(minus_dm, index=df.index)

    tr = true_range(df)
    atr_s = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * plus_dm_s.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_s.replace(0, np.nan)
    minus_di = 100.0 * minus_dm_s.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_s.replace(0, np.nan)

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx_line = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    return adx_line, plus_di, minus_di


def ema_crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """Returns +1 on golden cross, -1 on death cross, 0 otherwise."""
    cross = pd.Series(0, index=fast.index)
    cross[fast > slow] = 1
    cross[fast < slow] = -1
    return cross


def ema_cross_signal(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """Returns +1 exactly on golden cross bar, -1 on death cross bar."""
    state = ema_crossover(fast, slow)
    state_shifted = state.shift(1)
    signal = pd.Series(0, index=fast.index)
    signal[(state == 1) & (state_shifted != 1)] = 1
    signal[(state == -1) & (state_shifted != -1)] = -1
    return signal


# ─── Combined feature builder ─────────────────────────────────────────────────

def add_all_indicators(
    df: pd.DataFrame,
    ema_fast: int = 12,
    ema_slow: int = 26,
    adx_period: int = 14,
    atr_period: int = 14,
    rsi_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> pd.DataFrame:
    """Add common indicators to df in-place, return df."""
    df = df.copy()
    df["ema_fast"] = ema(df["close"], ema_fast)
    df["ema_slow"] = ema(df["close"], ema_slow)
    df["atr"] = atr(df, atr_period)
    df["adx"], df["plus_di"], df["minus_di"] = adx(df, adx_period)
    df["rsi"] = rsi(df["close"], rsi_period)
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = bollinger_bands(df["close"], bb_period, bb_std)
    df["bb_width"] = bb_width(df["close"], bb_period, bb_std)
    return df
