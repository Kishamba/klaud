"""Candlestick pattern detection — pure pandas, no TA-Lib."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Body / shadow helpers ────────────────────────────────────────────────────

def _body_size(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def _candle_range(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df["low"]


def _upper_shadow(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df[["open", "close"]].max(axis=1)


def _lower_shadow(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1) - df["low"]


def _is_bullish(df: pd.DataFrame) -> pd.Series:
    return df["close"] > df["open"]


def _is_bearish(df: pd.DataFrame) -> pd.Series:
    return df["close"] < df["open"]


# ─── Pattern definitions ──────────────────────────────────────────────────────

def is_doji(df: pd.DataFrame, tolerance: float = 0.1) -> pd.Series:
    """Body is very small relative to candle range."""
    body = _body_size(df)
    rng = _candle_range(df).replace(0, np.nan)
    return (body / rng) < tolerance


def is_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Current candle bullishly engulfs previous bearish candle."""
    prev_bearish = _is_bearish(df.shift(1))
    curr_bullish = _is_bullish(df)
    engulfs = (df["open"] <= df["close"].shift(1)) & (df["close"] >= df["open"].shift(1))
    return prev_bearish & curr_bullish & engulfs


def is_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Current candle bearishly engulfs previous bullish candle."""
    prev_bullish = _is_bullish(df.shift(1))
    curr_bearish = _is_bearish(df)
    engulfs = (df["open"] >= df["close"].shift(1)) & (df["close"] <= df["open"].shift(1))
    return prev_bullish & curr_bearish & engulfs


def is_hammer(df: pd.DataFrame, ratio: float = 2.0) -> pd.Series:
    """
    Bullish hammer: small body near top, long lower shadow.
    Lower shadow >= ratio * body, upper shadow <= body.
    """
    body = _body_size(df)
    lower = _lower_shadow(df)
    upper = _upper_shadow(df)
    rng = _candle_range(df).replace(0, np.nan)
    body_nonzero = body.replace(0, np.nan)
    return (lower >= ratio * body_nonzero) & (upper <= body_nonzero) & ((body / rng) < 0.4)


def is_hanging_man(df: pd.DataFrame, ratio: float = 2.0) -> pd.Series:
    """Same shape as hammer but appears after uptrend (bearish reversal)."""
    return is_hammer(df, ratio)


def is_shooting_star(df: pd.DataFrame, ratio: float = 2.0) -> pd.Series:
    """
    Bearish shooting star: small body near bottom, long upper shadow.
    Upper shadow >= ratio * body, lower shadow <= body.
    """
    body = _body_size(df)
    lower = _lower_shadow(df)
    upper = _upper_shadow(df)
    rng = _candle_range(df).replace(0, np.nan)
    body_nonzero = body.replace(0, np.nan)
    return (upper >= ratio * body_nonzero) & (lower <= body_nonzero) & ((body / rng) < 0.4)


def is_bull_pin_bar(df: pd.DataFrame, ratio: float = 2.5) -> pd.Series:
    """
    Bullish pin bar: long lower wick, small body and upper wick at top.
    Lower shadow >= ratio * (body + upper shadow).
    """
    body = _body_size(df)
    lower = _lower_shadow(df)
    upper = _upper_shadow(df)
    top_part = (body + upper).replace(0, np.nan)
    return lower >= ratio * top_part


def is_bear_pin_bar(df: pd.DataFrame, ratio: float = 2.5) -> pd.Series:
    """
    Bearish pin bar: long upper wick, small body and lower wick at bottom.
    Upper shadow >= ratio * (body + lower shadow).
    """
    body = _body_size(df)
    lower = _lower_shadow(df)
    upper = _upper_shadow(df)
    bottom_part = (body + lower).replace(0, np.nan)
    return upper >= ratio * bottom_part


def add_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add all pattern columns to DataFrame, return copy."""
    df = df.copy()
    df["pat_doji"] = is_doji(df)
    df["pat_bull_engulf"] = is_bullish_engulfing(df)
    df["pat_bear_engulf"] = is_bearish_engulfing(df)
    df["pat_hammer"] = is_hammer(df)
    df["pat_shooting_star"] = is_shooting_star(df)
    df["pat_bull_pin"] = is_bull_pin_bar(df)
    df["pat_bear_pin"] = is_bear_pin_bar(df)
    return df
