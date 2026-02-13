"""
Mean-reversion strategy: Bollinger Bands + RSI filter + ATR stops.

Rules:
  LONG  when close < lower_band AND RSI < rsi_oversold
  SHORT when close > upper_band AND RSI > rsi_overbought
  EXIT  when close crosses mid_band (mean)

Stop:  entry_price ∓ atr_multiplier_stop × ATR
Take:  Bollinger mid-band (dynamic)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators.technical import rsi, bollinger_bands, atr
from .base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    name = "MeanReversionStrategy"

    def _default_params(self) -> dict[str, Any]:
        return {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 35.0,
            "rsi_overbought": 65.0,
            "atr_period": 14,
            "atr_multiplier_stop": 1.5,
            "min_bb_width": 0.01,    # Minimum BB width to trade (volatility filter)
            "allow_short": True,
        }

    @property
    def warmup_bars(self) -> int:
        return max(self.params["bb_period"], self.params["rsi_period"]) * 3 + 10

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = bollinger_bands(
            df["close"], p["bb_period"], p["bb_std"]
        )
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
        df["rsi"] = rsi(df["close"], p["rsi_period"])
        df["atr"] = atr(df, p["atr_period"])
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        df["signal"] = 0
        df["stop_price"] = np.nan
        df["take_price"] = np.nan

        vol_filter = df["bb_width"] > p["min_bb_width"]

        long_entry = (
            (df["close"] < df["bb_lower"])
            & (df["rsi"] < p["rsi_oversold"])
            & vol_filter
        )
        short_entry = (
            (df["close"] > df["bb_upper"])
            & (df["rsi"] > p["rsi_overbought"])
            & vol_filter
        )

        df.loc[long_entry, "signal"] = 1
        if p["allow_short"]:
            df.loc[short_entry, "signal"] = -1

        # For longs: stop below entry by ATR; take = mid band
        long_mask = df["signal"] == 1
        df.loc[long_mask, "stop_price"] = (
            df.loc[long_mask, "close"] - p["atr_multiplier_stop"] * df.loc[long_mask, "atr"]
        )
        df.loc[long_mask, "take_price"] = df.loc[long_mask, "bb_mid"]

        # For shorts
        short_mask = df["signal"] == -1
        df.loc[short_mask, "stop_price"] = (
            df.loc[short_mask, "close"] + p["atr_multiplier_stop"] * df.loc[short_mask, "atr"]
        )
        df.loc[short_mask, "take_price"] = df.loc[short_mask, "bb_mid"]

        return df

    def param_space(self) -> dict[str, tuple]:
        return {
            "bb_period":           (10, 40, "int"),
            "bb_std":              (1.5, 3.0, "float"),
            "rsi_oversold":        (20.0, 45.0, "float"),
            "rsi_overbought":      (55.0, 80.0, "float"),
            "atr_multiplier_stop": (1.0, 3.0, "float"),
        }
