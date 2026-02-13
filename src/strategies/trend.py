"""
Trend-following strategy: EMA crossover + ADX filter + ATR stops.

Rules:
  LONG  when EMA_fast crosses above EMA_slow AND ADX > threshold
  SHORT when EMA_fast crosses below EMA_slow AND ADX > threshold
  (optional: confirm with bullish/bearish candle patterns)

Stop:  entry_price ∓ atr_multiplier_stop × ATR
Take:  entry_price ± atr_multiplier_tp   × ATR
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators.technical import ema, atr, adx
from src.indicators.patterns import add_patterns
from .base import BaseStrategy


class TrendStrategy(BaseStrategy):
    name = "TrendStrategy"

    def _default_params(self) -> dict[str, Any]:
        return {
            "ema_fast": 12,
            "ema_slow": 26,
            "adx_period": 14,
            "adx_threshold": 25.0,
            "atr_period": 14,
            "atr_multiplier_stop": 1.5,
            "atr_multiplier_tp": 3.0,
            "allow_short": True,
            "use_patterns": False,
        }

    @property
    def warmup_bars(self) -> int:
        return max(self.params["ema_slow"], self.params["adx_period"]) * 3 + 10

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        df["ema_fast"] = ema(df["close"], p["ema_fast"])
        df["ema_slow"] = ema(df["close"], p["ema_slow"])
        df["atr"] = atr(df, p["atr_period"])
        df["adx"], df["plus_di"], df["minus_di"] = adx(df, p["adx_period"])
        if p["use_patterns"]:
            df = add_patterns(df)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        df["signal"] = 0
        df["stop_price"] = np.nan
        df["take_price"] = np.nan

        fast = df["ema_fast"]
        slow = df["ema_slow"]
        adx_val = df["adx"]
        atr_val = df["atr"]

        # State: 1 when fast > slow, -1 when fast < slow
        trend_state = pd.Series(0, index=df.index)
        trend_state[fast > slow] = 1
        trend_state[fast < slow] = -1

        # Cross events (state changes)
        prev_state = trend_state.shift(1)

        trend_filter = adx_val > p["adx_threshold"]

        long_cross = (trend_state == 1) & (prev_state == -1) & trend_filter
        short_cross = (trend_state == -1) & (prev_state == 1) & trend_filter

        # Optional pattern confirmation
        if p["use_patterns"] and "pat_bull_engulf" in df.columns:
            long_cross = long_cross & (df["pat_bull_engulf"] | df["pat_hammer"] | df["pat_bull_pin"])
            short_cross = short_cross & (df["pat_bear_engulf"] | df["pat_shooting_star"] | df["pat_bear_pin"])

        # Assign signals
        df.loc[long_cross, "signal"] = 1
        if p["allow_short"]:
            df.loc[short_cross, "signal"] = -1

        # Compute stop/take for long entries
        long_mask = df["signal"] == 1
        df.loc[long_mask, "stop_price"] = df.loc[long_mask, "close"] - p["atr_multiplier_stop"] * atr_val[long_mask]
        df.loc[long_mask, "take_price"] = df.loc[long_mask, "close"] + p["atr_multiplier_tp"] * atr_val[long_mask]

        # Compute stop/take for short entries
        short_mask = df["signal"] == -1
        df.loc[short_mask, "stop_price"] = df.loc[short_mask, "close"] + p["atr_multiplier_stop"] * atr_val[short_mask]
        df.loc[short_mask, "take_price"] = df.loc[short_mask, "close"] - p["atr_multiplier_tp"] * atr_val[short_mask]

        # Exit signal: cross in opposite direction (or ADX falls below threshold)
        # We encode this as signal=0 which triggers exit in backtest engine
        # (engine exits when next signal differs from current position direction)
        return df

    def param_space(self) -> dict[str, tuple]:
        return {
            "ema_fast":             (5, 20, "int"),
            "ema_slow":             (20, 60, "int"),
            "adx_threshold":        (15.0, 40.0, "float"),
            "atr_multiplier_stop":  (1.0, 3.0, "float"),
            "atr_multiplier_tp":    (2.0, 6.0, "float"),
        }
