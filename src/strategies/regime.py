"""
Regime-switching strategy: detects trend vs. ranging market and
delegates to TrendStrategy or MeanReversionStrategy accordingly.

Regime detection:
  TRENDING if ADX > adx_trend_threshold
  RANGING  if ADX < adx_range_threshold
  (hysteresis zone between thresholds keeps previous regime)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators.technical import atr, adx
from .base import BaseStrategy
from .trend import TrendStrategy
from .mean_reversion import MeanReversionStrategy


class RegimeStrategy(BaseStrategy):
    name = "RegimeStrategy"

    def _default_params(self) -> dict[str, Any]:
        return {
            # Regime detection
            "adx_period": 14,
            "adx_trend_threshold": 25.0,
            "adx_range_threshold": 20.0,
            # Sub-strategy params are passed through with sub_ prefix
            # TrendStrategy params
            "ema_fast": 12,
            "ema_slow": 26,
            "adx_threshold": 25.0,
            "atr_multiplier_stop": 1.5,
            "atr_multiplier_tp": 3.0,
            # MeanReversionStrategy params
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 35.0,
            "rsi_overbought": 65.0,
            "allow_short": True,
            "use_patterns": False,
        }

    @property
    def warmup_bars(self) -> int:
        return max(self.params.get("ema_slow", 26), self.params.get("bb_period", 20)) * 3 + 20

    def _make_trend(self) -> TrendStrategy:
        p = self.params
        return TrendStrategy(params={
            "ema_fast": p["ema_fast"],
            "ema_slow": p["ema_slow"],
            "adx_period": p["adx_period"],
            "adx_threshold": p["adx_threshold"],
            "atr_period": p["adx_period"],
            "atr_multiplier_stop": p["atr_multiplier_stop"],
            "atr_multiplier_tp": p["atr_multiplier_tp"],
            "allow_short": p["allow_short"],
            "use_patterns": p["use_patterns"],
        })

    def _make_mr(self) -> MeanReversionStrategy:
        p = self.params
        return MeanReversionStrategy(params={
            "bb_period": p["bb_period"],
            "bb_std": p["bb_std"],
            "rsi_period": p["rsi_period"],
            "rsi_oversold": p["rsi_oversold"],
            "rsi_overbought": p["rsi_overbought"],
            "atr_period": p["adx_period"],
            "atr_multiplier_stop": p["atr_multiplier_stop"],
            "allow_short": p["allow_short"],
        })

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        df["adx"], df["plus_di"], df["minus_di"] = adx(df, p["adx_period"])

        # Regime: 1 = trending, 0 = ranging, -1 = unknown
        regime = pd.Series(np.nan, index=df.index)
        regime[df["adx"] > p["adx_trend_threshold"]] = 1.0
        regime[df["adx"] < p["adx_range_threshold"]] = 0.0
        df["regime"] = regime.ffill().fillna(0.0)

        # Compute features for both sub-strategies
        trend_strat = self._make_trend()
        mr_strat = self._make_mr()

        df = trend_strat.compute_features(df)
        df = mr_strat.compute_features(df)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        trend_strat = self._make_trend()
        mr_strat = self._make_mr()

        # Generate signals from both sub-strategies
        df_trend = trend_strat.generate_signals(df.copy())
        df_mr = mr_strat.generate_signals(df.copy())

        df["signal"] = 0
        df["stop_price"] = np.nan
        df["take_price"] = np.nan

        trending = df["regime"] == 1.0

        # In trending regime: use trend signals
        df.loc[trending, "signal"] = df_trend.loc[trending, "signal"]
        df.loc[trending, "stop_price"] = df_trend.loc[trending, "stop_price"]
        df.loc[trending, "take_price"] = df_trend.loc[trending, "take_price"]

        # In ranging regime: use mean-reversion signals
        ranging = df["regime"] == 0.0
        df.loc[ranging, "signal"] = df_mr.loc[ranging, "signal"]
        df.loc[ranging, "stop_price"] = df_mr.loc[ranging, "stop_price"]
        df.loc[ranging, "take_price"] = df_mr.loc[ranging, "take_price"]

        return df

    def param_space(self) -> dict[str, tuple]:
        return {
            "adx_trend_threshold": (20.0, 35.0, "float"),
            "ema_fast":            (5, 20, "int"),
            "ema_slow":            (20, 50, "int"),
            "atr_multiplier_stop": (1.0, 3.0, "float"),
            "atr_multiplier_tp":   (2.0, 5.0, "float"),
            "bb_period":           (10, 30, "int"),
            "rsi_oversold":        (25.0, 40.0, "float"),
            "rsi_overbought":      (60.0, 75.0, "float"),
        }
