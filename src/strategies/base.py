"""Abstract base class for all trading strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from pydantic import BaseModel


class BaseStrategy(ABC):
    """
    Interface every strategy must implement.

    Signals convention:
        signal column values: 1 = long, -1 = short, 0 = flat
        stop_price: float — required for every non-zero signal
        take_price: float — required for every non-zero signal
    """

    name: str = "BaseStrategy"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params: dict[str, Any] = params or {}
        self._apply_defaults()

    def _apply_defaults(self) -> None:
        for k, v in self._default_params().items():
            self.params.setdefault(k, v)

    def _default_params(self) -> dict[str, Any]:
        return {}

    @property
    def warmup_bars(self) -> int:
        """Number of bars needed before signals are valid."""
        return 50

    @abstractmethod
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator/feature columns to df. Must not look forward."""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add columns:
          signal (int): 1/−1/0
          stop_price (float): stop loss price
          take_price (float): take profit price
        Returns modified df.
        """

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline: compute features + generate signals."""
        df = self.compute_features(df.copy())
        df = self.generate_signals(df)
        return df

    def explain_last_signal(self, df: pd.DataFrame) -> str:
        """Human-readable explanation of the last non-zero signal."""
        if "signal" not in df.columns:
            return "No signal column. Run strategy first."
        last = df[df["signal"] != 0]
        if last.empty:
            return "No active signals."
        row = last.iloc[-1]
        return (
            f"Signal: {'LONG' if row.signal == 1 else 'SHORT'} | "
            f"Time: {row.name} | "
            f"Close: {row.close:.4f} | "
            f"Stop: {row.get('stop_price', '?'):.4f} | "
            f"Take: {row.get('take_price', '?'):.4f}"
        )

    def param_space(self) -> dict[str, tuple]:
        """
        Return Optuna search space as {param_name: (low, high, type)}.
        Override in subclass for optimization support.
        """
        return {}
