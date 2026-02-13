"""
Risk manager for live trading.
Tracks daily/weekly PnL, enforces kill-switch conditions,
and validates individual trade sizes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from loguru import logger

from src.config import RiskConfig

STOP_FILE = "STOP_TRADING"


class RiskManager:
    """
    Enforces risk limits during live trading.

    Kill-switch triggers:
      1. STOP_TRADING file exists in project root
      2. Daily loss exceeds max_daily_loss_pct
      3. Weekly loss exceeds max_weekly_loss_pct
      4. Drawdown from peak exceeds max_drawdown_pct
    """

    def __init__(self, config: RiskConfig, state_file: str = "live_state.json"):
        self.cfg = config
        self.state_file = Path(state_file)
        self.equity = config.initial_capital
        self.peak_equity = config.initial_capital
        self.daily_start_equity = config.initial_capital
        self.weekly_start_equity = config.initial_capital
        self.last_day = self._today()
        self.last_week = self._week()
        self._load_state()

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _week(self) -> str:
        d = datetime.now(timezone.utc)
        monday = d - timedelta(days=d.weekday())
        return monday.strftime("%Y-%m-%d")

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    s = json.load(f)
                self.equity = s.get("equity", self.equity)
                self.peak_equity = s.get("peak_equity", self.equity)
                self.daily_start_equity = s.get("daily_start_equity", self.equity)
                self.weekly_start_equity = s.get("weekly_start_equity", self.equity)
                self.last_day = s.get("last_day", self._today())
                self.last_week = s.get("last_week", self._week())
                logger.debug(f"Risk state loaded: equity={self.equity:.2f}")
            except Exception as e:
                logger.warning(f"Could not load risk state: {e}")

    def save_state(self) -> None:
        with open(self.state_file, "w") as f:
            json.dump({
                "equity": self.equity,
                "peak_equity": self.peak_equity,
                "daily_start_equity": self.daily_start_equity,
                "weekly_start_equity": self.weekly_start_equity,
                "last_day": self.last_day,
                "last_week": self.last_week,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)

    def update_equity(self, new_equity: float) -> None:
        today = self._today()
        week = self._week()

        if today != self.last_day:
            self.daily_start_equity = self.equity
            self.last_day = today
            logger.info(f"New trading day. Daily equity reset to {self.equity:.2f}")

        if week != self.last_week:
            self.weekly_start_equity = self.equity
            self.last_week = week
            logger.info(f"New trading week. Weekly equity reset to {self.equity:.2f}")

        self.equity = new_equity
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity

        self.save_state()

    # ── Kill-switch checks ────────────────────────────────────────────────────

    def should_stop(self) -> tuple[bool, str]:
        """Returns (should_stop: bool, reason: str)."""

        # Check STOP_TRADING file
        if Path(STOP_FILE).exists():
            return True, f"STOP_TRADING file found"

        if self.equity <= 0:
            return True, "Equity is zero or negative"

        # Daily loss
        daily_loss_pct = (self.equity - self.daily_start_equity) / self.daily_start_equity * 100
        if daily_loss_pct < -self.cfg.max_daily_loss_pct:
            return True, f"Daily loss {daily_loss_pct:.1f}% exceeds limit {self.cfg.max_daily_loss_pct}%"

        # Weekly loss
        weekly_loss_pct = (self.equity - self.weekly_start_equity) / self.weekly_start_equity * 100
        if weekly_loss_pct < -self.cfg.max_weekly_loss_pct:
            return True, f"Weekly loss {weekly_loss_pct:.1f}% exceeds limit {self.cfg.max_weekly_loss_pct}%"

        # Max drawdown
        dd_pct = (self.equity - self.peak_equity) / self.peak_equity * 100
        if dd_pct < -self.cfg.max_drawdown_pct:
            return True, f"Drawdown {dd_pct:.1f}% exceeds limit {self.cfg.max_drawdown_pct}%"

        return False, ""

    def validate_trade(
        self,
        notional: float,
        stop_distance: float,
    ) -> tuple[bool, str]:
        """Validate a potential trade. Returns (is_valid, reason)."""
        if stop_distance <= 0:
            return False, "Stop distance is zero or negative"

        if notional < self.cfg.min_notional:
            return False, f"Notional {notional:.2f} below min {self.cfg.min_notional}"

        max_notional = self.equity * self.cfg.max_leverage
        if notional > max_notional:
            return False, f"Notional {notional:.2f} exceeds max {max_notional:.2f} (leverage limit)"

        risk_amount = self.equity * self.cfg.risk_per_trade
        if stop_distance > 0:
            implied_size = risk_amount / stop_distance
            implied_notional = implied_size * (notional / (notional / (risk_amount / stop_distance)))
            # Just check risk fraction
        return True, ""

    def compute_position_size(self, entry_price: float, stop_price: float) -> float:
        """Compute position size (qty in base asset) using fixed fractional."""
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return 0.0
        risk_amount = self.equity * self.cfg.risk_per_trade
        size = risk_amount / stop_distance
        notional = size * entry_price
        # Cap at leverage limit
        max_notional = self.equity * self.cfg.max_leverage
        if notional > max_notional:
            size = max_notional / entry_price
        return size
