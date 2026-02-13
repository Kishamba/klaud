"""
Live trading engine.

Modes:
  paper   — Simulates orders locally, no API calls for orders (market data via API)
  testnet — Real Bybit testnet API
  live    — Real Bybit mainnet (requires LIVE_TRADING=YES env var)

Loop:
  1. Check kill-switch
  2. Fetch latest candles
  3. Compute features + signals
  4. Manage open position (trailing stop update, check stop/TP)
  5. Enter new trade if signalled
  6. Log state
  7. Sleep until next bar
"""
from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from src.config import AppConfig
from src.exchange.client import ExchangeClient
from src.data.downloader import DataDownloader
from src.strategies import get_strategy
from src.utils.helpers import tf_to_ms
from .risk import RiskManager, STOP_FILE

if TYPE_CHECKING:
    from src.strategies.base import BaseStrategy


class PaperPosition:
    """Track a paper/simulated position."""
    def __init__(self, symbol: str, direction: int, entry_price: float,
                 size: float, stop: float, take: float):
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.size = size
        self.stop = stop
        self.take = take
        self.entry_time = datetime.now(timezone.utc).isoformat()
        self.notional = size * entry_price

    @property
    def side_str(self) -> str:
        return "LONG" if self.direction == 1 else "SHORT"


class LiveTrader:
    """
    Live / paper / testnet trading engine.
    Call run() to start the main loop.
    """

    def __init__(self, config: AppConfig, mode: str = "paper"):
        """
        Parameters
        ----------
        config : AppConfig
        mode   : "paper" | "testnet" | "live"
        """
        assert mode in ("paper", "testnet", "live"), f"Invalid mode: {mode}"

        if mode == "live":
            live_ok = os.getenv("LIVE_TRADING", "NO").upper() == "YES"
            if not live_ok:
                raise EnvironmentError(
                    "LIVE_TRADING env var must be set to 'YES' to run in live mode."
                )

        self.mode = mode
        self.config = config
        self.risk = RiskManager(config.risk)
        self.events: list[str] = []  # Ring buffer for UI

        # Exchange client
        testnet = mode != "live"
        self.client = ExchangeClient(
            testnet=testnet,
            max_retries=config.exchange.max_retries,
            retry_delay=config.exchange.retry_delay,
        )

        # Data access (for loading stored candles as seed)
        self.downloader = DataDownloader(self.client, config.data)

        # Strategy
        self.strategy: BaseStrategy = get_strategy(
            config.strategy.name, config.strategy.params
        )

        # State
        self.paper_position: PaperPosition | None = None
        self.running = False

        logger.info(f"LiveTrader initialized: mode={mode}, strategy={self.strategy.name}")

    def _log_event(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.events.append(entry)
        if len(self.events) > 200:
            self.events = self.events[-200:]
        logger.info(msg)

    # ── Market data ───────────────────────────────────────────────────────────

    def _get_candles(self, symbol: str, tf: str, limit: int = 500) -> pd.DataFrame:
        """Fetch recent candles and convert to DataFrame."""
        raw = self.client.get_latest_klines(symbol, tf, limit=limit)
        if not raw:
            return pd.DataFrame()
        cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
        df = pd.DataFrame(raw, columns=cols)
        df = df.astype({c: "float64" for c in cols if c != "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    # ── Order management ──────────────────────────────────────────────────────

    def _enter_trade(self, symbol: str, direction: int, entry_price: float,
                     stop: float, take: float) -> bool:
        size = self.risk.compute_position_size(entry_price, stop)
        if size <= 0:
            return False

        notional = size * entry_price
        ok, reason = self.risk.validate_trade(notional, abs(entry_price - stop))
        if not ok:
            self._log_event(f"Trade rejected: {reason}")
            return False

        side = "Buy" if direction == 1 else "Sell"
        size_str = f"{size:.4f}"

        if self.mode == "paper":
            self.paper_position = PaperPosition(symbol, direction, entry_price, size, stop, take)
            self._log_event(
                f"PAPER {side} {size_str} {symbol} @ {entry_price:.4f} | stop={stop:.4f} take={take:.4f}"
            )
            return True
        else:
            try:
                # Set leverage
                self.client.set_leverage(symbol, int(self.config.risk.max_leverage))
                # Place market order
                order_link_id = f"cb_{uuid.uuid4().hex[:12]}"
                result = self.client.place_order(
                    symbol=symbol, side=side, qty=size_str,
                    order_type="Market", order_link_id=order_link_id
                )
                self._log_event(
                    f"{self.mode.upper()} {side} {size_str} {symbol} @ ~{entry_price:.4f} "
                    f"| orderId={result.get('orderId', '?')}"
                )
                return True
            except Exception as e:
                self._log_event(f"Order failed: {e}")
                return False

    def _exit_trade(self, symbol: str, direction: int, size: float, reason: str) -> bool:
        side = "Sell" if direction == 1 else "Buy"

        if self.mode == "paper":
            self.paper_position = None
            self._log_event(f"PAPER {side} {size:.4f} {symbol} | reason={reason}")
            return True
        else:
            try:
                result = self.client.place_order(
                    symbol=symbol, side=side, qty=f"{size:.4f}",
                    order_type="Market", reduce_only=True
                )
                self._log_event(
                    f"{self.mode.upper()} CLOSE {side} {size:.4f} {symbol} | reason={reason} "
                    f"orderId={result.get('orderId', '?')}"
                )
                return True
            except Exception as e:
                self._log_event(f"Exit order failed: {e}")
                return False

    # ── Main iteration ────────────────────────────────────────────────────────

    def _iterate(self, symbol: str, tf: str) -> None:
        """Single trading iteration for one symbol/tf."""

        # Check kill switch
        stop, reason = self.risk.should_stop()
        if stop:
            self._log_event(f"KILL SWITCH: {reason}")
            self.running = False
            return

        # Fetch candles
        df = self._get_candles(symbol, tf, limit=max(500, self.strategy.warmup_bars + 50))
        if df.empty or len(df) < self.strategy.warmup_bars + 5:
            self._log_event(f"Not enough candles for {symbol}/{tf}")
            return

        # Compute signals
        df = self.strategy.run(df)

        last_signal = int(df["signal"].iloc[-2])  # signal from last CLOSED bar
        last_close = float(df["close"].iloc[-1])
        last_stop = df["stop_price"].iloc[-2] if "stop_price" in df.columns else None
        last_take = df["take_price"].iloc[-2] if "take_price" in df.columns else None

        # ── Update equity ─────────────────────────────────────────────────
        if self.mode == "paper":
            eq = self.risk.equity
            if self.paper_position:
                unrealized = (last_close - self.paper_position.entry_price) * self.paper_position.direction * self.paper_position.size
                eq = self.risk.equity + unrealized
            self.risk.update_equity(eq)
        else:
            try:
                balance = self.client.get_balance()
                self.risk.update_equity(balance["equity"])
            except Exception as e:
                self._log_event(f"Balance fetch failed: {e}")

        # ── Check paper position against price ────────────────────────────
        if self.mode == "paper" and self.paper_position:
            pos = self.paper_position
            last_high = float(df["high"].iloc[-1])
            last_low = float(df["low"].iloc[-1])

            exit_reason = None
            exit_price = last_close

            if pos.direction == 1:
                if last_low <= pos.stop:
                    exit_reason = "stop"
                    exit_price = pos.stop
                elif last_high >= pos.take:
                    exit_reason = "take_profit"
                    exit_price = pos.take
            else:
                if last_high >= pos.stop:
                    exit_reason = "stop"
                    exit_price = pos.stop
                elif last_low <= pos.take:
                    exit_reason = "take_profit"
                    exit_price = pos.take

            if exit_reason:
                pnl = (exit_price - pos.entry_price) * pos.direction * pos.size
                fee = pos.notional * self.config.risk.fee_bps / 10_000
                net_pnl = pnl - fee
                self.risk.equity += net_pnl
                self._log_event(
                    f"PAPER EXIT {pos.side_str} {symbol} | reason={exit_reason} | "
                    f"pnl={net_pnl:+.4f} | equity={self.risk.equity:.2f}"
                )
                self.paper_position = None

        # ── Signal-based exit ─────────────────────────────────────────────
        if self.paper_position:
            pos = self.paper_position
            should_exit = (
                last_signal == 0
                or (pos.direction == 1 and last_signal == -1)
                or (pos.direction == -1 and last_signal == 1)
            )
            if should_exit:
                pnl = (last_close - pos.entry_price) * pos.direction * pos.size
                fee = pos.notional * self.config.risk.fee_bps / 10_000
                net_pnl = pnl - fee
                if self.mode == "paper":
                    self.risk.equity += net_pnl
                self._exit_trade(symbol, pos.direction, pos.size, "signal")

        # ── Entry ─────────────────────────────────────────────────────────
        has_position = self.paper_position is not None
        if not has_position and last_signal != 0 and last_stop is not None and last_take is not None:
            import numpy as np
            if not (pd.isna(last_stop) or pd.isna(last_take)):
                self._enter_trade(symbol, last_signal, last_close, float(last_stop), float(last_take))

        # Log status
        pos_str = f"{self.paper_position.side_str} {symbol}" if self.paper_position else "FLAT"
        self._log_event(
            f"Tick: {symbol}/{tf} close={last_close:.4f} signal={last_signal} "
            f"pos={pos_str} equity={self.risk.equity:.2f}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        symbol: str | None = None,
        tf: str | None = None,
    ) -> None:
        """Start the live trading loop (blocking)."""
        symbol = symbol or self.config.data.symbols[0]
        tf = tf or self.config.data.timeframes[-1]  # default: longer TF (1h)
        interval_ms = tf_to_ms(tf)
        sleep_sec = interval_ms / 1000

        self.running = True
        self._log_event(f"Starting live loop: {symbol}/{tf} mode={self.mode}")

        while self.running:
            if Path(STOP_FILE).exists():
                self._log_event("STOP_TRADING file detected. Halting.")
                self.running = False
                break
            try:
                self._iterate(symbol, tf)
            except Exception as e:
                self._log_event(f"ERROR in iteration: {e}")
                logger.exception(e)

            if not self.running:
                break

            self._log_event(f"Sleeping {sleep_sec:.0f}s until next bar...")
            time.sleep(sleep_sec)

        self._log_event("Live trading loop stopped.")

    def stop(self) -> None:
        self.running = False
        self._log_event("Stop requested.")

    def status(self) -> dict:
        pos_info = None
        if self.paper_position:
            p = self.paper_position
            pos_info = {
                "symbol": p.symbol, "side": p.side_str,
                "entry_price": p.entry_price, "size": p.size,
                "stop": p.stop, "take": p.take, "entry_time": p.entry_time,
            }
        return {
            "mode": self.mode,
            "running": self.running,
            "equity": self.risk.equity,
            "peak_equity": self.risk.peak_equity,
            "position": pos_info,
            "last_events": self.events[-20:],
        }
