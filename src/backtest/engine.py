"""
Bar-by-bar backtest engine.

Fill model:
  - Entry: next bar OPEN + slippage (signal generated at bar close)
  - Stop/TP: if hit intra-bar (using H/L), fill at stop/TP price (no slippage on stop)
  - Signal exit: next bar OPEN − slippage
  - Conservative: if both stop and TP hit in same bar, stop takes priority

Position: one open trade per symbol at a time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import BacktestConfig, RiskConfig
from src.strategies.base import BaseStrategy
from .metrics import compute_metrics, drawdown_series


@dataclass
class Trade:
    trade_id: int
    symbol: str
    direction: int          # 1=long, -1=short
    entry_time: str
    entry_price: float
    stop_price: float
    take_price: float
    size: float             # base currency qty
    notional: float         # USD value at entry
    entry_commission: float
    exit_time: str = ""
    exit_price: float = 0.0
    exit_commission: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""   # "stop", "take_profit", "signal", "end_of_data"
    is_open: bool = True


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    strategy_name: str
    strategy_params: dict
    trades: list[Trade]
    equity_curve: pd.Series
    metrics: dict[str, Any]
    config: dict[str, Any]

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([asdict(t) for t in self.trades])

    def save(self, output_dir: str) -> None:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)

        # Metrics JSON
        with open(p / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)

        # Trades CSV
        df = self.trades_df()
        if not df.empty:
            df.to_csv(p / "trades.csv", index=False)

        # Equity curve CSV
        eq_df = self.equity_curve.reset_index()
        eq_df.columns = ["timestamp", "equity"]
        eq_df.to_csv(p / "equity_curve.csv", index=False)

        logger.success(f"Backtest results saved to {p}")

    def html_report(self, output_dir: str) -> str:
        """Generate HTML report with Plotly charts. Returns path to HTML file."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=["Equity Curve", "Drawdown %", "Trade PnL"],
            row_heights=[0.5, 0.25, 0.25],
            vertical_spacing=0.05,
        )

        # Equity
        fig.add_trace(
            go.Scatter(x=self.equity_curve.index, y=self.equity_curve.values,
                       name="Equity", line=dict(color="royalblue")), row=1, col=1
        )

        # Drawdown
        dd = drawdown_series(self.equity_curve) * 100
        fig.add_trace(
            go.Scatter(x=dd.index, y=dd.values, name="Drawdown %",
                       fill="tozeroy", line=dict(color="crimson")), row=2, col=1
        )

        # Trade PnL bars
        trades_df = self.trades_df()
        if not trades_df.empty:
            colors = ["green" if p > 0 else "red" for p in trades_df["pnl"]]
            fig.add_trace(
                go.Bar(x=trades_df["exit_time"], y=trades_df["pnl"],
                       name="Trade PnL", marker_color=colors), row=3, col=1
            )

        title = (
            f"Backtest: {self.strategy_name} | {self.symbol}/{self.timeframe} | "
            f"Return: {self.metrics.get('total_return_pct', 0):.1f}% | "
            f"Sharpe: {self.metrics.get('sharpe', 0):.2f} | "
            f"MaxDD: {self.metrics.get('max_drawdown_pct', 0):.1f}%"
        )
        fig.update_layout(title=title, height=800, showlegend=True)

        html_path = p / "report.html"
        fig.write_html(str(html_path))
        logger.success(f"HTML report saved to {html_path}")
        return str(html_path)


class BacktestEngine:
    """
    Single-symbol, single-timeframe bar-by-bar backtest simulator.
    """

    def __init__(self, bt_config: BacktestConfig, risk_config: RiskConfig):
        self.bt_config = bt_config
        self.risk = risk_config

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        symbol: str = "UNKNOWN",
        timeframe: str = "?",
    ) -> BacktestResult:
        """Run full backtest pipeline."""
        logger.info(f"Backtest: {strategy.name} on {symbol}/{timeframe}, {len(df)} bars")

        if len(df) < strategy.warmup_bars + 10:
            raise ValueError(
                f"Not enough data: {len(df)} bars, need at least {strategy.warmup_bars + 10}"
            )

        # Compute features + signals
        df = strategy.run(df)

        trades: list[Trade] = []
        equity = self.risk.initial_capital
        equity_values: list[float] = []
        current_trade: Trade | None = None
        trade_id = 0

        slip_factor = self.bt_config.slippage_bps / 10_000
        fee_factor = self.bt_config.commission_bps / 10_000

        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]

            curr_time = str(curr.name)
            curr_open = float(curr["open"])
            curr_high = float(curr["high"])
            curr_low = float(curr["low"])

            # ── 1. Check stop / TP for open trade ────────────────────────────
            if current_trade is not None and current_trade.is_open:
                exited = False

                if current_trade.direction == 1:  # LONG
                    if curr_low <= current_trade.stop_price:
                        exit_price = current_trade.stop_price
                        exit_reason = "stop"
                        current_trade, equity = self._close_trade(
                            current_trade, exit_price, curr_time, equity, fee_factor, exit_reason
                        )
                        trades.append(current_trade)
                        current_trade = None
                        exited = True
                    elif curr_high >= current_trade.take_price:
                        exit_price = current_trade.take_price
                        exit_reason = "take_profit"
                        current_trade, equity = self._close_trade(
                            current_trade, exit_price, curr_time, equity, fee_factor, exit_reason
                        )
                        trades.append(current_trade)
                        current_trade = None
                        exited = True

                else:  # SHORT
                    if curr_high >= current_trade.stop_price:
                        exit_price = current_trade.stop_price
                        exit_reason = "stop"
                        current_trade, equity = self._close_trade(
                            current_trade, exit_price, curr_time, equity, fee_factor, exit_reason
                        )
                        trades.append(current_trade)
                        current_trade = None
                        exited = True
                    elif curr_low <= current_trade.take_price:
                        exit_price = current_trade.take_price
                        exit_reason = "take_profit"
                        current_trade, equity = self._close_trade(
                            current_trade, exit_price, curr_time, equity, fee_factor, exit_reason
                        )
                        trades.append(current_trade)
                        current_trade = None
                        exited = True

            # ── 2. Signal exit (if not already exited by stop/TP) ─────────────
            if current_trade is not None and current_trade.is_open:
                prev_sig = int(prev.get("signal", 0))
                should_exit = (
                    prev_sig == 0
                    or (current_trade.direction == 1 and prev_sig == -1)
                    or (current_trade.direction == -1 and prev_sig == 1)
                )
                if should_exit:
                    # Exit at current bar open + slippage
                    if current_trade.direction == 1:
                        exit_price = curr_open * (1 - slip_factor)
                    else:
                        exit_price = curr_open * (1 + slip_factor)
                    current_trade, equity = self._close_trade(
                        current_trade, exit_price, curr_time, equity, fee_factor, "signal"
                    )
                    trades.append(current_trade)
                    current_trade = None

            # ── 3. Entry ──────────────────────────────────────────────────────
            if current_trade is None and i >= strategy.warmup_bars:
                prev_sig = int(prev.get("signal", 0))
                prev_stop = float(prev.get("stop_price", np.nan))
                prev_take = float(prev.get("take_price", np.nan))

                if prev_sig != 0 and not np.isnan(prev_stop) and not np.isnan(prev_take):
                    direction = prev_sig

                    # Entry price at current bar open + slippage
                    if direction == 1:
                        entry_price = curr_open * (1 + slip_factor)
                        stop_price = prev_stop
                        take_price = prev_take
                        # Sanity: stop must be below entry
                        if stop_price >= entry_price:
                            equity_values.append(equity)
                            continue
                    else:
                        entry_price = curr_open * (1 - slip_factor)
                        stop_price = prev_stop
                        take_price = prev_take
                        if stop_price <= entry_price:
                            equity_values.append(equity)
                            continue

                    stop_dist = abs(entry_price - stop_price)
                    if stop_dist <= 0:
                        equity_values.append(equity)
                        continue

                    # Position sizing: fixed fractional
                    risk_amount = equity * self.risk.risk_per_trade
                    size = risk_amount / stop_dist
                    notional = size * entry_price

                    if notional < self.risk.min_notional:
                        equity_values.append(equity)
                        continue

                    # Leverage cap
                    max_notional = equity * self.risk.max_leverage
                    if notional > max_notional:
                        size = max_notional / entry_price
                        notional = size * entry_price

                    # Entry commission
                    entry_commission = notional * fee_factor
                    equity -= entry_commission

                    trade_id += 1
                    current_trade = Trade(
                        trade_id=trade_id,
                        symbol=symbol,
                        direction=direction,
                        entry_time=curr_time,
                        entry_price=entry_price,
                        stop_price=stop_price,
                        take_price=take_price,
                        size=size,
                        notional=notional,
                        entry_commission=entry_commission,
                    )

            # ── 4. Mark to market ─────────────────────────────────────────────
            mtm_equity = equity
            if current_trade is not None and current_trade.is_open:
                curr_close = float(curr["close"])
                unrealized = (curr_close - current_trade.entry_price) * current_trade.direction * current_trade.size
                mtm_equity = equity + unrealized

            equity_values.append(mtm_equity)

        # ── Close any open trade at end of data ────────────────────────────
        if current_trade is not None and current_trade.is_open:
            last_close = float(df.iloc[-1]["close"])
            if current_trade.direction == 1:
                exit_price = last_close * (1 - slip_factor)
            else:
                exit_price = last_close * (1 + slip_factor)
            current_trade, equity = self._close_trade(
                current_trade, exit_price, str(df.index[-1]), equity, fee_factor, "end_of_data"
            )
            trades.append(current_trade)

        # ── Build equity curve ─────────────────────────────────────────────
        eq_series = pd.Series(
            [self.risk.initial_capital] + equity_values,
            index=df.index,
        )

        # ── Trades DataFrame ──────────────────────────────────────────────
        trades_df = pd.DataFrame([asdict(t) for t in trades]) if trades else pd.DataFrame()

        # ── Periods per year ──────────────────────────────────────────────
        if len(df) > 1:
            interval_seconds = (df.index[1] - df.index[0]).total_seconds()
            periods_per_year = int(365.25 * 24 * 3600 / interval_seconds)
        else:
            periods_per_year = 8760

        # ── Metrics ───────────────────────────────────────────────────────
        metrics = compute_metrics(
            eq_series, trades_df, self.risk.initial_capital, periods_per_year
        )
        metrics["symbol"] = symbol
        metrics["timeframe"] = timeframe
        metrics["strategy"] = strategy.name

        logger.info(
            f"Backtest complete: {len(trades)} trades | "
            f"Return {metrics.get('total_return_pct', 0):.1f}% | "
            f"Sharpe {metrics.get('sharpe', 0):.2f} | "
            f"MaxDD {metrics.get('max_drawdown_pct', 0):.1f}%"
        )

        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            strategy_name=strategy.name,
            strategy_params=strategy.params,
            trades=trades,
            equity_curve=eq_series,
            metrics=metrics,
            config={"backtest": self.bt_config.model_dump(), "risk": self.risk.model_dump()},
        )

    @staticmethod
    def _close_trade(
        trade: Trade,
        exit_price: float,
        exit_time: str,
        equity: float,
        fee_factor: float,
        reason: str,
    ) -> tuple[Trade, float]:
        exit_commission = trade.notional * fee_factor
        gross_pnl = (exit_price - trade.entry_price) * trade.direction * trade.size
        net_pnl = gross_pnl - exit_commission
        equity += net_pnl

        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_commission = exit_commission
        trade.pnl = round(net_pnl, 6)
        trade.pnl_pct = round(net_pnl / trade.notional * 100, 4) if trade.notional else 0.0
        trade.exit_reason = reason
        trade.is_open = False
        return trade, equity
