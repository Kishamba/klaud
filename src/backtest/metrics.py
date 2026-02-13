"""Backtest performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any


def compute_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    initial_capital: float,
    periods_per_year: int = 8760,  # hourly; override for 15m = 35040
) -> dict[str, Any]:
    """
    Compute comprehensive performance metrics.

    Parameters
    ----------
    equity_curve : pd.Series  — equity at each bar (DatetimeIndex)
    trades       : pd.DataFrame — trades with columns: pnl, entry_time, exit_time
    initial_capital : float
    periods_per_year : int — bars per year (8760 for 1h, 35040 for 15m)
    """
    metrics: dict[str, Any] = {}

    if equity_curve.empty:
        return {"error": "empty equity curve"}

    eq = equity_curve.dropna()
    final_equity = float(eq.iloc[-1])
    metrics["initial_capital"] = initial_capital
    metrics["final_equity"] = round(final_equity, 2)
    metrics["total_return_pct"] = round((final_equity / initial_capital - 1) * 100, 2)

    # ── Drawdown ──────────────────────────────────────────────────────────────
    rolling_max = eq.cummax()
    drawdown = (eq - rolling_max) / rolling_max
    metrics["max_drawdown_pct"] = round(float(drawdown.min()) * 100, 2)

    # ── CAGR ──────────────────────────────────────────────────────────────────
    n_bars = len(eq)
    n_years = n_bars / periods_per_year
    if n_years > 0 and final_equity > 0:
        cagr = (final_equity / initial_capital) ** (1 / n_years) - 1
        metrics["cagr_pct"] = round(cagr * 100, 2)
    else:
        metrics["cagr_pct"] = 0.0

    # ── Returns for Sharpe/Sortino ────────────────────────────────────────────
    returns = eq.pct_change().dropna()
    if len(returns) > 1:
        ret_mean = returns.mean()
        ret_std = returns.std()
        if ret_std > 0:
            metrics["sharpe"] = round(float(ret_mean / ret_std * np.sqrt(periods_per_year)), 3)
        else:
            metrics["sharpe"] = 0.0

        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) > 1 else 0
        if downside_std > 0:
            metrics["sortino"] = round(float(ret_mean / downside_std * np.sqrt(periods_per_year)), 3)
        else:
            metrics["sortino"] = 0.0

        metrics["calmar"] = round(
            float(metrics["cagr_pct"] / abs(metrics["max_drawdown_pct"]))
            if metrics["max_drawdown_pct"] != 0 else 0.0, 3
        )
    else:
        metrics["sharpe"] = 0.0
        metrics["sortino"] = 0.0
        metrics["calmar"] = 0.0

    # ── Trade statistics ──────────────────────────────────────────────────────
    if trades.empty:
        metrics.update({
            "n_trades": 0, "win_rate_pct": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "profit_factor": 0.0, "avg_trade_pnl": 0.0, "max_consecutive_losses": 0,
        })
        return metrics

    n_trades = len(trades)
    metrics["n_trades"] = n_trades

    pnls = trades["pnl"].values
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    metrics["win_rate_pct"] = round(len(wins) / n_trades * 100, 2) if n_trades else 0.0
    metrics["avg_win"] = round(float(wins.mean()), 4) if len(wins) else 0.0
    metrics["avg_loss"] = round(float(losses.mean()), 4) if len(losses) else 0.0

    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    metrics["profit_factor"] = round(float(gross_profit / gross_loss), 3) if gross_loss > 0 else float("inf")
    metrics["avg_trade_pnl"] = round(float(pnls.mean()), 4)

    # Max consecutive losses
    consec = 0
    max_consec = 0
    for p in pnls:
        if p <= 0:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0
    metrics["max_consecutive_losses"] = max_consec

    # Avg trade duration
    if "entry_time" in trades.columns and "exit_time" in trades.columns:
        durations = (pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])).dt.total_seconds() / 3600
        metrics["avg_trade_duration_h"] = round(float(durations.mean()), 2)

    return metrics


def equity_curve_series(trades: pd.DataFrame, initial_capital: float, all_times: pd.DatetimeIndex) -> pd.Series:
    """Reconstruct equity curve at each bar from trades."""
    eq = pd.Series(initial_capital, index=all_times)
    cumulative = initial_capital
    for _, t in trades.iterrows():
        exit_t = pd.Timestamp(t["exit_time"])
        if exit_t in eq.index:
            cumulative += t["pnl"]
            eq.loc[exit_t:] = cumulative
    return eq


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Compute drawdown series as fraction of peak."""
    rolling_max = equity_curve.cummax()
    return (equity_curve - rolling_max) / rolling_max.replace(0, np.nan)
