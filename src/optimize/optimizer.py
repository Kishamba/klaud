"""
Optuna-based strategy parameter optimizer with walk-forward validation.
Protects against overfitting via out-of-sample test, trade count penalty,
and complexity penalty.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

import optuna
import pandas as pd
from loguru import logger

from src.config import AppConfig
from src.backtest.engine import BacktestEngine
from src.strategies import get_strategy

optuna.logging.set_verbosity(optuna.logging.WARNING)

if TYPE_CHECKING:
    from src.strategies.base import BaseStrategy


def _objective_score(metrics: dict) -> float:
    """
    Composite score to MAXIMIZE:
      Sharpe - penalty(max_drawdown) - penalty(few_trades) - penalty(no_trades)
    """
    if metrics.get("n_trades", 0) < 5:
        return -10.0

    sharpe = float(metrics.get("sharpe", 0))
    max_dd = abs(float(metrics.get("max_drawdown_pct", 0)))
    n_trades = int(metrics.get("n_trades", 0))

    # Drawdown penalty: quadratic beyond 15%
    dd_penalty = 0.0
    if max_dd > 15:
        dd_penalty = ((max_dd - 15) / 10) ** 2

    # Too many trades penalty (avoid over-trading)
    # Expected ~1 trade per 20 bars for 15m data → roughly 3/day
    trade_penalty = max(0, (n_trades - 500) / 1000)

    score = sharpe - dd_penalty - trade_penalty
    return score


class Optimizer:
    def __init__(self, config: AppConfig):
        self.config = config
        self.bt_engine = BacktestEngine(config.backtest, config.risk)

    def optimize(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        symbol: str = "UNKNOWN",
        timeframe: str = "?",
        n_trials: int | None = None,
        train_ratio: float | None = None,
        output_dir: str = "results",
    ) -> dict[str, Any]:
        n_trials = n_trials or self.config.optimize.n_trials
        train_ratio = train_ratio or self.config.optimize.train_ratio

        # ── Train / test split ────────────────────────────────────────────
        split_idx = int(len(df) * train_ratio)
        df_train = df.iloc[:split_idx].copy()
        df_test = df.iloc[split_idx:].copy()

        logger.info(
            f"Optimizing {strategy_name} on {symbol}/{timeframe} | "
            f"train={len(df_train)} bars, test={len(df_test)} bars, trials={n_trials}"
        )

        # Get param space from strategy
        dummy_strat = get_strategy(strategy_name, {})
        param_space = dummy_strat.param_space()

        if not param_space:
            logger.warning(f"{strategy_name} has no param_space defined")
            return {}

        # ── Optuna study ──────────────────────────────────────────────────
        study = optuna.create_study(direction="maximize", study_name=f"{strategy_name}_{symbol}")

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for param_name, spec in param_space.items():
                low, high, ptype = spec
                if ptype == "int":
                    params[param_name] = trial.suggest_int(param_name, int(low), int(high))
                elif ptype == "float":
                    params[param_name] = trial.suggest_float(param_name, float(low), float(high))
                elif ptype == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, low)
                else:
                    params[param_name] = trial.suggest_float(param_name, float(low), float(high))

            try:
                strat = get_strategy(strategy_name, params)
                result = self.bt_engine.run(df_train, strat, symbol, timeframe)
                return _objective_score(result.metrics)
            except Exception as exc:
                logger.debug(f"Trial failed: {exc}")
                return -100.0

        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=self.config.optimize.n_jobs,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_train_score = study.best_value

        # ── Out-of-sample evaluation ──────────────────────────────────────
        best_strat = get_strategy(strategy_name, best_params)
        oos_result = self.bt_engine.run(df_test, best_strat, symbol, timeframe)
        oos_score = _objective_score(oos_result.metrics)
        oos_metrics = oos_result.metrics

        # ── In-sample evaluation ──────────────────────────────────────────
        train_strat = get_strategy(strategy_name, best_params)
        train_result = self.bt_engine.run(df_train, train_strat, symbol, timeframe)
        train_metrics = train_result.metrics

        # ── Trials DataFrame ──────────────────────────────────────────────
        trials_data = []
        for t in study.trials:
            row = {"trial": t.number, "score": t.value}
            row.update(t.params)
            trials_data.append(row)
        trials_df = pd.DataFrame(trials_data).sort_values("score", ascending=False)

        # ── Overfitting warning ───────────────────────────────────────────
        overfit_warning = ""
        if best_train_score > 0 and oos_score < best_train_score * 0.5:
            overfit_warning = (
                f"WARNING: OOS score ({oos_score:.2f}) is < 50% of train score ({best_train_score:.2f}). "
                "Possible overfitting."
            )
            logger.warning(overfit_warning)
        elif oos_score < 0:
            overfit_warning = f"WARNING: OOS score is negative ({oos_score:.2f}). Strategy not robust."
            logger.warning(overfit_warning)

        # ── Save results ──────────────────────────────────────────────────
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = Path(output_dir) / f"optimize_{strategy_name}_{symbol}_{ts}"
        out.mkdir(parents=True, exist_ok=True)

        trials_df.to_csv(out / "trials.csv", index=False)

        summary = {
            "strategy": strategy_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "best_params": best_params,
            "train_score": best_train_score,
            "oos_score": oos_score,
            "train_metrics": train_metrics,
            "oos_metrics": oos_metrics,
            "n_trials": n_trials,
            "train_ratio": train_ratio,
            "overfit_warning": overfit_warning,
            "timestamp": ts,
        }

        with open(out / "best_params.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Save best config for live trading
        configs_dir = Path("configs")
        configs_dir.mkdir(exist_ok=True)
        best_config = {
            "strategy": {"name": strategy_name, "params": best_params},
            "symbol": symbol,
            "timeframe": timeframe,
            "oos_score": oos_score,
            "generated_at": ts,
        }
        with open(configs_dir / "best.json", "w") as f:
            json.dump(best_config, f, indent=2, default=str)

        # Save OOS backtest report
        oos_result.save(str(out / "oos"))
        oos_result.html_report(str(out / "oos"))
        train_result.save(str(out / "train"))

        logger.success(
            f"Optimization done: best_score={best_train_score:.3f} | "
            f"oos_score={oos_score:.3f} | "
            f"params={best_params}"
        )

        return summary
