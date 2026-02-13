"""Application configuration via Pydantic models + YAML."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()


class ExchangeConfig(BaseModel):
    testnet: bool = True
    rate_limit_sleep: float = 0.1
    max_retries: int = 3
    retry_delay: float = 1.0


class DataConfig(BaseModel):
    symbols: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    timeframes: list[str] = ["15", "60"]
    data_dir: str = "data/candles"

    @field_validator("timeframes", mode="before")
    @classmethod
    def coerce_tfs(cls, v: Any) -> list[str]:
        return [str(x) for x in v]


class RiskConfig(BaseModel):
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.01
    max_leverage: float = 3.0
    max_open_positions: int = 3
    max_daily_loss_pct: float = 5.0
    max_weekly_loss_pct: float = 12.0
    max_drawdown_pct: float = 20.0
    slippage_bps: float = 2.0
    fee_bps: float = 6.0
    min_notional: float = 10.0


class BacktestConfig(BaseModel):
    commission_bps: float = 6.0
    slippage_bps: float = 2.0


class OptimizeConfig(BaseModel):
    n_trials: int = 50
    train_ratio: float = 0.7
    n_jobs: int = 1


class StrategyConfig(BaseModel):
    name: str = "TrendStrategy"
    params: dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseModel):
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    optimize: OptimizeConfig = Field(default_factory=OptimizeConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)


def load_config(path: str = "configs/config.yaml") -> AppConfig:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            raw = yaml.safe_load(f) or {}
        return AppConfig.model_validate(raw)
    return AppConfig()
