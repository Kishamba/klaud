from .trend import TrendStrategy
from .mean_reversion import MeanReversionStrategy
from .regime import RegimeStrategy

STRATEGY_REGISTRY = {
    "TrendStrategy": TrendStrategy,
    "MeanReversionStrategy": MeanReversionStrategy,
    "RegimeStrategy": RegimeStrategy,
}

def get_strategy(name: str, params: dict | None = None):
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY)}")
    cls = STRATEGY_REGISTRY[name]
    return cls(params=params or {})

__all__ = ["TrendStrategy", "MeanReversionStrategy", "RegimeStrategy", "STRATEGY_REGISTRY", "get_strategy"]
