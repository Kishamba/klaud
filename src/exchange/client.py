"""Bybit exchange client wrapper with retries and rate limiting."""
from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class ExchangeClient:
    """Thin wrapper around pybit HTTP session with retry/backoff logic."""

    def __init__(self, testnet: bool = True, max_retries: int = 3, retry_delay: float = 1.0):
        from pybit.unified_trading import HTTP

        self.testnet = testnet
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        api_key = os.getenv("BYBIT_API_KEY", "") or None
        api_secret = os.getenv("BYBIT_API_SECRET", "") or None

        self._session = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret,
        )
        logger.debug(f"ExchangeClient init: testnet={testnet}, auth={'yes' if api_key else 'no'}")

    def _call(self, func_name: str, **kwargs: Any) -> dict:
        func = getattr(self._session, func_name)
        for attempt in range(self.max_retries):
            try:
                result = func(**kwargs)
                if result.get("retCode", -1) != 0:
                    msg = result.get("retMsg", "unknown error")
                    raise RuntimeError(f"Bybit API error [{result.get('retCode')}]: {msg}")
                return result
            except RuntimeError:
                raise
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise
                wait = self.retry_delay * (2 ** attempt)
                logger.warning(f"API call {func_name} failed (attempt {attempt + 1}): {exc}. Retrying in {wait:.1f}s")
                time.sleep(wait)
        raise RuntimeError(f"All {self.max_retries} retries exhausted for {func_name}")

    def get_klines(self, symbol: str, interval: str, start: int | None = None,
                   end: int | None = None, limit: int = 1000) -> list[list]:
        kwargs: dict[str, Any] = {
            "category": "linear", "symbol": symbol, "interval": interval, "limit": limit,
        }
        if start is not None:
            kwargs["start"] = start
        if end is not None:
            kwargs["end"] = end
        result = self._call("get_kline", **kwargs)
        return result["result"]["list"]

    def get_latest_klines(self, symbol: str, interval: str, limit: int = 200) -> list[list]:
        return self.get_klines(symbol=symbol, interval=interval, limit=limit)

    def place_order(self, symbol: str, side: str, qty: str, order_type: str = "Market",
                    reduce_only: bool = False, order_link_id: str | None = None) -> dict:
        kwargs: dict[str, Any] = {
            "category": "linear", "symbol": symbol, "side": side,
            "orderType": order_type, "qty": qty,
            "timeInForce": "IOC" if order_type == "Market" else "GTC",
            "reduceOnly": reduce_only,
        }
        if order_link_id:
            kwargs["orderLinkId"] = order_link_id
        return self._call("place_order", **kwargs)["result"]

    def cancel_order(self, symbol: str, order_id: str) -> dict:
        return self._call("cancel_order", category="linear", symbol=symbol, orderId=order_id)["result"]

    def get_positions(self, symbol: str | None = None) -> list[dict]:
        kwargs: dict[str, Any] = {"category": "linear", "settleCoin": "USDT"}
        if symbol:
            kwargs["symbol"] = symbol
        return self._call("get_positions", **kwargs)["result"]["list"]

    def get_balance(self) -> dict:
        result = self._call("get_wallet_balance", accountType="UNIFIED")
        coins = result["result"]["list"][0]["coin"]
        usdt = next((c for c in coins if c["coin"] == "USDT"), {})
        return {
            "equity": float(usdt.get("equity", 0)),
            "available": float(usdt.get("availableToWithdraw", 0)),
            "unrealised_pnl": float(usdt.get("unrealisedPnl", 0)),
        }

    def set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            self._call("set_leverage", category="linear", symbol=symbol,
                       buyLeverage=str(leverage), sellLeverage=str(leverage))
        except RuntimeError as e:
            if "leverage not modified" not in str(e).lower():
                raise

    def get_server_time(self) -> int:
        result = self._call("get_server_time")
        return int(result["result"]["timeNano"]) // 1_000_000
