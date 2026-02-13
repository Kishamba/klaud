"""OHLCV data downloader with parquet storage and incremental updates."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.utils.helpers import tf_to_ms, ms_to_dt, dt_to_ms, ensure_dirs, INTERVAL_LABELS

if TYPE_CHECKING:
    from src.exchange.client import ExchangeClient
    from src.config import DataConfig


COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]


class DataDownloader:
    def __init__(self, client: "ExchangeClient", config: "DataConfig"):
        self.client = client
        self.config = config
        self.data_dir = Path(config.data_dir)

    def _parquet_path(self, symbol: str, tf: str) -> Path:
        return self.data_dir / symbol / f"{tf}.parquet"

    def _meta_path(self, symbol: str, tf: str) -> Path:
        return self.data_dir / symbol / f"{tf}_meta.json"

    def _save(self, df: pd.DataFrame, symbol: str, tf: str) -> None:
        path = self._parquet_path(symbol, tf)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=True)
        meta = {
            "symbol": symbol, "tf": tf, "rows": len(df),
            "start": str(df.index[0]), "end": str(df.index[-1]),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._meta_path(symbol, tf), "w") as f:
            json.dump(meta, f, indent=2)
        logger.debug(f"Saved {symbol}/{tf}: {len(df)} rows")

    def load(self, symbol: str, tf: str) -> pd.DataFrame | None:
        path = self._parquet_path(symbol, tf)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def get_metadata(self, symbol: str, tf: str) -> dict | None:
        p = self._meta_path(symbol, tf)
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return None

    def _fetch_range(self, symbol: str, tf: str, start_ms: int, end_ms: int) -> list[list]:
        all_rows: dict[int, list] = {}
        current_end = end_ms

        with tqdm(desc=f"{symbol}/{INTERVAL_LABELS.get(tf, tf)}", unit=" candles", leave=False) as pbar:
            while current_end > start_ms:
                try:
                    raw = self.client.get_klines(symbol=symbol, interval=tf,
                                                  start=start_ms, end=current_end, limit=1000)
                except Exception as exc:
                    logger.error(f"Fetch error {symbol}/{tf}: {exc}")
                    break
                if not raw:
                    break
                added = 0
                for c in raw:
                    ts = int(c[0])
                    if start_ms <= ts < current_end and ts not in all_rows:
                        all_rows[ts] = c
                        added += 1
                pbar.update(added)
                oldest_ts = min(int(c[0]) for c in raw)
                if oldest_ts <= start_ms or len(raw) < 1000:
                    break
                current_end = oldest_ts
                time.sleep(0.05)

        return sorted(all_rows.values(), key=lambda x: int(x[0]))

    @staticmethod
    def _to_dataframe(raw: list[list]) -> pd.DataFrame:
        if not raw:
            return pd.DataFrame(columns=COLUMNS[1:])
        df = pd.DataFrame(raw, columns=COLUMNS)
        df = df.astype({"timestamp": "int64", "open": "float64", "high": "float64",
                         "low": "float64", "close": "float64", "volume": "float64", "turnover": "float64"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    @staticmethod
    def check_gaps(df: pd.DataFrame, tf: str) -> int:
        if len(df) < 2:
            return 0
        interval_ms = tf_to_ms(tf)
        expected = pd.date_range(start=df.index[0], end=df.index[-1], freq=f"{interval_ms}ms")
        return len(expected) - len(df)

    def download(self, symbol: str, tf: str, start: datetime | None = None,
                 end: datetime | None = None) -> pd.DataFrame:
        end_ms = dt_to_ms(end) if end else int(time.time() * 1000)
        existing = self.load(symbol, tf)

        if existing is not None and not existing.empty:
            last_ts = int(existing.index[-1].timestamp() * 1000)
            start_ms = last_ts + tf_to_ms(tf)
            logger.info(f"{symbol}/{tf}: incremental from {ms_to_dt(start_ms)}")
        else:
            start_ms = dt_to_ms(start) if start else end_ms - (2 * 365 * 24 * 60 * 60 * 1000)
            logger.info(f"{symbol}/{tf}: full download from {ms_to_dt(start_ms)}")

        if start_ms >= end_ms:
            logger.info(f"{symbol}/{tf}: already up to date")
            return existing if existing is not None else pd.DataFrame()

        raw = self._fetch_range(symbol, tf, start_ms, end_ms)
        if not raw:
            logger.warning(f"{symbol}/{tf}: no new data")
            return existing if existing is not None else pd.DataFrame()

        new_df = self._to_dataframe(raw)
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, new_df]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
        else:
            combined = new_df

        gaps = DataDownloader.check_gaps(combined, tf)
        if gaps > 0:
            logger.warning(f"{symbol}/{tf}: {gaps} missing bars")

        self._save(combined, symbol, tf)
        logger.success(f"{symbol}/{tf}: {len(combined)} total, {len(new_df)} new rows")
        return combined

    def download_all(self, symbols: list[str] | None = None, timeframes: list[str] | None = None,
                     start: datetime | None = None, end: datetime | None = None) -> dict:
        symbols = symbols or self.config.symbols
        timeframes = timeframes or self.config.timeframes
        results: dict = {}
        for sym in symbols:
            results[sym] = {}
            for tf in timeframes:
                logger.info(f"Downloading {sym}/{tf}...")
                try:
                    results[sym][tf] = self.download(sym, tf, start=start, end=end)
                except Exception as exc:
                    logger.error(f"Failed {sym}/{tf}: {exc}")
                    results[sym][tf] = pd.DataFrame()
        return results
