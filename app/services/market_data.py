from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import pandas as pd


class MarketDataError(RuntimeError):
    pass


@dataclass
class MarketDataClient:
    api_key: str
    base_url: str = "https://api.twelvedata.com"
    cache_ttl_seconds: int = 30
    cache: dict[str, pd.DataFrame] = field(default_factory=dict)
    cache_time: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        symbol = symbol.strip().upper().replace(" ", "")
        mapping = {
            "XAUUSD": "XAU/USD",
            "EURUSD": "EUR/USD",
            "GBPUSD": "GBP/USD",
            "USDJPY": "USD/JPY",
            "BTCUSD": "BTC/USD",
        }
        return mapping.get(symbol, symbol if "/" in symbol else f"{symbol[:3]}/{symbol[3:]}")

    async def fetch_candles(
        self,
        symbol: str,
        interval: str = "15min",
        outputsize: int = 300,
    ) -> pd.DataFrame:
        if not self.api_key:
            raise MarketDataError("TWELVEDATA_API_KEY eksik.")

        normalized = self.normalize_symbol(symbol)
        cache_key = f"{normalized}_{interval}_{outputsize}"
        now = time.time()

        if cache_key in self.cache and (now - self.cache_time.get(cache_key, 0) < self.cache_ttl_seconds):
            return self.cache[cache_key].copy()

        url = f"{self.base_url}/time_series"
        params = {
            "symbol": normalized,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "format": "JSON",
        }

        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            payload: dict[str, Any] = response.json()

        if "values" not in payload:
            raise MarketDataError(payload.get("message", f"Veri alınamadı: {normalized}"))

        df = pd.DataFrame(payload["values"])
        numeric_cols = ["open", "high", "low", "close"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        self.cache[cache_key] = df.copy()
        self.cache_time[cache_key] = now
        return df

    async def fetch_price(self, symbol: str) -> float:
        df = await self.fetch_candles(symbol=symbol, interval="1min", outputsize=2)
        if df.empty:
            raise MarketDataError(f"Anlık fiyat alınamadı: {symbol}")
        return float(df.iloc[-1]["close"])