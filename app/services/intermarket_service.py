"""
Intermarket Service — XAU/XAG ratio + US10Y real yield (FRED) + DXY confluence.

Gold real-rate short-duration bond karakteri taşır. En güçlü XAUUSD edge'leri:
  - DXY ters korelasyon (-0.8 tarihsel)
  - US10Y real yield (TIPS) ters korelasyon (-0.85)
  - XAU/XAG ratio göreli güç göstergesi

Kaynaklar:
  - DXY: TwelveData (mevcut market_data)
  - US10Y real yield: FRED API (DFII10), ücretsiz, API key olmadan çalışır
  - XAU/XAG: TwelveData (XAUUSD ve XAGUSD iki seri)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

# ── FRED API (US10Y real yield) ────────────────────────────────────────────
# DFII10 = 10-Year Treasury Inflation-Indexed Security (real yield)
# Ücretsiz fredgraph.csv endpoint, API key gerekmez
_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_REAL_YIELD_SERIES = "DFII10"

# Cache
_CACHE_TTL_REAL_YIELD = 3600 * 4  # 4 saat (günlük veri)
_CACHE_TTL_RATIO = 300            # 5 dk
_cache: dict = {}


@dataclass
class IntermarketSnapshot:
    """Tek bir zaman kesitinde tüm intermarket sinyalleri."""
    dxy_bias: str = "NEUTRAL"           # BULLISH / BEARISH / NEUTRAL
    real_yield_pct: float = 0.0          # 10Y TIPS yield %
    real_yield_delta_5d: float = 0.0     # 5-gün değişim (bps)
    real_yield_pressure: str = "NEUTRAL" # bullish XAU / bearish XAU / neutral
    xau_xag_ratio: float = 0.0           # XAU/XAG oranı
    xau_xag_zscore: float = 0.0          # 90-gün z-score
    xau_xag_signal: str = "NEUTRAL"      # gümüş XAU'ya göre zayıf → XAU bearish


async def fetch_us10y_real_yield() -> tuple[float, float] | None:
    """FRED'den son 10 günlük DFII10 serisini çek, son değer + 5-gün deltasını döndür.

    Returns:
        (latest_yield_pct, delta_5day_bps) veya None (hata).
    """
    cache_key = "real_yield_fred"
    cached = _cache.get(cache_key)
    if cached and time.time() - cached["ts"] < _CACHE_TTL_REAL_YIELD:
        return cached["value"]

    try:
        params = {"id": _REAL_YIELD_SERIES, "cosd": _recent_date(15), "coed": _recent_date(0)}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(_FRED_URL, params=params)
            resp.raise_for_status()
            text = resp.text

        # CSV format: DATE,DFII10\n2026-04-15,2.12\n...
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < 2:
            return None
        rows = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) != 2:
                continue
            val_str = parts[1].strip()
            if val_str in ("", "."):
                continue
            try:
                rows.append((parts[0], float(val_str)))
            except ValueError:
                continue

        if len(rows) < 2:
            return None

        latest = rows[-1][1]
        old = rows[max(0, len(rows) - 6)][1]  # ~5 iş günü önce
        delta_bps = (latest - old) * 100  # 1% = 100 bps

        result = (round(latest, 3), round(delta_bps, 1))
        _cache[cache_key] = {"ts": time.time(), "value": result}
        return result
    except Exception as exc:
        logger.warning("FRED real yield fetch failed: %s", exc)
        return None


def _recent_date(days_back: int) -> str:
    """YYYY-MM-DD formatında tarih (bugünden N gün önce)."""
    from datetime import date, timedelta
    return (date.today() - timedelta(days=days_back)).isoformat()


async def fetch_xau_xag_ratio(market_client) -> tuple[float, float, str] | None:
    """XAU/XAG oranı + 90-gün z-score.

    Returns:
        (ratio, zscore, signal) veya None.
        signal: "NEUTRAL" | "XAU_STRONG" (gümüş zayıf, XAU bullish) | "XAU_WEAK"
    """
    cache_key = "xau_xag_ratio"
    cached = _cache.get(cache_key)
    if cached and time.time() - cached["ts"] < _CACHE_TTL_RATIO:
        return cached["value"]

    try:
        xau_df = await market_client.fetch_candles("XAUUSD", interval="1day", outputsize=100)
        xag_df = await market_client.fetch_candles("XAGUSD", interval="1day", outputsize=100)
        if xau_df is None or xag_df is None or len(xau_df) < 30 or len(xag_df) < 30:
            return None

        # Align by datetime
        xau = xau_df.set_index("datetime")["close"].astype(float)
        xag = xag_df.set_index("datetime")["close"].astype(float)
        ratio_series = (xau / xag).dropna()
        if len(ratio_series) < 30:
            return None

        current = float(ratio_series.iloc[-1])
        mean = float(ratio_series.tail(90).mean())
        std = float(ratio_series.tail(90).std()) or 1e-9
        zscore = (current - mean) / std

        # z > 1: XAU/XAG yüksek → gümüş zayıf → risk-off/XAU bullish nötr-boğa
        # z < -1: XAU/XAG düşük → gümüş güçlü → risk-on → XAU hafif bearish
        if zscore > 1.0:
            signal = "XAU_STRONG"
        elif zscore < -1.0:
            signal = "XAU_WEAK"
        else:
            signal = "NEUTRAL"

        result = (round(current, 3), round(zscore, 2), signal)
        _cache[cache_key] = {"ts": time.time(), "value": result}
        return result
    except Exception as exc:
        logger.warning("XAU/XAG ratio fetch failed: %s", exc)
        return None


async def build_snapshot(market_client, dxy_bias: str = "NEUTRAL") -> IntermarketSnapshot:
    """Tüm intermarket göstergelerini tek snapshot'ta topla."""
    snap = IntermarketSnapshot(dxy_bias=dxy_bias)

    ry = await fetch_us10y_real_yield()
    if ry is not None:
        latest, delta_bps = ry
        snap.real_yield_pct = latest
        snap.real_yield_delta_5d = delta_bps
        # Real yield artıyor → XAU bearish (gold opportunity cost)
        # > +10 bps / 5d: bearish XAU
        # < -10 bps / 5d: bullish XAU
        if delta_bps > 10:
            snap.real_yield_pressure = "BEARISH_XAU"
        elif delta_bps < -10:
            snap.real_yield_pressure = "BULLISH_XAU"
        else:
            snap.real_yield_pressure = "NEUTRAL"

    xx = await fetch_xau_xag_ratio(market_client)
    if xx is not None:
        ratio, zscore, signal = xx
        snap.xau_xag_ratio = ratio
        snap.xau_xag_zscore = zscore
        snap.xau_xag_signal = signal

    return snap


def confluence_score(snap: IntermarketSnapshot, signal_direction: str) -> int:
    """Intermarket confluence skoru (-3 .. +3).

    LONG için: DXY bearish + real yield düşüşü + XAU_STRONG → +3
    SHORT için: DXY bullish + real yield artışı + XAU_WEAK → +3
    Karşıt sinyaller negatif.
    """
    score = 0
    dir_up = signal_direction == "LONG"

    # DXY (ters korelasyon XAU için)
    if dir_up and snap.dxy_bias == "BEARISH":
        score += 1
    elif not dir_up and snap.dxy_bias == "BULLISH":
        score += 1
    elif dir_up and snap.dxy_bias == "BULLISH":
        score -= 1
    elif not dir_up and snap.dxy_bias == "BEARISH":
        score -= 1

    # Real yield
    if dir_up and snap.real_yield_pressure == "BULLISH_XAU":
        score += 1
    elif not dir_up and snap.real_yield_pressure == "BEARISH_XAU":
        score += 1
    elif dir_up and snap.real_yield_pressure == "BEARISH_XAU":
        score -= 1
    elif not dir_up and snap.real_yield_pressure == "BULLISH_XAU":
        score -= 1

    # XAU/XAG
    if dir_up and snap.xau_xag_signal == "XAU_STRONG":
        score += 1
    elif not dir_up and snap.xau_xag_signal == "XAU_WEAK":
        score += 1
    elif dir_up and snap.xau_xag_signal == "XAU_WEAK":
        score -= 1
    elif not dir_up and snap.xau_xag_signal == "XAU_STRONG":
        score -= 1

    return score
