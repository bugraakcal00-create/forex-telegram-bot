from __future__ import annotations

"""
Retail Sentiment Service — Perakende trader pozisyonlama verisi.
Contrarian gösterge: retail çoğunluğunun TERSİ yönde işlem açmak karlıdır.

Veri kaynakları:
1. myfxbook.com/community/outlook (ücretsiz, scrape)
2. OANDA (ücretsiz json)

Çıktı: BULLISH/BEARISH/NEUTRAL (contrarian yorum)
  - Retail %65+ LONG → BEARISH (contrarian)
  - Retail %65+ SHORT → BULLISH (contrarian)
  - Arada → NEUTRAL
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# Cache: symbol -> (timestamp, result)
_cache: dict[str, tuple[datetime, "RetailSentimentResult"]] = {}
_CACHE_TTL = timedelta(minutes=30)

# Contrarian threshold
_CONTRARIAN_THRESHOLD = 65.0  # %65+ bir yönde ise ters sinyal


@dataclass(frozen=True)
class RetailSentimentResult:
    long_pct: float          # Retail long yüzdesi (0-100)
    short_pct: float         # Retail short yüzdesi (0-100)
    bias: str                # BULLISH / BEARISH / NEUTRAL (contrarian)
    source: str              # Veri kaynağı
    description: str         # Açıklama


def _symbol_to_myfxbook(symbol: str) -> str:
    """Sembol ismini myfxbook formatına çevir."""
    mapping = {
        "XAUUSD": "XAUUSD",
        "EURUSD": "EURUSD",
        "GBPUSD": "GBPUSD",
        "USDJPY": "USDJPY",
        "AUDUSD": "AUDUSD",
        "USDCHF": "USDCHF",
    }
    return mapping.get(symbol.upper(), symbol.upper())


async def get_retail_sentiment(symbol: str) -> RetailSentimentResult:
    """
    Retail trader pozisyonlama verisini çek.
    Contrarian yorum: retail çoğunluğunun tersi.
    """
    symbol_upper = symbol.upper()

    # Cache kontrolü
    if symbol_upper in _cache:
        cached_time, cached_result = _cache[symbol_upper]
        if datetime.now() - cached_time < _CACHE_TTL:
            return cached_result

    result = await _fetch_myfxbook_sentiment(symbol_upper)
    _cache[symbol_upper] = (datetime.now(), result)
    return result


async def _fetch_myfxbook_sentiment(symbol: str) -> RetailSentimentResult:
    """myfxbook community outlook verisini çek."""
    default = RetailSentimentResult(
        long_pct=50.0, short_pct=50.0,
        bias="NEUTRAL", source="default",
        description="Veri alinamadi — notr kabul edildi",
    )

    try:
        url = "https://www.myfxbook.com/community/outlook"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            if resp.status_code != 200:
                logger.debug("myfxbook returned %d", resp.status_code)
                return default

            # Parse the HTML for the symbol's long/short percentages
            text = resp.text
            sym = _symbol_to_myfxbook(symbol)

            # Look for the pattern: symbol name followed by percentage values
            import re
            # myfxbook uses format like: EURUSD ... Short 42% ... Long 58%
            # Try to find percentages near the symbol name
            idx = text.find(sym)
            if idx == -1:
                # Try without separator
                for alt in [f"{sym[:3]}/{sym[3:]}", sym.lower()]:
                    idx = text.find(alt)
                    if idx != -1:
                        break

            if idx == -1:
                logger.debug("Symbol %s not found in myfxbook outlook", symbol)
                return default

            # Extract nearby percentages
            snippet = text[idx:idx + 500]
            pcts = re.findall(r'(\d{1,3}(?:\.\d+)?)\s*%', snippet)

            if len(pcts) >= 2:
                # Usually first is short%, second is long% (or vice versa)
                val1 = float(pcts[0])
                val2 = float(pcts[1])

                # Determine which is long/short
                # If "Short" appears before first number, first is short
                short_idx = snippet.lower().find("short")
                long_idx = snippet.lower().find("long")

                if short_idx != -1 and long_idx != -1:
                    if short_idx < long_idx:
                        short_pct, long_pct = val1, val2
                    else:
                        long_pct, short_pct = val1, val2
                else:
                    # Assume first is short, second is long (common format)
                    short_pct, long_pct = val1, val2

                # Normalize
                total = short_pct + long_pct
                if total > 0:
                    long_pct = round(long_pct / total * 100, 1)
                    short_pct = round(short_pct / total * 100, 1)

                return _interpret_sentiment(long_pct, short_pct, "myfxbook")

            logger.debug("Could not parse percentages for %s from myfxbook", symbol)
            return default

    except Exception as exc:
        logger.debug("myfxbook fetch failed: %s", exc)
        return default


def _interpret_sentiment(long_pct: float, short_pct: float, source: str) -> RetailSentimentResult:
    """Contrarian yorum: retail çoğunluğunun tersi."""
    if long_pct >= _CONTRARIAN_THRESHOLD:
        bias = "BEARISH"  # Retail aşırı long → contrarian SHORT
        desc = f"Retail %{long_pct:.0f} LONG — kalabalığın tersi: SATIŞ baskısı beklenir"
    elif short_pct >= _CONTRARIAN_THRESHOLD:
        bias = "BULLISH"  # Retail aşırı short → contrarian LONG
        desc = f"Retail %{short_pct:.0f} SHORT — kalabalığın tersi: ALIŞ baskısı beklenir"
    else:
        bias = "NEUTRAL"
        desc = f"Retail dengeli (L:{long_pct:.0f}% / S:{short_pct:.0f}%) — net contrarian sinyal yok"

    return RetailSentimentResult(
        long_pct=long_pct,
        short_pct=short_pct,
        bias=bias,
        source=source,
        description=desc,
    )


def get_contrarian_filter(retail_bias: str, signal: str) -> bool:
    """
    Contrarian filtre: retail bias sinyal yönü ile AYNI ise engelle.
    Retail BULLISH + sinyal LONG → engelle (retail ile aynı yönde gitme)
    Retail BEARISH + sinyal SHORT → engelle
    """
    if retail_bias == "NEUTRAL":
        return False  # Engelleme yok
    if retail_bias == "BULLISH" and signal == "LONG":
        return True  # Retail zaten long, sen de long açma
    if retail_bias == "BEARISH" and signal == "SHORT":
        return True  # Retail zaten short, sen de short açma
    return False
