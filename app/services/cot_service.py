from __future__ import annotations

"""
COT (Commitment of Traders) Service
CFTC'nin haftalık yayınladığı büyük spekülatör pozisyon verilerinden
BULLISH / BEARISH / NEUTRAL bias üretir.
Veriler haftalık güncellenir → 24 saat önbelleklenir.
"""

import io
import logging
import zipfile
from datetime import datetime, timedelta, timezone

import pandas as pd

logger = logging.getLogger(__name__)

# CFTC Disaggregated COT kodları
_COT_CODES: dict[str, str] = {
    "XAUUSD": "088691",  # GOLD - COMMODITY EXCHANGE INC.
    "BTCUSD": "133741",  # BITCOIN - CME
    "EURUSD": "099741",  # EURO FX - CME
    "GBPUSD": "096742",  # BRITISH POUND STERLING - CME
    "USDJPY": "097741",  # JAPANESE YEN - CME
    "USDCHF": "092741",  # SWISS FRANC - CME
    "AUDUSD": "232741",  # AUSTRALIAN DOLLAR - CME
}

# Önbellek: symbol → (timestamp, bias)
_cache: dict[str, tuple[datetime, str]] = {}
_CACHE_TTL = timedelta(hours=24)


def get_cot_bias_sync(symbol: str) -> str:
    """
    Sync wrapper — bot'tan çağrılır.
    Önbellekte varsa döner, yoksa CFTC'den indirir.
    """
    symbol_upper = symbol.upper()

    # Önbellek kontrolü
    if symbol_upper in _cache:
        cached_time, cached_bias = _cache[symbol_upper]
        if datetime.now(timezone.utc) - cached_time < _CACHE_TTL:
            logger.debug("COT cache hit for %s: %s", symbol_upper, cached_bias)
            return cached_bias

    try:
        import requests  # httpx veya requests
        bias = _fetch_cot_bias(symbol_upper, requests)
        _cache[symbol_upper] = (datetime.now(timezone.utc), bias)
        logger.info("COT bias for %s: %s", symbol_upper, bias)
        return bias
    except Exception as exc:
        logger.warning("COT fetch failed for %s: %s", symbol_upper, exc)
        return "NEUTRAL"


def _fetch_cot_bias(symbol: str, requests_mod) -> str:
    code = _COT_CODES.get(symbol)
    if not code:
        return "NEUTRAL"

    year = datetime.now().year
    url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"

    resp = requests_mod.get(url, timeout=30)
    if resp.status_code != 200:
        # Geçen yılın datasını dene
        url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year - 1}.zip"
        resp = requests_mod.get(url, timeout=30)
        if resp.status_code != 200:
            return "NEUTRAL"

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        txt_files = [f for f in z.namelist() if f.lower().endswith(".txt")]
        if not txt_files:
            return "NEUTRAL"
        with z.open(txt_files[0]) as f:
            df = pd.read_csv(f, encoding="latin-1", low_memory=False)

    # Sütun isimlerini normalize et
    df.columns = [c.strip() for c in df.columns]

    # Kod sütununu bul
    code_col = next(
        (c for c in df.columns if "cftc" in c.lower() and "code" in c.lower()), None
    )
    if code_col is None:
        code_col = next(
            (c for c in df.columns if "market" in c.lower() and "code" in c.lower()), None
        )
    if code_col is None:
        return "NEUTRAL"

    df[code_col] = df[code_col].astype(str).str.strip()
    instrument_df = df[df[code_col] == code].copy()
    if instrument_df.empty:
        return "NEUTRAL"

    # Tarih sütunu bul ve sırala
    date_col = next(
        (c for c in df.columns if "report_date" in c.lower() or "date" in c.lower()), None
    )
    if date_col:
        instrument_df = instrument_df.sort_values(date_col, ascending=False).head(52)

    # Large Speculator long/short sütunları bul
    long_col = next(
        (c for c in df.columns if "noncomm" in c.lower() and "long" in c.lower() and "all" in c.lower()), None
    )
    short_col = next(
        (c for c in df.columns if "noncomm" in c.lower() and "short" in c.lower() and "all" in c.lower()), None
    )

    if not long_col or not short_col:
        return "NEUTRAL"

    net_positions = (
        pd.to_numeric(instrument_df[long_col], errors="coerce") -
        pd.to_numeric(instrument_df[short_col], errors="coerce")
    ).dropna()

    if len(net_positions) < 4:
        return "NEUTRAL"

    current_net = float(net_positions.iloc[0])
    min_net = float(net_positions.min())
    max_net = float(net_positions.max())

    if max_net == min_net:
        return "NEUTRAL"

    # Net Position Index (NPI): 0 = aşırı short, 1 = aşırı long
    npi = (current_net - min_net) / (max_net - min_net)

    # Kontrarian yorum (büyük spekülatörler trend takipçisidir → aşırı pozisyon karşı sinyal)
    if npi < 0.20:
        return "BULLISH"   # Specs çok short → dip yakın
    elif npi > 0.80:
        return "BEARISH"   # Specs çok long → tepe yakın

    # 4 haftalık momentum
    recent_change = float(net_positions.iloc[0]) - float(net_positions.iloc[3])
    total_range = max_net - min_net
    if abs(recent_change) > total_range * 0.10:
        return "BULLISH" if recent_change > 0 else "BEARISH"

    return "NEUTRAL"
