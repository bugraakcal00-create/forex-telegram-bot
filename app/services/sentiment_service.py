from __future__ import annotations

"""
Sentiment Service — Haber başlıklarından piyasa duyarlılığı ölçer.
İki katmanlı yaklaşım:
  1. Keyword tabanlı (hızlı, sıfır bağımlılık) — her zaman çalışır
  2. FinBERT tabanlı (derin NLP) — transformers kuruluysa devreye girer

Döndürülen değerler:
  gold_bias: BULLISH / BEARISH / NEUTRAL  (altın için)
  usd_bias:  BULLISH / BEARISH / NEUTRAL  (USD için)
  score:     -1.0 (çok düşüşçü) … +1.0 (çok yükselişçi) [altın için]
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ─── Keyword sözlükleri ────────────────────────────────────────────────────────
# (kelime_grubu, ağırlık)

_GOLD_BULLISH: dict[str, int] = {
    "inflation": 3, "stagflation": 3, "hyperinflation": 4,
    "safe haven": 3, "risk off": 3, "risk aversion": 2, "flight to safety": 3,
    "gold rally": 4, "gold surges": 4, "gold rises": 3, "gold gains": 3,
    "war": 3, "conflict": 2, "geopolitical": 2, "tension": 1,
    "recession": 2, "downturn": 1, "slowdown": 1, "contraction": 2,
    "rate cut": 3, "fed cut": 3, "dovish": 3, "easing": 2, "pivot": 2,
    "weaker dollar": 3, "dollar falls": 3, "dollar drops": 2, "dollar weakness": 3,
    "cpi above": 2, "inflation higher": 2, "hot cpi": 3,
    "debt ceiling": 2, "fiscal crisis": 2, "banking crisis": 3,
    "uncertainty": 1, "fear": 2, "panic": 3,
}

_GOLD_BEARISH: dict[str, int] = {
    "rate hike": 3, "hawkish": 3, "tightening": 2, "rate rise": 3,
    "stronger dollar": 3, "dollar rallies": 3, "dollar strength": 3,
    "gold falls": 4, "gold drops": 4, "gold declines": 3, "sell gold": 3,
    "risk on": 3, "risk appetite": 2, "stocks rally": 2,
    "nfp beat": 2, "jobs beat": 2, "strong jobs": 2, "unemployment falls": 2,
    "cpi below": 2, "inflation cools": 2, "disinflation": 2, "deflation": 2,
    "economic growth": 1, "gdp beat": 2, "strong economy": 2,
    "yield rises": 2, "yields surge": 3, "bond selloff": 2,
}

_USD_BULLISH: dict[str, int] = {
    "rate hike": 3, "hawkish fed": 4, "tightening": 2, "rate rise": 3,
    "strong jobs": 3, "nfp beat": 3, "unemployment falls": 2,
    "dollar strength": 3, "usd rally": 3, "dollar rallies": 3,
    "gdp beat": 2, "strong economy": 2, "growth beats": 2,
    "yield rises": 2, "safe haven dollar": 2,
    "cpi above": 1,  # mixed for USD: bad for economy, but causes rate hike expectations
}

_USD_BEARISH: dict[str, int] = {
    "rate cut": 3, "dovish fed": 4, "easing": 2, "pivot": 2,
    "dollar weakness": 3, "usd falls": 3, "dollar drops": 3,
    "weak jobs": 3, "nfp miss": 3, "unemployment rises": 2,
    "deficit": 2, "debt ceiling": 2, "fiscal concerns": 2,
    "trade war": 2, "tariffs": 1, "sanctions": 1,
    "fed pause": 2, "rate hold": 1,
}


# ─── Sonuç sınıfı ─────────────────────────────────────────────────────────────

@dataclass
class SentimentResult:
    gold_score: float        # -1.0 … +1.0
    usd_score: float         # -1.0 … +1.0
    gold_bias: str           # BULLISH / BEARISH / NEUTRAL
    usd_bias: str            # BULLISH / BEARISH / NEUTRAL
    headline_count: int
    method: str = "keyword"  # keyword | finbert
    top_keywords: list[str] = field(default_factory=list)


# ─── Önbellek (15 dakika) ─────────────────────────────────────────────────────

_sentiment_cache: tuple[datetime, int, SentimentResult] | None = None
_SENT_CACHE_TTL = timedelta(minutes=15)


def analyze_headlines(headlines: list[str]) -> SentimentResult:
    """
    Haber başlıklarını analiz et.
    Önce FinBERT dener, yoksa keyword tabanlı çalışır.
    """
    if not headlines:
        return SentimentResult(0.0, 0.0, "NEUTRAL", "NEUTRAL", 0)

    # FinBERT dene
    try:
        return _finbert_analyze(headlines)
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("FinBERT failed, falling back to keyword: %s", exc)

    return _keyword_analyze(headlines)


def analyze_news_list(news_items: list[dict]) -> SentimentResult:
    """
    news_service'ten gelen dict listesini analiz et.
    Her item'ın 'title' veya 'headline' alanı kullanılır.
    """
    global _sentiment_cache

    headlines = []
    for item in news_items:
        title = item.get("title") or item.get("headline") or item.get("description") or ""
        if title:
            headlines.append(str(title))

    # Önbellek kontrolü (input hash dahil)
    input_hash = hash(tuple(sorted(headlines)))
    if _sentiment_cache is not None:
        cached_time, cached_hash, cached_result = _sentiment_cache
        if datetime.now() - cached_time < _SENT_CACHE_TTL and cached_hash == input_hash:
            return cached_result

    result = analyze_headlines(headlines)
    _sentiment_cache = (datetime.now(), input_hash, result)
    return result


# ─── Keyword analizi ──────────────────────────────────────────────────────────

def _keyword_analyze(headlines: list[str]) -> SentimentResult:
    combined = " ".join(headlines).lower()

    gold_bull = sum(w for kw, w in _GOLD_BULLISH.items() if kw in combined)
    gold_bear = sum(w for kw, w in _GOLD_BEARISH.items() if kw in combined)
    usd_bull  = sum(w for kw, w in _USD_BULLISH.items()  if kw in combined)
    usd_bear  = sum(w for kw, w in _USD_BEARISH.items()  if kw in combined)

    def _score(bull: int, bear: int) -> float:
        total = bull + bear
        return round((bull - bear) / total, 3) if total else 0.0

    def _bias(score: float) -> str:
        if score >= 0.20:
            return "BULLISH"
        elif score <= -0.20:
            return "BEARISH"
        return "NEUTRAL"

    gold_score = _score(gold_bull, gold_bear)
    usd_score  = _score(usd_bull,  usd_bear)

    # En önemli anahtar kelimeleri topla
    top_kw: list[str] = []
    for kw in list(_GOLD_BULLISH) + list(_GOLD_BEARISH):
        if kw in combined and kw not in top_kw:
            top_kw.append(kw)
            if len(top_kw) >= 5:
                break

    return SentimentResult(
        gold_score=gold_score,
        usd_score=usd_score,
        gold_bias=_bias(gold_score),
        usd_bias=_bias(usd_score),
        headline_count=len(headlines),
        method="keyword",
        top_keywords=top_kw,
    )


# ─── FinBERT analizi (opsiyonel) ──────────────────────────────────────────────

_finbert_pipeline = None


def _finbert_analyze(headlines: list[str]) -> SentimentResult:
    """
    ProsusAI/finbert modeli ile analiz.
    `pip install transformers torch` gerektirir.
    """
    global _finbert_pipeline
    from transformers import pipeline  # type: ignore

    if _finbert_pipeline is None:
        _finbert_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
    clf = _finbert_pipeline

    scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    count = 0
    for headline in headlines[:20]:  # max 20 başlık
        result = clf(headline[:512])[0]
        label = str(result["label"]).lower()
        if label in scores:
            scores[label] += float(result["score"])
            count += 1

    if count == 0:
        return _keyword_analyze(headlines)

    pos = scores["positive"] / count
    neg = scores["negative"] / count
    total = pos + neg
    gold_score = round((pos - neg) / total, 3) if total else 0.0

    def _bias(score: float) -> str:
        if score >= 0.15:
            return "BULLISH"
        elif score <= -0.15:
            return "BEARISH"
        return "NEUTRAL"

    return SentimentResult(
        gold_score=gold_score,
        usd_score=-gold_score * 0.6,  # Altın ile USD ters korelasyonlu
        gold_bias=_bias(gold_score),
        usd_bias=_bias(-gold_score * 0.6),
        headline_count=len(headlines),
        method="finbert",
    )
