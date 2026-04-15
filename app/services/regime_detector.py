from __future__ import annotations

"""
Volatilite Rejim Tespiti — HMM (Hidden Markov Model) tabanlı.

3 gizli durum:
  - LOW_VOL: Düşük volatilite, range piyasa → dar stop, az sinyal
  - NORMAL: Normal piyasa koşulları → standart parametre
  - HIGH_VOL: Yüksek volatilite, kriz → geniş stop veya bekle

hmmlearn yoksa basit threshold bazlı fallback kullanır.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Cache
_regime_cache: dict[str, tuple[datetime, "RegimeResult"]] = {}
_CACHE_TTL = timedelta(minutes=15)


@dataclass(frozen=True)
class RegimeResult:
    regime: str           # LOW_VOL / NORMAL / HIGH_VOL
    confidence: float     # 0.0 - 1.0
    vol_percentile: float # Current vol percentile (0-100)
    description: str
    # Parametre çarpanları
    sl_multiplier: float  # SL mesafesi çarpanı
    tp_multiplier: float  # TP mesafesi çarpanı
    size_multiplier: float  # Pozisyon büyüklüğü çarpanı


def detect_regime(df: pd.DataFrame, symbol: str = "") -> RegimeResult:
    """
    Piyasa rejimini tespit et.
    HMM varsa kullanır, yoksa percentile bazlı fallback.
    """
    # Cache kontrolü
    cache_key = f"{symbol}_{len(df)}"
    if cache_key in _regime_cache:
        cached_time, cached_result = _regime_cache[cache_key]
        if datetime.now() - cached_time < _CACHE_TTL:
            return cached_result

    result = _hmm_detect(df) if _has_hmmlearn() else _threshold_detect(df)
    _regime_cache[cache_key] = (datetime.now(), result)
    return result


def _has_hmmlearn() -> bool:
    try:
        import hmmlearn  # noqa: F401
        return True
    except ImportError:
        return False


def _hmm_detect(df: pd.DataFrame) -> RegimeResult:
    """HMM tabanlı rejim tespiti."""
    try:
        from hmmlearn.hmm import GaussianHMM

        if len(df) < 50:
            return _threshold_detect(df)

        # Feature: log returns + ATR ratio
        close = df["close"].astype(float)
        returns = np.log(close / close.shift(1)).dropna()
        volatility = returns.rolling(14).std().dropna()

        if len(volatility) < 30:
            return _threshold_detect(df)

        X = volatility.values.reshape(-1, 1)

        model = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        model.fit(X)
        states = model.predict(X)
        current_state = int(states[-1])

        # Identify which state is which by mean volatility
        state_means = [float(X[states == i].mean()) for i in range(3)]
        sorted_states = sorted(range(3), key=lambda i: state_means[i])
        # sorted_states[0] = lowest vol, sorted_states[2] = highest vol

        regime_map = {
            sorted_states[0]: "LOW_VOL",
            sorted_states[1]: "NORMAL",
            sorted_states[2]: "HIGH_VOL",
        }
        regime = regime_map[current_state]

        # Confidence from posterior probabilities
        posteriors = model.predict_proba(X)
        confidence = float(posteriors[-1][current_state])

        vol_pctl = float(np.percentile(
            [state_means[sorted_states.index(i)] for i in range(3)],
            50,
        ))

        return _build_result(regime, confidence, float(np.mean(X[-5:])), float(np.mean(X)))

    except Exception as exc:
        logger.debug("HMM detection failed: %s, falling back to threshold", exc)
        return _threshold_detect(df)


def _threshold_detect(df: pd.DataFrame) -> RegimeResult:
    """Percentile bazlı basit rejim tespiti (fallback)."""
    if len(df) < 20:
        return _build_result("NORMAL", 0.5, 50.0, 50.0)

    close = df["close"].astype(float)
    returns = np.log(close / close.shift(1)).dropna()

    if len(returns) < 14:
        return _build_result("NORMAL", 0.5, 50.0, 50.0)

    rolling_vol = returns.rolling(14).std().dropna()
    if len(rolling_vol) < 5:
        return _build_result("NORMAL", 0.5, 50.0, 50.0)

    current_vol = float(rolling_vol.iloc[-1])
    avg_vol = float(rolling_vol.mean())

    # Percentile of current vol vs historical
    vol_percentile = float((rolling_vol <= current_vol).mean() * 100)

    if vol_percentile <= 25:
        regime = "LOW_VOL"
        confidence = (25 - vol_percentile) / 25
    elif vol_percentile >= 75:
        regime = "HIGH_VOL"
        confidence = (vol_percentile - 75) / 25
    else:
        regime = "NORMAL"
        confidence = 1.0 - abs(vol_percentile - 50) / 25

    return _build_result(regime, min(1.0, confidence), current_vol, avg_vol)


def _build_result(regime: str, confidence: float, current_vol: float, avg_vol: float) -> RegimeResult:
    """Rejim sonucunu parametre çarpanlarıyla birlikte oluştur."""
    if regime == "LOW_VOL":
        return RegimeResult(
            regime=regime,
            confidence=round(confidence, 3),
            vol_percentile=round(current_vol / max(avg_vol, 1e-9) * 50, 1),
            description="Düşük volatilite — dar range, dikkatli ol",
            sl_multiplier=0.8,   # Daha dar SL
            tp_multiplier=0.8,   # Daha dar TP
            size_multiplier=0.7, # Küçük pozisyon
        )
    elif regime == "HIGH_VOL":
        return RegimeResult(
            regime=regime,
            confidence=round(confidence, 3),
            vol_percentile=round(current_vol / max(avg_vol, 1e-9) * 50, 1),
            description="Yüksek volatilite — geniş stop gerekli veya bekle",
            sl_multiplier=1.5,   # Daha geniş SL
            tp_multiplier=1.3,   # Daha geniş TP
            size_multiplier=0.5, # Küçük pozisyon (risk azalt)
        )
    else:
        return RegimeResult(
            regime=regime,
            confidence=round(confidence, 3),
            vol_percentile=round(current_vol / max(avg_vol, 1e-9) * 50, 1),
            description="Normal piyasa koşulları",
            sl_multiplier=1.0,
            tp_multiplier=1.0,
            size_multiplier=1.0,
        )
