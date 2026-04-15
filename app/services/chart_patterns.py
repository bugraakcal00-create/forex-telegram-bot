"""
Kapsamli Chart Pattern Tespit Motoru
35+ klasik, harmonik ve gelismis grafik formasyonu.

Kategoriler:
  1. Reversal Patterns (Donus formasyonlari)
  2. Continuation Patterns (Devam formasyonlari)
  3. Harmonic Patterns (Harmonik formasyonlar)
  4. Candlestick Patterns (Mum formasyonlari)
  5. Gap Patterns (Bosluk formasyonlari)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PatternResult:
    """Tespit edilen pattern bilgisi."""
    name: str
    pattern_type: str        # reversal / continuation / harmonic / candlestick / gap
    direction: str           # bullish / bearish
    confidence: int          # 0-100
    entry_price: float       # Onerilen giris fiyati
    stop_loss: float         # Onerilen stop loss
    take_profit: float       # Onerilen take profit
    description: str         # Aciklama


def _swing_points(df: pd.DataFrame, left: int = 5, right: int = 5) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Swing high/low noktalarini bul."""
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    for i in range(left, len(df) - right):
        h = float(df.iloc[i]["high"])
        l = float(df.iloc[i]["low"])
        is_high = all(h >= float(df.iloc[i + d]["high"]) for d in range(-left, right + 1) if d != 0)
        is_low = all(l <= float(df.iloc[i + d]["low"]) for d in range(-left, right + 1) if d != 0)
        if is_high:
            highs.append((i, h))
        if is_low:
            lows.append((i, l))
    return highs, lows


def _tolerance(price: float, pct: float = 0.003) -> float:
    return abs(price * pct)


# ═══════════════════════════════════════════════════════════════════
#  1. REVERSAL PATTERNS
# ═══════════════════════════════════════════════════════════════════

def detect_head_and_shoulders(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """Head and Shoulders (Bas-Omuz) — bearish reversal."""
    if len(df) < 60:
        return None
    highs, lows = _swing_points(df.tail(80), left=4, right=4)
    if len(highs) < 3 or len(lows) < 2:
        return None
    # Son 3 tepe: left shoulder, head, right shoulder
    ls, head, rs = highs[-3], highs[-2], highs[-1]
    # Head en yuksek olmali
    if not (head[1] > ls[1] and head[1] > rs[1]):
        return None
    # Omuzlar yaklasik ayni seviyede
    tol = _tolerance(head[1], 0.015)
    if abs(ls[1] - rs[1]) > tol:
        return None
    # Neckline: iki dip arasi
    nl_lows = [l for l in lows if ls[0] < l[0] < rs[0]]
    if not nl_lows:
        return None
    neckline = min(l[1] for l in nl_lows)
    price = float(df.iloc[-1]["close"])
    # Kirilma kontrol: fiyat neckline'a yakin veya alti
    if price > neckline + atr * 0.5:
        return None
    height = head[1] - neckline
    return PatternResult(
        name="Head and Shoulders", pattern_type="reversal", direction="bearish",
        confidence=80, entry_price=neckline, stop_loss=round(rs[1] + atr * 0.3, 5),
        take_profit=round(neckline - height, 5),
        description=f"Bas-Omuz formasyonu: Neckline {neckline:.5f}, Hedef {neckline - height:.5f}",
    )


def detect_inverse_head_and_shoulders(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """Inverse Head and Shoulders — bullish reversal."""
    if len(df) < 60:
        return None
    highs, lows = _swing_points(df.tail(80), left=4, right=4)
    if len(lows) < 3 or len(highs) < 2:
        return None
    ls, head, rs = lows[-3], lows[-2], lows[-1]
    if not (head[1] < ls[1] and head[1] < rs[1]):
        return None
    tol = _tolerance(head[1], 0.015)
    if abs(ls[1] - rs[1]) > tol:
        return None
    nl_highs = [h for h in highs if ls[0] < h[0] < rs[0]]
    if not nl_highs:
        return None
    neckline = max(h[1] for h in nl_highs)
    price = float(df.iloc[-1]["close"])
    if price < neckline - atr * 0.5:
        return None
    height = neckline - head[1]
    return PatternResult(
        name="Inverse Head and Shoulders", pattern_type="reversal", direction="bullish",
        confidence=80, entry_price=neckline, stop_loss=round(rs[1] - atr * 0.3, 5),
        take_profit=round(neckline + height, 5),
        description=f"Ters Bas-Omuz: Neckline {neckline:.5f}, Hedef {neckline + height:.5f}",
    )


def detect_double_top(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """Double Top — bearish reversal."""
    if len(df) < 40:
        return None
    highs, lows = _swing_points(df.tail(60), left=3, right=3)
    if len(highs) < 2:
        return None
    top1, top2 = highs[-2], highs[-1]
    tol = _tolerance(top1[1], 0.005)
    if abs(top1[1] - top2[1]) > tol:
        return None
    if top2[0] - top1[0] < 5:
        return None
    mid_lows = [l for l in lows if top1[0] < l[0] < top2[0]]
    neckline = min(l[1] for l in mid_lows) if mid_lows else float(df.iloc[top1[0]:top2[0]]["low"].min())
    price = float(df.iloc[-1]["close"])
    if price > neckline + atr:
        return None
    height = top1[1] - neckline
    return PatternResult(
        name="Double Top", pattern_type="reversal", direction="bearish",
        confidence=75, entry_price=neckline, stop_loss=round(max(top1[1], top2[1]) + atr * 0.2, 5),
        take_profit=round(neckline - height, 5),
        description=f"Cift Tepe: {top1[1]:.5f} & {top2[1]:.5f}, Neckline {neckline:.5f}",
    )


def detect_double_bottom(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """Double Bottom — bullish reversal."""
    if len(df) < 40:
        return None
    highs, lows = _swing_points(df.tail(60), left=3, right=3)
    if len(lows) < 2:
        return None
    bot1, bot2 = lows[-2], lows[-1]
    tol = _tolerance(bot1[1], 0.005)
    if abs(bot1[1] - bot2[1]) > tol:
        return None
    if bot2[0] - bot1[0] < 5:
        return None
    mid_highs = [h for h in highs if bot1[0] < h[0] < bot2[0]]
    neckline = max(h[1] for h in mid_highs) if mid_highs else float(df.iloc[bot1[0]:bot2[0]]["high"].max())
    price = float(df.iloc[-1]["close"])
    if price < neckline - atr:
        return None
    height = neckline - bot1[1]
    return PatternResult(
        name="Double Bottom", pattern_type="reversal", direction="bullish",
        confidence=75, entry_price=neckline, stop_loss=round(min(bot1[1], bot2[1]) - atr * 0.2, 5),
        take_profit=round(neckline + height, 5),
        description=f"Cift Dip: {bot1[1]:.5f} & {bot2[1]:.5f}, Neckline {neckline:.5f}",
    )


def detect_triple_top(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """Triple Top — bearish reversal."""
    if len(df) < 60:
        return None
    highs, lows = _swing_points(df.tail(80), left=3, right=3)
    if len(highs) < 3:
        return None
    t1, t2, t3 = highs[-3], highs[-2], highs[-1]
    tol = _tolerance(t1[1], 0.008)
    if abs(t1[1] - t2[1]) > tol or abs(t2[1] - t3[1]) > tol:
        return None
    neckline = min(l[1] for l in lows if t1[0] < l[0] < t3[0]) if any(t1[0] < l[0] < t3[0] for l in lows) else float(df.iloc[t1[0]:t3[0]]["low"].min())
    price = float(df.iloc[-1]["close"])
    if price > neckline + atr:
        return None
    height = t1[1] - neckline
    return PatternResult(
        name="Triple Top", pattern_type="reversal", direction="bearish",
        confidence=82, entry_price=neckline, stop_loss=round(max(t1[1], t2[1], t3[1]) + atr * 0.2, 5),
        take_profit=round(neckline - height, 5),
        description=f"Uclu Tepe: ~{t1[1]:.5f}, Neckline {neckline:.5f}",
    )


def detect_triple_bottom(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """Triple Bottom — bullish reversal."""
    if len(df) < 60:
        return None
    highs, lows = _swing_points(df.tail(80), left=3, right=3)
    if len(lows) < 3:
        return None
    b1, b2, b3 = lows[-3], lows[-2], lows[-1]
    tol = _tolerance(b1[1], 0.008)
    if abs(b1[1] - b2[1]) > tol or abs(b2[1] - b3[1]) > tol:
        return None
    neckline = max(h[1] for h in highs if b1[0] < h[0] < b3[0]) if any(b1[0] < h[0] < b3[0] for h in highs) else float(df.iloc[b1[0]:b3[0]]["high"].max())
    price = float(df.iloc[-1]["close"])
    if price < neckline - atr:
        return None
    height = neckline - b1[1]
    return PatternResult(
        name="Triple Bottom", pattern_type="reversal", direction="bullish",
        confidence=82, entry_price=neckline, stop_loss=round(min(b1[1], b2[1], b3[1]) - atr * 0.2, 5),
        take_profit=round(neckline + height, 5),
        description=f"Uclu Dip: ~{b1[1]:.5f}, Neckline {neckline:.5f}",
    )


def detect_rounding_bottom(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """Rounding Bottom (Saucer) — bullish reversal."""
    if len(df) < 40:
        return None
    window = df.tail(50)
    n = len(window)
    closes = window["close"].astype(float).values
    # U-seklinde mi kontrol: ilk 1/3 dusus, orta 1/3 duz, son 1/3 yukselis
    third = n // 3
    first_avg = float(np.mean(closes[:third]))
    mid_avg = float(np.mean(closes[third:2 * third]))
    last_avg = float(np.mean(closes[2 * third:]))
    if not (first_avg > mid_avg and last_avg > mid_avg and last_avg > first_avg * 0.98):
        return None
    # Minimum nokta orta bolmede mi
    min_idx = int(np.argmin(closes))
    if not (third - 3 <= min_idx <= 2 * third + 3):
        return None
    price = float(closes[-1])
    low_point = float(np.min(closes))
    rim = max(first_avg, last_avg)
    return PatternResult(
        name="Rounding Bottom", pattern_type="reversal", direction="bullish",
        confidence=65, entry_price=price, stop_loss=round(low_point - atr * 0.3, 5),
        take_profit=round(price + (rim - low_point), 5),
        description=f"Yuvarlak Dip: Dip {low_point:.5f}, Rim ~{rim:.5f}",
    )


def detect_v_reversal(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """V-Bottom / V-Top — sharp reversal."""
    if len(df) < 15:
        return None
    window = df.tail(15)
    closes = window["close"].astype(float).values
    min_idx = int(np.argmin(closes))
    max_idx = int(np.argmax(closes))
    price = float(closes[-1])
    # V-Bottom: dip ortada, sert dusus + sert yukselis
    if 3 <= min_idx <= 11:
        drop = float(np.max(closes[:min_idx + 1]) - closes[min_idx])
        rise = float(closes[-1] - closes[min_idx])
        if drop > atr * 2 and rise > atr * 2 and abs(drop - rise) / max(drop, 1e-9) < 0.4:
            return PatternResult(
                name="V-Bottom Reversal", pattern_type="reversal", direction="bullish",
                confidence=70, entry_price=price, stop_loss=round(float(closes[min_idx]) - atr * 0.3, 5),
                take_profit=round(price + rise * 0.5, 5),
                description=f"V-Dip donus: Dip {closes[min_idx]:.5f}",
            )
    # V-Top
    if 3 <= max_idx <= 11:
        rise = float(closes[max_idx] - np.min(closes[:max_idx + 1]))
        drop = float(closes[max_idx] - closes[-1])
        if rise > atr * 2 and drop > atr * 2 and abs(rise - drop) / max(rise, 1e-9) < 0.4:
            return PatternResult(
                name="V-Top Reversal", pattern_type="reversal", direction="bearish",
                confidence=70, entry_price=price, stop_loss=round(float(closes[max_idx]) + atr * 0.3, 5),
                take_profit=round(price - drop * 0.5, 5),
                description=f"V-Tepe donus: Tepe {closes[max_idx]:.5f}",
            )
    return None


def detect_diamond(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """Diamond Top/Bottom — reversal pattern."""
    if len(df) < 40:
        return None
    window = df.tail(40)
    highs_arr = window["high"].astype(float).values
    lows_arr = window["low"].astype(float).values
    n = len(window)
    half = n // 2
    # Ilk yari: genisleyen range, ikinci yari: daralan range
    first_range = float(np.max(highs_arr[:half]) - np.min(lows_arr[:half]))
    mid_range = float(np.max(highs_arr[half // 2:half + half // 2]) - np.min(lows_arr[half // 2:half + half // 2]))
    last_range = float(np.max(highs_arr[half:]) - np.min(lows_arr[half:]))
    if not (mid_range > first_range * 0.8 and mid_range > last_range * 1.2):
        return None
    price = float(window.iloc[-1]["close"])
    mid_high = float(np.max(highs_arr))
    mid_low = float(np.min(lows_arr))
    height = mid_high - mid_low
    direction = "bearish" if price < (mid_high + mid_low) / 2 else "bullish"
    if direction == "bearish":
        return PatternResult(
            name="Diamond Top", pattern_type="reversal", direction="bearish",
            confidence=68, entry_price=price, stop_loss=round(mid_high + atr * 0.2, 5),
            take_profit=round(price - height * 0.6, 5),
            description=f"Elmas Tepe: Range {mid_low:.5f}-{mid_high:.5f}",
        )
    return PatternResult(
        name="Diamond Bottom", pattern_type="reversal", direction="bullish",
        confidence=68, entry_price=price, stop_loss=round(mid_low - atr * 0.2, 5),
        take_profit=round(price + height * 0.6, 5),
        description=f"Elmas Dip: Range {mid_low:.5f}-{mid_high:.5f}",
    )


# ═══════════════════════════════════════════════════════════════════
#  2. CONTINUATION PATTERNS
# ═══════════════════════════════════════════════════════════════════

def detect_ascending_triangle(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """Ascending Triangle — bullish continuation."""
    if len(df) < 30:
        return None
    highs, lows = _swing_points(df.tail(40), left=3, right=3)
    if len(highs) < 2 or len(lows) < 2:
        return None
    # Flat resistance: son 2+ tepe ayni seviyede
    tol = _tolerance(highs[-1][1], 0.004)
    flat_res = abs(highs[-1][1] - highs[-2][1]) <= tol
    # Rising lows: her dip bir oncekinden yuksek
    rising_lows = lows[-1][1] > lows[-2][1] if len(lows) >= 2 else False
    if not (flat_res and rising_lows):
        return None
    resistance = (highs[-1][1] + highs[-2][1]) / 2
    price = float(df.iloc[-1]["close"])
    height = resistance - lows[-1][1]
    return PatternResult(
        name="Ascending Triangle", pattern_type="continuation", direction="bullish",
        confidence=72, entry_price=round(resistance, 5),
        stop_loss=round(lows[-1][1] - atr * 0.2, 5),
        take_profit=round(resistance + height, 5),
        description=f"Yukselen Ucgen: Direnc {resistance:.5f}, Hedef {resistance + height:.5f}",
    )


def detect_descending_triangle(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """Descending Triangle — bearish continuation."""
    if len(df) < 30:
        return None
    highs, lows = _swing_points(df.tail(40), left=3, right=3)
    if len(highs) < 2 or len(lows) < 2:
        return None
    tol = _tolerance(lows[-1][1], 0.004)
    flat_sup = abs(lows[-1][1] - lows[-2][1]) <= tol
    falling_highs = highs[-1][1] < highs[-2][1]
    if not (flat_sup and falling_highs):
        return None
    support = (lows[-1][1] + lows[-2][1]) / 2
    price = float(df.iloc[-1]["close"])
    height = highs[-1][1] - support
    return PatternResult(
        name="Descending Triangle", pattern_type="continuation", direction="bearish",
        confidence=72, entry_price=round(support, 5),
        stop_loss=round(highs[-1][1] + atr * 0.2, 5),
        take_profit=round(support - height, 5),
        description=f"Azalan Ucgen: Destek {support:.5f}, Hedef {support - height:.5f}",
    )


def detect_symmetrical_triangle(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """Symmetrical Triangle — bilateral (trend devamı yonunde)."""
    if len(df) < 30:
        return None
    highs, lows = _swing_points(df.tail(40), left=3, right=3)
    if len(highs) < 2 or len(lows) < 2:
        return None
    falling_highs = highs[-1][1] < highs[-2][1]
    rising_lows = lows[-1][1] > lows[-2][1]
    if not (falling_highs and rising_lows):
        return None
    price = float(df.iloc[-1]["close"])
    apex = (highs[-1][1] + lows[-1][1]) / 2
    height = highs[-2][1] - lows[-2][1]
    # Trend yonunde kirilma beklenir
    ema50 = float(df["close"].ewm(span=50, adjust=False).mean().iloc[-1])
    direction = "bullish" if price > ema50 else "bearish"
    if direction == "bullish":
        return PatternResult(
            name="Symmetrical Triangle", pattern_type="continuation", direction="bullish",
            confidence=65, entry_price=round(highs[-1][1], 5),
            stop_loss=round(lows[-1][1] - atr * 0.2, 5),
            take_profit=round(highs[-1][1] + height * 0.7, 5),
            description=f"Simetrik Ucgen: Apex ~{apex:.5f}, Yukari kirilma beklenir",
        )
    return PatternResult(
        name="Symmetrical Triangle", pattern_type="continuation", direction="bearish",
        confidence=65, entry_price=round(lows[-1][1], 5),
        stop_loss=round(highs[-1][1] + atr * 0.2, 5),
        take_profit=round(lows[-1][1] - height * 0.7, 5),
        description=f"Simetrik Ucgen: Apex ~{apex:.5f}, Asagi kirilma beklenir",
    )


def detect_rising_wedge(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """Rising Wedge — bearish reversal."""
    if len(df) < 30:
        return None
    highs, lows = _swing_points(df.tail(40), left=3, right=3)
    if len(highs) < 2 or len(lows) < 2:
        return None
    rising_highs = highs[-1][1] > highs[-2][1]
    rising_lows = lows[-1][1] > lows[-2][1]
    if not (rising_highs and rising_lows):
        return None
    # Daraliyor mu: range azalmali
    range1 = highs[-2][1] - lows[-2][1]
    range2 = highs[-1][1] - lows[-1][1]
    if range2 >= range1:
        return None
    price = float(df.iloc[-1]["close"])
    height = highs[-1][1] - lows[-1][1]
    return PatternResult(
        name="Rising Wedge", pattern_type="reversal", direction="bearish",
        confidence=70, entry_price=round(lows[-1][1], 5),
        stop_loss=round(highs[-1][1] + atr * 0.2, 5),
        take_profit=round(lows[-1][1] - height, 5),
        description=f"Yukselen Kama: Kirilma {lows[-1][1]:.5f} altinda",
    )


def detect_falling_wedge(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """Falling Wedge — bullish reversal."""
    if len(df) < 30:
        return None
    highs, lows = _swing_points(df.tail(40), left=3, right=3)
    if len(highs) < 2 or len(lows) < 2:
        return None
    falling_highs = highs[-1][1] < highs[-2][1]
    falling_lows = lows[-1][1] < lows[-2][1]
    if not (falling_highs and falling_lows):
        return None
    range1 = highs[-2][1] - lows[-2][1]
    range2 = highs[-1][1] - lows[-1][1]
    if range2 >= range1:
        return None
    price = float(df.iloc[-1]["close"])
    height = highs[-1][1] - lows[-1][1]
    return PatternResult(
        name="Falling Wedge", pattern_type="reversal", direction="bullish",
        confidence=70, entry_price=round(highs[-1][1], 5),
        stop_loss=round(lows[-1][1] - atr * 0.2, 5),
        take_profit=round(highs[-1][1] + height, 5),
        description=f"Dusen Kama: Kirilma {highs[-1][1]:.5f} ustunde",
    )


def detect_flag(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """Bull/Bear Flag — continuation pattern."""
    if len(df) < 20:
        return None
    # Son 20 mum: ilk 5 mum guclu hareket (pole), sonra 15 mum kucuk kanal (flag)
    pole = df.iloc[-20:-15]
    flag = df.iloc[-15:]
    pole_range = float(pole["high"].max() - pole["low"].min())
    flag_range = float(flag["high"].max() - flag["low"].min())
    if flag_range >= pole_range * 0.5 or pole_range < atr * 1.5:
        return None
    pole_dir = float(pole.iloc[-1]["close"]) - float(pole.iloc[0]["close"])
    price = float(df.iloc[-1]["close"])
    if pole_dir > 0:  # Bull flag
        return PatternResult(
            name="Bull Flag", pattern_type="continuation", direction="bullish",
            confidence=72, entry_price=round(float(flag["high"].max()), 5),
            stop_loss=round(float(flag["low"].min()) - atr * 0.2, 5),
            take_profit=round(price + pole_range * 0.8, 5),
            description=f"Boga Bayragi: Pole {pole_range:.5f}, Hedef +{pole_range * 0.8:.5f}",
        )
    else:  # Bear flag
        return PatternResult(
            name="Bear Flag", pattern_type="continuation", direction="bearish",
            confidence=72, entry_price=round(float(flag["low"].min()), 5),
            stop_loss=round(float(flag["high"].max()) + atr * 0.2, 5),
            take_profit=round(price - pole_range * 0.8, 5),
            description=f"Ayi Bayragi: Pole {pole_range:.5f}, Hedef -{pole_range * 0.8:.5f}",
        )


def detect_pennant(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """Pennant — continuation (symmetric triangle after strong move)."""
    if len(df) < 20:
        return None
    pole = df.iloc[-20:-12]
    pennant = df.iloc[-12:]
    pole_range = float(pole["high"].max() - pole["low"].min())
    if pole_range < atr * 1.5:
        return None
    # Pennant: daralan range
    p_highs = pennant["high"].astype(float).values
    p_lows = pennant["low"].astype(float).values
    early_range = float(max(p_highs[:4]) - min(p_lows[:4]))
    late_range = float(max(p_highs[-4:]) - min(p_lows[-4:]))
    if late_range >= early_range * 0.8:
        return None
    pole_dir = float(pole.iloc[-1]["close"]) - float(pole.iloc[0]["close"])
    price = float(df.iloc[-1]["close"])
    direction = "bullish" if pole_dir > 0 else "bearish"
    if direction == "bullish":
        return PatternResult(
            name="Bull Pennant", pattern_type="continuation", direction="bullish",
            confidence=70, entry_price=round(float(max(p_highs)), 5),
            stop_loss=round(float(min(p_lows)) - atr * 0.2, 5),
            take_profit=round(price + pole_range * 0.7, 5),
            description=f"Boga Flamasi: Hedef +{pole_range * 0.7:.5f}",
        )
    return PatternResult(
        name="Bear Pennant", pattern_type="continuation", direction="bearish",
        confidence=70, entry_price=round(float(min(p_lows)), 5),
        stop_loss=round(float(max(p_highs)) + atr * 0.2, 5),
        take_profit=round(price - pole_range * 0.7, 5),
        description=f"Ayi Flamasi: Hedef -{pole_range * 0.7:.5f}",
    )


def detect_cup_and_handle(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """Cup and Handle — bullish continuation."""
    if len(df) < 50:
        return None
    window = df.tail(60)
    closes = window["close"].astype(float).values
    n = len(closes)
    # Cup: U-shape in first 40 bars
    cup = closes[:40] if n >= 50 else closes[:n - 10]
    handle = closes[-10:]
    cup_high_l = float(np.max(cup[:10]))
    cup_low = float(np.min(cup[8:32]))
    cup_high_r = float(np.max(cup[-10:]))
    # Cup: iki kenar yuksek, orta dusuk
    if not (cup_low < cup_high_l * 0.97 and cup_low < cup_high_r * 0.97):
        return None
    # Handle: kucuk dusus, cup_high_r'ye yakin
    handle_low = float(np.min(handle))
    handle_high = float(np.max(handle))
    rim = max(cup_high_l, cup_high_r)
    if handle_low < cup_low or (rim - handle_low) > (rim - cup_low) * 0.5:
        return None
    price = float(closes[-1])
    height = rim - cup_low
    return PatternResult(
        name="Cup and Handle", pattern_type="continuation", direction="bullish",
        confidence=75, entry_price=round(rim, 5),
        stop_loss=round(handle_low - atr * 0.2, 5),
        take_profit=round(rim + height, 5),
        description=f"Fincan-Kulp: Rim {rim:.5f}, Hedef {rim + height:.5f}",
    )


def detect_channel(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """Ascending/Descending Channel — continuation."""
    if len(df) < 25:
        return None
    highs, lows = _swing_points(df.tail(30), left=2, right=2)
    if len(highs) < 3 or len(lows) < 3:
        return None
    # Trend: highs ve lows'un yonu
    h_slope = (highs[-1][1] - highs[-3][1]) / max(highs[-1][0] - highs[-3][0], 1)
    l_slope = (lows[-1][1] - lows[-3][1]) / max(lows[-1][0] - lows[-3][0], 1)
    # Paralel mi: slope'lar benzer
    if abs(h_slope) < atr * 0.001 and abs(l_slope) < atr * 0.001:
        return None
    if h_slope > 0 and l_slope > 0:  # Ascending
        price = float(df.iloc[-1]["close"])
        ch_width = highs[-1][1] - lows[-1][1]
        if abs(price - lows[-1][1]) < ch_width * 0.3:
            return PatternResult(
                name="Ascending Channel", pattern_type="continuation", direction="bullish",
                confidence=65, entry_price=round(lows[-1][1], 5),
                stop_loss=round(lows[-1][1] - atr * 0.3, 5),
                take_profit=round(highs[-1][1], 5),
                description=f"Yukselen Kanal: Destek {lows[-1][1]:.5f}, Direnc {highs[-1][1]:.5f}",
            )
    elif h_slope < 0 and l_slope < 0:  # Descending
        price = float(df.iloc[-1]["close"])
        ch_width = highs[-1][1] - lows[-1][1]
        if abs(price - highs[-1][1]) < ch_width * 0.3:
            return PatternResult(
                name="Descending Channel", pattern_type="continuation", direction="bearish",
                confidence=65, entry_price=round(highs[-1][1], 5),
                stop_loss=round(highs[-1][1] + atr * 0.3, 5),
                take_profit=round(lows[-1][1], 5),
                description=f"Dusen Kanal: Direnc {highs[-1][1]:.5f}, Destek {lows[-1][1]:.5f}",
            )
    return None


def detect_rectangle(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """Rectangle / Range — breakout yonunde devam."""
    if len(df) < 25:
        return None
    window = df.tail(25)
    h = float(window["high"].max())
    l = float(window["low"].min())
    mid = (h + l) / 2
    rng = h - l
    # Range: son 15 mumun %80'i bu bant icinde mi
    in_range = sum(1 for _, r in window.tail(15).iterrows()
                   if float(r["low"]) >= l - rng * 0.05 and float(r["high"]) <= h + rng * 0.05)
    if in_range < 12:
        return None
    price = float(df.iloc[-1]["close"])
    if abs(price - h) < rng * 0.15:
        return PatternResult(
            name="Rectangle Breakout", pattern_type="continuation", direction="bullish",
            confidence=60, entry_price=round(h, 5), stop_loss=round(mid, 5),
            take_profit=round(h + rng, 5),
            description=f"Dikdortgen yukari kirilma: {l:.5f}-{h:.5f}",
        )
    elif abs(price - l) < rng * 0.15:
        return PatternResult(
            name="Rectangle Breakdown", pattern_type="continuation", direction="bearish",
            confidence=60, entry_price=round(l, 5), stop_loss=round(mid, 5),
            take_profit=round(l - rng, 5),
            description=f"Dikdortgen asagi kirilma: {l:.5f}-{h:.5f}",
        )
    return None


# ═══════════════════════════════════════════════════════════════════
#  3. CANDLESTICK PATTERNS (Genisletilmis)
# ═══════════════════════════════════════════════════════════════════

def detect_candlestick_patterns(df: pd.DataFrame, atr: float) -> list[PatternResult]:
    """Genisletilmis mum formasyonlari tespiti."""
    results: list[PatternResult] = []
    if len(df) < 5:
        return results

    c = df.iloc[-1]   # current
    p = df.iloc[-2]   # previous
    pp = df.iloc[-3]  # 2 bars ago

    price = float(c["close"])
    c_open, c_close, c_high, c_low = float(c["open"]), float(c["close"]), float(c["high"]), float(c["low"])
    p_open, p_close, p_high, p_low = float(p["open"]), float(p["close"]), float(p["high"]), float(p["low"])
    pp_open, pp_close = float(pp["open"]), float(pp["close"])

    c_body = abs(c_close - c_open)
    p_body = abs(p_close - p_open)
    c_range = c_high - c_low
    c_upper_wick = c_high - max(c_open, c_close)
    c_lower_wick = min(c_open, c_close) - c_low

    # 1. Morning Star (3 mum — bullish reversal)
    if (pp_close < pp_open and  # 1. mum bearish
        abs(float(p["close"]) - float(p["open"])) < atr * 0.3 and  # 2. mum kucuk gövde (yildiz)
        c_close > c_open and c_close > (pp_open + pp_close) / 2):  # 3. mum bullish, 1.'nin ortasina kadar
        results.append(PatternResult(
            name="Morning Star", pattern_type="candlestick", direction="bullish",
            confidence=78, entry_price=price, stop_loss=round(float(p["low"]) - atr * 0.2, 5),
            take_profit=round(price + atr * 2, 5), description="Sabah Yildizi: Guclu donus sinyali",
        ))

    # 2. Evening Star (3 mum — bearish reversal)
    if (pp_close > pp_open and
        abs(float(p["close"]) - float(p["open"])) < atr * 0.3 and
        c_close < c_open and c_close < (pp_open + pp_close) / 2):
        results.append(PatternResult(
            name="Evening Star", pattern_type="candlestick", direction="bearish",
            confidence=78, entry_price=price, stop_loss=round(float(p["high"]) + atr * 0.2, 5),
            take_profit=round(price - atr * 2, 5), description="Aksam Yildizi: Guclu donus sinyali",
        ))

    # 3. Three White Soldiers (3 ardisik buyuk bullish mum)
    if len(df) >= 4:
        m1, m2, m3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        if (float(m1["close"]) > float(m1["open"]) and float(m2["close"]) > float(m2["open"]) and
            float(m3["close"]) > float(m3["open"]) and
            float(m2["close"]) > float(m1["close"]) and float(m3["close"]) > float(m2["close"]) and
            abs(float(m1["close"]) - float(m1["open"])) > atr * 0.3):
            results.append(PatternResult(
                name="Three White Soldiers", pattern_type="candlestick", direction="bullish",
                confidence=80, entry_price=price, stop_loss=round(float(m1["low"]) - atr * 0.2, 5),
                take_profit=round(price + atr * 2.5, 5), description="Uc Beyaz Asker: Guclu yukselis",
            ))

    # 4. Three Black Crows (3 ardisik buyuk bearish mum)
    if len(df) >= 4:
        m1, m2, m3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        if (float(m1["close"]) < float(m1["open"]) and float(m2["close"]) < float(m2["open"]) and
            float(m3["close"]) < float(m3["open"]) and
            float(m2["close"]) < float(m1["close"]) and float(m3["close"]) < float(m2["close"]) and
            abs(float(m1["close"]) - float(m1["open"])) > atr * 0.3):
            results.append(PatternResult(
                name="Three Black Crows", pattern_type="candlestick", direction="bearish",
                confidence=80, entry_price=price, stop_loss=round(float(m1["high"]) + atr * 0.2, 5),
                take_profit=round(price - atr * 2.5, 5), description="Uc Kara Karga: Guclu dusus",
            ))

    # 5. Tweezer Top
    if (abs(p_high - c_high) <= atr * 0.1 and p_close > p_open and c_close < c_open):
        results.append(PatternResult(
            name="Tweezer Top", pattern_type="candlestick", direction="bearish",
            confidence=65, entry_price=price, stop_loss=round(c_high + atr * 0.2, 5),
            take_profit=round(price - atr * 1.5, 5), description="Cimbiz Tepe: Ayni high, yon degisimi",
        ))

    # 6. Tweezer Bottom
    if (abs(p_low - c_low) <= atr * 0.1 and p_close < p_open and c_close > c_open):
        results.append(PatternResult(
            name="Tweezer Bottom", pattern_type="candlestick", direction="bullish",
            confidence=65, entry_price=price, stop_loss=round(c_low - atr * 0.2, 5),
            take_profit=round(price + atr * 1.5, 5), description="Cimbiz Dip: Ayni low, yon degisimi",
        ))

    # 7. Inside Bar
    if (c_high <= p_high and c_low >= p_low):
        direction = "bullish" if c_close > c_open else "bearish"
        results.append(PatternResult(
            name="Inside Bar", pattern_type="candlestick", direction=direction,
            confidence=60, entry_price=price,
            stop_loss=round(c_low - atr * 0.1 if direction == "bullish" else c_high + atr * 0.1, 5),
            take_profit=round(price + atr * 1.5 if direction == "bullish" else price - atr * 1.5, 5),
            description="Ic Bar: Konsolidasyon, kirilma beklenir",
        ))

    # 8. Marubozu (gövde = range, fitil yok)
    if c_body > c_range * 0.9 and c_body > atr * 0.5:
        direction = "bullish" if c_close > c_open else "bearish"
        results.append(PatternResult(
            name="Marubozu", pattern_type="candlestick", direction=direction,
            confidence=70, entry_price=price,
            stop_loss=round(c_low - atr * 0.2 if direction == "bullish" else c_high + atr * 0.2, 5),
            take_profit=round(price + c_body if direction == "bullish" else price - c_body, 5),
            description=f"Marubozu: Fitilsiz guclu {'yukselis' if direction == 'bullish' else 'dusus'}",
        ))

    # 9. Harami (Inside bar + yon degisimi)
    if (c_high <= p_high and c_low >= p_low and
        c_body < p_body * 0.5 and
        ((p_close > p_open and c_close < c_open) or (p_close < p_open and c_close > c_open))):
        direction = "bullish" if c_close > c_open else "bearish"
        results.append(PatternResult(
            name=f"{'Bullish' if direction == 'bullish' else 'Bearish'} Harami",
            pattern_type="candlestick", direction=direction,
            confidence=62, entry_price=price,
            stop_loss=round(c_low - atr * 0.2 if direction == "bullish" else c_high + atr * 0.2, 5),
            take_profit=round(price + atr * 1.5 if direction == "bullish" else price - atr * 1.5, 5),
            description="Harami: Ic bar ile yon degisimi",
        ))

    # 10. Piercing Line (bullish — 2 mum)
    if (p_close < p_open and c_close > c_open and
        c_open < p_low and c_close > (p_open + p_close) / 2 and c_close < p_open):
        results.append(PatternResult(
            name="Piercing Line", pattern_type="candlestick", direction="bullish",
            confidence=68, entry_price=price, stop_loss=round(c_low - atr * 0.2, 5),
            take_profit=round(price + atr * 2, 5), description="Delici Cizgi: Guclu bullish donus",
        ))

    # 11. Dark Cloud Cover (bearish — 2 mum)
    if (p_close > p_open and c_close < c_open and
        c_open > p_high and c_close < (p_open + p_close) / 2 and c_close > p_open):
        results.append(PatternResult(
            name="Dark Cloud Cover", pattern_type="candlestick", direction="bearish",
            confidence=68, entry_price=price, stop_loss=round(c_high + atr * 0.2, 5),
            take_profit=round(price - atr * 2, 5), description="Kara Bulut: Guclu bearish donus",
        ))

    return results


# ═══════════════════════════════════════════════════════════════════
#  4. GAP PATTERNS
# ═══════════════════════════════════════════════════════════════════

def detect_gap_patterns(df: pd.DataFrame, atr: float) -> list[PatternResult]:
    """Gap pattern detection."""
    results: list[PatternResult] = []
    if len(df) < 5:
        return results
    c = df.iloc[-1]
    p = df.iloc[-2]
    price = float(c["close"])

    # Gap Up
    if float(c["low"]) > float(p["high"]):
        gap_size = float(c["low"]) - float(p["high"])
        if gap_size > atr * 0.3:
            results.append(PatternResult(
                name="Bullish Gap", pattern_type="gap", direction="bullish",
                confidence=60, entry_price=price, stop_loss=round(float(p["high"]) - atr * 0.1, 5),
                take_profit=round(price + gap_size, 5),
                description=f"Yukari Bosluk: {gap_size:.5f} buyuklugunde",
            ))

    # Gap Down
    if float(c["high"]) < float(p["low"]):
        gap_size = float(p["low"]) - float(c["high"])
        if gap_size > atr * 0.3:
            results.append(PatternResult(
                name="Bearish Gap", pattern_type="gap", direction="bearish",
                confidence=60, entry_price=price, stop_loss=round(float(p["low"]) + atr * 0.1, 5),
                take_profit=round(price - gap_size, 5),
                description=f"Asagi Bosluk: {gap_size:.5f} buyuklugunde",
            ))

    return results


# ═══════════════════════════════════════════════════════════════════
#  5. HARMONIC PATTERNS (Temel)
# ═══════════════════════════════════════════════════════════════════

def _fib_ratio_match(actual: float, target: float, tolerance: float = 0.05) -> bool:
    return abs(actual - target) <= tolerance


def detect_abcd_pattern(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """AB=CD harmonic pattern."""
    if len(df) < 30:
        return None
    highs, lows = _swing_points(df.tail(40), left=3, right=3)
    if len(highs) < 2 or len(lows) < 2:
        return None
    # Bullish AB=CD: A(high) -> B(low) -> C(high) -> D(low)
    if len(highs) >= 2 and len(lows) >= 2:
        a = highs[-2][1]
        b = lows[-2][1]
        c = highs[-1][1]
        d_expected = b - (a - b)  # AB = CD
        price = float(df.iloc[-1]["close"])
        if a > b and c > b and c < a:  # Valid structure
            ab = a - b
            bc = c - b
            bc_ratio = bc / ab if ab > 0 else 0
            if _fib_ratio_match(bc_ratio, 0.618, 0.1) or _fib_ratio_match(bc_ratio, 0.786, 0.1):
                d_target = c - ab
                if abs(price - d_target) <= atr * 1.5:
                    return PatternResult(
                        name="Bullish AB=CD", pattern_type="harmonic", direction="bullish",
                        confidence=68, entry_price=round(d_target, 5),
                        stop_loss=round(d_target - atr * 0.5, 5),
                        take_profit=round(d_target + ab * 0.618, 5),
                        description=f"AB=CD Harmonik: D noktasi {d_target:.5f}",
                    )
    # Bearish AB=CD
    if len(lows) >= 2 and len(highs) >= 2:
        a = lows[-2][1]
        b = highs[-2][1]
        c = lows[-1][1]
        price = float(df.iloc[-1]["close"])
        if b > a and c > a and c < b:
            ab = b - a
            bc = b - c
            bc_ratio = bc / ab if ab > 0 else 0
            if _fib_ratio_match(bc_ratio, 0.618, 0.1) or _fib_ratio_match(bc_ratio, 0.786, 0.1):
                d_target = c + ab
                if abs(price - d_target) <= atr * 1.5:
                    return PatternResult(
                        name="Bearish AB=CD", pattern_type="harmonic", direction="bearish",
                        confidence=68, entry_price=round(d_target, 5),
                        stop_loss=round(d_target + atr * 0.5, 5),
                        take_profit=round(d_target - ab * 0.618, 5),
                        description=f"AB=CD Harmonik: D noktasi {d_target:.5f}",
                    )
    return None


def detect_gartley(df: pd.DataFrame, atr: float) -> PatternResult | None:
    """Gartley harmonic pattern (222 pattern)."""
    if len(df) < 40:
        return None
    highs, lows = _swing_points(df.tail(50), left=3, right=3)
    if len(highs) < 2 or len(lows) < 2:
        return None
    # Bullish Gartley: X(low)->A(high)->B(low)->C(high)->D(low near X)
    # AB = 0.618 XA, BC = 0.382-0.886 AB, CD = 1.272-1.618 BC, D = 0.786 XA
    # Simplified check
    if len(lows) >= 3 and len(highs) >= 2:
        x = lows[-3][1]
        a = highs[-2][1]
        b = lows[-2][1]
        c = highs[-1][1]
        price = float(df.iloc[-1]["close"])
        xa = a - x
        ab = a - b
        if xa > 0 and ab > 0:
            ab_ratio = ab / xa
            if _fib_ratio_match(ab_ratio, 0.618, 0.08):
                d_target = a - xa * 0.786
                if abs(price - d_target) <= atr * 1.5 and d_target > x:
                    return PatternResult(
                        name="Bullish Gartley", pattern_type="harmonic", direction="bullish",
                        confidence=72, entry_price=round(d_target, 5),
                        stop_loss=round(x - atr * 0.3, 5),
                        take_profit=round(d_target + xa * 0.382, 5),
                        description=f"Gartley 222: D={d_target:.5f} (0.786 XA)",
                    )
    return None


# ═══════════════════════════════════════════════════════════════════
#  ANA TESPIT FONKSIYONU
# ═══════════════════════════════════════════════════════════════════

ALL_PATTERN_DETECTORS = [
    # Reversal
    detect_head_and_shoulders,
    detect_inverse_head_and_shoulders,
    detect_double_top,
    detect_double_bottom,
    detect_triple_top,
    detect_triple_bottom,
    detect_rounding_bottom,
    detect_v_reversal,
    detect_diamond,
    # Continuation
    detect_ascending_triangle,
    detect_descending_triangle,
    detect_symmetrical_triangle,
    detect_rising_wedge,
    detect_falling_wedge,
    detect_flag,
    detect_pennant,
    detect_cup_and_handle,
    detect_channel,
    detect_rectangle,
    # Harmonic
    detect_abcd_pattern,
    detect_gartley,
]


def detect_all_patterns(df: pd.DataFrame, atr: float) -> list[PatternResult]:
    """
    Tum chart pattern'lari tarar ve tespit edilenleri dondurur.

    35+ pattern:
    - 9 Reversal: H&S, Inv H&S, Double Top/Bottom, Triple Top/Bottom,
                  Rounding Bottom, V-Reversal, Diamond
    - 10 Continuation: Asc/Desc/Sym Triangle, Rising/Falling Wedge,
                       Flag, Pennant, Cup&Handle, Channel, Rectangle
    - 2 Harmonic: AB=CD, Gartley
    - 11 Candlestick: Morning/Evening Star, 3 White Soldiers, 3 Black Crows,
                      Tweezer Top/Bottom, Inside Bar, Marubozu, Harami,
                      Piercing Line, Dark Cloud Cover
    - 2 Gap: Bullish/Bearish Gap
    """
    patterns: list[PatternResult] = []

    # Classic + Harmonic patterns
    for detector in ALL_PATTERN_DETECTORS:
        try:
            result = detector(df, atr)
            if result is not None:
                patterns.append(result)
        except Exception:
            pass

    # Candlestick patterns
    try:
        patterns.extend(detect_candlestick_patterns(df, atr))
    except Exception:
        pass

    # Gap patterns
    try:
        patterns.extend(detect_gap_patterns(df, atr))
    except Exception:
        pass

    # Confidence'a gore sirala
    patterns.sort(key=lambda p: p.confidence, reverse=True)
    return patterns
