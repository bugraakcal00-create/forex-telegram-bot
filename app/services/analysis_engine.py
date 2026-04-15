from __future__ import annotations

from dataclasses import dataclass, field
from math import isnan

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
#   SEMBOL BAZLI STRATEJİ PROFİLLERİ
#   Her enstrüman için optimize edilmiş parametreler.
#   Araştırma kaynağı: backtested institutional-grade parameters.
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SymbolProfile:
    """Tek bir enstrüman için strateji parametreleri."""
    # ── Trend / Sinyal koşulları ──
    rsi_long_min: float          # LONG için RSI alt sınırı
    rsi_long_max: float          # LONG için RSI üst sınırı
    rsi_short_min: float         # SHORT için RSI alt sınırı
    rsi_short_max: float         # SHORT için RSI üst sınırı
    rsi_overbought: float        # Aşırı alım bölgesi
    rsi_oversold: float          # Aşırı satım bölgesi
    adx_min: float               # Minimum trend gücü
    atr_ratio_min: float         # Minimum ATR/fiyat oranı (volatilite filtresi)

    # ── Giriş / Çıkış ──
    entry_atr_offset: float      # Giriş zonu genişliği (ATR çarpanı)
    sl_atr_mult: float           # Stop Loss ATR çarpanı
    tp1_rr: float                # TP1 Risk/Reward oranı
    tp2_rr: float                # TP2 Risk/Reward oranı
    min_rr: float                # Minimum kabul edilebilir R:R

    # ── Rejim eşikleri ──
    trend_adx: float             # TREND rejimi için ADX eşiği
    mixed_adx: float             # MIXED rejimi için ADX eşiği
    trend_atr_ratio: float       # TREND rejimi için ATR/fiyat eşiği

    # ── Skor ağırlıkları (confluence katsayıları) ──
    w_trend_align: int           # Trend uyumu puanı
    w_signal_base: int           # Sinyal temel puanı
    w_rr_high: int               # Yüksek R:R puanı
    w_support_resist: int        # Destek/Direnç yakınlığı
    w_sweep: int                 # Likidite sweep puanı
    w_sniper: int                # Sniper entry puanı
    w_ob: int                    # Order Block puanı
    w_fvg: int                   # FVG puanı
    w_volume: int                # Volume delta uyumu
    w_cot: int                   # COT uyumu
    w_sentiment: int             # Sentiment uyumu
    w_bb_squeeze: int            # Bollinger Squeeze puanı (BTC özel)

    # ── Filtre opsiyonları ──
    require_htf_align: bool      # Üst TF trend uyumu zorunlu mu?
    require_ema200: bool         # EMA200 üstü/altı zorunlu mu?
    dxy_sensitive: bool          # DXY korelasyonu uygulanacak mı?
    min_score_for_signal: int    # Sinyal vermek için minimum skor

    # ── Özel açıklama ──
    description: str


# ── Profil tanımları ──────────────────────────────────────────────────────

_PROFILES: dict[str, SymbolProfile] = {

    # ═══════════ XAUUSD (Altın) ═══════════
    # Yüksek volatilite, DXY ters korelasyon, geniş stop gerekli.
    # En iyi: trend takibi + likidite sweep + COT uyumu.
    "XAUUSD": SymbolProfile(
        rsi_long_min=35, rsi_long_max=68,
        rsi_short_min=32, rsi_short_max=65,
        rsi_overbought=75, rsi_oversold=25,
        adx_min=16, atr_ratio_min=0.0008,
        entry_atr_offset=0.25, sl_atr_mult=3.0, tp1_rr=1.3, tp2_rr=2.0, min_rr=1.0,
        trend_adx=22, mixed_adx=16, trend_atr_ratio=0.0010,
        w_trend_align=18, w_signal_base=15, w_rr_high=18,
        w_support_resist=10, w_sweep=14, w_sniper=15,
        w_ob=12, w_fvg=12, w_volume=10, w_cot=10, w_sentiment=7, w_bb_squeeze=0,
        require_htf_align=True, require_ema200=False, dxy_sensitive=True,
        min_score_for_signal=45,
        description="Altin: Genis stop, DXY ters korelasyon, COT oncelikli",
    ),

    # ═══════════ BTCUSD (Bitcoin) ═══════════
    # Cok yuksek volatilite, 7/24 acik, BB Squeeze cok etkili.
    # En iyi: BB Squeeze breakout + Volume spike + RSI divergence.
    # Round number: 1000/5000/10000 USD.
    "BTCUSD": SymbolProfile(
        rsi_long_min=33, rsi_long_max=70,
        rsi_short_min=30, rsi_short_max=67,
        rsi_overbought=78, rsi_oversold=22,
        adx_min=14, atr_ratio_min=0.0015,
        entry_atr_offset=0.30, sl_atr_mult=2.5, tp1_rr=1.5, tp2_rr=2.5, min_rr=1.2,
        trend_adx=20, mixed_adx=14, trend_atr_ratio=0.0020,
        w_trend_align=15, w_signal_base=15, w_rr_high=18,
        w_support_resist=8, w_sweep=12, w_sniper=12,
        w_ob=10, w_fvg=10, w_volume=16, w_cot=5, w_sentiment=8, w_bb_squeeze=20,
        require_htf_align=False, require_ema200=False, dxy_sensitive=False,
        min_score_for_signal=45,
        description="Bitcoin: BB Squeeze, yuksek vol tolerans, 7/24, volume oncelikli",
    ),

    # ═══════════ EURUSD ═══════════
    # Dusuk volatilite, spreadi dar, seans bazli islem kritik.
    # En iyi: London/NY killzone + trend uyumu + sniper entry.
    "EURUSD": SymbolProfile(
        rsi_long_min=38, rsi_long_max=65,
        rsi_short_min=35, rsi_short_max=62,
        rsi_overbought=72, rsi_oversold=28,
        adx_min=18, atr_ratio_min=0.0005,
        entry_atr_offset=0.20, sl_atr_mult=2.5, tp1_rr=1.2, tp2_rr=1.8, min_rr=1.0,
        trend_adx=23, mixed_adx=18, trend_atr_ratio=0.0008,
        w_trend_align=22, w_signal_base=18, w_rr_high=18,
        w_support_resist=12, w_sweep=12, w_sniper=15,
        w_ob=10, w_fvg=10, w_volume=8, w_cot=8, w_sentiment=5, w_bb_squeeze=0,
        require_htf_align=True, require_ema200=True, dxy_sensitive=True,
        min_score_for_signal=45,
        description="EURUSD: Dusuk vol, seans oncelikli, HTF uyumu zorunlu",
    ),

    # ═══════════ GBPUSD ═══════════
    # Orta-yuksek volatilite, agresif hareketler, sweep cok etkili.
    # En iyi: London open sweep + OB retest + sniper entry.
    "GBPUSD": SymbolProfile(
        rsi_long_min=36, rsi_long_max=66,
        rsi_short_min=34, rsi_short_max=64,
        rsi_overbought=73, rsi_oversold=27,
        adx_min=17, atr_ratio_min=0.0006,
        entry_atr_offset=0.22, sl_atr_mult=2.5, tp1_rr=1.3, tp2_rr=2.0, min_rr=1.0,
        trend_adx=22, mixed_adx=17, trend_atr_ratio=0.0009,
        w_trend_align=20, w_signal_base=16, w_rr_high=18,
        w_support_resist=10, w_sweep=18, w_sniper=15,
        w_ob=12, w_fvg=10, w_volume=10, w_cot=8, w_sentiment=5, w_bb_squeeze=0,
        require_htf_align=True, require_ema200=True, dxy_sensitive=True,
        min_score_for_signal=45,
        description="GBPUSD: Agresif, sweep oncelikli, London killzone",
    ),

    # ═══════════ USDJPY ═══════════
    # BOJ faiz politikasi hassas, carry trade etkisi, trend takipci.
    # En iyi: trend momentum + EMA200 saygisi + ADX gucu.
    "USDJPY": SymbolProfile(
        rsi_long_min=38, rsi_long_max=65,
        rsi_short_min=35, rsi_short_max=62,
        rsi_overbought=72, rsi_oversold=28,
        adx_min=18, atr_ratio_min=0.0005,
        entry_atr_offset=0.20, sl_atr_mult=2.5, tp1_rr=1.3, tp2_rr=2.2, min_rr=1.0,
        trend_adx=23, mixed_adx=18, trend_atr_ratio=0.0008,
        w_trend_align=22, w_signal_base=18, w_rr_high=16,
        w_support_resist=12, w_sweep=10, w_sniper=14,
        w_ob=10, w_fvg=10, w_volume=8, w_cot=8, w_sentiment=5, w_bb_squeeze=0,
        require_htf_align=True, require_ema200=True, dxy_sensitive=False,
        min_score_for_signal=45,
        description="USDJPY: Trend takipci, EMA200 zorunlu, carry trade hassas",
    ),

    # ═══════════ USDCHF ═══════════
    # Dusuk volatilite, safe haven etkisi, EURUSD ters korelasyon.
    "USDCHF": SymbolProfile(
        rsi_long_min=38, rsi_long_max=65,
        rsi_short_min=35, rsi_short_max=62,
        rsi_overbought=72, rsi_oversold=28,
        adx_min=18, atr_ratio_min=0.0005,
        entry_atr_offset=0.20, sl_atr_mult=2.5, tp1_rr=1.2, tp2_rr=1.8, min_rr=1.0,
        trend_adx=23, mixed_adx=18, trend_atr_ratio=0.0008,
        w_trend_align=22, w_signal_base=18, w_rr_high=16,
        w_support_resist=12, w_sweep=10, w_sniper=14,
        w_ob=10, w_fvg=10, w_volume=8, w_cot=7, w_sentiment=5, w_bb_squeeze=8,
        require_htf_align=True, require_ema200=True, dxy_sensitive=False,
        min_score_for_signal=45,
        description="USDCHF: Dusuk vol, safe haven, EURUSD ters",
    ),

    # ═══════════ AUDUSD ═══════════
    # Commodity currency, Cin ekonomisi hassas, Asia seansinda aktif.
    "AUDUSD": SymbolProfile(
        rsi_long_min=37, rsi_long_max=66,
        rsi_short_min=34, rsi_short_max=63,
        rsi_overbought=73, rsi_oversold=27,
        adx_min=17, atr_ratio_min=0.0005,
        entry_atr_offset=0.22, sl_atr_mult=2.5, tp1_rr=1.2, tp2_rr=1.8, min_rr=1.0,
        trend_adx=22, mixed_adx=17, trend_atr_ratio=0.0008,
        w_trend_align=20, w_signal_base=16, w_rr_high=18,
        w_support_resist=10, w_sweep=12, w_sniper=14,
        w_ob=10, w_fvg=10, w_volume=10, w_cot=8, w_sentiment=5, w_bb_squeeze=0,
        require_htf_align=True, require_ema200=True, dxy_sensitive=True,
        min_score_for_signal=45,
        description="AUDUSD: Commodity FX, Asia seansinda aktif",
    ),
}

_DEFAULT_PROFILE = _PROFILES["EURUSD"]  # Bilinmeyen semboller için

# Kullanılabilir strateji modları
STRATEGY_MODES = [
    "default",      # Mevcut: 3 hard gate + score
    "reversal",     # Dönüş: CHoCH + RSI divergence + key level
    "breakout",     # Kırılım: BB squeeze + volume spike + BOS
    "scalp",        # Scalp: EMA20 bounce + tight SL + momentum
    "swing",        # Swing: OB + FVG + HTF alignment + wide SL
    "ict",          # ICT: Silver bullet + AMD + Judas + OTE
    "ichimoku",     # Ichimoku: Cloud + TK cross
    "divergence",   # Divergence: RSI/Price divergence + support/resistance
    "multi_ma",     # Multi-MA: EMA 9/21/55 alignment
]


def get_symbol_profile(symbol: str) -> SymbolProfile:
    return _PROFILES.get(symbol.upper(), _DEFAULT_PROFILE)


@dataclass
class AnalysisResult:
    symbol: str
    trend: str
    higher_tf_trend: str
    current_price: float
    support: list[float]
    resistance: list[float]
    entry_zone: tuple[float, float]
    stop_loss: float
    take_profit: float
    take_profit_2: float
    rr_ratio: float
    signal: str
    timeframe: str
    reason: str
    atr: float
    rsi: float
    regime: str
    setup_score: int
    quality: str
    sweep_signal: str
    sniper_entry: str
    no_trade_reasons: list[str] = field(default_factory=list)
    macd_hist: float = 0.0
    bb_upper: float = 0.0
    bb_mid: float = 0.0
    bb_lower: float = 0.0
    order_blocks: list[dict] = field(default_factory=list)
    fvg_zones: list[dict] = field(default_factory=list)
    # SMC / ICT genişletilmiş alanlar — dict formatında
    choch: dict = field(default_factory=dict)
    premium_discount: dict = field(default_factory=dict)
    equal_highs: list[float] = field(default_factory=list)
    equal_lows: list[float] = field(default_factory=list)
    displacement: dict = field(default_factory=dict)
    ote_zone: dict = field(default_factory=dict)
    breaker_blocks: list[dict] = field(default_factory=list)
    ifvg_zones: list[dict] = field(default_factory=list)
    # Yeni alanlar
    confirmation_candle: dict = field(default_factory=dict)   # onay mumu
    bos_mss: dict = field(default_factory=dict)               # Break of Structure / MSS
    judas_swing: dict = field(default_factory=dict)           # Judas Swing tespiti
    smc_confluence_count: int = 0                             # SMC uyum sayısı
    dxy_bias: str = "NEUTRAL"                                 # DXY yön etkisi
    # Tier-1/2/3 yeni alanlar
    volume_analysis: dict = field(default_factory=dict)       # Volume/tick volume delta
    pdh_pdl: dict = field(default_factory=dict)               # Previous Day/Week H/L
    round_numbers: list[float] = field(default_factory=list)  # Yakın round number seviyeleri
    cot_bias: str = "NEUTRAL"                                 # COT raporu bias
    sentiment_score: float = 0.0                              # Haber sentiment (-1..+1)
    ml_probability: float = 0.55                              # ML filtre olasılığı
    unicorn_model: dict = field(default_factory=dict)
    vwap: float = 0.0
    silver_bullet: dict = field(default_factory=dict)
    ipda_levels: dict = field(default_factory=dict)
    amd_phase: dict = field(default_factory=dict)


class AnalysisEngine:

    # ─────────────────────────── Temel indikatörler ────────────────────────

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        # Wilder's EMA (ewm with alpha=1/period)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _stoch_rsi(series: pd.Series, rsi_period: int = 14, stoch_period: int = 14) -> pd.Series:
        rsi = AnalysisEngine._rsi(series, rsi_period)
        rsi_min = rsi.rolling(stoch_period).min()
        rsi_max = rsi.rolling(stoch_period).max()
        denom = (rsi_max - rsi_min).replace(0, np.nan)
        return ((rsi - rsi_min) / denom) * 100

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        # Wilder's smoothing (ewm with alpha=1/period)
        return true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        tr = pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        # Wilder's smoothing for all components
        _alpha = 1.0 / period
        atr_ = tr.ewm(alpha=_alpha, min_periods=period, adjust=False).mean()
        plus_di  = 100 * (plus_dm.ewm(alpha=_alpha, min_periods=period, adjust=False).mean() / atr_.replace(0, np.nan))
        minus_di = 100 * (minus_dm.ewm(alpha=_alpha, min_periods=period, adjust=False).mean() / atr_.replace(0, np.nan))
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        return dx.ewm(alpha=_alpha, min_periods=period, adjust=False).mean()

    @staticmethod
    def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
        fast_ema = series.ewm(span=fast, adjust=False).mean()
        slow_ema = series.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        return macd_line, signal_line, macd_line - signal_line

    @staticmethod
    def _bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
        mid = series.rolling(period).mean()
        std_dev = series.rolling(period).std()
        return mid + num_std * std_dev, mid, mid - num_std * std_dev

    # ─────────────────────────── Mum yardımcıları ──────────────────────────

    @staticmethod
    def _body_size(candle: pd.Series) -> float:
        return abs(float(candle["close"]) - float(candle["open"]))

    @staticmethod
    def _upper_wick(candle: pd.Series) -> float:
        return float(candle["high"]) - max(float(candle["open"]), float(candle["close"]))

    @staticmethod
    def _lower_wick(candle: pd.Series) -> float:
        return min(float(candle["open"]), float(candle["close"])) - float(candle["low"])

    @staticmethod
    def _close_location(candle: pd.Series) -> float:
        rng = max(float(candle["high"]) - float(candle["low"]), 1e-9)
        return (float(candle["close"]) - float(candle["low"])) / rng

    @staticmethod
    def _is_bullish(candle: pd.Series) -> bool:
        return float(candle["close"]) > float(candle["open"])

    @staticmethod
    def _bos(df: pd.DataFrame) -> bool:
        """Swing-based Break of Structure: son mum önceki swing high/low'u kırdı mı."""
        if len(df) < 7:
            return False
        window = df.tail(20).reset_index(drop=True)
        swing_highs: list[float] = []
        swing_lows: list[float] = []
        for i in range(2, len(window) - 2):
            h = float(window.iloc[i]["high"])
            l = float(window.iloc[i]["low"])
            if (h >= float(window.iloc[i - 1]["high"]) and h >= float(window.iloc[i + 1]["high"])
                    and h >= float(window.iloc[i - 2]["high"]) and h >= float(window.iloc[i + 2]["high"])):
                swing_highs.append(h)
            if (l <= float(window.iloc[i - 1]["low"]) and l <= float(window.iloc[i + 1]["low"])
                    and l <= float(window.iloc[i - 2]["low"]) and l <= float(window.iloc[i + 2]["low"])):
                swing_lows.append(l)
        if not swing_highs and not swing_lows:
            return False
        last = window.iloc[-1]
        last_close = float(last["close"])
        if swing_highs and last_close > swing_highs[-1]:
            return True
        if swing_lows and last_close < swing_lows[-1]:
            return True
        return False

    @staticmethod
    def _cluster_levels(values: list[float], tolerance_ratio: float = 0.0025) -> list[float]:
        if not values:
            return []
        sorted_vals = sorted(values)
        clusters: list[list[float]] = [[sorted_vals[0]]]
        for value in sorted_vals[1:]:
            anchor = float(np.mean(clusters[-1]))
            if abs(value - anchor) / anchor <= tolerance_ratio:
                clusters[-1].append(value)
            else:
                clusters.append([value])
        return [round(float(np.mean(c)), 5) for c in clusters]

    @staticmethod
    def _find_levels(df: pd.DataFrame, lookback: int = 120) -> tuple[list[float], list[float]]:
        window = df.tail(lookback).reset_index(drop=True)
        supports, resistances = [], []
        # 8-bar pivot (her iki yanda 8 bar = anlamlı seviye)
        pivot_range = 8
        for i in range(pivot_range, len(window) - pivot_range):
            low = float(window.loc[i, "low"])
            if all(low <= float(window.loc[i + d, "low"]) for d in range(-pivot_range, pivot_range + 1) if d != 0):
                supports.append(low)
            high = float(window.loc[i, "high"])
            if all(high >= float(window.loc[i + d, "high"]) for d in range(-pivot_range, pivot_range + 1) if d != 0):
                resistances.append(high)
        return AnalysisEngine._cluster_levels(supports), AnalysisEngine._cluster_levels(resistances)

    @staticmethod
    def _trend_from_df(df: pd.DataFrame) -> str:
        data = df.copy()
        data["ema_50"] = data["close"].ewm(span=50, adjust=False).mean()
        data["ema_200"] = data["close"].ewm(span=200, adjust=False).mean()
        last = data.iloc[-1]
        return "Yukari" if float(last["ema_50"]) > float(last["ema_200"]) else "Asagi"

    @staticmethod
    def _quality_from_score(score: int) -> str:
        if score >= 85: return "A"
        if score >= 70: return "B"
        if score >= 55: return "C"
        return "D"

    # ─────────────────────────── SMC / ICT Tespitleri ──────────────────────

    @staticmethod
    def _detect_order_blocks(df: pd.DataFrame, atr: float, current_price: float) -> list[dict]:
        """Order Block: güçlü impuls öncesi son ters yönlü mum."""
        blocks: list[dict] = []
        if len(df) < 5:
            return blocks
        tail = df.tail(80).reset_index(drop=True)
        for i in range(1, len(tail) - 2):
            candle = tail.iloc[i]
            nxt    = tail.iloc[i + 1]
            body      = abs(float(candle["close"]) - float(candle["open"]))
            next_body = abs(float(nxt["close"]) - float(nxt["open"]))
            if body < 1e-9 or next_body < body * 1.4:
                continue

            future = tail.iloc[i + 2:]

            if float(candle["close"]) < float(candle["open"]) and float(nxt["close"]) > float(nxt["open"]):
                top, bot = float(candle["open"]), float(candle["close"])
                broken = any(float(r["low"]) < bot for _, r in future.iterrows())
                near = abs(current_price - (top + bot) / 2) <= atr * 2
                blocks.append({"type": "bullish_ob", "top": round(top, 5), "bottom": round(bot, 5),
                                "broken": broken, "near_price": near, "age_bars": len(tail) - i})

            elif float(candle["close"]) > float(candle["open"]) and float(nxt["close"]) < float(nxt["open"]):
                top, bot = float(candle["close"]), float(candle["open"])
                broken = any(float(r["high"]) > top for _, r in future.iterrows())
                near = abs(current_price - (top + bot) / 2) <= atr * 2
                blocks.append({"type": "bearish_ob", "top": round(top, 5), "bottom": round(bot, 5),
                                "broken": broken, "near_price": near, "age_bars": len(tail) - i})

            if len(blocks) >= 6:
                break
        return blocks

    @staticmethod
    def _detect_fvg(df: pd.DataFrame, atr: float, current_price: float) -> list[dict]:
        """FVG: 3 mumlu yapıda oluşan doldurulamayan boşluk."""
        gaps: list[dict] = []
        if len(df) < 5:
            return gaps
        tail = df.tail(60).reset_index(drop=True)
        n = len(tail)
        for i in range(1, n - 1):
            prev, nxt = tail.iloc[i - 1], tail.iloc[i + 1]

            if float(nxt["low"]) > float(prev["high"]):
                top, bot = float(nxt["low"]), float(prev["high"])
                size_pct = (top - bot) / max(current_price, 1e-9) * 100
                future = tail.iloc[i + 2:]
                filled = any(float(r["low"]) <= top and float(r["high"]) >= bot for _, r in future.iterrows())
                near = abs(current_price - (top + bot) / 2) <= atr * 1.5
                gaps.append({"type": "bullish_fvg", "top": round(top, 5), "bottom": round(bot, 5),
                             "size_pct": round(size_pct, 4), "filled": filled,
                             "near_price": near, "age_bars": n - i})

            elif float(nxt["high"]) < float(prev["low"]):
                top, bot = float(prev["low"]), float(nxt["high"])
                size_pct = (top - bot) / max(current_price, 1e-9) * 100
                future = tail.iloc[i + 2:]
                filled = any(float(r["high"]) >= bot and float(r["low"]) <= top for _, r in future.iterrows())
                near = abs(current_price - (top + bot) / 2) <= atr * 1.5
                gaps.append({"type": "bearish_fvg", "top": round(top, 5), "bottom": round(bot, 5),
                             "size_pct": round(size_pct, 4), "filled": filled,
                             "near_price": near, "age_bars": n - i})

            if len(gaps) >= 6:
                break
        return gaps

    @staticmethod
    def _detect_ifvg(fvg_zones: list[dict], current_price: float, atr: float) -> list[dict]:
        """Inversion FVG: doldurulan FVG'nin ters yönde S/R olması."""
        ifvgs: list[dict] = []
        for fvg in fvg_zones:
            if not fvg.get("filled"):
                continue
            mid = (fvg["top"] + fvg["bottom"]) / 2
            near = abs(current_price - mid) <= atr * 2
            ifvgs.append({
                "type": "bearish_ifvg" if fvg["type"] == "bullish_fvg" else "bullish_ifvg",
                "top": fvg["top"], "bottom": fvg["bottom"],
                "near_price": near,
            })
        return ifvgs[:3]

    @staticmethod
    def _detect_choch(df: pd.DataFrame, trend: str) -> dict:
        """Change of Character: mevcut trendin kırılma sinyali. Dict döner."""
        default = {"detected": False, "type": "", "price": 0.0, "description": "Yok"}
        if len(df) < 12:
            return default
        window = df.tail(30).reset_index(drop=True)
        highs: list[tuple[int, float]] = []
        lows:  list[tuple[int, float]] = []
        for i in range(2, len(window) - 2):
            h = float(window.iloc[i]["high"])
            l = float(window.iloc[i]["low"])
            if (h >= float(window.iloc[i-1]["high"]) and h >= float(window.iloc[i+1]["high"])
                    and h >= float(window.iloc[i-2]["high"]) and h >= float(window.iloc[i+2]["high"])):
                highs.append((i, h))
            if (l <= float(window.iloc[i-1]["low"]) and l <= float(window.iloc[i+1]["low"])
                    and l <= float(window.iloc[i-2]["low"]) and l <= float(window.iloc[i+2]["low"])):
                lows.append((i, l))

        if trend == "Yukari" and len(highs) >= 2 and len(lows) >= 2:
            if highs[-1][1] < highs[-2][1] and lows[-1][1] < lows[-2][1]:
                return {"detected": True, "type": "bearish_choch",
                        "price": round(lows[-1][1], 5),
                        "description": "Yukari trend kırılıyor — Düşüş dönüşü sinyali"}
        elif trend == "Asagi" and len(highs) >= 2 and len(lows) >= 2:
            if lows[-1][1] > lows[-2][1] and highs[-1][1] > highs[-2][1]:
                return {"detected": True, "type": "bullish_choch",
                        "price": round(highs[-1][1], 5),
                        "description": "Asagi trend kırılıyor — Yükseliş dönüşü sinyali"}
        return default

    @staticmethod
    def _detect_premium_discount(df: pd.DataFrame, current_price: float) -> dict:
        """Premium/Discount bölgesi. Dict döner."""
        default = {"zone": "DENGE", "range_low": 0.0, "range_high": 0.0, "pct": 0.5}
        if len(df) < 20:
            return default
        window = df.tail(50)
        swing_high = float(window["high"].max())
        swing_low  = float(window["low"].min())
        rng = swing_high - swing_low
        if rng < 1e-9:
            return default
        pct = (current_price - swing_low) / rng
        zone = "PREMIUM" if pct >= 0.618 else ("DISCOUNT" if pct <= 0.382 else "DENGE")
        return {"zone": zone, "range_low": round(swing_low, 5),
                "range_high": round(swing_high, 5), "pct": round(pct, 4)}

    @staticmethod
    def _detect_equal_levels(df: pd.DataFrame, atr: float) -> tuple[list[float], list[float]]:
        """Equal Highs/Lows: likidite havuzları."""
        if len(df) < 20:
            return [], []
        window = df.tail(60).reset_index(drop=True)
        tolerance = atr * 0.25
        swing_highs, swing_lows = [], []
        for i in range(2, len(window) - 2):
            h = float(window.iloc[i]["high"])
            l = float(window.iloc[i]["low"])
            if h >= float(window.iloc[i-1]["high"]) and h >= float(window.iloc[i+1]["high"]):
                swing_highs.append(h)
            if l <= float(window.iloc[i-1]["low"]) and l <= float(window.iloc[i+1]["low"]):
                swing_lows.append(l)

        eq_highs, eq_lows = [], []
        for i in range(len(swing_highs)):
            for j in range(i + 1, len(swing_highs)):
                if abs(swing_highs[i] - swing_highs[j]) <= tolerance:
                    lvl = round((swing_highs[i] + swing_highs[j]) / 2, 5)
                    if lvl not in eq_highs:
                        eq_highs.append(lvl)
        for i in range(len(swing_lows)):
            for j in range(i + 1, len(swing_lows)):
                if abs(swing_lows[i] - swing_lows[j]) <= tolerance:
                    lvl = round((swing_lows[i] + swing_lows[j]) / 2, 5)
                    if lvl not in eq_lows:
                        eq_lows.append(lvl)
        return eq_highs[:3], eq_lows[:3]

    @staticmethod
    def _detect_displacement(df: pd.DataFrame, atr: float) -> dict:
        """Displacement: güçlü impulsif hareket. Dict döner."""
        default = {"detected": False, "direction": "", "strength": ""}
        if len(df) < 4:
            return default
        last3 = df.tail(3)
        closes = [float(r["close"]) for _, r in last3.iterrows()]
        opens  = [float(r["open"])  for _, r in last3.iterrows()]
        bodies = [abs(c - o) for c, o in zip(closes, opens)]
        avg_body = sum(bodies) / len(bodies)

        all_bull = all(c > o for c, o in zip(closes, opens))
        all_bear = all(c < o for c, o in zip(closes, opens))

        if avg_body >= atr * 0.55:
            if all_bull:
                return {"detected": True, "direction": "bullish", "strength": "Güçlü 3-mum Yukari Displacement"}
            if all_bear:
                return {"detected": True, "direction": "bearish", "strength": "Güçlü 3-mum Asagi Displacement"}

        last = df.iloc[-1]
        last_body = abs(float(last["close"]) - float(last["open"]))
        if last_body >= atr * 1.1:
            direction = "bullish" if float(last["close"]) > float(last["open"]) else "bearish"
            return {"detected": True, "direction": direction, "strength": "Tek Mum Displacement"}
        return default

    @staticmethod
    def _detect_ote(df: pd.DataFrame, trend: str) -> dict:
        """OTE: ICT Fibonacci %61.8–78.6 geri çekilme bölgesi. Dict döner."""
        default = {"valid": False, "ote_low": 0.0, "ote_high": 0.0}
        if len(df) < 20:
            return default
        window = df.tail(40)
        swing_high = float(window["high"].max())
        swing_low  = float(window["low"].min())
        rng = swing_high - swing_low
        if rng < 1e-9:
            return default
        if trend == "Yukari":
            ote_low  = round(swing_high - rng * 0.786, 5)
            ote_high = round(swing_high - rng * 0.618, 5)
        else:
            ote_low  = round(swing_low + rng * 0.618, 5)
            ote_high = round(swing_low + rng * 0.786, 5)
        return {"valid": True, "ote_low": ote_low, "ote_high": ote_high}

    @staticmethod
    def _detect_breaker_blocks(df: pd.DataFrame, order_blocks: list[dict],
                                current_price: float, atr: float) -> list[dict]:
        """Breaker Block: kırılan OB'nin ters yönde S/R olması."""
        breakers: list[dict] = []
        for ob in order_blocks:
            if not ob.get("broken"):
                continue
            mid = (ob["top"] + ob["bottom"]) / 2
            near = abs(current_price - mid) <= atr * 2
            if ob["type"] == "bullish_ob":
                breakers.append({"type": "bearish_breaker", "top": ob["top"], "bottom": ob["bottom"],
                                  "near_price": near,
                                  "description": "Kırılan Bullish OB → Direnç (Bearish Breaker)"})
            else:
                breakers.append({"type": "bullish_breaker", "top": ob["top"], "bottom": ob["bottom"],
                                  "near_price": near,
                                  "description": "Kırılan Bearish OB → Destek (Bullish Breaker)"})
        return breakers[:3]

    @staticmethod
    def _detect_confirmation_candle(df: pd.DataFrame, signal: str, atr: float) -> dict:
        """
        Onay mumu tespiti: engulfing, pin bar, hammer, shooting star.
        Signal yönüyle uyumlu bir onay mumu var mı kontrol eder.
        """
        default = {"detected": False, "type": "", "direction": "", "strength": 0}
        if len(df) < 3:
            return default
        last = df.iloc[-1]
        prev = df.iloc[-2]
        body      = abs(float(last["close"]) - float(last["open"]))
        prev_body = abs(float(prev["close"]) - float(prev["open"]))
        wick_up   = AnalysisEngine._upper_wick(last)
        wick_dn   = AnalysisEngine._lower_wick(last)
        cl        = AnalysisEngine._close_location(last)
        total_range = float(last["high"]) - float(last["low"])

        # Bullish Engulfing
        if (signal == "LONG"
                and float(last["close"]) > float(last["open"])
                and float(last["close"]) > float(prev["open"])
                and float(last["open"]) < float(prev["close"])
                and body > prev_body * 1.1):
            return {"detected": True, "type": "Bullish Engulfing",
                    "direction": "bullish", "strength": 90}

        # Bearish Engulfing
        if (signal == "SHORT"
                and float(last["close"]) < float(last["open"])
                and float(last["close"]) < float(prev["open"])
                and float(last["open"]) > float(prev["close"])
                and body > prev_body * 1.1):
            return {"detected": True, "type": "Bearish Engulfing",
                    "direction": "bearish", "strength": 90}

        # Hammer (Bullish)
        if (signal == "LONG"
                and wick_dn >= body * 2.0
                and wick_up <= body * 0.5
                and total_range > atr * 0.3):
            return {"detected": True, "type": "Hammer",
                    "direction": "bullish", "strength": 80}

        # Shooting Star (Bearish)
        if (signal == "SHORT"
                and wick_up >= body * 2.0
                and wick_dn <= body * 0.5
                and total_range > atr * 0.3):
            return {"detected": True, "type": "Shooting Star",
                    "direction": "bearish", "strength": 80}

        # Bullish Pin Bar
        if (signal == "LONG"
                and wick_dn >= total_range * 0.6
                and cl >= 0.6):
            return {"detected": True, "type": "Bullish Pin Bar",
                    "direction": "bullish", "strength": 75}

        # Bearish Pin Bar
        if (signal == "SHORT"
                and wick_up >= total_range * 0.6
                and cl <= 0.4):
            return {"detected": True, "type": "Bearish Pin Bar",
                    "direction": "bearish", "strength": 75}

        # Doji sonrası yön (belirsizlik → karar)
        if body <= total_range * 0.1 and total_range > atr * 0.25:
            nxt_dir = ""
            if len(df) >= 2:
                prev2 = df.iloc[-2]
                nxt_dir = "bullish" if float(last["close"]) > float(prev2["close"]) else "bearish"
            if ((signal == "LONG" and nxt_dir == "bullish")
                    or (signal == "SHORT" and nxt_dir == "bearish")):
                return {"detected": True, "type": "Doji Dönüşü",
                        "direction": nxt_dir, "strength": 60}

        return default

    @staticmethod
    def _detect_bos_mss(df: pd.DataFrame, trend: str, atr: float) -> dict:
        """
        Break of Structure (BOS) ve Market Structure Shift (MSS) tespiti.
        BOS: trendde devam, MSS: trend kırılması (CHoCH öncesi).
        """
        default = {"bos": False, "mss": False, "type": "", "level": 0.0}
        if len(df) < 10:
            return default
        window = df.tail(20).reset_index(drop=True)
        swing_highs = []
        swing_lows  = []
        for i in range(1, len(window) - 1):
            h = float(window.iloc[i]["high"])
            l = float(window.iloc[i]["low"])
            if h >= float(window.iloc[i-1]["high"]) and h >= float(window.iloc[i+1]["high"]):
                swing_highs.append((i, h))
            if l <= float(window.iloc[i-1]["low"]) and l <= float(window.iloc[i+1]["low"]):
                swing_lows.append((i, l))

        last_close = float(window.iloc[-1]["close"])

        # Bullish BOS: yeni yüksek yapıldı, trend devam
        if trend == "Yukari" and len(swing_highs) >= 2:
            if swing_highs[-1][1] > swing_highs[-2][1] and last_close > swing_highs[-2][1]:
                return {"bos": True, "mss": False, "type": "bullish_bos",
                        "level": round(swing_highs[-2][1], 5)}

        # Bearish BOS: yeni düşük yapıldı, trend devam
        if trend == "Asagi" and len(swing_lows) >= 2:
            if swing_lows[-1][1] < swing_lows[-2][1] and last_close < swing_lows[-2][1]:
                return {"bos": True, "mss": False, "type": "bearish_bos",
                        "level": round(swing_lows[-2][1], 5)}

        # Bullish MSS: aşağı trendde yukarı kırılım (trend değişimi)
        if trend == "Asagi" and len(swing_highs) >= 1:
            if last_close > swing_highs[-1][1]:
                return {"bos": False, "mss": True, "type": "bullish_mss",
                        "level": round(swing_highs[-1][1], 5)}

        # Bearish MSS: yukarı trendde aşağı kırılım (trend değişimi)
        if trend == "Yukari" and len(swing_lows) >= 1:
            if last_close < swing_lows[-1][1]:
                return {"bos": False, "mss": True, "type": "bearish_mss",
                        "level": round(swing_lows[-1][1], 5)}

        return default

    @staticmethod
    def _detect_judas_swing(df: pd.DataFrame, trend: str, atr: float) -> dict:
        """
        Judas Swing: seans açılışında önce yanlış yöne hareket,
        sonra gerçek yönde güçlü dönüş. ICT'nin temel kavramı.
        """
        default = {"detected": False, "direction": "", "sweep_level": 0.0}
        if len(df) < 10:
            return default
        # Son 8 mumda: ilk 2-3 mumda yanlış yön, sonra sert dönüş
        window = df.tail(8).reset_index(drop=True)
        early_high = float(window.iloc[:3]["high"].max())
        early_low  = float(window.iloc[:3]["low"].min())
        late_close = float(window.iloc[-1]["close"])
        late_low   = float(window.iloc[-2:]["low"].min())
        late_high  = float(window.iloc[-2:]["high"].max())

        # Bearish Judas: erken mumlar yukarı sweep (fake move), sonra aşağı dönüş
        # Gerçek trend aşağı, erken mumlar yukarı aldatma yapıyor
        if (trend == "Asagi"
                and early_high > float(window.iloc[3:]["high"].max())
                and late_close < early_low + atr * 0.3):
            return {"detected": True, "direction": "bearish",
                    "sweep_level": round(early_high, 5)}

        # Bullish Judas: erken mumlar aşağı sweep (fake move), sonra yukarı dönüş
        # Gerçek trend yukarı, erken mumlar aşağı aldatma yapıyor
        if (trend == "Yukari"
                and early_low < float(window.iloc[3:]["low"].min())
                and late_close > early_high - atr * 0.3):
            return {"detected": True, "direction": "bullish",
                    "sweep_level": round(early_low, 5)}

        return default

    def _detect_liquidity_sweep(self, df: pd.DataFrame, supports: list[float],
                                 resistances: list[float]) -> str:
        if len(df) < 3:
            return "Yok"
        last, prev = df.iloc[-1], df.iloc[-2]
        if resistances:
            nearest_res = min(resistances, key=lambda x: abs(x - float(last["close"])))
            if (float(last["high"]) > nearest_res and float(last["close"]) < nearest_res
                    and float(last["close"]) < float(last["open"])
                    and self._upper_wick(last) > self._body_size(last) * 1.2):
                return "Direnc ustu likidite alinip geri donuldu"
        if supports:
            nearest_sup = min(supports, key=lambda x: abs(x - float(last["close"])))
            if (float(last["low"]) < nearest_sup and float(last["close"]) > nearest_sup
                    and float(last["close"]) > float(last["open"])
                    and self._lower_wick(last) > self._body_size(last) * 1.2):
                return "Destek alti likidite alinip geri donuldu"
        if float(last["high"]) > float(prev["high"]) and float(last["close"]) < float(prev["high"]):
            return "Yukari fake breakout ihtimali"
        if float(last["low"]) < float(prev["low"]) and float(last["close"]) > float(prev["low"]):
            return "Asagi fake breakout ihtimali"
        return "Yok"

    def _detect_sniper_entry(self, df: pd.DataFrame, trend: str, higher_tf_trend: str,
                              supports: list[float], resistances: list[float]) -> str:
        if len(df) < 3:
            return "Yok"
        last, prev = df.iloc[-1], df.iloc[-2]
        bos = self._bos(df)

        if trend == "Asagi" and higher_tf_trend == "Asagi" and resistances:
            nearest_res = min(resistances, key=lambda x: abs(x - float(last["close"])))
            if (float(last["high"]) >= nearest_res and float(last["close"]) < nearest_res
                    and float(last["close"]) < float(last["open"])
                    and (self._upper_wick(last) > self._body_size(last) * 1.5
                         or (float(last["high"]) > float(prev["high"])
                             and float(last["close"]) < float(prev["high"])))
                    and bos and self._close_location(last) <= 0.45):
                return "SHORT sniper: direnc sweep + rejection"

        if trend == "Yukari" and higher_tf_trend == "Yukari" and supports:
            nearest_sup = min(supports, key=lambda x: abs(x - float(last["close"])))
            if (float(last["low"]) <= nearest_sup and float(last["close"]) > nearest_sup
                    and float(last["close"]) > float(last["open"])
                    and (self._lower_wick(last) > self._body_size(last) * 1.5
                         or (float(last["low"]) < float(prev["low"])
                             and float(last["close"]) > float(prev["low"])))
                    and bos and self._close_location(last) >= 0.55):
                return "LONG sniper: destek sweep + rejection"
        return "Yok"

    @staticmethod
    def _count_smc_confluence(
        signal: str,
        choch: dict,
        displacement: dict,
        ote_zone: dict,
        current_price: float,
        premium_discount: dict,
        order_blocks: list[dict],
        fvg_zones: list[dict],
        breaker_blocks: list[dict],
        ifvg_zones: list[dict],
        eq_highs: list[float],
        eq_lows: list[float],
        atr: float,
        bos_mss: dict,
        confirmation_candle: dict,
        sweep_signal: str,
        judas_swing: dict,
    ) -> int:
        """Kaç SMC sinyali uyumlu? Minimum 2 gerekli."""
        count = 0
        if signal == "NO TRADE":
            return 0

        # CHoCH uyumu
        if choch.get("detected"):
            if signal == "LONG" and choch.get("type") == "bullish_choch": count += 1
            if signal == "SHORT" and choch.get("type") == "bearish_choch": count += 1

        # Displacement uyumu
        if displacement.get("detected"):
            if signal == "LONG"  and displacement.get("direction") == "bullish": count += 1
            if signal == "SHORT" and displacement.get("direction") == "bearish": count += 1

        # OTE bölgesinde
        if ote_zone.get("valid"):
            if ote_zone["ote_low"] <= current_price <= ote_zone["ote_high"]: count += 1

        # Premium / Discount uyumu
        pd_zone = premium_discount.get("zone", "DENGE")
        if signal == "LONG"  and pd_zone == "DISCOUNT": count += 1
        if signal == "SHORT" and pd_zone == "PREMIUM":  count += 1

        # Kırılmamış OB üzerinde
        fresh_bull = [ob for ob in order_blocks if ob["type"] == "bullish_ob" and not ob.get("broken") and ob.get("near_price")]
        fresh_bear = [ob for ob in order_blocks if ob["type"] == "bearish_ob" and not ob.get("broken") and ob.get("near_price")]
        if signal == "LONG"  and fresh_bull: count += 1
        if signal == "SHORT" and fresh_bear: count += 1

        # Unfilled FVG yakını
        near_bull_fvg = [g for g in fvg_zones if g["type"] == "bullish_fvg" and g.get("near_price") and not g.get("filled")]
        near_bear_fvg = [g for g in fvg_zones if g["type"] == "bearish_fvg" and g.get("near_price") and not g.get("filled")]
        if signal == "LONG"  and near_bull_fvg: count += 1
        if signal == "SHORT" and near_bear_fvg: count += 1

        # Breaker Block
        bull_bb = [b for b in breaker_blocks if b["type"] == "bullish_breaker" and b.get("near_price")]
        bear_bb = [b for b in breaker_blocks if b["type"] == "bearish_breaker" and b.get("near_price")]
        if signal == "LONG"  and bull_bb: count += 1
        if signal == "SHORT" and bear_bb: count += 1

        # iFVG
        near_bull_ifvg = [g for g in ifvg_zones if g["type"] == "bullish_ifvg" and g.get("near_price")]
        near_bear_ifvg = [g for g in ifvg_zones if g["type"] == "bearish_ifvg" and g.get("near_price")]
        if signal == "LONG"  and near_bull_ifvg: count += 1
        if signal == "SHORT" and near_bear_ifvg: count += 1

        # Equal Highs/Lows likidite sweep
        if signal == "LONG"  and any(abs(current_price - lvl) <= atr * 0.5 for lvl in eq_lows):  count += 1
        if signal == "SHORT" and any(abs(current_price - lvl) <= atr * 0.5 for lvl in eq_highs): count += 1

        # BOS/MSS uyumu
        if bos_mss.get("bos"):
            if signal == "LONG"  and bos_mss.get("type") == "bullish_bos": count += 1
            if signal == "SHORT" and bos_mss.get("type") == "bearish_bos": count += 1
        if bos_mss.get("mss"):
            if signal == "LONG"  and bos_mss.get("type") == "bullish_mss": count += 1
            if signal == "SHORT" and bos_mss.get("type") == "bearish_mss": count += 1

        # Onay mumu
        if confirmation_candle.get("detected"):
            if signal == "LONG"  and confirmation_candle.get("direction") == "bullish": count += 1
            if signal == "SHORT" and confirmation_candle.get("direction") == "bearish": count += 1

        # Likidite sweep
        if sweep_signal != "Yok": count += 1

        # Judas Swing
        if judas_swing.get("detected"):
            if signal == "LONG"  and judas_swing.get("direction") == "bullish": count += 1
            if signal == "SHORT" and judas_swing.get("direction") == "bearish": count += 1

        return count

    # ─────────────────────────── Tier-1/2/3 Yeni Tespitler ────────────────────

    @staticmethod
    def _detect_volume_analysis(df: pd.DataFrame, atr: float) -> dict:
        """
        Tick volume tabanlı delta analizi.
        Volume sütunu varsa kullanır, yoksa gövde büyüklüğünden proxy türetir.
        """
        default = {
            "volume_trend": "NEUTRAL",
            "last_volume_ratio": 1.0,
            "volume_spike": False,
            "delta_bias": "NEUTRAL",
            "high_volume_bars": 0,
        }
        if len(df) < 10:
            return default

        # Volume sütunu kontrolü
        has_volume = "volume" in df.columns and df["volume"].sum() > 0

        window = df.tail(40).copy().reset_index(drop=True)

        if has_volume:
            vols = window["volume"].astype(float)
        else:
            # Proxy: mum gövde büyüklüğü × fiyat aralığı
            vols = (abs(window["close"] - window["open"]) * (window["high"] - window["low"])).astype(float)

        if vols.mean() < 1e-9:
            return default

        avg_vol = float(vols.mean())
        last_vol = float(vols.iloc[-1])
        last_ratio = round(last_vol / avg_vol, 2)

        # Spike: son mum ortalamanın 1.8 katından fazla
        volume_spike = last_ratio >= 1.8

        # Son 5 mumda yüksek hacim sayısı (>1.3x avg)
        high_vol_bars = int(sum(1 for v in vols.tail(5) if float(v) > avg_vol * 1.3))

        # Delta bias: yükselen mumların hacmi vs düşen mumların hacmi
        bull_vol = float(vols[window["close"] >= window["open"]].sum())
        bear_vol = float(vols[window["close"] < window["open"]].sum())
        total_vol = bull_vol + bear_vol
        if total_vol > 0:
            bull_ratio = bull_vol / total_vol
            if bull_ratio >= 0.60:
                delta_bias = "BULLISH"
            elif bull_ratio <= 0.40:
                delta_bias = "BEARISH"
            else:
                delta_bias = "NEUTRAL"
        else:
            delta_bias = "NEUTRAL"

        # Volume trendi: son 10 mum ortalaması vs önceki 10 mum
        if len(vols) >= 20:
            recent_avg = float(vols.tail(10).mean())
            prior_avg  = float(vols.iloc[-20:-10].mean())
            if prior_avg > 0:
                if recent_avg > prior_avg * 1.15:
                    vol_trend = "INCREASING"
                elif recent_avg < prior_avg * 0.85:
                    vol_trend = "DECREASING"
                else:
                    vol_trend = "NEUTRAL"
            else:
                vol_trend = "NEUTRAL"
        else:
            vol_trend = "NEUTRAL"

        return {
            "volume_trend": vol_trend,
            "last_volume_ratio": last_ratio,
            "volume_spike": volume_spike,
            "delta_bias": delta_bias,
            "high_volume_bars": high_vol_bars,
        }

    @staticmethod
    def _detect_pdh_pdl(df: pd.DataFrame) -> dict:
        """
        Previous Day High/Low ve Previous Week High/Low.
        'datetime' sütunu varsa kullanır, yoksa bar sayısına göre proxy.
        """
        default = {
            "prev_day_high": 0.0,
            "prev_day_low": 0.0,
            "prev_week_high": 0.0,
            "prev_week_low": 0.0,
            "today_high": 0.0,
            "today_low": 0.0,
        }
        if len(df) < 10:
            return default

        try:
            if "datetime" in df.columns:
                df2 = df.copy()
                df2["datetime"] = pd.to_datetime(df2["datetime"], utc=True, errors="coerce")
                df2 = df2.dropna(subset=["datetime"])
                if df2.empty:
                    raise ValueError("empty after parse")

                df2["date"] = df2["datetime"].dt.date
                today = df2["date"].iloc[-1]

                today_df    = df2[df2["date"] == today]
                prev_days   = df2[df2["date"] < today]

                if prev_days.empty:
                    raise ValueError("no prev days")

                prev_day    = prev_days["date"].max()
                prev_day_df = prev_days[prev_days["date"] == prev_day]

                # Geçen hafta (yıl geçişini hesaba kat)
                _iso = df2["datetime"].dt.isocalendar()
                df2["iso_year"] = _iso.year.astype(int)
                df2["iso_week"] = _iso.week.astype(int)
                this_year = int(df2["iso_year"].iloc[-1])
                this_week = int(df2["iso_week"].iloc[-1])
                prev_week_df = df2[
                    (df2["iso_year"] < this_year) |
                    ((df2["iso_year"] == this_year) & (df2["iso_week"] < this_week))
                ]

                return {
                    "prev_day_high": round(float(prev_day_df["high"].max()), 5),
                    "prev_day_low":  round(float(prev_day_df["low"].min()),  5),
                    "prev_week_high": round(float(prev_week_df["high"].max()), 5) if not prev_week_df.empty else 0.0,
                    "prev_week_low":  round(float(prev_week_df["low"].min()),  5) if not prev_week_df.empty else 0.0,
                    "today_high": round(float(today_df["high"].max()), 5),
                    "today_low":  round(float(today_df["low"].min()),  5),
                }
            else:
                raise ValueError("no datetime column")
        except Exception:
            # Proxy: son 24 bar = "dün", önceki 24 bar = "önceki gün"
            # (5 dakika TF için 24 bar ≈ 2 saat, 1 saatlik TF için 24 bar = 1 gün)
            tail80 = df.tail(80).reset_index(drop=True)
            n = len(tail80)
            today_df    = tail80.tail(min(24, n // 2))
            prev_day_df = tail80.iloc[max(0, n // 2 - 24) : n // 2]

            return {
                "prev_day_high": round(float(prev_day_df["high"].max()), 5),
                "prev_day_low":  round(float(prev_day_df["low"].min()),  5),
                "prev_week_high": round(float(tail80["high"].max()), 5),
                "prev_week_low":  round(float(tail80["low"].min()),  5),
                "today_high": round(float(today_df["high"].max()), 5),
                "today_low":  round(float(today_df["low"].min()),  5),
            }

    @staticmethod
    def _detect_round_numbers(current_price: float, atr: float, symbol: str = "") -> list[float]:
        """
        Fiyata yakın round number seviyeleri (stop cluster noktaları).
        BTC: 1000/5000/10000 USD. Altın: 50/100 USD. Forex: pip bazlı.
        """
        levels: list[float] = []
        search_range = atr * 5
        sym = symbol.upper()

        if sym == "BTCUSD" or current_price > 10000:
            # BTC: 1000, 5000, 10000 USD round numbers
            for step in [1000.0, 2500.0, 5000.0, 10000.0]:
                base = round(current_price / step) * step
                for mult in range(-3, 4):
                    lvl = round(base + mult * step, 0)
                    if abs(lvl - current_price) <= search_range and lvl not in levels:
                        levels.append(lvl)
        elif current_price > 500:
            # XAUUSD: 50, 100, 500 USD round numbers
            for step in [50.0, 100.0, 500.0, 1000.0]:
                base = round(current_price / step) * step
                for mult in range(-3, 4):
                    lvl = round(base + mult * step, 2)
                    if abs(lvl - current_price) <= search_range and lvl not in levels:
                        levels.append(lvl)
        else:
            # Forex: pip bazlı
            for step in [0.001, 0.005, 0.01, 0.05, 0.10]:
                base = round(current_price / step) * step
                for mult in range(-3, 4):
                    lvl = round(base + mult * step, 5)
                    if abs(lvl - current_price) <= search_range and lvl not in levels:
                        levels.append(lvl)

        levels.sort(key=lambda x: abs(x - current_price))
        return levels[:6]

    # ─────────────────── ICT Unicorn Model ──────────────────────────────────

    @staticmethod
    def _detect_unicorn_model(breaker_blocks: list[dict], fvg_zones: list[dict],
                               current_price: float, atr: float) -> dict:
        """ICT Unicorn Model: Breaker Block + FVG çakışması — en yüksek olasılıklı setup."""
        default = {"detected": False, "type": "", "zone_top": 0.0, "zone_bottom": 0.0, "near_price": False}
        for bb in breaker_blocks:
            if not bb.get("near_price"):
                continue
            for fvg in fvg_zones:
                if fvg.get("filled"):
                    continue
                # Check overlap between breaker block and FVG
                overlap_top = min(bb["top"], fvg["top"])
                overlap_bot = max(bb["bottom"], fvg["bottom"])
                if overlap_top > overlap_bot:  # There IS overlap
                    near = abs(current_price - (overlap_top + overlap_bot) / 2) <= atr * 2
                    if bb["type"] == "bullish_breaker":
                        return {"detected": True, "type": "bullish_unicorn",
                                "zone_top": round(overlap_top, 5), "zone_bottom": round(overlap_bot, 5),
                                "near_price": near}
                    elif bb["type"] == "bearish_breaker":
                        return {"detected": True, "type": "bearish_unicorn",
                                "zone_top": round(overlap_top, 5), "zone_bottom": round(overlap_bot, 5),
                                "near_price": near}
        return default

    # ─────────────────── VWAP Hesaplama ───────────────────────────────────

    @staticmethod
    def _calculate_vwap(df: pd.DataFrame) -> float:
        """VWAP hesaplama — kurumsal benchmark seviyesi."""
        if len(df) < 10:
            return float(df["close"].iloc[-1]) if len(df) > 0 else 0.0

        # Use available volume or proxy (body size)
        has_volume = "volume" in df.columns and df["volume"].sum() > 0

        # Try to get today's data only
        if "datetime" in df.columns:
            try:
                df2 = df.copy()
                df2["datetime"] = pd.to_datetime(df2["datetime"], errors="coerce")
                today = df2["datetime"].dt.date.iloc[-1]
                today_df = df2[df2["datetime"].dt.date == today]
                if len(today_df) >= 5:
                    typical_price = (today_df["high"] + today_df["low"] + today_df["close"]) / 3
                    if has_volume:
                        vol = today_df["volume"].astype(float)
                    else:
                        vol = (today_df["high"] - today_df["low"]).abs().astype(float)
                    vol_sum = vol.sum()
                    if vol_sum > 0:
                        return round(float((typical_price * vol).sum() / vol_sum), 5)
            except Exception:
                pass

        # Fallback: last 40 bars
        window = df.tail(40)
        typical_price = (window["high"] + window["low"] + window["close"]) / 3
        if has_volume:
            vol = window["volume"].astype(float)
        else:
            vol = (window["high"] - window["low"]).abs().astype(float)
        vol_sum = vol.sum()
        if vol_sum > 0:
            return round(float((typical_price * vol).sum() / vol_sum), 5)
        return round(float(typical_price.iloc[-1]), 5)

    # ─────────────────── ICT Silver Bullet ────────────────────────────────

    @staticmethod
    def _detect_silver_bullet(df: pd.DataFrame, fvg_zones: list[dict], atr: float) -> dict:
        """ICT Silver Bullet: belirli saat pencerelerinde FVG + likidite sweep."""
        default = {"active": False, "window": "", "fvg_count": 0}
        if "datetime" not in df.columns or len(df) < 5:
            return default
        try:
            last_dt = pd.to_datetime(df.iloc[-1]["datetime"])
            hour = last_dt.hour
            minute = last_dt.minute
            t = hour * 60 + minute

            # Silver Bullet windows (UTC):
            # London: 10:00-11:00 (600-660)
            # NY AM: 14:00-15:00 (840-900)
            # NY PM: 19:00-20:00 (1140-1200)
            windows = [
                (600, 660, "London Silver Bullet"),
                (840, 900, "NY AM Silver Bullet"),
                (1140, 1200, "NY PM Silver Bullet"),
            ]

            for w_start, w_end, w_name in windows:
                if w_start <= t <= w_end:
                    # Count fresh unfilled FVGs in this window
                    fresh_fvg = [f for f in fvg_zones if not f.get("filled") and f.get("near_price") and f.get("age_bars", 99) <= 5]
                    if fresh_fvg:
                        return {"active": True, "window": w_name, "fvg_count": len(fresh_fvg)}
        except Exception:
            pass
        return default

    # ─────────────────── IPDA Seviyeleri ──────────────────────────────────

    @staticmethod
    def _detect_ipda_levels(df: pd.DataFrame, current_price: float, atr: float) -> dict:
        """IPDA Data Range: 20/40/60 günlük kurumsal hedef seviyeleri."""
        default = {"levels": [], "nearest": 0.0, "distance_atr": 0.0}
        if len(df) < 60:
            return default

        levels = []
        for period_name, n_bars in [("20D", 20*24), ("40D", 40*24), ("60D", 60*24)]:
            # Adapt to available data
            actual_bars = min(n_bars, len(df) - 1)
            if actual_bars < 10:
                continue
            window = df.tail(actual_bars)
            period_high = float(window["high"].max())
            period_low = float(window["low"].min())
            levels.append({"period": period_name, "high": round(period_high, 5), "low": round(period_low, 5)})

        if not levels:
            return default

        # Find nearest IPDA level
        all_levels = []
        for lvl in levels:
            all_levels.append(lvl["high"])
            all_levels.append(lvl["low"])

        nearest = min(all_levels, key=lambda x: abs(x - current_price))
        distance = abs(nearest - current_price)

        return {
            "levels": levels,
            "nearest": round(nearest, 5),
            "distance_atr": round(distance / atr, 2) if atr > 0 else 0.0,
        }

    # ─────────────────── AMD Faz Tespiti ──────────────────────────────────

    @staticmethod
    def _detect_amd_phase(df: pd.DataFrame, session_hour: int = -1) -> dict:
        """AMD (Power of Three): Accumulation/Manipulation/Distribution faz tespiti."""
        default = {"phase": "UNKNOWN", "asia_high": 0.0, "asia_low": 0.0, "asia_swept": False}
        if len(df) < 20 or "datetime" not in df.columns:
            return default
        try:
            df2 = df.copy()
            df2["datetime"] = pd.to_datetime(df2["datetime"], errors="coerce")
            df2["hour"] = df2["datetime"].dt.hour

            # Asia session: 00:00-08:00 UTC
            today = df2["datetime"].dt.date.iloc[-1]
            today_data = df2[df2["datetime"].dt.date == today]

            asia_data = today_data[(today_data["hour"] >= 0) & (today_data["hour"] < 8)]
            if len(asia_data) < 3:
                # Fallback: use early bars
                asia_data = today_data.head(max(3, len(today_data) // 4))

            asia_high = float(asia_data["high"].max())
            asia_low = float(asia_data["low"].min())

            current_hour = int(df2.iloc[-1]["hour"])
            current_high = float(today_data["high"].max())
            current_low = float(today_data["low"].min())

            # Was Asia range swept?
            asia_swept = current_high > asia_high or current_low < asia_low

            # Phase determination (UTC hours)
            if current_hour < 8:
                phase = "ACCUMULATION"
            elif current_hour < 13:
                phase = "MANIPULATION"  # London session — fake moves
            else:
                phase = "DISTRIBUTION"  # NY session — real move

            return {
                "phase": phase,
                "asia_high": round(asia_high, 5),
                "asia_low": round(asia_low, 5),
                "asia_swept": asia_swept,
            }
        except Exception:
            return default

    # ─────────────────── BB Squeeze tespiti (BTC için kritik) ──────────────

    @staticmethod
    def _detect_bb_squeeze(df: pd.DataFrame, bb_upper: pd.Series, bb_lower: pd.Series, bb_mid: pd.Series) -> dict:
        """
        Bollinger Band Squeeze: bandwidth < %4 → sıkışma.
        Breakout: sıkışma sonrası ilk mum BB dışına çıkarsa.
        BTC'de %60-64 win rate ile kanıtlanmış strateji.
        """
        bandwidth = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
        bw = bandwidth.dropna()
        if len(bw) < 5:
            return {"squeeze": False, "breakout": None, "bandwidth": 0.0}

        current_bw = float(bw.iloc[-1])
        prev_bw = float(bw.iloc[-2]) if len(bw) >= 2 else current_bw
        avg_bw = float(bw.tail(20).mean()) if len(bw) >= 20 else float(bw.mean())

        is_squeeze = current_bw < 0.04 or current_bw < avg_bw * 0.6

        # Breakout: bandwidth genişliyor VE fiyat BB dışına çıktı
        breakout = None
        if prev_bw < 0.04 and current_bw > prev_bw * 1.2:
            close = float(df.iloc[-1]["close"])
            upper = float(bb_upper.iloc[-1])
            lower = float(bb_lower.iloc[-1])
            if close > upper:
                breakout = "BULLISH"
            elif close < lower:
                breakout = "BEARISH"

        return {"squeeze": is_squeeze, "breakout": breakout, "bandwidth": round(current_bw, 5)}

    # ─────────────────────────── Ana analiz ────────────────────────────────

    # ─────────────── Ichimoku Cloud hesaplama ────────────────────────────
    @staticmethod
    def _ichimoku(df: pd.DataFrame) -> dict:
        """Ichimoku Cloud: Tenkan, Kijun, Senkou A/B, Chikou."""
        if len(df) < 52:
            return {"valid": False}
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        chikou = close.shift(-26)
        last_i = len(df) - 1
        return {
            "valid": True,
            "tenkan": float(tenkan.iloc[last_i]) if not isnan(float(tenkan.iloc[last_i])) else 0.0,
            "kijun": float(kijun.iloc[last_i]) if not isnan(float(kijun.iloc[last_i])) else 0.0,
            "senkou_a": float(senkou_a.iloc[last_i]) if last_i < len(senkou_a) and not isnan(float(senkou_a.iloc[last_i])) else 0.0,
            "senkou_b": float(senkou_b.iloc[last_i]) if last_i < len(senkou_b) and not isnan(float(senkou_b.iloc[last_i])) else 0.0,
            "price_above_cloud": float(close.iloc[-1]) > max(
                float(senkou_a.iloc[last_i]) if not isnan(float(senkou_a.iloc[last_i])) else 0,
                float(senkou_b.iloc[last_i]) if not isnan(float(senkou_b.iloc[last_i])) else 0,
            ),
            "price_below_cloud": float(close.iloc[-1]) < min(
                float(senkou_a.iloc[last_i]) if not isnan(float(senkou_a.iloc[last_i])) else float("inf"),
                float(senkou_b.iloc[last_i]) if not isnan(float(senkou_b.iloc[last_i])) else float("inf"),
            ),
            "tenkan_above_kijun": float(tenkan.iloc[last_i]) > float(kijun.iloc[last_i]) if not isnan(float(tenkan.iloc[last_i])) else False,
            "cloud_bullish": (float(senkou_a.iloc[last_i]) if not isnan(float(senkou_a.iloc[last_i])) else 0) > (float(senkou_b.iloc[last_i]) if not isnan(float(senkou_b.iloc[last_i])) else 0),
        }

    @staticmethod
    def _detect_divergence(df: pd.DataFrame, rsi: pd.Series) -> dict:
        """RSI/Price divergence detection."""
        default = {"bullish_div": False, "bearish_div": False}
        if len(df) < 30:
            return default
        window = df.tail(30).reset_index(drop=True)
        rsi_vals = rsi.tail(30).reset_index(drop=True)
        # Find swing lows/highs in price and RSI
        price_lows = []
        price_highs = []
        for i in range(2, len(window) - 2):
            if (float(window.iloc[i]["low"]) <= float(window.iloc[i-1]["low"]) and
                float(window.iloc[i]["low"]) <= float(window.iloc[i+1]["low"])):
                price_lows.append((i, float(window.iloc[i]["low"]), float(rsi_vals.iloc[i]) if not isnan(float(rsi_vals.iloc[i])) else 50))
            if (float(window.iloc[i]["high"]) >= float(window.iloc[i-1]["high"]) and
                float(window.iloc[i]["high"]) >= float(window.iloc[i+1]["high"])):
                price_highs.append((i, float(window.iloc[i]["high"]), float(rsi_vals.iloc[i]) if not isnan(float(rsi_vals.iloc[i])) else 50))
        # Bullish divergence: price makes lower low, RSI makes higher low
        if len(price_lows) >= 2:
            if price_lows[-1][1] < price_lows[-2][1] and price_lows[-1][2] > price_lows[-2][2]:
                default["bullish_div"] = True
        # Bearish divergence: price makes higher high, RSI makes lower high
        if len(price_highs) >= 2:
            if price_highs[-1][1] > price_highs[-2][1] and price_highs[-1][2] < price_highs[-2][2]:
                default["bearish_div"] = True
        return default

    def analyze(
        self,
        symbol: str,
        df: pd.DataFrame,
        timeframe: str = "15min",
        higher_tf_df: pd.DataFrame | None = None,
        high_impact_events: list[dict[str, str]] | None = None,
        dxy_bias: str = "NEUTRAL",
        cot_bias: str = "NEUTRAL",
        sentiment_score: float = 0.0,
        strategy_mode: str = "default",
    ) -> AnalysisResult:
        prof = get_symbol_profile(symbol)

        data = df.copy()
        data["ema_20"]  = data["close"].ewm(span=20,  adjust=False).mean()
        data["ema_50"]  = data["close"].ewm(span=50,  adjust=False).mean()
        data["ema_200"] = data["close"].ewm(span=200, adjust=False).mean()
        data["rsi"]     = self._rsi(data["close"])
        data["atr"]     = self._atr(data)
        data["adx"]     = self._adx(data)
        _macd_line, _macd_sig, _macd_hist = self._macd(data["close"])
        _bb_upper, _bb_mid, _bb_lower     = self._bollinger(data["close"])

        last          = data.iloc[-1]
        current_price = float(last["close"])
        ema20         = float(last["ema_20"])
        ema50         = float(last["ema_50"])
        ema200        = float(last["ema_200"]) if not isnan(float(last["ema_200"])) else current_price
        rsi           = float(last["rsi"])     if not isnan(float(last["rsi"]))     else 50.0
        atr           = float(last["atr"])     if not isnan(float(last["atr"]))     else max(current_price * 0.002, 0.001)
        adx           = float(last["adx"])     if not isnan(float(last["adx"]))     else 18.0
        macd_hist_val = float(_macd_hist.iloc[-1]) if not isnan(float(_macd_hist.iloc[-1])) else 0.0
        bb_upper_val  = float(_bb_upper.iloc[-1])  if not isnan(float(_bb_upper.iloc[-1]))  else current_price
        bb_mid_val    = float(_bb_mid.iloc[-1])    if not isnan(float(_bb_mid.iloc[-1]))    else current_price
        bb_lower_val  = float(_bb_lower.iloc[-1])  if not isnan(float(_bb_lower.iloc[-1]))  else current_price

        # ── SMC / ICT hesaplamaları ──
        order_blocks     = self._detect_order_blocks(data, atr, current_price)
        fvg_zones        = self._detect_fvg(data, atr, current_price)
        ifvg_zones       = self._detect_ifvg(fvg_zones, current_price, atr)
        breaker_blocks   = self._detect_breaker_blocks(data, order_blocks, current_price, atr)
        displacement     = self._detect_displacement(data, atr)
        premium_discount = self._detect_premium_discount(data, current_price)
        eq_highs, eq_lows = self._detect_equal_levels(data, atr)

        supports, resistances = self._find_levels(data)
        nearest_supports     = [lvl for lvl in supports     if lvl < current_price][-3:]
        nearest_resistances  = [lvl for lvl in resistances  if lvl > current_price][:3]

        lower_support    = nearest_supports[-1]    if nearest_supports    else round(current_price - atr * 1.5, 5)
        upper_resistance = nearest_resistances[0]  if nearest_resistances else round(current_price + atr * 1.5, 5)

        # Trend: HTF'den (1H) al — 15min trend çok gürültülü
        higher_tf_trend = self._trend_from_df(higher_tf_df) if higher_tf_df is not None else ("Yukari" if ema50 > ema200 else "Asagi")
        trend = higher_tf_trend  # Ana trend = HTF trend
        _ema_momentum = "Yukari" if ema20 > ema50 else "Asagi"
        choch           = self._detect_choch(data, trend)
        ote_zone        = self._detect_ote(data, trend)

        near_support    = abs(current_price - lower_support)    <= atr * 1.5
        near_resistance = abs(upper_resistance - current_price) <= atr * 1.5

        sweep_signal = self._detect_liquidity_sweep(data.tail(10), supports, resistances)
        sniper_entry = self._detect_sniper_entry(data.tail(10), trend, higher_tf_trend, supports, resistances)

        # ── Tier-1/2/3 yeni tespitler ──
        volume_analysis = self._detect_volume_analysis(data, atr)
        pdh_pdl         = self._detect_pdh_pdl(data)
        round_numbers   = self._detect_round_numbers(current_price, atr, symbol)

        # ── BB Squeeze (özellikle BTC için) ──
        bb_squeeze = self._detect_bb_squeeze(data, _bb_upper, _bb_lower, _bb_mid)

        # ── Yeni ICT/Kurumsal tespitler ──
        unicorn_model = self._detect_unicorn_model(breaker_blocks, fvg_zones, current_price, atr)
        vwap = self._calculate_vwap(data)
        silver_bullet = self._detect_silver_bullet(data, fvg_zones, atr)
        ipda_levels = self._detect_ipda_levels(data, current_price, atr)
        amd_phase = self._detect_amd_phase(data)

        # ── Volatilite Rejim Tespiti → SL/TP çarpanı ──
        try:
            from app.services.regime_detector import detect_regime as _detect_regime
            _regime_result = _detect_regime(data, symbol)
            _sl_mult = _regime_result.sl_multiplier
            _tp_mult = _regime_result.tp_multiplier
        except Exception:
            _sl_mult, _tp_mult = 1.0, 1.0

        # ── Sinyal mantığı (SEMBOL BAZLI PROFİL) ──
        signal     = "NO TRADE"
        reason     = "Kaliteli kurulum olusmadi."
        no_trade_reasons: list[str] = []

        entry_zone    = (round(current_price - atr * prof.entry_atr_offset, 5),
                         round(current_price + atr * prof.entry_atr_offset, 5))
        # Default TP/SL: rejim çarpanlı, dengeli SL
        _default_long = (trend == "Yukari")
        _sl_m = prof.sl_atr_mult * _sl_mult
        if _default_long:
            stop_loss     = round(current_price - atr * _sl_m, 5)
            take_profit   = round(current_price + atr * _sl_m * prof.tp1_rr * _tp_mult, 5)
            take_profit_2 = round(current_price + atr * _sl_m * prof.tp2_rr * _tp_mult, 5)
        else:
            stop_loss     = round(current_price + atr * _sl_m, 5)
            take_profit   = round(current_price - atr * _sl_m * prof.tp1_rr * _tp_mult, 5)
            take_profit_2 = round(current_price - atr * _sl_m * prof.tp2_rr * _tp_mult, 5)

        # DXY bias filtresi (sadece hassas enstrümanlar)
        dxy_block_long  = prof.dxy_sensitive and dxy_bias == "BULLISH"
        dxy_block_short = prof.dxy_sensitive and dxy_bias == "BEARISH"

        # COT bias filtresi
        cot_block_long  = cot_bias == "BEARISH"
        cot_block_short = cot_bias == "BULLISH"

        # Volume uyumu
        vol_confirms_long  = volume_analysis.get("delta_bias") == "BULLISH"
        vol_confirms_short = volume_analysis.get("delta_bias") == "BEARISH"

        trend_strength_ok = adx >= prof.adx_min

        # ══════════════════════════════════════════════════════════════
        # YENİ SİNYAL SİSTEMİ: 3 Hard Gate + Score Bazlı
        # Araştırma: optimal koşul sayısı 3-5, geri kalanı score
        # ══════════════════════════════════════════════════════════════

        # Hard Gate 1: Trend yönü (EMA50 > EMA200 = stabil trend)
        # Hard Gate 2: RSI aşırı bölgede DEĞİL
        # Hard Gate 3: Minimum volatilite
        _rsi_not_extreme = prof.rsi_oversold < rsi < prof.rsi_overbought
        atr_ratio = atr / current_price if current_price else 0.0
        _vol_ok = atr_ratio >= prof.atr_ratio_min

        # Pullback entry: trend yönünde + anahtar seviyeye çekilme
        _near_ema20 = abs(current_price - ema20) <= atr * 0.5
        _at_key_level_long = near_support or _near_ema20 or (bb_squeeze.get("breakout") == "BULLISH")
        _at_key_level_short = near_resistance or _near_ema20 or (bb_squeeze.get("breakout") == "BEARISH")

        # ── Yeni stratejiler için gereken tespitler (strateji seçiminden ÖNCE) ──
        ichimoku = self._ichimoku(data)
        divergence = self._detect_divergence(data, data["rsi"])
        bos_mss = self._detect_bos_mss(data, trend, atr)
        confirmation_candle = self._detect_confirmation_candle(data, "LONG" if trend == "Yukari" else "SHORT", atr)
        judas_swing = self._detect_judas_swing(data, trend, atr)

        # ── Strateji Modu Seçimi ──
        if strategy_mode == "default":
            long_ok = (trend == "Yukari" and _rsi_not_extreme and _vol_ok and _at_key_level_long)
            short_ok = (trend == "Asagi" and _rsi_not_extreme and _vol_ok and _at_key_level_short)

        elif strategy_mode == "reversal":
            # Reversal: CHoCH + divergence at key levels + RSI extreme
            _rsi_oversold = rsi < prof.rsi_oversold + 5
            _rsi_overbought = rsi > prof.rsi_overbought - 5
            long_ok = (_rsi_oversold and _vol_ok and (near_support or divergence.get("bullish_div"))
                       and (choch.get("type") == "bullish_choch" or divergence.get("bullish_div")))
            short_ok = (_rsi_overbought and _vol_ok and (near_resistance or divergence.get("bearish_div"))
                        and (choch.get("type") == "bearish_choch" or divergence.get("bearish_div")))

        elif strategy_mode == "breakout":
            # Breakout: BB squeeze breakout + volume spike + BOS
            _bb_bull_break = bb_squeeze.get("breakout") == "BULLISH"
            _bb_bear_break = bb_squeeze.get("breakout") == "BEARISH"
            _vol_spike = volume_analysis.get("volume_spike", False)
            _bos_bull = bos_mss.get("bos") and bos_mss.get("type") == "bullish_bos"
            _bos_bear = bos_mss.get("bos") and bos_mss.get("type") == "bearish_bos"
            long_ok = (_vol_ok and (_bb_bull_break or (_bos_bull and _vol_spike))
                       and adx >= prof.adx_min)
            short_ok = (_vol_ok and (_bb_bear_break or (_bos_bear and _vol_spike))
                        and adx >= prof.adx_min)

        elif strategy_mode == "scalp":
            # Scalp: EMA20 bounce + tight conditions + momentum confirmation
            _near_ema20_tight = abs(current_price - ema20) <= atr * 0.3
            long_ok = (trend == "Yukari" and _rsi_not_extreme and _vol_ok
                       and _near_ema20_tight and _ema_momentum == "Yukari"
                       and macd_hist_val > 0)
            short_ok = (trend == "Asagi" and _rsi_not_extreme and _vol_ok
                        and _near_ema20_tight and _ema_momentum == "Asagi"
                        and macd_hist_val < 0)

        elif strategy_mode == "swing":
            # Swing: OB + FVG + HTF alignment + wide SL
            _fresh_bull_ob = any(ob["type"] == "bullish_ob" and not ob.get("broken") and ob.get("near_price") for ob in order_blocks)
            _fresh_bear_ob = any(ob["type"] == "bearish_ob" and not ob.get("broken") and ob.get("near_price") for ob in order_blocks)
            _near_bull_fvg = any(f["type"] == "bullish_fvg" and not f.get("filled") and f.get("near_price") for f in fvg_zones)
            _near_bear_fvg = any(f["type"] == "bearish_fvg" and not f.get("filled") and f.get("near_price") for f in fvg_zones)
            long_ok = (trend == "Yukari" and higher_tf_trend == "Yukari" and _vol_ok
                       and (_fresh_bull_ob or _near_bull_fvg) and _rsi_not_extreme)
            short_ok = (trend == "Asagi" and higher_tf_trend == "Asagi" and _vol_ok
                        and (_fresh_bear_ob or _near_bear_fvg) and _rsi_not_extreme)

        elif strategy_mode == "ict":
            # ICT: Silver bullet + AMD + Judas swing + OTE
            _in_ote = ote_zone.get("valid") and ote_zone["ote_low"] <= current_price <= ote_zone["ote_high"]
            _sb_active = silver_bullet.get("active", False)
            _judas_bull = judas_swing.get("detected") and judas_swing.get("direction") == "bullish"
            _judas_bear = judas_swing.get("detected") and judas_swing.get("direction") == "bearish"
            _amd_dist = amd_phase.get("phase") == "DISTRIBUTION"
            long_ok = (_vol_ok and _rsi_not_extreme and trend == "Yukari"
                       and (_in_ote or _sb_active or _judas_bull)
                       and (confirmation_candle.get("detected") or displacement.get("detected")))
            short_ok = (_vol_ok and _rsi_not_extreme and trend == "Asagi"
                        and (_in_ote or _sb_active or _judas_bear)
                        and (confirmation_candle.get("detected") or displacement.get("detected")))

        elif strategy_mode == "ichimoku":
            # Ichimoku Cloud: price above/below cloud + TK cross
            if ichimoku.get("valid"):
                long_ok = (ichimoku["price_above_cloud"] and ichimoku["tenkan_above_kijun"]
                           and ichimoku["cloud_bullish"] and _vol_ok and _rsi_not_extreme)
                short_ok = (ichimoku["price_below_cloud"] and not ichimoku["tenkan_above_kijun"]
                            and not ichimoku["cloud_bullish"] and _vol_ok and _rsi_not_extreme)
            else:
                long_ok = False
                short_ok = False

        elif strategy_mode == "divergence":
            # Pure divergence: RSI divergence + confirmation candle
            long_ok = (divergence.get("bullish_div") and _vol_ok
                       and (rsi < 45) and near_support)
            short_ok = (divergence.get("bearish_div") and _vol_ok
                        and (rsi > 55) and near_resistance)

        elif strategy_mode == "multi_ma":
            # Multi-MA Crossover: EMA 9/21/55 alignment
            ema9 = float(data["close"].ewm(span=9, adjust=False).mean().iloc[-1])
            ema21 = float(data["close"].ewm(span=21, adjust=False).mean().iloc[-1])
            ema55 = float(data["close"].ewm(span=55, adjust=False).mean().iloc[-1])
            long_ok = (ema9 > ema21 > ema55 and _vol_ok and _rsi_not_extreme
                       and current_price > ema9 and adx >= prof.adx_min)
            short_ok = (ema9 < ema21 < ema55 and _vol_ok and _rsi_not_extreme
                        and current_price < ema9 and adx >= prof.adx_min)

        else:
            # Fallback to default
            long_ok = (trend == "Yukari" and _rsi_not_extreme and _vol_ok and _at_key_level_long)
            short_ok = (trend == "Asagi" and _rsi_not_extreme and _vol_ok and _at_key_level_short)

        # Haber kilidi — hard gate (sinyal verme)
        if high_impact_events:
            long_ok = False
            short_ok = False
            no_trade_reasons.append("Yuksek etkili haber riski")

        # SL/TP multipliers per strategy mode
        _mode_sl_mult = {"scalp": 0.6, "swing": 1.4, "breakout": 0.8, "ict": 1.0,
                         "reversal": 1.0, "ichimoku": 1.1, "divergence": 1.0,
                         "multi_ma": 0.9}.get(strategy_mode, 1.0)
        _mode_tp_mult = {"scalp": 0.7, "swing": 1.5, "breakout": 1.3, "ict": 1.2,
                         "reversal": 1.1, "ichimoku": 1.2, "divergence": 1.1,
                         "multi_ma": 1.0}.get(strategy_mode, 1.0)

        if long_ok:
            signal    = "LONG"
            reason    = f"{symbol} long [{strategy_mode}]: trend yukari."
            entry_zone = (round(current_price - atr * 0.05, 5),
                          round(current_price + atr * 0.05, 5))
            stop_loss  = round(current_price - atr * prof.sl_atr_mult * _sl_mult * _mode_sl_mult, 5)
            risk       = abs(current_price - stop_loss)
            _rr_tp1   = current_price + risk * prof.tp1_rr * _mode_tp_mult
            _max_tp1  = current_price + risk * 2.5
            if upper_resistance > current_price and upper_resistance <= _max_tp1:
                take_profit = round(upper_resistance - atr * 0.05, 5)
            else:
                take_profit = round(_rr_tp1, 5)
            take_profit_2 = round(current_price + risk * prof.tp2_rr * _mode_tp_mult, 5)

        elif short_ok:
            signal    = "SHORT"
            reason    = f"{symbol} short [{strategy_mode}]: trend asagi."
            entry_zone = (round(current_price - atr * 0.05, 5),
                          round(current_price + atr * 0.05, 5))
            stop_loss  = round(current_price + atr * prof.sl_atr_mult * _sl_mult * _mode_sl_mult, 5)
            risk       = abs(stop_loss - current_price)
            _rr_tp1   = current_price - risk * prof.tp1_rr * _mode_tp_mult
            _max_tp1  = current_price - risk * 2.5
            if lower_support < current_price and lower_support >= _max_tp1:
                take_profit = round(lower_support + atr * 0.05, 5)
            else:
                take_profit = round(_rr_tp1, 5)
            take_profit_2 = round(current_price - risk * prof.tp2_rr * _mode_tp_mult, 5)

        risk   = abs(np.mean(entry_zone) - stop_loss)
        reward = abs(take_profit - np.mean(entry_zone))
        rr_ratio = round(reward / risk, 2) if risk else 0.0

        if adx >= prof.trend_adx and atr_ratio >= prof.trend_atr_ratio:
            regime = "TREND"
        elif adx >= prof.mixed_adx and atr_ratio >= prof.atr_ratio_min:
            regime = "MIXED"
        else:
            regime = "RANGE"

        # Onay mumu (signal belirlendikten sonra yeniden hesapla — doğru yönle)
        if signal in ("LONG", "SHORT"):
            confirmation_candle = self._detect_confirmation_candle(data, signal, atr)

        # SMC Confluence Sayısı
        smc_confluence_count = self._count_smc_confluence(
            signal=signal, choch=choch, displacement=displacement,
            ote_zone=ote_zone, current_price=current_price,
            premium_discount=premium_discount, order_blocks=order_blocks,
            fvg_zones=fvg_zones, breaker_blocks=breaker_blocks,
            ifvg_zones=ifvg_zones, eq_highs=eq_highs, eq_lows=eq_lows,
            atr=atr, bos_mss=bos_mss, confirmation_candle=confirmation_candle,
            sweep_signal=sweep_signal, judas_swing=judas_swing,
        )

        # ── No-trade sebepleri (sadece bilgilendirme) ──
        if signal == "NO TRADE":
            no_trade_reasons.append("Ana trend kosullari olusmadi")
        if rr_ratio < prof.min_rr and signal in {"LONG", "SHORT"}:
            no_trade_reasons.append(f"R/R yetersiz ({rr_ratio:.2f} < {prof.min_rr})")

        # ══════════════════════════════════════════════════════════════
        # SCORE SİSTEMİ: Normalize 0-100 (8 kategori, her biri max ~12 puan)
        # Sadece signal LONG/SHORT ise hesaplanır
        # ══════════════════════════════════════════════════════════════
        score = 0

        if signal not in {"LONG", "SHORT"}:
            # NO TRADE sinyallerine score verme (anlamlı değil)
            score = 0
        else:
            # ── Kategori 1: Trend Uyumu (max 15) ──
            if trend == higher_tf_trend:                           score += 12
            else:                                                  score -= 10  # HTF uyumsuz = ciddi ceza

            # ── Kategori 2: Momentum (max 15) ──
            if macd_hist_val > 0 and signal == "LONG":            score += 5
            elif macd_hist_val < 0 and signal == "SHORT":         score += 5
            elif macd_hist_val < 0 and signal == "LONG":          score -= 5  # MACD ters
            elif macd_hist_val > 0 and signal == "SHORT":         score -= 5
            if adx >= prof.trend_adx:                              score += 5
            elif adx >= prof.adx_min:                              score += 2
            # EMA20 momentum doğrulaması (kritik)
            _signal_dir = "Yukari" if signal == "LONG" else "Asagi"
            if _ema_momentum == _signal_dir:
                score += 6  # Momentum uyumlu
            else:
                score -= 4  # Momentum ters = dikkatli ol

            # ── Kategori 3: Yapısal Konum (max 12) ──
            if signal == "LONG" and near_support:                  score += 8
            if signal == "SHORT" and near_resistance:              score += 8
            if regime == "TREND":                                  score += 4
            elif regime == "MIXED":                                score += 2

            # ── Kategori 4: R:R Kalitesi (max 10) ──
            if rr_ratio >= 1.8:                                    score += 10
            elif rr_ratio >= 1.5:                                  score += 8
            elif rr_ratio >= 1.2:                                  score += 5
            elif rr_ratio >= 1.0:                                  score += 3

            # ── Kategori 5: SMC/ICT Confluence (max 15) ──
            if sweep_signal != "Yok":                              score += 5
            if confirmation_candle.get("detected"):                score += 4
            if signal == "LONG" and premium_discount.get("zone") == "DISCOUNT": score += 4
            elif signal == "SHORT" and premium_discount.get("zone") == "PREMIUM": score += 4
            elif signal == "LONG" and premium_discount.get("zone") == "PREMIUM": score -= 4
            elif signal == "SHORT" and premium_discount.get("zone") == "DISCOUNT": score -= 4

            # ── Kategori 6: Kurumsal Seviye (max 12) ──
            if vwap > 0:
                if signal == "LONG" and current_price > vwap:      score += 4
                elif signal == "SHORT" and current_price < vwap:   score += 4
                else:                                              score -= 3
            pdh = pdh_pdl.get("prev_day_high", 0)
            pdl = pdh_pdl.get("prev_day_low", 0)
            if pdh > 0 and signal == "SHORT" and abs(current_price - pdh) <= atr: score += 4
            if pdl > 0 and signal == "LONG" and abs(current_price - pdl) <= atr: score += 4

            # ── Kategori 7: Dış Doğrulama (max 10) ──
            if not dxy_block_long and signal == "LONG":            score += 3
            elif dxy_block_long and signal == "LONG":              score -= 5
            if not dxy_block_short and signal == "SHORT":          score += 3
            elif dxy_block_short and signal == "SHORT":            score -= 5
            if cot_bias == "BULLISH" and signal == "LONG":         score += 4
            if cot_bias == "BEARISH" and signal == "SHORT":        score += 4
            if cot_block_long and signal == "LONG":                score -= 4
            if cot_block_short and signal == "SHORT":              score -= 4

            # ── Kategori 8: Seans/Zamanlama (max 8) ──
            from datetime import datetime as _dt, timezone as _tz
            _now_utc = _dt.now(_tz.utc).hour
            _sym_upper = symbol.upper()
            if _sym_upper in ("XAUUSD", "EURUSD", "GBPUSD", "USDCHF"):
                if 13 <= _now_utc < 16:   score += 6
                elif 8 <= _now_utc < 13:  score += 3
                elif 0 <= _now_utc < 7:   score -= 5
            elif _sym_upper == "BTCUSD":
                if 13 <= _now_utc < 17:   score += 4
                elif 8 <= _now_utc < 13:  score += 2
            elif _sym_upper == "USDJPY":
                if 0 <= _now_utc < 3:     score += 5
                elif 13 <= _now_utc < 16: score += 5
            elif _sym_upper == "AUDUSD":
                if 0 <= _now_utc < 4:     score += 5
                elif 13 <= _now_utc < 16: score += 3

            # ── Ek SMC bonuslar (küçük) ──
            if bb_squeeze.get("breakout") and signal in {"LONG", "SHORT"}: score += 4
            if sniper_entry != "Yok":                              score += 3

        # SMC confluence ek bonus (normalize: max 6 puan)
        if signal in {"LONG", "SHORT"}:
            score += min(smc_confluence_count * 2, 6)

        # Clamp: 0-100 arası
        score = max(0, min(100, score))
        quality = self._quality_from_score(score)

        # ── Score bazlı NO TRADE kararı (threshold: min_score_for_signal) ──
        if signal in {"LONG", "SHORT"} and score < prof.min_score_for_signal:
            no_trade_reasons.append(f"Score yetersiz ({score}/{prof.min_score_for_signal})")
            signal = "NO TRADE"
            reason = " | ".join(no_trade_reasons)
        elif signal in {"LONG", "SHORT"} and rr_ratio < prof.min_rr:
            no_trade_reasons.append(f"R/R yetersiz ({rr_ratio:.2f} < {prof.min_rr})")
            signal = "NO TRADE"
            reason = " | ".join(no_trade_reasons)
        elif signal == "NO TRADE":
            reason = " | ".join(no_trade_reasons) if no_trade_reasons else reason

        # Reason zenginleştirme
        extras = []
        if sweep_signal != "Yok":        extras.append(f"Sweep: {sweep_signal}")
        if sniper_entry != "Yok":        extras.append(f"Sniper: {sniper_entry}")
        if choch.get("detected"):        extras.append(choch.get("description", "CHoCH"))
        if displacement.get("detected"): extras.append(displacement.get("strength", "Displacement"))
        if judas_swing.get("detected"):  extras.append(f"Judas Swing ({judas_swing.get('direction','')})")
        if bos_mss.get("bos"):           extras.append(f"BOS: {bos_mss.get('type','')}")
        if bos_mss.get("mss"):           extras.append(f"MSS: {bos_mss.get('type','')}")
        if confirmation_candle.get("detected"): extras.append(f"Onay: {confirmation_candle.get('type','')}")
        if extras:
            reason = reason + " | " + " | ".join(extras)

        return AnalysisResult(
            symbol=symbol,
            trend=trend,
            higher_tf_trend=higher_tf_trend,
            current_price=round(current_price, 5),
            support=nearest_supports,
            resistance=nearest_resistances,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            take_profit=take_profit,
            take_profit_2=take_profit_2,
            rr_ratio=rr_ratio,
            signal=signal,
            timeframe=timeframe,
            reason=reason,
            atr=round(atr, 5),
            rsi=round(rsi, 2),
            regime=regime,
            setup_score=score,
            quality=quality,
            sweep_signal=sweep_signal,
            sniper_entry=sniper_entry,
            no_trade_reasons=no_trade_reasons,
            macd_hist=round(macd_hist_val, 6),
            bb_upper=round(bb_upper_val, 5),
            bb_mid=round(bb_mid_val, 5),
            bb_lower=round(bb_lower_val, 5),
            order_blocks=order_blocks,
            fvg_zones=fvg_zones,
            choch=choch,
            premium_discount=premium_discount,
            equal_highs=eq_highs,
            equal_lows=eq_lows,
            displacement=displacement,
            ote_zone=ote_zone,
            breaker_blocks=breaker_blocks,
            ifvg_zones=ifvg_zones,
            confirmation_candle=confirmation_candle,
            bos_mss=bos_mss,
            judas_swing=judas_swing,
            smc_confluence_count=smc_confluence_count,
            dxy_bias=dxy_bias,
            volume_analysis=volume_analysis,
            pdh_pdl=pdh_pdl,
            round_numbers=round_numbers,
            cot_bias=cot_bias,
            sentiment_score=round(float(sentiment_score), 3),
            unicorn_model=unicorn_model,
            vwap=vwap,
            silver_bullet=silver_bullet,
            ipda_levels=ipda_levels,
            amd_phase=amd_phase,
        )

    @staticmethod
    def risk_text(balance: float, risk_percent: float, entry: float, stop_loss: float) -> str:
        risk_amount = balance * (risk_percent / 100)
        stop_distance = abs(entry - stop_loss)
        if stop_distance == 0:
            return "Stop mesafesi 0 olamaz."
        units = risk_amount / stop_distance
        return (
            f"Bakiye: {balance:.2f}\n"
            f"Risk: %{risk_percent:.2f} = {risk_amount:.2f}\n"
            f"Stop mesafesi: {stop_distance:.5f}\n"
            f"Pozisyon boyutu: {units:.2f} birim\n"
        )
