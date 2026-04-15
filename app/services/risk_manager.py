from __future__ import annotations

"""
Risk Manager — Profesyonel seviye risk yönetimi.

Özellikler:
  - Kelly Criterion (fraksiyonel)
  - Drawdown tabanlı pozisyon küçültme
  - Kısmi TP yapısı (50% TP1, trail geri kalanı)
  - Monte Carlo simülasyonu
  - ATR dinamik SL/TP çarpanları (enstrümana göre)
  - Korelasyon tabanlı toplam risk uyarısı
"""

import math
import random
from dataclasses import dataclass, field


# ─── ATR Çarpanları (enstrümana göre) ────────────────────────────────────────
# Research: XAUUSD daha geniş stop gerektirir

_ATR_MULTIPLIERS: dict[str, dict[str, float]] = {
    "XAUUSD": {"sl": 2.0, "tp1": 3.0, "tp2": 5.0},
    "BTCUSD": {"sl": 2.5, "tp1": 3.5, "tp2": 6.0},
    "GBPUSD": {"sl": 1.5, "tp1": 2.5, "tp2": 4.0},
    "EURUSD": {"sl": 1.2, "tp1": 2.2, "tp2": 3.5},
    "USDJPY": {"sl": 1.2, "tp1": 2.2, "tp2": 3.5},
    "USDCHF": {"sl": 1.2, "tp1": 2.2, "tp2": 3.5},
    "AUDUSD": {"sl": 1.3, "tp1": 2.3, "tp2": 3.8},
    "_default": {"sl": 1.5, "tp1": 2.5, "tp2": 4.0},
}


# ─── Veri sınıfları ───────────────────────────────────────────────────────────

@dataclass
class PositionSizeRec:
    base_risk_pct: float        # Temel risk (ör: 0.01 = %1)
    adjusted_risk_pct: float    # Drawdown + Kelly sonrası ayarlanmış
    kelly_fraction: float       # Kelly f* (fraksiyonel)
    drawdown_scale: float       # Drawdown çarpanı (0.0–1.0)
    lot_size_hint: str          # Örnek: "0.05 lot for $10,000 account"
    notes: list[str] = field(default_factory=list)


@dataclass
class PartialTPStructure:
    entry: float
    stop_loss: float
    tp1: float              # Birinci hedef
    tp2: float              # İkinci hedef
    tp1_size_pct: float     # TP1'de kapatılacak % (0.5 = %50)
    tp2_size_pct: float     # TP2'ye taşınacak %
    breakeven_at: float     # BE için fiyat seviyesi
    expected_rr: float      # Beklenen ağırlıklı R/R


@dataclass
class MonteCarloResult:
    median_return: float        # Medyan yıllık getiri
    pct5_return: float          # %5'lik kötü senaryo
    pct95_return: float         # %95'lik iyi senaryo
    max_dd_median: float        # Medyan maksimum drawdown
    max_dd_pct95: float         # %95 senaryo max drawdown
    ruin_probability: float     # P(DD > %50) yıkım olasılığı
    sharpe_ratio: float         # Basit Sharpe
    expected_trades_per_year: int
    recommendation: str         # "DEVAM", "DİKKAT", "DUR"


@dataclass
class DynamicLevels:
    symbol: str
    atr: float
    sl_distance: float
    tp1_distance: float
    tp2_distance: float
    sl_multiplier: float
    rr_ratio: float


# ─── Ana sınıf ────────────────────────────────────────────────────────────────

class RiskManager:

    # --- Kelly Criterion ---

    def kelly_criterion(
        self,
        win_rate: float,
        avg_win_r: float = 2.0,
        avg_loss_r: float = 1.0,
        fraction: float = 0.25,
    ) -> float:
        """
        Kelly f* = (bp - q) / b
        b = avg_win / avg_loss
        fraction: quarter-kelly (0.25) önerilir
        """
        if not (0 < win_rate < 1):
            return 0.01
        q = 1.0 - win_rate
        b = avg_win_r / max(avg_loss_r, 1e-9)
        f_star = (b * win_rate - q) / b
        if f_star <= 0:
            return 0.005  # Negatif beklenti
        adjusted = f_star * fraction
        return round(max(0.005, min(adjusted, 0.03)), 4)  # %0.5 – %3 arası

    # --- Drawdown skalası ---

    def drawdown_scale(self, current_drawdown_pct: float) -> float:
        """
        Drawdown'a göre pozisyon büyüklüğü çarpanı.
        DD < 5%:  1.00x (tam boyut)
        DD < 10%: 0.75x
        DD < 15%: 0.50x
        DD < 20%: 0.25x
        DD >= 20%: 0.00 (işlem yapma)
        """
        dd = abs(current_drawdown_pct)
        if dd < 0.05:    return 1.00
        elif dd < 0.10:  return 0.75
        elif dd < 0.15:  return 0.50
        elif dd < 0.20:  return 0.25
        return 0.0

    # --- Pozisyon büyüklüğü önerisi ---

    def get_position_size_rec(
        self,
        balance: float = 10000.0,
        win_rate: float = 0.55,
        avg_win_r: float = 2.0,
        avg_loss_r: float = 1.0,
        current_drawdown_pct: float = 0.0,
        base_risk_pct: float = 0.01,
        entry: float = 0.0,
        stop_loss: float = 0.0,
    ) -> PositionSizeRec:
        kelly_f  = self.kelly_criterion(win_rate, avg_win_r, avg_loss_r)
        dd_scale = self.drawdown_scale(current_drawdown_pct)

        # En muhafazakar olanı al
        effective_risk = min(base_risk_pct, kelly_f) * dd_scale
        effective_risk = max(0.0, effective_risk)

        notes: list[str] = []
        if dd_scale < 1.0:
            notes.append(f"DD skalasi: x{dd_scale} (Drawdown: {current_drawdown_pct*100:.1f}%)")
        if kelly_f < base_risk_pct:
            notes.append(f"Kelly onerisi: %{kelly_f*100:.2f} (baz riskten dusuk)")
        if dd_scale == 0.0:
            notes.append("MAX DD ASILDI — ISLEM YAPMA")

        # Lot hint
        lot_hint = ""
        if entry > 0 and stop_loss > 0 and balance > 0:
            risk_amount = balance * effective_risk
            pip_value  = abs(entry - stop_loss)
            if pip_value > 0:
                units = risk_amount / pip_value
                lots  = units / 100_000 if pip_value < 1.0 else units / 100
                lot_hint = f"${balance:.0f} bakiye icin ~{lots:.2f} lot ({effective_risk*100:.2f}% risk)"

        return PositionSizeRec(
            base_risk_pct=base_risk_pct,
            adjusted_risk_pct=round(effective_risk, 4),
            kelly_fraction=kelly_f,
            drawdown_scale=dd_scale,
            lot_size_hint=lot_hint,
            notes=notes,
        )

    # --- Kısmi TP yapısı ---

    def partial_tp_structure(
        self,
        entry: float,
        stop_loss: float,
        direction: str,
        tp1_r: float = 1.5,
        tp2_r: float = 2.5,
        tp1_split: float = 0.50,
    ) -> PartialTPStructure:
        """
        TP1'de pozisyonun %50'sini kapat, geri kalanını TP2'ye taşı.
        BE: TP1 vurulunca SL entry'ye çekilir.
        """
        risk = abs(entry - stop_loss) or entry * 0.001

        if direction.lower() == "long":
            tp1 = round(entry + tp1_r * risk, 5)
            tp2 = round(entry + tp2_r * risk, 5)
            be  = round(entry + risk * 0.20, 5)  # %20 karda BE
        else:
            tp1 = round(entry - tp1_r * risk, 5)
            tp2 = round(entry - tp2_r * risk, 5)
            be  = round(entry - risk * 0.20, 5)

        # Ağırlıklı beklenen R/R
        # TP1: tp1_split * tp1_r olasılıklı kazanç
        # TP2: (1-tp1_split) * tp2_r olasılıklı kazanç (BE'den sonra)
        exp_rr = round(tp1_split * tp1_r + (1 - tp1_split) * tp2_r, 2)

        return PartialTPStructure(
            entry=entry,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            tp1_size_pct=tp1_split,
            tp2_size_pct=1.0 - tp1_split,
            breakeven_at=be,
            expected_rr=exp_rr,
        )

    # --- ATR dinamik seviyeleri ---

    def get_dynamic_levels(
        self,
        symbol: str,
        entry: float,
        direction: str,
        atr: float,
    ) -> DynamicLevels:
        """ATR'ye göre enstrüman özelinde SL/TP hesapla."""
        mults = _ATR_MULTIPLIERS.get(symbol.upper(), _ATR_MULTIPLIERS["_default"])
        sl_mult  = mults["sl"]
        tp1_mult = mults["tp1"]
        tp2_mult = mults["tp2"]

        sl_dist  = round(atr * sl_mult,  5)
        tp1_dist = round(atr * tp1_mult, 5)
        tp2_dist = round(atr * tp2_mult, 5)
        rr_ratio = round(tp1_dist / sl_dist, 2) if sl_dist > 0 else 0.0

        return DynamicLevels(
            symbol=symbol,
            atr=atr,
            sl_distance=sl_dist,
            tp1_distance=tp1_dist,
            tp2_distance=tp2_dist,
            sl_multiplier=sl_mult,
            rr_ratio=rr_ratio,
        )

    # --- Monte Carlo simülasyonu ---

    def monte_carlo(
        self,
        win_rate: float = 0.55,
        avg_win_r: float = 2.0,
        avg_loss_r: float = 1.0,
        risk_per_trade: float = 0.01,
        trades_per_year: int = 100,
        n_simulations: int = 4000,
    ) -> MonteCarloResult:
        """
        Monte Carlo ile yıllık getiri ve max drawdown dağılımı.
        Gerçek trade geçmişi yoksa win_rate + R parametrelerinden simüle eder.
        """
        # Trade sonuç havuzu oluştur
        n_wins  = int(win_rate * 100)
        n_loss  = 100 - n_wins
        pool    = [avg_win_r] * n_wins + [-avg_loss_r] * n_loss

        all_returns:  list[float] = []
        all_max_dds:  list[float] = []
        ruin_count = 0

        for _ in range(n_simulations):
            sampled = random.choices(pool, k=trades_per_year)
            equity  = 1.0
            peak    = 1.0
            max_dd  = 0.0

            for r in sampled:
                equity = equity * (1.0 + r * risk_per_trade)
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd

            all_returns.append(equity - 1.0)
            all_max_dds.append(max_dd)
            if equity < 0.50:
                ruin_count += 1

        all_returns.sort()
        all_max_dds.sort()
        n = len(all_returns)

        median_ret   = all_returns[n // 2]
        pct5_ret     = all_returns[int(n * 0.05)]
        pct95_ret    = all_returns[int(n * 0.95)]
        dd_median    = all_max_dds[n // 2]
        dd_pct95     = all_max_dds[int(n * 0.95)]
        ruin_prob    = ruin_count / n_simulations

        avg_r = sum(all_returns) / n
        std_r = math.sqrt(sum((r - avg_r) ** 2 for r in all_returns) / n) or 1e-9
        sharpe = round(avg_r / std_r, 3)

        # Öneri
        if ruin_prob > 0.05 or dd_pct95 > 0.30:
            rec = "DUR — Risk cok yuksek"
        elif dd_pct95 > 0.20:
            rec = "DIKKAT — Risk azalt"
        else:
            rec = "DEVAM — Risk kabul edilebilir"

        return MonteCarloResult(
            median_return=round(median_ret, 4),
            pct5_return=round(pct5_ret, 4),
            pct95_return=round(pct95_ret, 4),
            max_dd_median=round(dd_median, 4),
            max_dd_pct95=round(dd_pct95, 4),
            ruin_probability=round(ruin_prob, 4),
            sharpe_ratio=sharpe,
            expected_trades_per_year=trades_per_year,
            recommendation=rec,
        )


# Global singleton
risk_manager = RiskManager()
