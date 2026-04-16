from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from app.services.analysis_engine import AnalysisEngine


# ── Execution cost modeli: spread + slippage (pips) per-sembol ────────────
_SPREAD_PIPS = {
    "XAUUSD": 1.5, "XAGUSD": 3.0,
    "EURUSD": 0.8, "GBPUSD": 1.2, "USDCHF": 1.2, "AUDUSD": 1.0, "NZDUSD": 1.5,
    "USDCAD": 1.2, "USDJPY": 1.0, "EURJPY": 1.5, "GBPJPY": 2.0,
    "BTCUSD": 5.0,
}
_SLIPPAGE_PIPS = 0.4  # giriş + çıkışta toplam slippage

_PIP_SIZE = {
    "XAUUSD": 0.1, "XAGUSD": 0.01,
    "USDJPY": 0.01, "EURJPY": 0.01, "GBPJPY": 0.01,
    "BTCUSD": 1.0,
}


def _exec_cost(symbol: str) -> float:
    """Tek yönlü fiyat cinsinden maliyet (spread + slippage)."""
    sym = symbol.upper()
    pip = _PIP_SIZE.get(sym, 0.0001)
    spread = _SPREAD_PIPS.get(sym, 1.0)
    return (spread + _SLIPPAGE_PIPS) * pip


def _spans_weekend(entry_dt) -> bool:
    """Trade açılış zamanı hafta sonu likidite boşluğunda mı?"""
    if entry_dt is None:
        return False
    try:
        wd = entry_dt.weekday()  # 0=Mon, 6=Sun
        h = entry_dt.hour
    except Exception:
        return False
    if wd == 4 and h >= 20:  # Cuma 20:00 GMT sonrası
        return True
    if wd == 5:  # Cumartesi
        return True
    if wd == 6 and h < 22:  # Pazar 22:00 GMT öncesi
        return True
    return False


def bootstrap_wr_ci(wins: int, total: int, n_iter: int = 10000, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap resampling ile WR için %95 güven aralığı."""
    if total == 0:
        return (0.0, 0.0)
    outcomes = [1] * wins + [0] * (total - wins)
    wrs = []
    for _ in range(n_iter):
        sample = random.choices(outcomes, k=total)
        wrs.append(sum(sample) / total * 100.0)
    wrs.sort()
    lo = wrs[int((1 - ci) / 2 * n_iter)]
    hi = wrs[int((1 + ci) / 2 * n_iter)]
    return (round(lo, 2), round(hi, 2))


@dataclass
class BacktestResult:
    tested_signals: int
    wins: int
    losses: int
    no_result: int
    winrate: float
    avg_rr: float
    expectancy: float
    # --- v2 ek metrikler ---
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    max_consecutive_losses: int = 0
    monthly_returns: dict[str, float] = field(default_factory=dict)
    equity_curve: list[float] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)
    # --- Gerçekçilik metrikleri ---
    wr_ci_low: float = 0.0
    wr_ci_high: float = 0.0
    weekend_skipped: int = 0
    news_skipped: int = 0


class BacktestService:
    def __init__(self, engine: AnalysisEngine) -> None:
        self.engine = engine

    def run(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        higher_df: pd.DataFrame,
        lookahead_bars: int = 40,
        warmup_bars: int = 220,
        strategy_mode: str = "default",
        high_impact_events: list[dict] | None = None,
        news_blackout_minutes: int = 15,
    ) -> BacktestResult:
        if len(df) <= warmup_bars + 5:
            return BacktestResult(
                tested_signals=0,
                wins=0,
                losses=0,
                no_result=0,
                winrate=0.0,
                avg_rr=0.0,
                expectancy=0.0,
            )

        wins = 0
        losses = 0
        no_result = 0
        weekend_skipped = 0
        news_skipped = 0
        rr_values: list[float] = []
        trade_log: list[dict] = []

        # Haber zamanlarını tek seferde parse et
        news_times: list = []
        for ev in (high_impact_events or []):
            ev_time = ev.get("datetime") or ev.get("time") or ev.get("ts")
            if ev_time is not None:
                try:
                    news_times.append(pd.to_datetime(ev_time))
                except Exception:
                    continue
        blackout = timedelta(minutes=news_blackout_minutes)
        cost = _exec_cost(symbol)

        for idx in range(warmup_bars, len(df) - lookahead_bars):
            slice_df = df.iloc[: idx + 1].copy()
            if "datetime" in slice_df.columns and "datetime" in higher_df.columns:
                end_time = slice_df.iloc[-1]["datetime"]
                htf_slice = higher_df[higher_df["datetime"] <= end_time].copy()
            else:
                htf_slice = higher_df.iloc[: idx + 1].copy()
            result = self.engine.analyze(
                symbol=symbol,
                df=slice_df,
                timeframe=timeframe,
                higher_tf_df=htf_slice,
                high_impact_events=[],
                strategy_mode=strategy_mode,
            )

            if result.signal not in {"LONG", "SHORT"}:
                continue

            # Giriş zamanı
            entry_dt = None
            if "datetime" in df.columns:
                try:
                    entry_dt = pd.to_datetime(df.iloc[idx]["datetime"])
                except Exception:
                    entry_dt = None

            # Hafta sonu likidite boşluğu — trade atla
            if _spans_weekend(entry_dt):
                weekend_skipped += 1
                continue

            # Haber blackout — ±N dk içinde yüksek etkili event varsa atla
            if entry_dt is not None and news_times:
                if any(abs(nt - entry_dt) <= blackout for nt in news_times):
                    news_skipped += 1
                    continue

            future = df.iloc[idx + 1 : idx + 1 + lookahead_bars]
            outcome = self._evaluate_trade(
                result.signal, result.stop_loss, result.take_profit, future, cost=cost,
            )

            # Trade log kaydı
            trade_dt = ""
            if "datetime" in df.columns:
                trade_dt = str(df.iloc[idx]["datetime"])

            if outcome == "win":
                wins += 1
                rr_values.append(float(result.rr_ratio))
                trade_log.append({
                    "dt": trade_dt, "signal": result.signal, "quality": result.quality,
                    "score": result.setup_score, "rr": round(result.rr_ratio, 2),
                    "outcome": "win",
                })
            elif outcome == "loss":
                losses += 1
                rr_values.append(-1.0)
                trade_log.append({
                    "dt": trade_dt, "signal": result.signal, "quality": result.quality,
                    "score": result.setup_score, "rr": -1.0,
                    "outcome": "loss",
                })
            else:
                no_result += 1

        tested_signals = wins + losses + no_result
        winrate = round((wins / tested_signals) * 100, 2) if tested_signals else 0.0
        avg_rr = round(sum(rr_values) / len(rr_values), 3) if rr_values else 0.0
        expectancy = round(sum(rr_values) / tested_signals, 3) if tested_signals else 0.0

        # --- Gelismis metrikler ---
        equity_curve = _build_equity_curve(rr_values)
        sharpe_ratio = _calc_sharpe(rr_values)
        profit_factor = _calc_profit_factor(rr_values)
        max_drawdown_pct = _calc_max_drawdown(equity_curve)
        max_consecutive_losses = _calc_max_consec_losses(rr_values)
        monthly_returns = _calc_monthly_returns(trade_log)

        # Bootstrap %95 CI — sadece çözülmüş işlemler (win+loss)
        resolved = wins + losses
        wr_ci_low, wr_ci_high = bootstrap_wr_ci(wins, resolved) if resolved >= 10 else (0.0, 0.0)

        return BacktestResult(
            tested_signals=tested_signals,
            wins=wins,
            losses=losses,
            no_result=no_result,
            winrate=winrate,
            avg_rr=avg_rr,
            expectancy=expectancy,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            max_drawdown_pct=max_drawdown_pct,
            max_consecutive_losses=max_consecutive_losses,
            monthly_returns=monthly_returns,
            equity_curve=equity_curve,
            trade_log=trade_log,
            wr_ci_low=wr_ci_low,
            wr_ci_high=wr_ci_high,
            weekend_skipped=weekend_skipped,
            news_skipped=news_skipped,
        )

    @staticmethod
    def _evaluate_trade(
        signal: str,
        stop_loss: float,
        take_profit: float,
        future: pd.DataFrame,
        cost: float = 0.0,
    ) -> str:
        """Spread + slippage dahil gerçekçi TP/SL değerlendirmesi.
        LONG: TP zor, SL kolay (her ikisi de cost kadar yukarı kayar).
        SHORT: TP zor, SL kolay (her ikisi de cost kadar aşağı kayar).
        """
        if signal == "LONG":
            tp_eff = take_profit + cost
            sl_eff = stop_loss + cost
        elif signal == "SHORT":
            tp_eff = take_profit - cost
            sl_eff = stop_loss - cost
        else:
            tp_eff, sl_eff = take_profit, stop_loss

        for _, candle in future.iterrows():
            high = float(candle["high"])
            low = float(candle["low"])
            open_price = float(candle["open"])
            if signal == "LONG":
                tp_hit = high >= tp_eff
                sl_hit = low <= sl_eff
            elif signal == "SHORT":
                tp_hit = low <= tp_eff
                sl_hit = high >= sl_eff
            else:
                continue

            if tp_hit and sl_hit:
                dist_tp = abs(open_price - tp_eff)
                dist_sl = abs(open_price - sl_eff)
                return "win" if dist_tp <= dist_sl else "loss"
            if tp_hit:
                return "win"
            if sl_hit:
                return "loss"
        return "no_result"


# ── Yardımcı hesaplama fonksiyonları ─────────────────────────────────────────

def _build_equity_curve(rr_values: list[float]) -> list[float]:
    """Kümülatif RR eğrisi (başlangıç = 0)."""
    curve = [0.0]
    cumulative = 0.0
    for rr in rr_values:
        cumulative += rr
        curve.append(round(cumulative, 3))
    return curve


def _calc_sharpe(rr_values: list[float], annualization: float = 252.0) -> float:
    """Yıllıklaştırılmış Sharpe oranı."""
    if len(rr_values) < 2:
        return 0.0
    mean_rr = sum(rr_values) / len(rr_values)
    variance = sum((r - mean_rr) ** 2 for r in rr_values) / (len(rr_values) - 1)
    std_rr = math.sqrt(variance) if variance > 0 else 0.0
    if std_rr == 0:
        return 0.0
    return round((mean_rr / std_rr) * math.sqrt(annualization), 3)


def _calc_profit_factor(rr_values: list[float]) -> float:
    """Brüt kâr / brüt zarar."""
    gross_profit = sum(r for r in rr_values if r > 0)
    gross_loss = abs(sum(r for r in rr_values if r < 0))
    if gross_loss == 0:
        return round(gross_profit, 2) if gross_profit > 0 else 0.0
    return round(gross_profit / gross_loss, 3)


def _calc_max_drawdown(equity_curve: list[float]) -> float:
    """Equity curve'dan maksimum drawdown yüzdesi."""
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
    # Drawdown'u peak'e oranla göster
    if peak > 0:
        return round((max_dd / peak) * 100, 2)
    return round(max_dd, 2)


def _calc_max_consec_losses(rr_values: list[float]) -> int:
    """Maksimum ardışık kayıp sayısı."""
    max_streak = 0
    current = 0
    for rr in rr_values:
        if rr < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _calc_monthly_returns(trade_log: list[dict]) -> dict[str, float]:
    """Trade log'dan aylık toplam RR."""
    monthly: dict[str, float] = {}
    for t in trade_log:
        dt = t.get("dt", "")
        if len(dt) >= 7:
            month_key = dt[:7]  # YYYY-MM
        else:
            continue
        monthly[month_key] = round(monthly.get(month_key, 0.0) + float(t.get("rr", 0)), 3)
    return monthly
