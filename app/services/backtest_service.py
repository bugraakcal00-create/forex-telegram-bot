from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import pandas as pd

from app.services.analysis_engine import AnalysisEngine


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
        rr_values: list[float] = []
        trade_log: list[dict] = []

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

            future = df.iloc[idx + 1 : idx + 1 + lookahead_bars]
            outcome = self._evaluate_trade(result.signal, result.stop_loss, result.take_profit, future)

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
        )

    @staticmethod
    def _evaluate_trade(signal: str, stop_loss: float, take_profit: float, future: pd.DataFrame) -> str:
        for _, candle in future.iterrows():
            high = float(candle["high"])
            low = float(candle["low"])
            open_price = float(candle["open"])
            if signal == "LONG":
                tp_hit = high >= take_profit
                sl_hit = low <= stop_loss
            elif signal == "SHORT":
                tp_hit = low <= take_profit
                sl_hit = high >= stop_loss
            else:
                continue

            if tp_hit and sl_hit:
                dist_tp = abs(open_price - take_profit)
                dist_sl = abs(open_price - stop_loss)
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
