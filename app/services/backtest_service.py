from __future__ import annotations

from dataclasses import dataclass

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


class BacktestService:
    def __init__(self, engine: AnalysisEngine) -> None:
        self.engine = engine

    def run(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        higher_df: pd.DataFrame,
        lookahead_bars: int = 12,
        warmup_bars: int = 220,
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
            )

            if result.signal not in {"LONG", "SHORT"}:
                continue

            future = df.iloc[idx + 1 : idx + 1 + lookahead_bars]
            outcome = self._evaluate_trade(result.signal, result.stop_loss, result.take_profit, future)

            if outcome == "win":
                wins += 1
                rr_values.append(float(result.rr_ratio))
            elif outcome == "loss":
                losses += 1
                rr_values.append(-1.0)
            else:
                no_result += 1

        tested_signals = wins + losses + no_result
        winrate = round((wins / tested_signals) * 100, 2) if tested_signals else 0.0
        avg_rr = round(sum(rr_values) / len(rr_values), 3) if rr_values else 0.0
        expectancy = round(sum(rr_values) / tested_signals, 3) if tested_signals else 0.0

        return BacktestResult(
            tested_signals=tested_signals,
            wins=wins,
            losses=losses,
            no_result=no_result,
            winrate=winrate,
            avg_rr=avg_rr,
            expectancy=expectancy,
        )

    @staticmethod
    def _evaluate_trade(signal: str, stop_loss: float, take_profit: float, future: pd.DataFrame) -> str:
        for _, candle in future.iterrows():
            high = float(candle["high"])
            low = float(candle["low"])
            if signal == "LONG":
                if low <= stop_loss:
                    return "loss"
                if high >= take_profit:
                    return "win"
            if signal == "SHORT":
                if high >= stop_loss:
                    return "loss"
                if low <= take_profit:
                    return "win"
        return "no_result"
