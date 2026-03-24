from __future__ import annotations

from dataclasses import dataclass, field
from math import isnan

import numpy as np
import pandas as pd


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
    setup_score: int
    quality: str
    sweep_signal: str
    sniper_entry: str
    no_trade_reasons: list[str] = field(default_factory=list)


class AnalysisEngine:
    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()

    @staticmethod
    def _cluster_levels(values: list[float], tolerance_ratio: float = 0.0025) -> list[float]:
        if not values:
            return []
        sorted_vals = sorted(values)
        clusters: list[list[float]] = [[sorted_vals[0]]]
        for value in sorted_vals[1:]:
            anchor = np.mean(clusters[-1])
            if abs(value - anchor) / anchor <= tolerance_ratio:
                clusters[-1].append(value)
            else:
                clusters.append([value])
        return [round(float(np.mean(cluster)), 5) for cluster in clusters]

    @staticmethod
    def _find_levels(df: pd.DataFrame, lookback: int = 80) -> tuple[list[float], list[float]]:
        window = df.tail(lookback).reset_index(drop=True)
        supports: list[float] = []
        resistances: list[float] = []

        for i in range(2, len(window) - 2):
            low = window.loc[i, "low"]
            if (
                low <= window.loc[i - 1, "low"]
                and low <= window.loc[i - 2, "low"]
                and low <= window.loc[i + 1, "low"]
                and low <= window.loc[i + 2, "low"]
            ):
                supports.append(float(low))

            high = window.loc[i, "high"]
            if (
                high >= window.loc[i - 1, "high"]
                and high >= window.loc[i - 2, "high"]
                and high >= window.loc[i + 1, "high"]
                and high >= window.loc[i + 2, "high"]
            ):
                resistances.append(float(high))

        return AnalysisEngine._cluster_levels(supports), AnalysisEngine._cluster_levels(resistances)

    @staticmethod
    def _trend_from_df(df: pd.DataFrame) -> str:
        data = df.copy()
        data["ema_20"] = data["close"].ewm(span=20, adjust=False).mean()
        data["ema_50"] = data["close"].ewm(span=50, adjust=False).mean()
        last = data.iloc[-1]
        return "Yukarı" if float(last["ema_20"]) > float(last["ema_50"]) else "Aşağı"

    @staticmethod
    def _quality_from_score(score: int) -> str:
        if score >= 85:
            return "A"
        if score >= 70:
            return "B"
        if score >= 55:
            return "C"
        return "D"

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
    def _bos(df: pd.DataFrame) -> bool:
        if len(df) < 5:
            return False
        last = df.iloc[-1]
        prev = df.iloc[-2]
        return bool(float(last["high"]) > float(prev["high"]) or float(last["low"]) < float(prev["low"]))

    def _detect_liquidity_sweep(self, df: pd.DataFrame, supports: list[float], resistances: list[float]) -> str:
        if len(df) < 3:
            return "Yok"

        last = df.iloc[-1]
        prev = df.iloc[-2]

        if resistances:
            nearest_res = min(resistances, key=lambda x: abs(x - float(last["close"])))
            wick_above = float(last["high"]) > nearest_res
            close_back_below = float(last["close"]) < nearest_res
            bearish_close = float(last["close"]) < float(last["open"])
            long_upper_wick = self._upper_wick(last) > self._body_size(last) * 1.2
            if wick_above and close_back_below and bearish_close and long_upper_wick:
                return "Direnç üstü likidite alınıp geri dönüldü"

        if supports:
            nearest_sup = min(supports, key=lambda x: abs(x - float(last["close"])))
            wick_below = float(last["low"]) < nearest_sup
            close_back_above = float(last["close"]) > nearest_sup
            bullish_close = float(last["close"]) > float(last["open"])
            long_lower_wick = self._lower_wick(last) > self._body_size(last) * 1.2
            if wick_below and close_back_above and bullish_close and long_lower_wick:
                return "Destek altı likidite alınıp geri dönüldü"

        if float(last["high"]) > float(prev["high"]) and float(last["close"]) < float(prev["high"]):
            return "Yukarı fake breakout ihtimali"
        if float(last["low"]) < float(prev["low"]) and float(last["close"]) > float(prev["low"]):
            return "Aşağı fake breakout ihtimali"

        return "Yok"

    def _detect_sniper_entry(
        self,
        df: pd.DataFrame,
        trend: str,
        higher_tf_trend: str,
        supports: list[float],
        resistances: list[float],
    ) -> str:
        if len(df) < 3:
            return "Yok"

        last = df.iloc[-1]
        prev = df.iloc[-2]
        bos = self._bos(df)

        if trend == "Aşağı" and higher_tf_trend == "Aşağı" and resistances:
            nearest_res = min(resistances, key=lambda x: abs(x - float(last["close"])))
            touched_above = float(last["high"]) >= nearest_res
            closed_below = float(last["close"]) < nearest_res
            bearish_candle = float(last["close"]) < float(last["open"])
            wick_rejection = self._upper_wick(last) > self._body_size(last) * 1.5
            fake_break = float(last["high"]) > float(prev["high"]) and float(last["close"]) < float(prev["high"])
            if touched_above and closed_below and bearish_candle and (wick_rejection or fake_break) and bos:
                return "SHORT sniper: direnç sweep + rejection"

        if trend == "Yukarı" and higher_tf_trend == "Yukarı" and supports:
            nearest_sup = min(supports, key=lambda x: abs(x - float(last["close"])))
            touched_below = float(last["low"]) <= nearest_sup
            closed_above = float(last["close"]) > nearest_sup
            bullish_candle = float(last["close"]) > float(last["open"])
            wick_rejection = self._lower_wick(last) > self._body_size(last) * 1.5
            fake_break = float(last["low"]) < float(prev["low"]) and float(last["close"]) > float(prev["low"])
            if touched_below and closed_above and bullish_candle and (wick_rejection or fake_break) and bos:
                return "LONG sniper: destek sweep + rejection"

        return "Yok"

    def analyze(
        self,
        symbol: str,
        df: pd.DataFrame,
        timeframe: str = "15min",
        higher_tf_df: pd.DataFrame | None = None,
        high_impact_events: list[dict[str, str]] | None = None,
    ) -> AnalysisResult:
        data = df.copy()
        data["ema_20"] = data["close"].ewm(span=20, adjust=False).mean()
        data["ema_50"] = data["close"].ewm(span=50, adjust=False).mean()
        data["ema_200"] = data["close"].ewm(span=200, adjust=False).mean()
        data["rsi"] = self._rsi(data["close"])
        data["atr"] = self._atr(data)

        last = data.iloc[-1]
        current_price = float(last["close"])
        ema20 = float(last["ema_20"])
        ema50 = float(last["ema_50"])
        ema200 = float(last["ema_200"]) if not isnan(float(last["ema_200"])) else current_price
        rsi = float(last["rsi"]) if not isnan(float(last["rsi"])) else 50.0
        atr = float(last["atr"]) if not isnan(float(last["atr"])) else max(current_price * 0.002, 0.001)

        supports, resistances = self._find_levels(data)
        nearest_supports = [lvl for lvl in supports if lvl < current_price][-3:]
        nearest_resistances = [lvl for lvl in resistances if lvl > current_price][:3]

        lower_support = nearest_supports[-1] if nearest_supports else round(current_price - atr * 1.5, 5)
        upper_resistance = nearest_resistances[0] if nearest_resistances else round(current_price + atr * 1.5, 5)

        trend = "Yukarı" if ema20 > ema50 else "Aşağı"
        higher_tf_trend = self._trend_from_df(higher_tf_df) if higher_tf_df is not None else trend

        price_to_support = abs(current_price - lower_support)
        price_to_resistance = abs(upper_resistance - current_price)
        near_support = price_to_support <= atr * 0.9
        near_resistance = price_to_resistance <= atr * 0.9

        sweep_signal = self._detect_liquidity_sweep(data.tail(10), supports, resistances)
        sniper_entry = self._detect_sniper_entry(data.tail(10), trend, higher_tf_trend, supports, resistances)

        signal = "NO TRADE"
        reason = "Kaliteli scalp kurulum oluşmadı."
        no_trade_reasons: list[str] = []

        entry_zone = (round(current_price - atr * 0.2, 5), round(current_price + atr * 0.2, 5))
        stop_loss = round(current_price - atr, 5)
        take_profit = round(current_price + atr, 5)
        take_profit_2 = round(current_price + atr * 1.5, 5)

        long_ok = (
            trend == "Yukarı"
            and higher_tf_trend == "Yukarı"
            and current_price > ema200
            and 50 <= rsi <= 62
            and near_support
        )

        short_ok = (
            trend == "Aşağı"
            and higher_tf_trend == "Aşağı"
            and current_price < ema200
            and 38 <= rsi <= 50
            and near_resistance
        )

        if long_ok:
            signal = "LONG"
            reason = "Scalp long kurulumu: trend yukarı, üst TF onaylı, fiyat destek yakınında."
            entry_zone = (
                round(lower_support + atr * 0.10, 5),
                round(lower_support + atr * 0.25, 5),
            )
            stop_loss = round(lower_support - atr * 0.35, 5)
            risk = abs(np.mean(entry_zone) - stop_loss)
            take_profit = round(max(upper_resistance, np.mean(entry_zone) + risk * 2.0), 5)
            take_profit_2 = round(max(take_profit + atr * 0.8, np.mean(entry_zone) + risk * 2.8), 5)

        elif short_ok:
            signal = "SHORT"
            reason = "Scalp short kurulumu: trend aşağı, üst TF onaylı, fiyat direnç yakınında."
            entry_zone = (
                round(upper_resistance - atr * 0.25, 5),
                round(upper_resistance - atr * 0.10, 5),
            )
            stop_loss = round(upper_resistance + atr * 0.35, 5)
            risk = abs(stop_loss - np.mean(entry_zone))
            take_profit = round(min(lower_support, np.mean(entry_zone) - risk * 2.0), 5)
            take_profit_2 = round(min(take_profit - atr * 0.8, np.mean(entry_zone) - risk * 2.8), 5)

        risk = abs(np.mean(entry_zone) - stop_loss)
        reward = abs(take_profit - np.mean(entry_zone))
        rr_ratio = round(reward / risk, 2) if risk else 0.0

        atr_ratio = atr / current_price if current_price else 0.0

        if atr_ratio < 0.0012:
            no_trade_reasons.append("Scalp için volatilite düşük")
        if rr_ratio < 1.8:
            no_trade_reasons.append("R/R yetersiz")
        if rsi > 65 or rsi < 35:
            no_trade_reasons.append("RSI aşırı bölgede")
        if signal == "LONG" and not near_support:
            no_trade_reasons.append("Fiyat destek bölgesine yakın değil")
        if signal == "SHORT" and not near_resistance:
            no_trade_reasons.append("Fiyat direnç bölgesine yakın değil")
        if signal != "NO TRADE" and trend != higher_tf_trend:
            no_trade_reasons.append("Üst zaman dilimi trend onayı yok")
        if high_impact_events:
            no_trade_reasons.append("Yüksek etkili haber riski")
        if signal == "NO TRADE":
            no_trade_reasons.append("Ana kurulum şartları oluşmadı")

        score = 0
        if trend == higher_tf_trend:
            score += 20
        if signal in {"LONG", "SHORT"}:
            score += 20
        if rr_ratio >= 2.2:
            score += 20
        elif rr_ratio >= 2.0:
            score += 16
        elif rr_ratio >= 1.8:
            score += 12
        if atr_ratio >= 0.0012:
            score += 10
        if signal == "LONG" and near_support:
            score += 10
        if signal == "SHORT" and near_resistance:
            score += 10
        if signal == "LONG" and 52 <= rsi <= 60:
            score += 8
        if signal == "SHORT" and 40 <= rsi <= 48:
            score += 8
        if sweep_signal != "Yok":
            score += 12
        if sniper_entry != "Yok":
            score += 15
        if not high_impact_events:
            score += 10

        score = max(0, min(100, score))
        quality = self._quality_from_score(score)

        if no_trade_reasons and (score < 70 or signal == "NO TRADE"):
            signal = "NO TRADE"
            reason = " | ".join(no_trade_reasons)
        elif no_trade_reasons:
            reason = f"{reason} Riskler: {' | '.join(no_trade_reasons)}"

        if sweep_signal != "Yok":
            reason = f"{reason} | Sweep: {sweep_signal}"
        if sniper_entry != "Yok":
            reason = f"{reason} | Sniper: {sniper_entry}"

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
            setup_score=score,
            quality=quality,
            sweep_signal=sweep_signal,
            sniper_entry=sniper_entry,
            no_trade_reasons=no_trade_reasons,
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
            f"Entry: {entry:.5f}\n"
            f"SL: {stop_loss:.5f}\n"
            f"Stop mesafesi: {stop_distance:.5f}\n"
            f"Teorik pozisyon büyüklüğü: {units:.2f} birim\n"
            f"Not: Lot hesaplaması broker sözleşme boyutuna göre ayrıca uyarlanmalı."
        )