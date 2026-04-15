"""
Strateji Optimizer: Tum semboller x timeframe x strategy_mode kombinasyonlarini
backtest ederek en iyi yapilandirmayi bulur.

Kullanim:
    python optimize_strategies.py

Sonuclar: data/optimization_results.json
En iyi config: data/best_strategies.json (bot tarafindan okunur)
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Proje kökünü path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.services.analysis_engine import AnalysisEngine, STRATEGY_MODES
from app.services.backtest_service import BacktestService
from app.services.market_data import MarketDataClient
from app.config import settings

SYMBOLS = ["XAUUSD", "BTCUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"]
TIMEFRAMES = ["5min", "15min", "30min", "1h", "4h"]
HIGHER_TF = {
    "5min": "15min", "15min": "1h", "30min": "1h",
    "1h": "4h", "4h": "1day",
}

OUTPUT_DIR = Path("data")
RESULTS_FILE = OUTPUT_DIR / "optimization_results.json"
BEST_FILE = OUTPUT_DIR / "best_strategies.json"


async def run_single_backtest(
    market: MarketDataClient,
    engine: AnalysisEngine,
    symbol: str,
    timeframe: str,
    strategy_mode: str,
) -> dict | None:
    """Tek bir sembol/TF/strateji kombinasyonunu backtest et."""
    higher_tf = HIGHER_TF.get(timeframe, "1h")
    try:
        df, higher_df = await asyncio.gather(
            market.fetch_candles(symbol, interval=timeframe, outputsize=2000),
            market.fetch_candles(symbol, interval=higher_tf, outputsize=2000),
        )
    except Exception as e:
        print(f"  [HATA] Veri alinamadi: {symbol}/{timeframe} - {e}")
        return None

    if len(df) < 250:
        print(f"  [ATLA] Yetersiz veri: {symbol}/{timeframe} ({len(df)} bar)")
        return None

    bt = BacktestService(engine=engine)
    try:
        result = bt.run(
            symbol=symbol,
            timeframe=timeframe,
            df=df,
            higher_df=higher_df,
            strategy_mode=strategy_mode,
        )
    except Exception as e:
        print(f"  [HATA] Backtest hatasi: {symbol}/{timeframe}/{strategy_mode} - {e}")
        return None

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy_mode": strategy_mode,
        "tested_signals": result.tested_signals,
        "wins": result.wins,
        "losses": result.losses,
        "no_result": result.no_result,
        "winrate": result.winrate,
        "avg_rr": result.avg_rr,
        "expectancy": result.expectancy,
        "sharpe_ratio": result.sharpe_ratio,
        "profit_factor": result.profit_factor,
        "max_drawdown_pct": result.max_drawdown_pct,
        "max_consecutive_losses": result.max_consecutive_losses,
    }


def score_result(r: dict) -> float:
    """Optimizasyon skoru: expectancy * sqrt(sinyal sayisi) * profit_factor bonus."""
    if r["tested_signals"] < 3:
        return -999
    signals = r["tested_signals"]
    wr = r["winrate"] / 100
    exp = r["expectancy"]
    pf = r["profit_factor"]
    mdd = r["max_drawdown_pct"]

    # Ana skor: expectancy * karekök(sinyal sayısı)
    base = exp * (signals ** 0.5)
    # Profit factor bonusu
    if pf > 1.5:
        base *= 1.2
    elif pf > 1.0:
        base *= 1.0
    else:
        base *= 0.5
    # Drawdown cezası
    if mdd > 50:
        base *= 0.5
    elif mdd > 30:
        base *= 0.7
    # WR bonusu
    if wr > 0.55:
        base *= 1.1
    elif wr < 0.35:
        base *= 0.6
    return round(base, 4)


async def main():
    print("=" * 70)
    print("  FOREX BOT STRATEJI OPTIMIZER")
    print(f"  {len(SYMBOLS)} sembol x {len(TIMEFRAMES)} TF x {len(STRATEGY_MODES)} strateji = {len(SYMBOLS)*len(TIMEFRAMES)*len(STRATEGY_MODES)} kombinasyon")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    market = MarketDataClient(api_key=settings.twelvedata_api_key)
    engine = AnalysisEngine()

    all_results: list[dict] = []
    total = len(SYMBOLS) * len(TIMEFRAMES) * len(STRATEGY_MODES)
    done = 0

    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            # Her sembol/TF icin tum stratejileri test et
            print(f"\n{'─'*50}")
            print(f"  {symbol} / {timeframe}")
            print(f"{'─'*50}")

            for mode in STRATEGY_MODES:
                done += 1
                print(f"  [{done}/{total}] {mode}...", end=" ", flush=True)
                start = time.time()

                result = await run_single_backtest(market, engine, symbol, timeframe, mode)
                elapsed = time.time() - start

                if result:
                    result["score"] = score_result(result)
                    all_results.append(result)
                    wr = result["winrate"]
                    exp = result["expectancy"]
                    sig = result["tested_signals"]
                    sc = result["score"]
                    print(f"WR:{wr:.1f}% Exp:{exp:.3f} Sig:{sig} Score:{sc:.2f} ({elapsed:.1f}s)")
                else:
                    print(f"ATLA ({elapsed:.1f}s)")

                # API rate limit: kisa bekleme
                await asyncio.sleep(1.5)

    # Sonuclari kaydet
    all_results.sort(key=lambda x: x.get("score", -999), reverse=True)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Tum sonuclar kaydedildi: {RESULTS_FILE}")

    # Her sembol/TF icin en iyi stratejiyi sec
    best_strategies: dict[str, dict] = {}
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            candidates = [r for r in all_results
                         if r["symbol"] == symbol and r["timeframe"] == timeframe
                         and r.get("score", -999) > 0]
            if candidates:
                best = max(candidates, key=lambda x: x.get("score", -999))
                key = f"{symbol}_{timeframe}"
                best_strategies[key] = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "strategy_mode": best["strategy_mode"],
                    "winrate": best["winrate"],
                    "expectancy": best["expectancy"],
                    "profit_factor": best["profit_factor"],
                    "tested_signals": best["tested_signals"],
                    "score": best["score"],
                }

    with open(BEST_FILE, "w", encoding="utf-8") as f:
        json.dump(best_strategies, f, indent=2, ensure_ascii=False)

    # Ozet rapor
    print("\n" + "=" * 70)
    print("  EN IYI STRATEJILER")
    print("=" * 70)
    for key, val in sorted(best_strategies.items()):
        print(f"  {val['symbol']:8s} {val['timeframe']:6s} → {val['strategy_mode']:12s} "
              f"WR:{val['winrate']:5.1f}% Exp:{val['expectancy']:.3f} PF:{val['profit_factor']:.2f} "
              f"Sig:{val['tested_signals']}")

    # Genel en iyi 10
    print(f"\n{'─'*70}")
    print("  TOP 10 GENEL (en yuksek skor)")
    print(f"{'─'*70}")
    for i, r in enumerate(all_results[:10], 1):
        print(f"  {i:2d}. {r['symbol']:8s} {r['timeframe']:6s} {r['strategy_mode']:12s} "
              f"WR:{r['winrate']:5.1f}% Exp:{r['expectancy']:.3f} PF:{r['profit_factor']:.2f} "
              f"Score:{r['score']:.2f}")

    print(f"\n  Toplam test: {len(all_results)}")
    print(f"  Sonuclar: {RESULTS_FILE}")
    print(f"  En iyiler: {BEST_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
