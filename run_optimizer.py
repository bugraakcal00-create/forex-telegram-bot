"""
Hizli optimizer: Her sembol icin her stratejiyi her TF'de test eder.
Her sembol icin yeni httpx client kullanir.
"""
from __future__ import annotations
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx
import pandas as pd
from app.services.analysis_engine import AnalysisEngine, STRATEGY_MODES
from app.services.backtest_service import BacktestService
from app.config import settings

# TwelveData API 30 istekten sonra hang yapiyor
# Iki turda calistirilir: ilk 3 + son 3
SYMBOLS = ["XAUUSD", "BTCUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF"]
# Sadece ilk 3'u calistir (ikinci tur icin yorum satirini degistirin)
# SYMBOLS = ["XAUUSD", "BTCUSD", "EURUSD"]
# SYMBOLS = ["GBPUSD", "USDJPY", "USDCHF"]
# 4h atlandı: higher TF=1day fetch TwelveData'da hang yapıyor
TIMEFRAMES = ["5min", "15min", "30min", "1h"]
HIGHER_TF = {
    "5min": "15min", "15min": "1h", "30min": "1h",
    "1h": "4h", "4h": "1day",
}

SYM_MAP = {
    "XAUUSD": "XAU/USD", "BTCUSD": "BTC/USD", "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY", "USDCHF": "USD/CHF",
    "AUDUSD": "AUD/USD",
}


def _fetch_sync(symbol: str, interval: str, outputsize: int = 800) -> pd.DataFrame:
    """Senkron veri cekme - urllib ile, thread-safe, timeout korumalı."""
    import urllib.request
    import urllib.parse
    normalized = SYM_MAP.get(symbol.upper(), symbol)
    params = urllib.parse.urlencode({
        "symbol": normalized,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": settings.twelvedata_api_key,
        "format": "JSON",
    })
    url = f"https://api.twelvedata.com/time_series?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 ForexBot/2.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)

    if "values" not in data:
        raise Exception(data.get("message", "Veri yok"))

    df = pd.DataFrame(data["values"])
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True) if "datetime" in df.columns else df.iloc[::-1].reset_index(drop=True)
    return df


async def fetch_raw(symbol: str, interval: str, outputsize: int = 800) -> pd.DataFrame:
    """Async wrapper - senkron fetch'i executor'da calistirir."""
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(None, _fetch_sync, symbol, interval, outputsize),
        timeout=20,
    )


async def main():
    engine = AnalysisEngine()
    bt_svc = BacktestService(engine=engine)

    all_results = []
    total = len(SYMBOLS) * len(TIMEFRAMES) * len(STRATEGY_MODES)

    data_cache: dict[str, dict] = {}

    print("=" * 80)
    print(f"  STRATEJI OPTIMIZER - {len(SYMBOLS)} sembol x {len(TIMEFRAMES)} TF x {len(STRATEGY_MODES)} strateji = {total} test")
    print("=" * 80)
    # Her sembol icin benzersiz TF'leri belirle (ortaklastir)
    needed_tfs: dict[str, set[str]] = {}
    for symbol in SYMBOLS:
        s = set()
        for tf in TIMEFRAMES:
            s.add(tf)
            s.add(HIGHER_TF.get(tf, "1h"))
        needed_tfs[symbol] = s

    total_fetches = sum(len(v) for v in needed_tfs.values())
    print(f"\n[1/2] Veri indiriliyor ({total_fetches} benzersiz fetch, ~10s arasi)...")

    raw_cache: dict[str, dict[str, pd.DataFrame]] = {}

    for symbol in SYMBOLS:
        raw_cache[symbol] = {}
        for tf in sorted(needed_tfs[symbol]):
            sys.stdout.write(f"  {symbol} {tf}... ")
            sys.stdout.flush()
            try:
                df = await fetch_raw(symbol, tf, 800)
                raw_cache[symbol][tf] = df
                print(f"{len(df)} bar OK")
            except asyncio.TimeoutError:
                print("TIMEOUT")
            except Exception as e:
                err = str(e)[:60]
                if "credit" in err.lower() or "limit" in err.lower():
                    print(f"RATE LIMIT - bekleniyor...")
                    await asyncio.sleep(65)
                    try:
                        df = await fetch_raw(symbol, tf, 800)
                        raw_cache[symbol][tf] = df
                        print(f"  -> Retry OK: {len(df)} bar")
                    except Exception:
                        print(f"  -> Retry HATA")
                else:
                    print(f"HATA: {err}")
            await asyncio.sleep(10)

    # Cache'den data_cache'e aktar
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            htf = HIGHER_TF.get(tf, "1h")
            key = f"{symbol}_{tf}"
            if tf in raw_cache.get(symbol, {}) and htf in raw_cache.get(symbol, {}):
                data_cache[key] = {"df": raw_cache[symbol][tf], "hdf": raw_cache[symbol][htf]}

    print(f"\n  {len(data_cache)}/{len(SYMBOLS)*len(TIMEFRAMES)} veri seti indirildi")
    print(f"\n[2/2] Backtest basliyor...\n")

    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"  {symbol}")
        print(f"{'='*60}")

        for tf in TIMEFRAMES:
            key = f"{symbol}_{tf}"
            if key not in data_cache:
                continue

            df = data_cache[key]["df"]
            hdf = data_cache[key]["hdf"]

            if len(df) < 250:
                print(f"  {tf}: Yetersiz veri ({len(df)} bar)")
                continue

            print(f"\n  {tf} ({len(df)} bar):")

            for mode in STRATEGY_MODES:
                t0 = time.time()
                try:
                    result = bt_svc.run(
                        symbol=symbol, timeframe=tf, df=df,
                        higher_df=hdf, strategy_mode=mode,
                    )
                    elapsed = time.time() - t0
                    r = {
                        "symbol": symbol,
                        "timeframe": tf,
                        "strategy": mode,
                        "signals": result.tested_signals,
                        "wins": result.wins,
                        "losses": result.losses,
                        "wr": result.winrate,
                        "avg_rr": result.avg_rr,
                        "exp": result.expectancy,
                        "pf": result.profit_factor,
                        "sharpe": result.sharpe_ratio,
                        "mdd": result.max_drawdown_pct,
                        "mcl": result.max_consecutive_losses,
                    }
                    all_results.append(r)
                    sig = result.tested_signals
                    if sig > 0:
                        wr_mark = "*" if result.winrate >= 55 else (" " if result.winrate >= 45 else "!")
                        print(f"    {wr_mark} {mode:12s} WR:{result.winrate:5.1f}% "
                              f"Exp:{result.expectancy:+.3f} PF:{result.profit_factor:.2f} "
                              f"Sig:{sig:3d} MDD:{result.max_drawdown_pct:.0f}% ({elapsed:.1f}s)")
                    else:
                        print(f"      {mode:12s} Sinyal yok ({elapsed:.1f}s)")
                except Exception as e:
                    print(f"    X {mode:12s} HATA: {e}")

    # Sonuclari kaydet
    Path("data").mkdir(exist_ok=True)
    with open("data/optimization_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # En iyi stratejiyi bul - her sembol/TF icin
    best_strategies = {}
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            candidates = [r for r in all_results
                         if r["symbol"] == symbol and r["timeframe"] == tf
                         and r["signals"] >= 3 and r["wr"] > 0]
            if candidates:
                for c in candidates:
                    c["score"] = round(c["wr"] * (c["signals"] ** 0.3) * max(0.1, 1 + c["exp"]), 2)
                best = max(candidates, key=lambda x: x["score"])
                best_strategies[f"{symbol}_{tf}"] = best

    with open("data/best_strategies.json", "w", encoding="utf-8") as f:
        json.dump(best_strategies, f, indent=2, ensure_ascii=False)

    # OZET RAPOR
    print("\n\n" + "=" * 80)
    print("  EN IYI STRATEJILER (Sembol bazli)")
    print("=" * 80)

    for symbol in SYMBOLS:
        print(f"\n  {symbol}:")
        sym_results = [(k, v) for k, v in best_strategies.items() if v["symbol"] == symbol]
        sym_results.sort(key=lambda x: x[1]["score"], reverse=True)
        for key, val in sym_results:
            tf = val["timeframe"]
            print(f"    {tf:6s} -> {val['strategy']:12s} WR:{val['wr']:5.1f}% "
                  f"Exp:{val['exp']:+.3f} PF:{val['pf']:.2f} Sig:{val['signals']:3d} "
                  f"Score:{val['score']:.1f}")

    all_best = sorted(best_strategies.values(), key=lambda x: x.get("score", 0), reverse=True)
    print(f"\n{'='*80}")
    print("  TOP 20 EN IYI KOMBINASYONLAR")
    print(f"{'='*80}")
    for i, r in enumerate(all_best[:20], 1):
        print(f"  {i:2d}. {r['symbol']:8s} {r['timeframe']:6s} {r['strategy']:12s} "
              f"WR:{r['wr']:5.1f}% Exp:{r['exp']:+.3f} PF:{r['pf']:.2f} Sig:{r['signals']:3d}")

    print(f"\n  Toplam test: {len(all_results)}")
    print(f"  Sonuc dosyalari: data/optimization_results.json, data/best_strategies.json")


if __name__ == "__main__":
    asyncio.run(main())
