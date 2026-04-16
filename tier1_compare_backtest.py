"""TIER-1 sonrası backtest — gerçekçi cost + filter'larla WR/PF/Sharpe raporu."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.services.analysis_engine import AnalysisEngine
from app.services.backtest_service import BacktestService, _exec_cost
from app.services.market_data import MarketDataClient

TEST_CASES = [
    ("XAUUSD", "30min", "default"),
    ("XAUUSD", "1h", "ichimoku"),
    ("USDJPY", "30min", "multi_ma"),
    ("EURUSD", "30min", "default"),
]


async def main():
    client = MarketDataClient(api_key=settings.twelvedata_api_key)
    engine = AnalysisEngine()
    service = BacktestService(engine)

    out = {}
    for symbol, tf, mode in TEST_CASES:
        print(f"\n=== {symbol} {tf} [{mode}] ===", flush=True)
        try:
            htf = {"15min": "1h", "30min": "1h", "1h": "4h", "5min": "15min"}.get(tf, "1h")
            df = await client.fetch_candles(symbol, interval=tf, outputsize=2000)
            hdf = await client.fetch_candles(symbol, interval=htf, outputsize=2000)
        except Exception as e:
            print(f"  fetch failed: {e}", flush=True)
            continue

        r = service.run(symbol=symbol, timeframe=tf, df=df, higher_df=hdf, strategy_mode=mode)
        resolved = r.wins + r.losses
        resolved_wr = (r.wins / resolved * 100) if resolved else 0.0

        cost = _exec_cost(symbol)
        long_count = sum(1 for t in r.trade_log if t.get("signal") == "LONG")
        short_count = sum(1 for t in r.trade_log if t.get("signal") == "SHORT")

        print(f"  Toplam sinyal:      {r.tested_signals}", flush=True)
        print(f"  LONG / SHORT:       {long_count} / {short_count}", flush=True)
        print(f"  Win / Loss / NoRes: {r.wins} / {r.losses} / {r.no_result}", flush=True)
        print(f"  WR (tum):           {r.winrate}%", flush=True)
        print(f"  WR (resolved):      {resolved_wr:.2f}%  CI=({r.wr_ci_low},{r.wr_ci_high})", flush=True)
        print(f"  PF / AvgRR:         {r.profit_factor} / {r.avg_rr}", flush=True)
        print(f"  Sharpe / MaxDD:     {r.sharpe_ratio} / {r.max_drawdown_pct}%", flush=True)
        print(f"  Exec cost / trade:  {cost:.5f}", flush=True)
        print(f"  Skipped weekend:    {r.weekend_skipped}", flush=True)
        print(f"  Skipped news:       {r.news_skipped}", flush=True)

        out[f"{symbol}_{tf}_{mode}"] = {
            "tested": r.tested_signals,
            "long": long_count, "short": short_count,
            "wins": r.wins, "losses": r.losses, "no_result": r.no_result,
            "wr": r.winrate, "wr_resolved": round(resolved_wr, 2),
            "wr_ci_low": r.wr_ci_low, "wr_ci_high": r.wr_ci_high,
            "pf": r.profit_factor, "sharpe": r.sharpe_ratio,
            "max_dd": r.max_drawdown_pct, "cost": cost,
            "weekend_skipped": r.weekend_skipped, "news_skipped": r.news_skipped,
        }

    Path("data/tier1_backtest_report.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== OZET ===", flush=True)
    for k, v in out.items():
        print(f"  {k}: WR {v['wr_resolved']}% (CI {v['wr_ci_low']}-{v['wr_ci_high']}), PF {v['pf']}, L/S: {v['long']}/{v['short']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
