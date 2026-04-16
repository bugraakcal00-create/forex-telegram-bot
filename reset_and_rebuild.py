"""TIER-1 canli test oncesi: tum simulasyon/sinyal/islem verilerini sifirla + watchlist'i
yalnizca best_strategies.json'daki WR>=50% kombinasyonlarla yeniden insa et.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB = "data/bot.db"
BEST = "data/best_strategies.json"
CHAT_ID = 7817666361  # mevcut tek chat


def main():
    best = json.loads(Path(BEST).read_text(encoding="utf-8"))
    # WR >= 50 + signals >= 3 filtre
    qualifying = []
    for key, val in best.items():
        wr = float(val.get("wr", 0))
        sig = int(val.get("signals", 0))
        if wr >= 50 and sig >= 3:
            qualifying.append((val["symbol"], val["timeframe"], val["strategy"], wr))
    qualifying.sort(key=lambda x: -x[3])

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # ── 1. Tum isem/simulasyon gecmisini sil ─────────────────────────────
    print("[1/4] Veri sifirlama...")
    cur.execute("DELETE FROM sim_trades")
    cur.execute("DELETE FROM sim_equity_log")
    cur.execute("DELETE FROM sim_account")
    cur.execute("DELETE FROM signal_logs")
    cur.execute("DELETE FROM trades")
    cur.execute("DELETE FROM equity_snapshots")
    cur.execute("DELETE FROM backtest_results")
    conn.commit()

    # sim_account default'u yeniden olustur
    now = datetime.now().isoformat(timespec="seconds")
    cur.execute(
        "INSERT INTO sim_account (id, initial_balance, balance, peak_balance, "
        "total_trades, wins, losses, risk_pct, created_at, updated_at) "
        "VALUES (1, 100.0, 100.0, 100.0, 0, 0, 0, 1.0, ?, ?)",
        (now, now),
    )
    conn.commit()
    print("  sim_trades, sim_equity_log, sim_account, signal_logs, trades, equity_snapshots, backtest_results temizlendi")
    print("  sim_account $100 baslangic bakiye ile sifirlandi")

    # ── 2. Watchlist sifirlama ───────────────────────────────────────────
    print("\n[2/4] Watchlist temizleniyor...")
    cur.execute("DELETE FROM watchlists WHERE chat_id = ?", (CHAT_ID,))
    conn.commit()

    # ── 3. Kaliteli kombinasyonlari ekle ──────────────────────────────────
    print(f"\n[3/4] {len(qualifying)} kaliteli kombinasyon ekleniyor (chat {CHAT_ID}):")
    for sym, tf, strat, wr in qualifying:
        cur.execute(
            "INSERT OR IGNORE INTO watchlists (chat_id, symbol, timeframe, created_at) VALUES (?,?,?,?)",
            (CHAT_ID, sym, tf, now),
        )
        print(f"  {sym:>8} {tf:>7} -> {strat:>10} (WR {wr}%)")
    conn.commit()

    # ── 4. Dogrulama ─────────────────────────────────────────────────────
    print("\n[4/4] Dogrulama:")
    watch_count = cur.execute("SELECT COUNT(*) FROM watchlists").fetchone()[0]
    sim_trades_count = cur.execute("SELECT COUNT(*) FROM sim_trades").fetchone()[0]
    signal_count = cur.execute("SELECT COUNT(*) FROM signal_logs").fetchone()[0]
    sim_balance = cur.execute("SELECT balance FROM sim_account WHERE id=1").fetchone()[0]
    print(f"  watchlists: {watch_count}")
    print(f"  sim_trades: {sim_trades_count}")
    print(f"  signal_logs: {signal_count}")
    print(f"  sim_account balance: ${sim_balance:.2f}")

    conn.close()
    print("\nHAZIR — simdi botu yeniden baslat, test temiz basliyor.")


if __name__ == "__main__":
    main()
