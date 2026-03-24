from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class BotRepository:
    db_path: Path

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._seed_defaults()
        self._migrate_json_if_needed()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS watchlists (
                    chat_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(chat_id, symbol, timeframe)
                );

                CREATE TABLE IF NOT EXISTS daily_subscribers (
                    chat_id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    result TEXT NOT NULL,
                    rr REAL NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS signal_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER,
                    source TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    quality TEXT NOT NULL,
                    setup_score INTEGER NOT NULL,
                    rr_ratio REAL NOT NULL,
                    current_price REAL NOT NULL,
                    trend TEXT NOT NULL,
                    higher_tf_trend TEXT NOT NULL,
                    sweep_signal TEXT NOT NULL,
                    sniper_entry TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    no_trade_reasons TEXT NOT NULL,
                    session_name TEXT NOT NULL,
                    is_session_open INTEGER NOT NULL,
                    had_high_impact_event INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runtime_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_trades_chat_created
                ON trades(chat_id, created_at);

                CREATE INDEX IF NOT EXISTS idx_signals_created
                ON signal_logs(created_at);
                """
            )

    def _seed_defaults(self) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runtime_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO NOTHING
                """,
                ("session_filter_enabled", "1", now),
            )
            conn.execute(
                """
                INSERT INTO runtime_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO NOTHING
                """,
                ("min_quality_for_alert", "A", now),
            )
            conn.execute(
                """
                INSERT INTO runtime_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO NOTHING
                """,
                ("min_score_for_alert", "80", now),
            )
            conn.execute(
                """
                INSERT INTO runtime_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO NOTHING
                """,
                ("min_rr_for_alert", "2.0", now),
            )

    def _migrate_json_if_needed(self) -> None:
        with self._connect() as conn:
            watch_count = conn.execute("SELECT COUNT(*) AS c FROM watchlists").fetchone()["c"]
            trade_count = conn.execute("SELECT COUNT(*) AS c FROM trades").fetchone()["c"]

        watch_path = self.db_path.parent / "watchlists.json"
        trade_path = self.db_path.parent / "trade_journal.json"

        if int(watch_count) == 0 and watch_path.exists():
            try:
                payload = json.loads(watch_path.read_text(encoding="utf-8"))
                for chat_id_str, items in payload.get("watchlists", {}).items():
                    chat_id = int(chat_id_str)
                    for item in items:
                        self.add_watch(chat_id, str(item.get("symbol", "")), str(item.get("timeframe", "5min")))
                for chat_id in payload.get("daily_subscribers", []):
                    self.subscribe_daily(int(chat_id))
            except Exception:
                pass

        if int(trade_count) == 0 and trade_path.exists():
            try:
                rows = json.loads(trade_path.read_text(encoding="utf-8"))
                for row in rows:
                    self.add_trade(
                        chat_id=int(row.get("chat_id")),
                        symbol=str(row.get("symbol", "XAUUSD")),
                        timeframe=str(row.get("timeframe", "5min")),
                        result=str(row.get("result", "loss")),
                        rr=float(row.get("rr", 0)),
                    )
            except Exception:
                pass

    def get_setting(self, key: str, default: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM runtime_settings WHERE key = ?",
                (key,),
            ).fetchone()
        if not row:
            return default
        return str(row["value"])

    def set_setting(self, key: str, value: str) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runtime_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, value, now),
            )

    def get_all_settings(self) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM runtime_settings").fetchall()
        return {str(r["key"]): str(r["value"]) for r in rows}

    def add_watch(self, chat_id: int, symbol: str, timeframe: str) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO watchlists (chat_id, symbol, timeframe, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chat_id, symbol, timeframe) DO NOTHING
                """,
                (chat_id, symbol.upper(), timeframe.lower(), now),
            )

    def remove_watch(self, chat_id: int, symbol: str, timeframe: str | None = None) -> bool:
        with self._connect() as conn:
            if timeframe:
                cursor = conn.execute(
                    """
                    DELETE FROM watchlists
                    WHERE chat_id = ? AND symbol = ? AND timeframe = ?
                    """,
                    (chat_id, symbol.upper(), timeframe.lower()),
                )
            else:
                cursor = conn.execute(
                    """
                    DELETE FROM watchlists
                    WHERE chat_id = ? AND symbol = ?
                    """,
                    (chat_id, symbol.upper()),
                )
        return cursor.rowcount > 0

    def get_watches(self, chat_id: int) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT symbol, timeframe FROM watchlists
                WHERE chat_id = ?
                ORDER BY symbol, timeframe
                """,
                (chat_id,),
            ).fetchall()
        return [{"symbol": str(r["symbol"]), "timeframe": str(r["timeframe"])} for r in rows]

    def iter_all_watches(self) -> dict[str, list[dict[str, str]]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chat_id, symbol, timeframe FROM watchlists ORDER BY chat_id"
            ).fetchall()
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            key = str(row["chat_id"])
            grouped.setdefault(key, []).append(
                {"symbol": str(row["symbol"]), "timeframe": str(row["timeframe"])}
            )
        return grouped

    def subscribe_daily(self, chat_id: int) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO daily_subscribers (chat_id, created_at)
                VALUES (?, ?)
                ON CONFLICT(chat_id) DO NOTHING
                """,
                (chat_id, now),
            )

    def unsubscribe_daily(self, chat_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM daily_subscribers WHERE chat_id = ?", (chat_id,))

    def get_daily_subscribers(self) -> list[int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT chat_id FROM daily_subscribers ORDER BY chat_id").fetchall()
        return [int(r["chat_id"]) for r in rows]

    def add_trade(
        self,
        chat_id: int,
        symbol: str,
        timeframe: str,
        result: str,
        rr: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trades (chat_id, symbol, timeframe, result, rr, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    symbol.upper(),
                    timeframe.lower(),
                    result.lower(),
                    float(rr),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def get_trade_stats(self, chat_id: int) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT symbol, result, rr FROM trades WHERE chat_id = ?",
                (chat_id,),
            ).fetchall()
        return self._stats_from_rows(rows)

    def get_today_trade_stats(self, chat_id: int) -> dict[str, Any]:
        today = datetime.now().date().isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT symbol, result, rr FROM trades
                WHERE chat_id = ? AND created_at LIKE ?
                """,
                (chat_id, f"{today}%"),
            ).fetchall()
        base = self._stats_from_rows(rows)
        return {
            "total": base["total"],
            "wins": base["wins"],
            "losses": base["losses"],
            "winrate": base["winrate"],
            "net_rr": base["net_rr"],
        }

    @staticmethod
    def _stats_from_rows(rows: list[sqlite3.Row]) -> dict[str, Any]:
        total = len(rows)
        wins = sum(1 for row in rows if str(row["result"]) == "win")
        losses = sum(1 for row in rows if str(row["result"]) == "loss")
        net_rr = round(sum(float(row["rr"]) for row in rows), 2)
        winrate = round((wins / total) * 100, 2) if total else 0.0

        by_symbol: dict[str, dict[str, Any]] = {}
        for row in rows:
            symbol = str(row["symbol"])
            by_symbol.setdefault(symbol, {"total": 0, "wins": 0, "losses": 0, "net_rr": 0.0})
            by_symbol[symbol]["total"] += 1
            if str(row["result"]) == "win":
                by_symbol[symbol]["wins"] += 1
            elif str(row["result"]) == "loss":
                by_symbol[symbol]["losses"] += 1
            by_symbol[symbol]["net_rr"] += float(row["rr"])

        for symbol, item in by_symbol.items():
            item["net_rr"] = round(float(item["net_rr"]), 2)
            total_symbol = int(item["total"])
            item["winrate"] = round((int(item["wins"]) / total_symbol) * 100, 2) if total_symbol else 0.0

        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "winrate": winrate,
            "net_rr": net_rr,
            "by_symbol": by_symbol,
        }

    def add_signal_log(
        self,
        *,
        chat_id: int | None,
        source: str,
        symbol: str,
        timeframe: str,
        signal: str,
        quality: str,
        setup_score: int,
        rr_ratio: float,
        current_price: float,
        trend: str,
        higher_tf_trend: str,
        sweep_signal: str,
        sniper_entry: str,
        reason: str,
        no_trade_reasons: list[str],
        session_name: str,
        is_session_open: bool,
        had_high_impact_event: bool,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO signal_logs (
                    chat_id, source, symbol, timeframe, signal, quality, setup_score, rr_ratio,
                    current_price, trend, higher_tf_trend, sweep_signal, sniper_entry, reason,
                    no_trade_reasons, session_name, is_session_open, had_high_impact_event, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    source,
                    symbol.upper(),
                    timeframe.lower(),
                    signal,
                    quality,
                    int(setup_score),
                    float(rr_ratio),
                    float(current_price),
                    trend,
                    higher_tf_trend,
                    sweep_signal,
                    sniper_entry,
                    reason,
                    json.dumps(no_trade_reasons, ensure_ascii=False),
                    session_name,
                    1 if is_session_open else 0,
                    1 if had_high_impact_event else 0,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def get_recent_signal_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, source, symbol, timeframe, signal, quality, setup_score, rr_ratio,
                       current_price, session_name, is_session_open, had_high_impact_event, created_at
                FROM signal_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_dashboard_summary(self) -> dict[str, Any]:
        with self._connect() as conn:
            trade_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_trades,
                    SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) AS losses,
                    COALESCE(SUM(rr), 0) AS net_rr
                FROM trades
                """
            ).fetchone()
            signal_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_signals,
                    SUM(CASE WHEN signal = 'NO TRADE' THEN 1 ELSE 0 END) AS no_trade_count
                FROM signal_logs
                """
            ).fetchone()

        total_trades = int(trade_row["total_trades"] or 0)
        wins = int(trade_row["wins"] or 0)
        losses = int(trade_row["losses"] or 0)
        net_rr = round(float(trade_row["net_rr"] or 0.0), 2)
        winrate = round((wins / total_trades) * 100, 2) if total_trades else 0.0

        total_signals = int(signal_row["total_signals"] or 0)
        no_trade_count = int(signal_row["no_trade_count"] or 0)
        no_trade_rate = round((no_trade_count / total_signals) * 100, 2) if total_signals else 0.0

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "net_rr": net_rr,
            "winrate": winrate,
            "total_signals": total_signals,
            "no_trade_count": no_trade_count,
            "no_trade_rate": no_trade_rate,
        }
