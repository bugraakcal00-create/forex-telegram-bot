"""
Paper Trading Simulation Engine
$100 baslangic bakiyesi, %1 risk per trade.
Sinyaller otomatik olarak acilir, TP/SL ile kapanir.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class SimTrade:
    id: int
    symbol: str
    timeframe: str
    direction: str  # LONG / SHORT
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    position_size: float
    status: str  # open / tp_hit / sl_hit / expired
    pnl: float
    opened_at: str
    closed_at: str | None
    signal_log_id: int | None
    strategy_mode: str


class SimulationService:
    """Paper trading simulation engine with $100 initial balance and 1% risk."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sim_account (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    initial_balance REAL NOT NULL DEFAULT 100.0,
                    balance REAL NOT NULL DEFAULT 100.0,
                    risk_pct REAL NOT NULL DEFAULT 1.0,
                    total_trades INTEGER NOT NULL DEFAULT 0,
                    wins INTEGER NOT NULL DEFAULT 0,
                    losses INTEGER NOT NULL DEFAULT 0,
                    peak_balance REAL NOT NULL DEFAULT 100.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sim_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id INTEGER NOT NULL DEFAULT 1,
                    signal_log_id INTEGER,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    strategy_mode TEXT NOT NULL DEFAULT 'default',
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    risk_amount REAL NOT NULL,
                    position_size REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    pnl REAL NOT NULL DEFAULT 0.0,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT,
                    FOREIGN KEY (account_id) REFERENCES sim_account(id)
                );

                CREATE TABLE IF NOT EXISTS sim_equity_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id INTEGER NOT NULL DEFAULT 1,
                    balance REAL NOT NULL,
                    equity REAL NOT NULL,
                    drawdown_pct REAL NOT NULL DEFAULT 0.0,
                    open_positions INTEGER NOT NULL DEFAULT 0,
                    trade_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sim_trades_status
                ON sim_trades(status);

                CREATE INDEX IF NOT EXISTS idx_sim_equity_created
                ON sim_equity_log(created_at);
            """)

    def get_or_create_account(self, initial_balance: float = 100.0, risk_pct: float = 1.0) -> dict[str, Any]:
        """Get existing account or create new one."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sim_account WHERE id = 1").fetchone()
            if row:
                return dict(row)
            now = datetime.now().isoformat(timespec="seconds")
            conn.execute(
                "INSERT INTO sim_account (id, initial_balance, balance, risk_pct, peak_balance, created_at, updated_at) "
                "VALUES (1, ?, ?, ?, ?, ?, ?)",
                (initial_balance, initial_balance, risk_pct, initial_balance, now, now),
            )
            return {
                "id": 1, "initial_balance": initial_balance, "balance": initial_balance,
                "risk_pct": risk_pct, "total_trades": 0, "wins": 0, "losses": 0,
                "peak_balance": initial_balance, "created_at": now, "updated_at": now,
            }

    def reset_account(self, initial_balance: float = 100.0, risk_pct: float = 1.0) -> None:
        """Reset simulation - start fresh."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute("DELETE FROM sim_trades")
            conn.execute("DELETE FROM sim_equity_log")
            conn.execute("DELETE FROM sim_account")
            conn.execute(
                "INSERT INTO sim_account (id, initial_balance, balance, risk_pct, peak_balance, created_at, updated_at) "
                "VALUES (1, ?, ?, ?, ?, ?, ?)",
                (initial_balance, initial_balance, risk_pct, initial_balance, now, now),
            )

    def open_trade(
        self,
        symbol: str,
        timeframe: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        signal_log_id: int | None = None,
        strategy_mode: str = "default",
    ) -> int | None:
        """Open a new simulated trade with 1% risk."""
        account = self.get_or_create_account()
        balance = float(account["balance"])
        risk_pct = float(account["risk_pct"])

        if balance <= 0:
            return None

        # Calculate risk amount and position size
        risk_amount = balance * (risk_pct / 100.0)
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance <= 0:
            return None

        position_size = risk_amount / stop_distance

        # Check if we already have an open trade for this symbol
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT COUNT(*) AS c FROM sim_trades WHERE symbol = ? AND status = 'open'",
                (symbol.upper(),),
            ).fetchone()
            if int(existing["c"]) > 0:
                return None  # Don't double up

        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO sim_trades
                   (account_id, signal_log_id, symbol, timeframe, direction, strategy_mode,
                    entry_price, stop_loss, take_profit, risk_amount, position_size, status, opened_at)
                   VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)""",
                (signal_log_id, symbol.upper(), timeframe.lower(), direction.upper(),
                 strategy_mode, entry_price, stop_loss, take_profit,
                 round(risk_amount, 4), round(position_size, 6), now),
            )
            return int(cursor.lastrowid)

    def close_trade(self, trade_id: int, outcome: str, exit_price: float) -> float:
        """Close a trade and update account balance. Returns PnL."""
        with self._connect() as conn:
            trade = conn.execute("SELECT * FROM sim_trades WHERE id = ?", (trade_id,)).fetchone()
            if not trade or str(trade["status"]) != "open":
                return 0.0

            entry = float(trade["entry_price"])
            direction = str(trade["direction"])
            position_size = float(trade["position_size"])
            risk_amount = float(trade["risk_amount"])

            # Calculate PnL — spread/slippage düşülerek gerçekçi
            from app.services.backtest_service import _exec_cost
            cost = _exec_cost(str(trade["symbol"]))
            if direction == "LONG":
                pnl = (exit_price - entry - cost) * position_size
            else:
                pnl = (entry - exit_price - cost) * position_size

            pnl = round(pnl, 4)
            now = datetime.now().isoformat(timespec="seconds")

            conn.execute(
                "UPDATE sim_trades SET status = ?, pnl = ?, closed_at = ? WHERE id = ?",
                (outcome, pnl, now, trade_id),
            )

            # Update account
            account = conn.execute("SELECT * FROM sim_account WHERE id = 1").fetchone()
            new_balance = round(float(account["balance"]) + pnl, 4)
            new_total = int(account["total_trades"]) + 1
            new_wins = int(account["wins"]) + (1 if pnl > 0 else 0)
            new_losses = int(account["losses"]) + (1 if pnl < 0 else 0)
            new_peak = max(float(account["peak_balance"]), new_balance)

            conn.execute(
                """UPDATE sim_account SET balance = ?, total_trades = ?, wins = ?,
                   losses = ?, peak_balance = ?, updated_at = ? WHERE id = 1""",
                (new_balance, new_total, new_wins, new_losses, new_peak, now),
            )

            # Log equity
            open_count = conn.execute(
                "SELECT COUNT(*) AS c FROM sim_trades WHERE status = 'open'"
            ).fetchone()["c"]
            dd = round((1 - new_balance / new_peak) * 100, 2) if new_peak > 0 else 0
            conn.execute(
                "INSERT INTO sim_equity_log (account_id, balance, equity, drawdown_pct, open_positions, trade_count, created_at) "
                "VALUES (1, ?, ?, ?, ?, ?, ?)",
                (new_balance, new_balance, dd, int(open_count), new_total, now),
            )

            return pnl

    def check_and_resolve_trades(self, candle_data: dict[str, dict]) -> list[dict]:
        """
        Check open trades against current candle data.
        candle_data: {symbol: {high, low, open, close}}
        Returns list of resolved trades.
        """
        resolved = []
        with self._connect() as conn:
            open_trades = conn.execute(
                "SELECT * FROM sim_trades WHERE status = 'open'"
            ).fetchall()

        for trade in open_trades:
            symbol = str(trade["symbol"])
            if symbol not in candle_data:
                continue

            candle = candle_data[symbol]
            high = float(candle.get("high", 0))
            low = float(candle.get("low", 0))
            direction = str(trade["direction"])
            tp = float(trade["take_profit"])
            sl = float(trade["stop_loss"])
            trade_id = int(trade["id"])

            if direction == "LONG":
                tp_hit = high >= tp
                sl_hit = low <= sl
            else:
                tp_hit = low <= tp
                sl_hit = high >= sl

            if tp_hit and sl_hit:
                # Both hit in same candle - use distance from open
                candle_open = float(candle.get("open", 0))
                if abs(candle_open - tp) <= abs(candle_open - sl):
                    pnl = self.close_trade(trade_id, "tp_hit", tp)
                    resolved.append({"trade_id": trade_id, "symbol": symbol, "outcome": "tp_hit", "pnl": pnl})
                else:
                    pnl = self.close_trade(trade_id, "sl_hit", sl)
                    resolved.append({"trade_id": trade_id, "symbol": symbol, "outcome": "sl_hit", "pnl": pnl})
            elif tp_hit:
                pnl = self.close_trade(trade_id, "tp_hit", tp)
                resolved.append({"trade_id": trade_id, "symbol": symbol, "outcome": "tp_hit", "pnl": pnl})
            elif sl_hit:
                pnl = self.close_trade(trade_id, "sl_hit", sl)
                resolved.append({"trade_id": trade_id, "symbol": symbol, "outcome": "sl_hit", "pnl": pnl})

        return resolved

    def get_open_trades(self) -> list[dict[str, Any]]:
        """Get all open simulation trades."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sim_trades WHERE status = 'open' ORDER BY opened_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_trade_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get closed trades history."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sim_trades WHERE status != 'open' ORDER BY closed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_trades(self, limit: int = 200) -> list[dict[str, Any]]:
        """Get all trades."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sim_trades ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_equity_curve(self, limit: int = 500) -> list[dict[str, Any]]:
        """Get equity curve data."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sim_equity_log ORDER BY created_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_account_summary(self) -> dict[str, Any]:
        """Get full account summary with statistics."""
        account = self.get_or_create_account()
        balance = float(account["balance"])
        initial = float(account["initial_balance"])
        peak = float(account["peak_balance"])
        total = int(account["total_trades"])
        wins = int(account["wins"])
        losses = int(account["losses"])

        winrate = round((wins / total) * 100, 2) if total > 0 else 0.0
        total_return = round(((balance - initial) / initial) * 100, 2) if initial > 0 else 0.0
        drawdown = round((1 - balance / peak) * 100, 2) if peak > 0 else 0.0
        profit = round(balance - initial, 2)

        # Open positions
        open_trades = self.get_open_trades()

        # Best/worst trade
        with self._connect() as conn:
            best = conn.execute(
                "SELECT MAX(pnl) AS best FROM sim_trades WHERE status != 'open'"
            ).fetchone()
            worst = conn.execute(
                "SELECT MIN(pnl) AS worst FROM sim_trades WHERE status != 'open'"
            ).fetchone()
            avg_pnl = conn.execute(
                "SELECT AVG(pnl) AS avg FROM sim_trades WHERE status != 'open'"
            ).fetchone()
            # By symbol stats
            symbol_stats = conn.execute("""
                SELECT symbol,
                       COUNT(*) AS total,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                       SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) AS losses,
                       ROUND(SUM(pnl), 2) AS net_pnl,
                       ROUND(AVG(pnl), 4) AS avg_pnl
                FROM sim_trades WHERE status != 'open'
                GROUP BY symbol ORDER BY net_pnl DESC
            """).fetchall()

        # Gelismis metrikler — Sharpe, Profit Factor, Expectancy, Max Consec Losses
        with self._connect() as conn:
            pnl_rows = conn.execute(
                "SELECT pnl FROM sim_trades WHERE status != 'open' ORDER BY closed_at ASC"
            ).fetchall()

        pnls = [float(r["pnl"] or 0) for r in pnl_rows]
        sharpe = 0.0
        profit_factor = 0.0
        expectancy = 0.0
        max_consec_loss = 0

        if pnls:
            import math as _m
            # Sharpe (trade-bazli, annualized yok — saf ratio)
            mean = sum(pnls) / len(pnls)
            if len(pnls) > 1:
                var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
                std = _m.sqrt(var) if var > 0 else 0.0
                sharpe = round((mean / std), 3) if std > 0 else 0.0
            # Profit Factor
            gp = sum(p for p in pnls if p > 0)
            gl = abs(sum(p for p in pnls if p < 0))
            profit_factor = round(gp / gl, 3) if gl > 0 else (round(gp, 2) if gp > 0 else 0.0)
            # Expectancy (ortalama PnL per trade)
            expectancy = round(mean, 4)
            # Max consecutive losses
            cur, mx = 0, 0
            for p in pnls:
                if p < 0:
                    cur += 1
                    mx = max(mx, cur)
                else:
                    cur = 0
            max_consec_loss = mx

        # Max drawdown dollar (peak - trough)
        max_dd_dollar = round(peak - balance, 2) if peak > balance else 0.0

        return {
            "balance": balance,
            "initial_balance": initial,
            "profit": profit,
            "total_return_pct": total_return,
            "peak_balance": peak,
            "drawdown_pct": drawdown,
            "max_dd_dollar": max_dd_dollar,
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "winrate": winrate,
            "open_positions": len(open_trades),
            "open_trades": open_trades,
            "best_trade": float(best["best"] or 0) if best else 0,
            "worst_trade": float(worst["worst"] or 0) if worst else 0,
            "avg_pnl": round(float(avg_pnl["avg"] or 0), 4) if avg_pnl else 0,
            "risk_pct": float(account["risk_pct"]),
            # v2 metrikleri
            "sharpe": sharpe,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "max_consec_loss": max_consec_loss,
            "by_symbol": [dict(r) for r in symbol_stats],
            "created_at": account["created_at"],
            "updated_at": account["updated_at"],
        }

    def get_strategy_stats(self) -> list[dict[str, Any]]:
        """Get performance stats per strategy mode."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT strategy_mode,
                       COUNT(*) AS total,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                       ROUND(SUM(pnl), 2) AS net_pnl,
                       ROUND(AVG(pnl), 4) AS avg_pnl
                FROM sim_trades WHERE status != 'open'
                GROUP BY strategy_mode ORDER BY net_pnl DESC
            """).fetchall()
        return [dict(r) for r in rows]
