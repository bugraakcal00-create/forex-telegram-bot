from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class TradeJournal:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def _read(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _write(self, data: list[dict]) -> None:
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_trade(
        self,
        chat_id: int,
        symbol: str,
        timeframe: str,
        result: str,
        rr: float,
    ) -> None:
        data = self._read()
        data.append(
            {
                "chat_id": chat_id,
                "symbol": symbol.upper(),
                "timeframe": timeframe.lower(),
                "result": result.lower(),   # win / loss
                "rr": rr,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self._write(data)

    def get_stats(self, chat_id: int) -> dict:
        data = [x for x in self._read() if x.get("chat_id") == chat_id]

        total = len(data)
        wins = sum(1 for x in data if x.get("result") == "win")
        losses = sum(1 for x in data if x.get("result") == "loss")
        net_rr = round(sum(float(x.get("rr", 0)) for x in data), 2)

        winrate = round((wins / total) * 100, 2) if total else 0.0

        by_symbol: dict[str, dict] = {}
        for item in data:
            symbol = item["symbol"]
            if symbol not in by_symbol:
                by_symbol[symbol] = {
                    "total": 0,
                    "wins": 0,
                    "losses": 0,
                    "net_rr": 0.0,
                }
            by_symbol[symbol]["total"] += 1
            if item["result"] == "win":
                by_symbol[symbol]["wins"] += 1
            else:
                by_symbol[symbol]["losses"] += 1
            by_symbol[symbol]["net_rr"] += float(item["rr"])

        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "winrate": winrate,
            "net_rr": net_rr,
            "by_symbol": by_symbol,
        }

    def get_today_stats(self, chat_id: int) -> dict:
        today = datetime.now().date().isoformat()
        data = [
            x for x in self._read()
            if x.get("chat_id") == chat_id and str(x.get("created_at", "")).startswith(today)
        ]

        total = len(data)
        wins = sum(1 for x in data if x.get("result") == "win")
        losses = sum(1 for x in data if x.get("result") == "loss")
        net_rr = round(sum(float(x.get("rr", 0)) for x in data), 2)
        winrate = round((wins / total) * 100, 2) if total else 0.0

        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "winrate": winrate,
            "net_rr": net_rr,
        }