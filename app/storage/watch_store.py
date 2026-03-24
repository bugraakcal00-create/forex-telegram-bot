from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WatchStore:
    path: Path
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                self.payload = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.payload = {"watchlists": {}, "daily_subscribers": []}
        else:
            self.payload = {"watchlists": {}, "daily_subscribers": []}
            self._save()

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_watch(self, chat_id: int, symbol: str, timeframe: str) -> None:
        watchlists = self.payload.setdefault("watchlists", {})
        entries = watchlists.setdefault(str(chat_id), [])
        item = {"symbol": symbol, "timeframe": timeframe}
        if item not in entries:
            entries.append(item)
            self._save()

    def remove_watch(self, chat_id: int, symbol: str) -> bool:
        watchlists = self.payload.setdefault("watchlists", {})
        entries = watchlists.get(str(chat_id), [])
        initial = len(entries)
        entries[:] = [x for x in entries if x["symbol"] != symbol]
        if len(entries) != initial:
            self._save()
            return True
        return False

    def get_watches(self, chat_id: int) -> list[dict[str, str]]:
        return self.payload.setdefault("watchlists", {}).get(str(chat_id), [])

    def iter_all(self) -> dict[str, list[dict[str, str]]]:
        return self.payload.setdefault("watchlists", {})

    def subscribe_daily(self, chat_id: int) -> None:
        subscribers = self.payload.setdefault("daily_subscribers", [])
        if chat_id not in subscribers:
            subscribers.append(chat_id)
            self._save()

    def unsubscribe_daily(self, chat_id: int) -> None:
        subscribers = self.payload.setdefault("daily_subscribers", [])
        if chat_id in subscribers:
            subscribers.remove(chat_id)
            self._save()

    def get_daily_subscribers(self) -> list[int]:
        return self.payload.setdefault("daily_subscribers", [])
