from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.storage.sqlite_store import BotRepository


@dataclass
class WatchStore:
    path: Path
    repo: BotRepository | None = None

    def __post_init__(self) -> None:
        db_path = self.path.parent / "bot.db"
        self.repo = BotRepository(db_path=db_path)

    def add_watch(self, chat_id: int, symbol: str, timeframe: str) -> None:
        assert self.repo is not None
        self.repo.add_watch(chat_id=chat_id, symbol=symbol, timeframe=timeframe)

    def remove_watch(self, chat_id: int, symbol: str, timeframe: str | None = None) -> bool:
        assert self.repo is not None
        return self.repo.remove_watch(chat_id=chat_id, symbol=symbol, timeframe=timeframe)

    def get_watches(self, chat_id: int) -> list[dict[str, str]]:
        assert self.repo is not None
        return self.repo.get_watches(chat_id=chat_id)

    def iter_all(self) -> dict[str, list[dict[str, str]]]:
        assert self.repo is not None
        return self.repo.iter_all_watches()

    def subscribe_daily(self, chat_id: int) -> None:
        assert self.repo is not None
        self.repo.subscribe_daily(chat_id=chat_id)

    def unsubscribe_daily(self, chat_id: int) -> None:
        assert self.repo is not None
        self.repo.unsubscribe_daily(chat_id=chat_id)

    def get_daily_subscribers(self) -> list[int]:
        assert self.repo is not None
        return self.repo.get_daily_subscribers()
