from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.storage.sqlite_store import BotRepository


@dataclass
class TradeJournal:
    path: Path
    repo: BotRepository | None = None

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db_path = self.path.parent / "bot.db"
        self.repo = BotRepository(db_path=db_path)

    def add_trade(
        self,
        chat_id: int,
        symbol: str,
        timeframe: str,
        result: str,
        rr: float,
    ) -> None:
        assert self.repo is not None
        self.repo.add_trade(
            chat_id=chat_id,
            symbol=symbol,
            timeframe=timeframe,
            result=result,
            rr=rr,
        )

    def get_stats(self, chat_id: int) -> dict:
        assert self.repo is not None
        return self.repo.get_trade_stats(chat_id=chat_id)

    def get_today_stats(self, chat_id: int) -> dict:
        assert self.repo is not None
        return self.repo.get_today_trade_stats(chat_id=chat_id)
