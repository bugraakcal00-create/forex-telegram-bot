from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class NewsService:
    api_key: str
    base_url: str = "https://newsapi.org/v2/everything"

    async def get_forex_news(self, query: str = "forex OR gold OR dollar OR fed", limit: int = 5) -> list[dict[str, str]]:
        if not self.api_key:
            return []

        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit,
            "apiKey": self.api_key,
        }
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
            payload: dict[str, Any] = response.json()

        articles = []
        for item in payload.get("articles", [])[:limit]:
            articles.append(
                {
                    "title": item.get("title", "Başlık yok"),
                    "source": item.get("source", {}).get("name", "Bilinmiyor"),
                    "url": item.get("url", ""),
                    "published_at": item.get("publishedAt", ""),
                }
            )
        return articles
