from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import httpx


# Ücretsiz Forex RSS kaynakları (API anahtarı gerektirmez)
_RSS_FEEDS = [
    ("FXStreet",    "https://www.fxstreet.com/rss/news"),
    ("Investing",   "https://www.investing.com/rss/news_25.rss"),
    ("DailyFX",    "https://www.dailyfx.com/feeds/all"),
    ("ForexLive",  "https://www.forexlive.com/feed/news"),
    ("Reuters FX", "https://feeds.reuters.com/reuters/businessNews"),
]


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


async def _fetch_rss(client: httpx.AsyncClient, name: str, url: str, limit: int) -> list[dict[str, str]]:
    try:
        r = await client.get(url, timeout=10, follow_redirects=True,
                             headers={"User-Agent": "Mozilla/5.0 ForexBot/2.0"})
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.text)
        items = root.findall(".//item")
        results = []
        for item in items[:limit]:
            title = _strip_html(item.findtext("title", ""))
            link  = (item.findtext("link", "") or "").strip()
            pub   = item.findtext("pubDate", "")
            if title and link:
                results.append({
                    "title": title,
                    "source": name,
                    "url": link,
                    "published_at": pub[:25] if pub else "",
                })
        return results
    except Exception:
        return []


@dataclass
class NewsService:
    api_key: str
    base_url: str = "https://newsapi.org/v2/everything"

    async def get_forex_news(
        self,
        query: str = "forex OR \"gold price\" OR \"US dollar\" OR \"Federal Reserve\" OR \"interest rate\" OR XAUUSD OR EUR/USD",
        limit: int = 10,
    ) -> list[dict[str, str]]:
        # Önce NewsAPI dene (anahtar varsa)
        if self.api_key:
            try:
                result = await self._fetch_newsapi(query, limit)
                if result:
                    return result
            except Exception:
                pass

        # NewsAPI yoksa / başarısız olursa RSS'e düş
        return await self._fetch_rss_all(limit)

    async def _fetch_newsapi(self, query: str, limit: int) -> list[dict[str, str]]:
        params: dict[str, Any] = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit,
            "apiKey": self.api_key,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
            payload: dict[str, Any] = response.json()

        articles = []
        for item in payload.get("articles", [])[:limit]:
            articles.append({
                "title": item.get("title", "Başlık yok"),
                "source": item.get("source", {}).get("name", "Bilinmiyor"),
                "url": item.get("url", ""),
                "published_at": item.get("publishedAt", ""),
            })
        return articles

    async def _fetch_rss_all(self, limit: int) -> list[dict[str, str]]:
        import asyncio
        per_feed = max(3, limit // len(_RSS_FEEDS) + 1)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=6.0, read=10.0, write=5.0, pool=5.0),
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 ForexBot/2.0"},
        ) as client:
            tasks = [_fetch_rss(client, name, url, per_feed) for name, url in _RSS_FEEDS]
            batches = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[dict[str, str]] = []
        for batch in batches:
            if isinstance(batch, list):
                results.extend(batch)
        return results[:limit]
