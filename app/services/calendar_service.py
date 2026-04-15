from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx


# ForexFactory ücretsiz JSON feed (API anahtarı gerektirmez)
_FF_THIS_WEEK = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_FF_NEXT_WEEK = "https://nfs.faireconomy.media/ff_calendar_nextweek.json"

# FMP endpoint (ücretli planda çalışır)
_FMP_URL = "https://financialmodelingprep.com/stable/economic-calendar"


def _parse_ff_date(raw: str) -> datetime | None:
    """ForexFactory ISO tarihini UTC datetime'a çevirir."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


@dataclass
class CalendarService:
    api_key: str
    base_url: str = _FMP_URL

    async def get_upcoming_high_impact_events(
        self,
        hours_ahead: int = 12,
        limit: int = 8,
    ) -> list[dict[str, str]]:
        # Önce ForexFactory'yi dene (ücretsiz, güvenilir)
        try:
            result = await self._fetch_forexfactory(hours_ahead, limit)
            if result:
                return result
        except Exception:
            pass

        # ForexFactory başarısız olursa FMP'yi dene (ücretli plan gerekir)
        if self.api_key:
            try:
                result = await self._fetch_fmp(hours_ahead, limit)
                if result:
                    return result
            except Exception:
                pass

        return []

    async def _fetch_forexfactory(
        self, hours_ahead: int, limit: int
    ) -> list[dict[str, str]]:
        now_utc = datetime.now(timezone.utc)
        end_utc = now_utc + timedelta(hours=hours_ahead)

        results: list[dict[str, str]] = []
        async with httpx.AsyncClient(
            timeout=10,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 ForexBot/2.0"},
        ) as client:
            for url in (_FF_THIS_WEEK, _FF_NEXT_WEEK):
                try:
                    r = await client.get(url)
                    if r.status_code != 200:
                        continue
                    events: list[dict[str, Any]] = r.json()
                except Exception:
                    continue

                for item in events:
                    impact = str(item.get("impact", "")).lower()
                    if impact != "high":
                        continue

                    dt = _parse_ff_date(str(item.get("date", "")))
                    if dt is None:
                        continue
                    if not (now_utc <= dt <= end_utc):
                        continue

                    results.append({
                        "date": dt.strftime("%Y-%m-%d %H:%M"),
                        "country": str(item.get("country", "")),
                        "event": str(item.get("title", "")),
                        "impact": "high",
                        "actual":   str(item.get("actual",   "") or ""),
                        "forecast": str(item.get("forecast", "") or ""),
                        "previous": str(item.get("previous", "") or ""),
                    })

                    if len(results) >= limit:
                        return results

        return results

    async def _fetch_fmp(self, hours_ahead: int, limit: int) -> list[dict[str, str]]:
        now = datetime.now(timezone.utc)
        end = now + timedelta(hours=hours_ahead)
        params = {
            "from": now.date().isoformat(),
            "to": end.date().isoformat(),
            "apikey": self.api_key,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(self.base_url, params=params)
            if response.status_code in (401, 402, 403, 429):
                return []
            response.raise_for_status()
            payload: list[dict[str, Any]] = response.json()

        results: list[dict[str, str]] = []
        for item in payload:
            impact = str(item.get("impact", "")).lower()
            if impact not in {"high", "3"}:
                continue
            results.append({
                "date":     str(item.get("date", "")),
                "country":  str(item.get("country", "")),
                "event":    str(item.get("event", "")),
                "impact":   str(item.get("impact", "")),
                "actual":   str(item.get("actual",   "") or ""),
                "forecast": str(item.get("estimate", "") or ""),
                "previous": str(item.get("previous", "") or ""),
            })
            if len(results) >= limit:
                break
        return results
