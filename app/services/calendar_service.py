from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx


@dataclass
class CalendarService:
    api_key: str
    base_url: str = "https://financialmodelingprep.com/stable/economic-calendar"

    async def get_upcoming_high_impact_events(
        self,
        hours_ahead: int = 12,
        limit: int = 8,
    ) -> list[dict[str, str]]:
        if not self.api_key:
            return []

        now = datetime.now(timezone.utc)
        end = now + timedelta(hours=hours_ahead)
        params = {
            "from": now.date().isoformat(),
            "to": end.date().isoformat(),
            "apikey": self.api_key,
        }

        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(self.base_url, params=params)

                # FMP erişim/veri planı yoksa sinyali bozmasın
                if response.status_code in (401, 402, 403, 429):
                    return []

                response.raise_for_status()
                payload: list[dict[str, Any]] = response.json()

        except Exception:
            return []

        results: list[dict[str, str]] = []
        for item in payload:
            impact = str(item.get("impact", "")).lower()
            if impact not in {"high", "3"}:
                continue

            results.append(
                {
                    "date": str(item.get("date", "")),
                    "country": str(item.get("country", "")),
                    "event": str(item.get("event", "")),
                    "impact": str(item.get("impact", "")),
                    "actual": str(item.get("actual", "")),
                    "forecast": str(item.get("estimate", "")),
                    "previous": str(item.get("previous", "")),
                }
            )

            if len(results) >= limit:
                break

        return results