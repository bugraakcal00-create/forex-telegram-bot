from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class SessionStatus:
    now_text: str
    timezone: str
    session_name: str
    is_open: bool
    next_open_text: str


def _is_between(now_time: time, start: time, end: time) -> bool:
    return start <= now_time < end


def get_session_status(timezone_name: str) -> SessionStatus:
    now = datetime.now(ZoneInfo(timezone_name))
    t = now.time()

    london_start = time(10, 0)
    london_end = time(19, 0)
    ny_start = time(15, 30)
    ny_end = time(23, 59, 59)

    in_london = _is_between(t, london_start, london_end)
    in_ny = _is_between(t, ny_start, ny_end)

    if in_london and in_ny:
        session_name = "Londra + New York Kesisimi"
        is_open = True
        next_open_text = "Seans acik"
    elif in_london:
        session_name = "Londra"
        is_open = True
        next_open_text = "Seans acik"
    elif in_ny:
        session_name = "New York"
        is_open = True
        next_open_text = "Seans acik"
    else:
        session_name = "Kapali"
        is_open = False
        if t < london_start:
            next_open_text = f"Bugun {london_start.strftime('%H:%M')} ({timezone_name})"
        else:
            next_open_text = f"Yarin {london_start.strftime('%H:%M')} ({timezone_name})"

    return SessionStatus(
        now_text=now.strftime("%Y-%m-%d %H:%M:%S"),
        timezone=timezone_name,
        session_name=session_name,
        is_open=is_open,
        next_open_text=next_open_text,
    )
