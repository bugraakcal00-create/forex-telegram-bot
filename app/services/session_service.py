from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo


# ── Likidite Blackout Pencereleri (UTC) ───────────────────────────────────
# Market maker manipülasyonu yoğun — bu zamanlarda sinyal alma.
_BLACKOUT_WINDOWS: list[tuple[str, time, time]] = [
    ("London Fix",       time(15, 45), time(16, 15)),  # 4pm London fix manipülasyon
    ("NY Option Cut",    time(14, 55), time(15, 5)),   # NY option cut
    ("Rollover",         time(21, 55), time(22, 5)),   # günlük swap
    ("Sunday Open",      time(22, 0),  time(23, 59)),  # sadece pazar
]


def is_in_blackout(utc_dt: datetime | None = None) -> tuple[bool, str]:
    """Likidite blackout penceresinde miyiz?

    Returns:
        (in_blackout, window_name). Değilse (False, "").
    """
    if utc_dt is None:
        utc_dt = datetime.now(ZoneInfo("UTC"))
    t = utc_dt.time()
    wd = utc_dt.weekday()  # 0=Mon, 6=Sun

    # Sunday open: sadece pazar günü
    for name, start, end in _BLACKOUT_WINDOWS:
        if name == "Sunday Open":
            if wd == 6 and _is_between(t, start, end):
                return True, name
        else:
            if _is_between(t, start, end):
                return True, name

    # Hafta sonu hard-block: Cuma 20:00 UTC - Pazar 22:00 UTC
    if wd == 4 and t >= time(20, 0):
        return True, "Friday Close"
    if wd == 5:
        return True, "Weekend"
    if wd == 6 and t < time(22, 0):
        return True, "Sunday (pre-open)"

    return False, ""


# ── Round Number Tespiti ──────────────────────────────────────────────────
# Stop hunt zone: $2000, $2050, $2100, 1.1000, 1.2000 etc.
_ROUND_NUMBER_STEPS = {
    "XAUUSD": 50.0,   # her $50'de round
    "XAGUSD": 1.0,
    "BTCUSD": 1000.0,
    "USDJPY": 1.0,    # 150.00, 151.00
    "EURJPY": 1.0,
    "GBPJPY": 1.0,
    # EUR/GBP/AUD etc.: 0.0100 = 100 pip round (1.1000, 1.2000)
}


def near_round_number(symbol: str, price: float, atr: float, tolerance_atr: float = 0.3) -> bool:
    """Fiyat round number'a yakın mı (tolerance_atr × ATR mesafesi içinde)?

    Round number'lar bankaların stop hunt bölgesi. Bu zonlarda sinyal riskli.
    """
    sym = symbol.upper()
    step = _ROUND_NUMBER_STEPS.get(sym, 0.0100)  # EUR/GBP/AUD default
    if step <= 0 or atr <= 0:
        return False
    # En yakın round number
    nearest = round(price / step) * step
    distance = abs(price - nearest)
    return distance <= atr * tolerance_atr


@dataclass(frozen=True)
class SessionStatus:
    now_text: str
    timezone: str
    session_name: str
    is_open: bool
    next_open_text: str
    # ICT Killzone bilgisi
    in_killzone: bool = False
    killzone_name: str = "—"


def _is_between(now_time: time, start: time, end: time) -> bool:
    if end < start:  # gece yarısı geçişi
        return now_time >= start or now_time < end
    return start <= now_time < end


# Tüm saatler UTC cinsinden
_KILLZONES: list[tuple[str, time, time]] = [
    ("London Open KZ",   time(7,  0), time(9, 30)),   # 07:00–09:30 UTC
    ("Silver Bullet 1",  time(9, 30), time(11, 0)),   # 09:30–11:00 UTC
    ("NY Open KZ",       time(12, 30), time(15, 0)),  # 12:30–15:00 UTC
    ("Silver Bullet 2",  time(13, 30), time(14, 30)), # 13:30–14:30 UTC
    ("NY Afternoon",     time(15,  0), time(16, 30)), # 15:00–16:30 UTC
    ("London Close",     time(15, 30), time(17, 0)),  # 15:30–17:00 UTC
]

# London + NY seans pencereleri — UTC
_SESSION_WINDOWS: list[tuple[str, time, time]] = [
    ("Londra",    time(8,  0), time(17, 0)),
    ("New York",  time(13, 0), time(22, 0)),
]


def get_session_status(timezone_name: str) -> SessionStatus:
    now_utc   = datetime.now(ZoneInfo("UTC"))
    now_local = now_utc.astimezone(ZoneInfo(timezone_name))
    t_utc     = now_utc.time()

    # Seans tespiti (UTC bazlı — timezone bağımsız)
    in_london = False
    in_ny = False
    for sess_name, sess_start, sess_end in _SESSION_WINDOWS:
        if _is_between(t_utc, sess_start, sess_end):
            if "Londra" in sess_name:
                in_london = True
            elif "New York" in sess_name:
                in_ny = True

    # Yerel saate çevrilmiş seans bilgisi
    london_local_start = datetime.combine(now_utc.date(), _SESSION_WINDOWS[0][1], tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo(timezone_name)).time()

    if in_london and in_ny:
        session_name    = "Londra + New York Kesisimi"
        is_open         = True
        next_open_text  = "Seans acik"
    elif in_london:
        session_name    = "Londra"
        is_open         = True
        next_open_text  = "Seans acik"
    elif in_ny:
        session_name    = "New York"
        is_open         = True
        next_open_text  = "Seans acik"
    else:
        session_name    = "Kapali (Asya)"
        is_open         = False
        t_local = now_local.time()
        if t_local < london_local_start:
            next_open_text = f"Bugun {london_local_start.strftime('%H:%M')} ({timezone_name})"
        else:
            next_open_text = f"Yarin {london_local_start.strftime('%H:%M')} ({timezone_name})"

    # ICT Killzone tespiti (UTC saatine göre)
    in_killzone   = False
    killzone_name = "—"
    for kz_name, kz_start, kz_end in _KILLZONES:
        if _is_between(t_utc, kz_start, kz_end):
            in_killzone   = True
            killzone_name = kz_name
            break

    return SessionStatus(
        now_text=now_local.strftime("%Y-%m-%d %H:%M:%S"),
        timezone=timezone_name,
        session_name=session_name,
        is_open=is_open,
        next_open_text=next_open_text,
        in_killzone=in_killzone,
        killzone_name=killzone_name,
    )
