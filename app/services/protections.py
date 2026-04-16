"""
Protections — Prop Firm disiplin kuralları (Freqtrade pattern).

Her koruma fonksiyonu (chat_id, repo) alıp (allowed: bool, reason: str) döner.
Bot alert_scan_job içinde sinyal göndermeden önce tüm korumaları kontrol eder.

Kapsam:
  - StoplossGuard: bugün N SL → trade yok
  - MaxDailyLoss: bugün net RR <= -threshold → dur
  - MaxDrawdown: peak equity'den -%N → dur
  - MaxDailyTrades: günde N+ işlem → dur
  - CooldownPeriod: son SL'den sonra M dk bekle
  - MinRRGate: sinyal R:R < 2.0 → reddet (hard gate)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


# ── Konfigürasyon ──────────────────────────────────────────────────────────
DEFAULT_MAX_DAILY_SL_COUNT = 2           # günde 2 SL → dur
DEFAULT_MAX_DAILY_LOSS_RR = -3.0         # günlük net -3R → dur (≈ -3% equity %1 risk ile)
DEFAULT_MAX_DRAWDOWN_PCT = 6.0           # peak'ten -%6 → dur
DEFAULT_MAX_DAILY_TRADES = 2             # günde max 2 işlem (kalite > miktar)
DEFAULT_COOLDOWN_AFTER_SL_MINUTES = 60   # SL sonrası 60 dk bekle
DEFAULT_MIN_RR = 2.0                     # minimum 2:1 R:R


@dataclass
class ProtectionCheck:
    allowed: bool
    reason: str = ""
    protection: str = ""  # hangi koruma tetikledi


def check_stoploss_guard(repo, chat_id: int, max_sl: int = DEFAULT_MAX_DAILY_SL_COUNT) -> ProtectionCheck:
    """Bugün belirli sayıdan fazla SL yendiyse durdur."""
    try:
        today = repo.get_today_trade_stats(chat_id)
        sl_count = int(today.get("losses", 0))
    except Exception:
        return ProtectionCheck(allowed=True)
    if sl_count >= max_sl:
        return ProtectionCheck(
            allowed=False,
            reason=f"Gunluk SL limiti ({sl_count}/{max_sl}) — trade yok",
            protection="StoplossGuard",
        )
    return ProtectionCheck(allowed=True)


def check_max_daily_loss(repo, chat_id: int, max_loss_rr: float = DEFAULT_MAX_DAILY_LOSS_RR) -> ProtectionCheck:
    """Bugünkü net RR belirli eşiğin altına düştüyse durdur."""
    try:
        today = repo.get_today_trade_stats(chat_id)
        net_rr = float(today.get("net_rr", 0.0))
    except Exception:
        return ProtectionCheck(allowed=True)
    if net_rr <= max_loss_rr:
        return ProtectionCheck(
            allowed=False,
            reason=f"Gunluk zarar limiti asildi (net RR={net_rr:.1f} <= {max_loss_rr})",
            protection="MaxDailyLoss",
        )
    return ProtectionCheck(allowed=True)


def check_max_drawdown(repo, chat_id: int, max_dd_pct: float = DEFAULT_MAX_DRAWDOWN_PCT) -> ProtectionCheck:
    """Tüm trade geçmişinde peak equity'den drawdown kontrolü.

    Equity = kümülatif RR (1R = %1 risk varsayımı). Peak'ten düşüş > max_dd_pct → dur.
    """
    try:
        stats = repo.get_trade_stats(chat_id)
        # Kronolojik RR serisi üret
        with repo._connect() as conn:
            rows = conn.execute(
                "SELECT rr FROM trades WHERE chat_id = ? ORDER BY created_at ASC",
                (chat_id,),
            ).fetchall()
        if not rows:
            return ProtectionCheck(allowed=True)

        equity = 0.0
        peak = 0.0
        for r in rows:
            equity += float(r["rr"])
            if equity > peak:
                peak = equity
        dd = peak - equity  # bugünkü drawdown (R cinsinden)
        # 1R ≈ %1 equity varsayımı
        if dd >= max_dd_pct:
            return ProtectionCheck(
                allowed=False,
                reason=f"Max drawdown limiti ({dd:.1f}R >= {max_dd_pct}%) — dur",
                protection="MaxDrawdown",
            )
    except Exception:
        pass
    return ProtectionCheck(allowed=True)


def check_max_daily_trades(repo, chat_id: int, max_trades: int = DEFAULT_MAX_DAILY_TRADES) -> ProtectionCheck:
    """Günlük toplam sinyal sayısı aşıldıysa dur.

    Hem kazanan hem kaybeden hem pending trade'leri sayar.
    """
    try:
        today = datetime.now().date().isoformat()
        with repo._connect() as conn:
            # Bugünkü LONG/SHORT sinyal sayısı (signal_logs)
            cnt = conn.execute(
                "SELECT COUNT(*) AS c FROM signal_logs WHERE chat_id = ? "
                "AND signal IN ('LONG','SHORT') AND created_at LIKE ?",
                (chat_id, f"{today}%"),
            ).fetchone()
            total = int(cnt["c"]) if cnt else 0
    except Exception:
        return ProtectionCheck(allowed=True)
    if total >= max_trades:
        return ProtectionCheck(
            allowed=False,
            reason=f"Gunluk max islem sayisi ({total}/{max_trades}) — kalite > miktar",
            protection="MaxDailyTrades",
        )
    return ProtectionCheck(allowed=True)


def check_cooldown_after_sl(repo, chat_id: int, minutes: int = DEFAULT_COOLDOWN_AFTER_SL_MINUTES) -> ProtectionCheck:
    """Son SL'den sonra belirli dakika bekle (revenge trade önleme)."""
    try:
        with repo._connect() as conn:
            row = conn.execute(
                "SELECT created_at FROM trades WHERE chat_id = ? AND result = 'loss' "
                "ORDER BY created_at DESC LIMIT 1",
                (chat_id,),
            ).fetchone()
        if row is None:
            return ProtectionCheck(allowed=True)
        last_sl_str = str(row["created_at"])
        try:
            last_sl = datetime.fromisoformat(last_sl_str)
        except Exception:
            return ProtectionCheck(allowed=True)
        if last_sl.tzinfo is not None:
            # naive'e indirge (her ikisi de lokal varsayımı)
            last_sl = last_sl.replace(tzinfo=None)
        diff = (datetime.now() - last_sl).total_seconds() / 60.0
        if diff < minutes:
            return ProtectionCheck(
                allowed=False,
                reason=f"SL sonrasi bekleme ({diff:.0f}dk/{minutes}dk)",
                protection="CooldownAfterSL",
            )
    except Exception:
        pass
    return ProtectionCheck(allowed=True)


def check_min_rr(rr_ratio: float, min_rr: float = DEFAULT_MIN_RR) -> ProtectionCheck:
    """Sinyal R:R hard gate."""
    if rr_ratio < min_rr:
        return ProtectionCheck(
            allowed=False,
            reason=f"R:R cok dusuk ({rr_ratio:.2f} < {min_rr})",
            protection="MinRRGate",
        )
    return ProtectionCheck(allowed=True)


# ── Ana dispatcher ─────────────────────────────────────────────────────────
def run_all_protections(repo, chat_id: int, rr_ratio: float | None = None) -> ProtectionCheck:
    """Tüm korumaları sırayla kontrol et. İlk başarısızlıkta dön."""
    checks = [
        check_stoploss_guard(repo, chat_id),
        check_max_daily_loss(repo, chat_id),
        check_max_drawdown(repo, chat_id),
        check_max_daily_trades(repo, chat_id),
        check_cooldown_after_sl(repo, chat_id),
    ]
    if rr_ratio is not None:
        checks.append(check_min_rr(rr_ratio))

    for c in checks:
        if not c.allowed:
            return c
    return ProtectionCheck(allowed=True)
