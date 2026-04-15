from __future__ import annotations

import logging
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CallbackContext, CallbackQueryHandler, CommandHandler, ContextTypes, Defaults

from app.logging_config import setup_logging
setup_logging()

from app.config import settings
from app.services.analysis_engine import AnalysisEngine, AnalysisResult
from app.services.backtest_service import BacktestService
from app.services.calendar_service import CalendarService
from app.services.cot_service import get_cot_bias_sync
from app.services.market_data import MarketDataClient, MarketDataError
from app.services.ml_filter import ml_filter
from app.services.news_service import NewsService
from app.services.risk_manager import risk_manager
from app.services.sentiment_service import analyze_news_list
from app.services.session_service import get_session_status
from app.services.retail_sentiment import get_retail_sentiment, get_contrarian_filter
from app.services.simulation_service import SimulationService
from app.storage.sqlite_store import BotRepository

# ── En iyi strateji haritası (optimizer sonuçlarından) ──────────────────────
# Sadece WR >= 50% ve sinyal >= 3 olan kombinasyonlar kabul edilir.
# Diğer sembol/TF kombinasyonları sinyal ÜRETMEZ.
_best_strategies_cache: dict[str, dict] = {}  # key -> {strategy, wr, pf, signals}
_best_strategies_loaded = False
_MIN_WR_FOR_SIGNAL = 50.0  # Minimum win rate eşiği


def _load_best_strategies() -> dict[str, dict]:
    """data/best_strategies.json'dan yüksek WR'li stratejileri yükle."""
    global _best_strategies_cache, _best_strategies_loaded
    if _best_strategies_loaded:
        return _best_strategies_cache
    try:
        import json as _j
        best_file = Path("data/best_strategies.json")
        if best_file.exists():
            data = _j.loads(best_file.read_text(encoding="utf-8"))
            for key, val in data.items():
                wr = float(val.get("wr", 0))
                signals = int(val.get("signals", 0))
                # Sadece WR >= 50% ve en az 3 sinyal olanları kabul et
                if wr >= _MIN_WR_FOR_SIGNAL and signals >= 3:
                    _best_strategies_cache[key] = {
                        "strategy": str(val.get("strategy", "default")),
                        "wr": wr,
                        "pf": float(val.get("pf", 0)),
                        "signals": signals,
                    }
            logging.getLogger(__name__).info(
                "Loaded %d high-WR strategies (>=%s%%) from optimizer",
                len(_best_strategies_cache), _MIN_WR_FOR_SIGNAL,
            )
    except Exception:
        pass
    _best_strategies_loaded = True
    return _best_strategies_cache


def get_best_strategy(symbol: str, timeframe: str) -> str | None:
    """Sembol/TF için en iyi strateji modunu döndür. WR düşükse None döner."""
    strategies = _load_best_strategies()
    key = f"{symbol.upper()}_{timeframe.lower()}"
    entry = strategies.get(key)
    if entry:
        return entry["strategy"]
    return None  # Bu combo'dan sinyal gelmesin


def is_high_wr_combo(symbol: str, timeframe: str) -> bool:
    """Bu sembol/TF kombinasyonu yüksek WR'li mi?"""
    strategies = _load_best_strategies()
    key = f"{symbol.upper()}_{timeframe.lower()}"
    return key in strategies

logger = logging.getLogger(__name__)

market = MarketDataClient(api_key=settings.twelvedata_api_key)
news_service = NewsService(api_key=settings.newsapi_api_key)
calendar_service = CalendarService(api_key=settings.fmp_api_key)
engine = AnalysisEngine()
backtest_service = BacktestService(engine=engine)
repo = BotRepository(db_path=Path(settings.db_path))
sim_service = SimulationService(db_path=Path(settings.db_path))

QUALITY_RANK = {"A": 4, "B": 3, "C": 2, "D": 1}
_MAX_TG_MSG = 4096


async def _safe_reply(message, text: str, **kwargs) -> None:
    """Telegram 4096 karakter limitini aşan mesajları böler."""
    if message is None:
        return
    if len(text) <= _MAX_TG_MSG:
        await message.reply_text(text, **kwargs)
        return
    # HTML parse modunda bölme — tag'ler kırılabilir, düz metin olarak gönder
    chunks = [text[i:i + _MAX_TG_MSG] for i in range(0, len(text), _MAX_TG_MSG)]
    for i, chunk in enumerate(chunks):
        kw = kwargs.copy() if i == len(chunks) - 1 else {k: v for k, v in kwargs.items() if k != "reply_markup"}
        try:
            await message.reply_text(chunk, **kw)
        except Exception:
            await message.reply_text(chunk)


def parse_symbol_and_tf(args: list[str]) -> tuple[str, str]:
    symbol = args[0].upper() if args else "XAUUSD"
    timeframe = args[1].lower() if len(args) > 1 else "5min"
    return symbol, timeframe


def session_filter_enabled() -> bool:
    return repo.get_setting("session_filter_enabled", "1") == "1"


def quality_meets_min(current_quality: str, min_quality: str) -> bool:
    return QUALITY_RANK.get(current_quality.upper(), 0) >= QUALITY_RANK.get(min_quality.upper(), 0)


def signal_label(signal: str) -> str:
    mapping = {
        "LONG": "AL",
        "SHORT": "SAT",
        "NO TRADE": "ISLEM YOK",
    }
    return mapping.get(signal, signal)


def session_text() -> str:
    status = get_session_status(settings.default_timezone)
    return (
        f"Sunucu zamani: {status.now_text} ({status.timezone})\n"
        "Londra: 10:00 - 19:00\n"
        "New York: 15:30 - 24:00\n"
        f"Durum: {status.session_name}\n"
        f"Seans filtresi: {'ACIK' if session_filter_enabled() else 'KAPALI'}\n"
        f"Ultra secici mod: {'ACIK' if settings.ultra_selective_mode else 'KAPALI'}\n"
        f"Sonraki acilis: {status.next_open_text}"
    )


def higher_timeframe_for(timeframe: str) -> str:
    mapping = {
        "1min": "5min",
        "5min": "15min",
        "15min": "1h",
        "30min": "4h",
        "45min": "4h",
        "1h": "4h",
        "2h": "1day",
        "4h": "1day",
        "1day": "1week",
    }
    return mapping.get(timeframe.lower(), "1h")


def _parse_event_datetime(raw: str) -> datetime | None:
    text = str(raw).strip()
    if not text:
        return None
    candidates = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in candidates:
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def news_lock_events(events: list[dict[str, str]], lock_minutes: int) -> list[dict[str, str]]:
    now_utc = datetime.now(timezone.utc)
    limit = now_utc + timedelta(minutes=lock_minutes)
    locked: list[dict[str, str]] = []
    for event in events:
        dt = _parse_event_datetime(event.get("date", ""))
        if dt is None:
            continue
        if now_utc <= dt <= limit:
            locked.append(event)
    return locked


async def _fetch_dxy_bias() -> str:
    """DXY trendini tespit eder. XAUUSD/majors için bias filtresi."""
    try:
        dxy_df = await market.fetch_candles("DXY", interval="1h", outputsize=60)
        if dxy_df is not None and len(dxy_df) >= 20:
            ema20 = dxy_df["close"].ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = dxy_df["close"].ewm(span=50, adjust=False).mean().iloc[-1]
            if ema20 > ema50:
                return "BULLISH"
            elif ema20 < ema50:
                return "BEARISH"
    except Exception:
        pass
    return "NEUTRAL"


async def _fetch_cot_bias(symbol: str) -> str:
    """COT (CFTC) bias — sync wrapper, thread pool'da çalışır."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, get_cot_bias_sync, symbol)
    except Exception:
        return "NEUTRAL"


async def _fetch_sentiment() -> float:
    """Haber sentiment skoru (-1.0 … +1.0)."""
    try:
        articles = await news_service.get_forex_news()
        if articles:
            result = analyze_news_list(articles)
            return result.gold_score
    except Exception:
        pass
    return 0.0


async def _fetch_retail_bias(symbol: str) -> str:
    """Retail trader pozisyonlama — contrarian bias."""
    try:
        result = await get_retail_sentiment(symbol)
        return result.bias
    except Exception:
        return "NEUTRAL"


async def build_signal_result(symbol: str, timeframe: str) -> tuple[AnalysisResult, list[dict[str, str]]]:
    import asyncio as _asyncio
    events = await calendar_service.get_upcoming_high_impact_events(hours_ahead=2, limit=3)
    locked_events = news_lock_events(events, settings.news_lock_minutes)
    df, higher_df, dxy_bias, cot_bias, sentiment_score, retail_bias = await _asyncio.gather(
        market.fetch_candles(symbol=symbol, interval=timeframe, outputsize=settings.candle_output_size),
        market.fetch_candles(symbol=symbol, interval=higher_timeframe_for(timeframe), outputsize=settings.candle_output_size),
        _fetch_dxy_bias(),
        _fetch_cot_bias(symbol),
        _fetch_sentiment(),
        _fetch_retail_bias(symbol),
    )

    result = engine.analyze(
        symbol=symbol.upper(),
        df=df,
        timeframe=timeframe,
        higher_tf_df=higher_df,
        high_impact_events=locked_events,
        dxy_bias=dxy_bias,
        cot_bias=cot_bias,
        sentiment_score=sentiment_score,
    )

    # ML filtresi uygula (eğitilmişse)
    ml_result = ml_filter.predict(result, df=df)
    from dataclasses import replace
    result = replace(result, ml_probability=ml_result.probability)
    if ml_result.is_trained and not ml_result.should_trade and result.signal != "NO TRADE":
        result = replace(
            result,
            signal="NO TRADE",
            no_trade_reasons=result.no_trade_reasons + [f"ML filtre: P(win)={ml_result.probability:.2f} < esik"],
            reason=f"ML filtre engelledi (P={ml_result.probability:.2f}) | {result.reason}",
        )

    # Retail contrarian filtre
    if result.signal != "NO TRADE" and get_contrarian_filter(retail_bias, result.signal):
        result = replace(
            result,
            signal="NO TRADE",
            no_trade_reasons=result.no_trade_reasons + [f"Retail kalabalık aynı yönde ({retail_bias})"],
            reason=f"Contrarian filtre: retail {retail_bias} → {result.signal} engellendi | {result.reason}",
        )

    return result, events


def format_signal(result: AnalysisResult, events: list[dict[str, str]]) -> str:
    support_text    = ", ".join(f"{x:.5f}" for x in result.support)    if result.support    else "Yok"
    resistance_text = ", ".join(f"{x:.5f}" for x in result.resistance) if result.resistance else "Yok"
    macd_arrow = ">" if result.macd_hist > 0 else "<"
    conf_str   = f"x{result.smc_confluence_count}" if result.smc_confluence_count else "--"

    # Sentiment gösterimi
    sent = getattr(result, "sentiment_score", 0.0)
    if sent >= 0.2:
        sent_str = f"+{sent:.2f} Yukselis"
    elif sent <= -0.2:
        sent_str = f"{sent:.2f} Dusus"
    else:
        sent_str = f"{sent:.2f} Notr"

    # Volume analizi
    vol = getattr(result, "volume_analysis", {}) or {}
    vol_str = f"{vol.get('delta_bias','?')} | Trend:{vol.get('volume_trend','?')} | Ratio:{vol.get('last_volume_ratio',1.0):.1f}x"
    if vol.get("volume_spike"):
        vol_str += " [SPIKE]"

    # ML filtresi
    ml_prob = getattr(result, "ml_probability", 0.55)
    ml_str  = f"{ml_prob:.0%}" if ml_filter.is_trained() else "Egitilmedi"

    cot_bias = getattr(result, "cot_bias", "NEUTRAL")

    msg = (
        f"<b>{result.symbol} - {result.timeframe}</b>\n"
        f"Sinyal: <b>{signal_label(result.signal)}</b>  |  Kalite: <b>{result.quality}</b>  |  Skor: <b>{result.setup_score}/100</b>\n"
        f"SMC Uyum: <b>{conf_str}</b>  |  DXY: <b>{result.dxy_bias}</b>  |  COT: <b>{cot_bias}</b>\n"
        f"Sentiment: {sent_str}  |  ML: {ml_str}  |  Rejim: {result.regime}\n"
        f"Ana trend: {result.trend}  |  Ust TF: {result.higher_tf_trend}\n"
        f"Fiyat: <code>{result.current_price:.5f}</code>  |  ATR: {result.atr:.5f}\n"
        f"RSI: {result.rsi:.2f}  |  MACD: {macd_arrow} {result.macd_hist:+.6f}\n"
        f"BB: {result.bb_lower:.2f} / {result.bb_mid:.2f} / {result.bb_upper:.2f}\n"
        f"Hacim: {vol_str}\n"
        f"Destekler: {support_text}\n"
        f"Direncler: {resistance_text}\n"
        f"Giris: <code>{result.entry_zone[0]:.5f} - {result.entry_zone[1]:.5f}</code>\n"
        f"SL: <code>{result.stop_loss:.5f}</code>  |  TP1: <code>{result.take_profit:.5f}</code>  |  TP2: <code>{result.take_profit_2:.5f}</code>\n"
        f"R/R: <b>{result.rr_ratio}</b>  |  Sweep: {result.sweep_signal}\n"
        f"Sniper: {result.sniper_entry}\n"
    )

    # Kismi TP yapisi (sadece LONG/SHORT icin)
    if result.signal in ("LONG", "SHORT"):
        direction = "long" if result.signal == "LONG" else "short"
        ptp = risk_manager.partial_tp_structure(
            entry=float(result.entry_zone[0] + result.entry_zone[1]) / 2,
            stop_loss=result.stop_loss,
            direction=direction,
        )
        msg += (
            f"\n<b>Kismi TP Plani</b>\n"
            f"TP1 (%50 kapat): <code>{ptp.tp1:.5f}</code>  [1.5R]\n"
            f"TP2 (trail et): <code>{ptp.tp2:.5f}</code>  [2.5R]\n"
            f"BE Tetikleyici: <code>{ptp.breakeven_at:.5f}</code>\n"
            f"Agirlikli R/R: <b>{ptp.expected_rr}</b>\n"
        )

    # PDH/PDL bilgisi
    pdh_pdl = getattr(result, "pdh_pdl", {}) or {}
    pdh = pdh_pdl.get("prev_day_high", 0)
    pdl = pdh_pdl.get("prev_day_low", 0)
    if pdh > 0 or pdl > 0:
        msg += f"\n<b>Gunluk Seviyeler</b>\n"
        if pdh > 0: msg += f"PDH: <code>{pdh:.5f}</code>\n"
        if pdl > 0: msg += f"PDL: <code>{pdl:.5f}</code>\n"

    # ── Yeni Kurumsal Seviyeler ──
    # VWAP
    vwap = getattr(result, "vwap", 0.0) or 0.0
    if vwap > 0:
        vwap_pos = "ÜSTÜ" if result.current_price > vwap else "ALTI"
        msg += f"VWAP: <code>{vwap:.5f}</code> (fiyat {vwap_pos})\n"

    # Unicorn Model
    unicorn = getattr(result, "unicorn_model", {}) or {}
    if unicorn.get("detected"):
        msg += f"🦄 <b>UNICORN MODEL:</b> {unicorn['type']} @ {unicorn['zone_bottom']:.2f}-{unicorn['zone_top']:.2f}\n"

    # Silver Bullet
    sb = getattr(result, "silver_bullet", {}) or {}
    if sb.get("active"):
        msg += f"🔫 <b>{sb['window']}</b> aktif ({sb['fvg_count']} FVG)\n"

    # AMD Phase
    amd = getattr(result, "amd_phase", {}) or {}
    if amd.get("phase") and amd["phase"] != "UNKNOWN":
        phase_icon = {"ACCUMULATION": "📦", "MANIPULATION": "🎭", "DISTRIBUTION": "💰"}.get(amd["phase"], "❓")
        swept_tag = " [Asia Swept]" if amd.get("asia_swept") else ""
        msg += f"{phase_icon} Faz: {amd['phase']}{swept_tag}\n"

    # IPDA Levels
    ipda = getattr(result, "ipda_levels", {}) or {}
    if ipda.get("nearest") and ipda.get("distance_atr", 99) <= 3.0:
        msg += f"🏦 IPDA yakın seviye: <code>{ipda['nearest']:.5f}</code> ({ipda['distance_atr']:.1f} ATR uzakta)\n"

    # ── SMC Blok ──
    smc_lines = []
    if result.premium_discount and result.premium_discount.get("zone"):
        icon = "🔴" if result.premium_discount["zone"] == "PREMIUM" else "🟢"
        smc_lines.append(f"{icon} P/D: {result.premium_discount['zone']}")
    if result.choch and result.choch.get("detected"):
        smc_lines.append(f"⚡ CHoCH: {result.choch.get('type','')} @ {result.choch.get('price',0):.5f}")
    if result.displacement and result.displacement.get("detected"):
        disp_dir = "▲" if result.displacement.get("direction") == "bullish" else "▼"
        smc_lines.append(f"💥 Displacement: {disp_dir} {result.displacement.get('strength','')}")
    if result.ote_zone and result.ote_zone.get("valid"):
        smc_lines.append(f"🎯 OTE: {result.ote_zone['ote_low']:.5f} - {result.ote_zone['ote_high']:.5f}")
    if result.bos_mss and result.bos_mss.get("bos"):
        smc_lines.append(f"📈 BOS: {result.bos_mss.get('type','')} @ {result.bos_mss.get('level',0):.5f}")
    if result.bos_mss and result.bos_mss.get("mss"):
        smc_lines.append(f"🔄 MSS: {result.bos_mss.get('type','')} @ {result.bos_mss.get('level',0):.5f}")
    if result.judas_swing and result.judas_swing.get("detected"):
        smc_lines.append(f"🎭 Judas Swing ({result.judas_swing.get('direction','')}) → {result.judas_swing.get('sweep_level',0):.5f}")
    if result.confirmation_candle and result.confirmation_candle.get("detected"):
        smc_lines.append(f"✅ Onay: {result.confirmation_candle.get('type','')} ({result.confirmation_candle.get('strength',0)}%)")
    if result.equal_highs:
        smc_lines.append(f"EH: {', '.join(f'{v:.5f}' for v in result.equal_highs[:2])}")
    if result.equal_lows:
        smc_lines.append(f"EL: {', '.join(f'{v:.5f}' for v in result.equal_lows[:2])}")

    if smc_lines:
        msg += "\n<b>SMC/ICT</b>\n" + "\n".join(f"• {l}" for l in smc_lines) + "\n"

    # ── Yapılar ──
    if result.order_blocks:
        msg += "\n<b>Order Block'lar</b>\n"
        for ob in result.order_blocks[:3]:
            ob_type = "Bullish OB" if ob["type"] == "bullish_ob" else "Bearish OB"
            broken_tag = " [KIRILI→BB]" if ob.get("broken") else " [TAZE]"
            near_tag = " ★" if ob.get("near_price") else ""
            msg += f"• {ob_type}{broken_tag}{near_tag}: {ob['bottom']:.2f}–{ob['top']:.2f}\n"

    if result.breaker_blocks:
        msg += "\n<b>Breaker Block'lar</b>\n"
        for bb in result.breaker_blocks[:2]:
            bb_type = "Bullish BB" if bb["type"] == "bullish_breaker" else "Bearish BB"
            near_tag = " ★" if bb.get("near_price") else ""
            msg += f"• {bb_type}{near_tag}: {bb['bottom']:.2f}–{bb['top']:.2f}\n"

    if result.fvg_zones:
        open_fvg = [f for f in result.fvg_zones if not f.get("filled")]
        if open_fvg:
            msg += "\n<b>FVG (Açık)</b>\n"
            for fvg in open_fvg[:3]:
                fvg_type = "Bullish" if fvg["type"] == "bullish_fvg" else "Bearish"
                near_tag = " ★" if fvg.get("near_price") else ""
                msg += f"• {fvg_type} FVG{near_tag}: {fvg['bottom']:.2f}–{fvg['top']:.2f}\n"

    if result.ifvg_zones:
        msg += "\n<b>iFVG (Inversion)</b>\n"
        for ifvg in result.ifvg_zones[:2]:
            ifvg_type = "Bullish" if ifvg["type"] == "bullish_ifvg" else "Bearish"
            near_tag = " ★" if ifvg.get("near_price") else ""
            msg += f"• {ifvg_type} iFVG{near_tag}: {ifvg['bottom']:.2f}–{ifvg['top']:.2f}\n"

    if result.no_trade_reasons:
        msg += "\n<b>Filtreler</b>\n"
        for reason in result.no_trade_reasons:
            msg += f"⚠ {reason}\n"

    if events:
        msg += "\n<b>Yaklaşan Haberler</b>\n"
        for event in events:
            msg += f"• {event.get('date','')[:16]} | {event.get('country','')} | {event.get('event','')}\n"

    msg += f"\nSebep: {result.reason}"
    return msg


def signal_keyboard(symbol: str, timeframe: str, show_result_buttons: bool = False) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton("📊 Grafik", callback_data=f"chart_{symbol}_{timeframe}"),
            InlineKeyboardButton("➕ Izlemeye Ekle", callback_data=f"watch_{symbol}_{timeframe}"),
        ],
        [
            InlineKeyboardButton("🔄 Yenile", callback_data=f"refresh_{symbol}_{timeframe}"),
            InlineKeyboardButton("📈 Backtest", callback_data=f"bt_{symbol}_{timeframe}"),
        ],
    ]
    if show_result_buttons:
        rows.append([
            InlineKeyboardButton("✅ TP1 Aldı", callback_data=f"tp1_{symbol}_{timeframe}"),
            InlineKeyboardButton("✅ TP2 Aldı", callback_data=f"tp2_{symbol}_{timeframe}"),
        ])
        rows.append([
            InlineKeyboardButton("❌ SL Yedi", callback_data=f"slhit_{symbol}_{timeframe}"),
            InlineKeyboardButton("🔄 BE Kapattı", callback_data=f"be_{symbol}_{timeframe}"),
        ])
    return InlineKeyboardMarkup(rows)


def _apply_ultra_selective_gate(result: AnalysisResult, events: list[dict[str, str]]) -> AnalysisResult:
    if not settings.ultra_selective_mode:
        return result

    reasons: list[str] = []
    if result.signal == "NO TRADE":
        reasons.append("Ana kurulum yok")
    if result.quality not in ("A", "B", "C"):
        reasons.append("Kalite D — yetersiz")
    if result.setup_score < 55:
        reasons.append("Setup score 55 alti")
    if events:
        reasons.append("Yuksek etkili haber riski")

    if not reasons:
        return result

    from dataclasses import replace
    merged_reasons = list(dict.fromkeys(result.no_trade_reasons + reasons))
    gated_reason = "ULTRA_SELECTIVE filtre: " + " | ".join(merged_reasons)
    return replace(
        result,
        signal="NO TRADE",
        reason=gated_reason,
        no_trade_reasons=merged_reasons,
    )


def log_signal(
    *,
    source: str,
    chat_id: int | None,
    symbol: str,
    timeframe: str,
    result: AnalysisResult,
    events: list[dict[str, str]],
) -> None:
    status = get_session_status(settings.default_timezone)
    repo.add_signal_log(
        chat_id=chat_id,
        source=source,
        symbol=symbol,
        timeframe=timeframe,
        signal=result.signal,
        quality=result.quality,
        setup_score=result.setup_score,
        rr_ratio=result.rr_ratio,
        current_price=result.current_price,
        trend=result.trend,
        higher_tf_trend=result.higher_tf_trend,
        sweep_signal=result.sweep_signal,
        sniper_entry=result.sniper_entry,
        reason=result.reason,
        no_trade_reasons=result.no_trade_reasons,
        session_name=status.session_name,
        is_session_open=status.is_open,
        had_high_impact_event=bool(events),
        entry_zone=result.entry_zone,
        stop_loss=result.stop_loss,
        take_profit=result.take_profit,
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    text = (
        "Forex scalp bot hazir.\n\n"
        "Komutlar:\n"
        "/signal XAUUSD 5min\n"
        "/levels XAUUSD 15min\n"
        "/news\n"
        "/session\n"
        "/session_filter_on\n"
        "/session_filter_off\n"
        "/plan XAUUSD 5min\n"
        "/risk 1000 1 3030 3018\n"
        "/watch XAUUSD 5min\n"
        "/unwatch XAUUSD 5min\n"
        "/watchlist\n"
        "/subscribe_daily\n"
        "/unsubscribe_daily\n"
        "/logwin XAUUSD 5min 2.1\n"
        "/logloss XAUUSD 5min -1\n"
        "/stats\n"
        "/todaystats\n"
        "/backtest XAUUSD 5min\n"
        "/chart XAUUSD 5min\n"
        "/weeklyreport"
    )
    await update.message.reply_text(text)


async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    try:
        symbol, timeframe = parse_symbol_and_tf(context.args)
        status = get_session_status(settings.default_timezone)
        if session_filter_enabled() and not status.is_open:
            await update.message.reply_text(
                f"Seans disi. Simdi: {status.session_name}\nSonraki acilis: {status.next_open_text}"
            )
            return

        # Günlük risk limiti kontrolü
        chat_id = update.effective_chat.id if update.effective_chat else None
        if chat_id:
            today_stats = repo.get_today_trade_stats(chat_id)
            daily_loss = int(today_stats.get("losses", 0))
            if daily_loss >= 3:
                await update.message.reply_text(
                    "⛔ Günlük kayıp limiti aşıldı (3 SL). Bugün yeni işlem açmayın.\n"
                    f"Bugünkü durum: {today_stats.get('wins',0)}W / {daily_loss}L"
                )
                return

        result, events = await build_signal_result(symbol, timeframe)
        result = _apply_ultra_selective_gate(result, events)
        log_signal(
            source="manual_signal",
            chat_id=chat_id,
            symbol=symbol,
            timeframe=timeframe,
            result=result,
            events=events,
        )
        show_result = result.signal in ("LONG", "SHORT")
        await _safe_reply(
            update.message,
            format_signal(result, events),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=signal_keyboard(symbol, timeframe, show_result_buttons=show_result),
        )
    except MarketDataError as exc:
        await update.message.reply_text(f"Hata: {exc}")
    except Exception as exc:
        logger.exception("signal error")
        await update.message.reply_text(f"Beklenmeyen hata: {exc}")


async def levels(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        symbol, timeframe = parse_symbol_and_tf(context.args)
        df = await market.fetch_candles(symbol=symbol, interval=timeframe, outputsize=settings.candle_output_size)
        result = engine.analyze(symbol=symbol.upper(), df=df, timeframe=timeframe)
        text = (
            f"{result.symbol} {result.timeframe}\n"
            f"Destekler: {', '.join(f'{x:.5f}' for x in result.support) if result.support else 'Yok'}\n"
            f"Direncler: {', '.join(f'{x:.5f}' for x in result.resistance) if result.resistance else 'Yok'}"
        )
        await update.message.reply_text(text)
    except Exception as exc:
        logger.exception("levels error")
        await update.message.reply_text(f"Hata: {exc}")


async def news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    try:
        articles = await news_service.get_forex_news()
        events = await calendar_service.get_upcoming_high_impact_events()

        text = "<b>Forex Haber Ozeti</b>\n"
        if articles:
            for item in articles:
                text += f"- <a href='{item['url']}'>{item['title']}</a> | {item['source']}\n"
        else:
            text += "- Haber API anahtari yok ya da veri gelmedi.\n"

        text += "\n<b>Yaklasan yuksek etkili veriler</b>\n"
        if events:
            for event in events:
                text += f"- {event['date']} | {event['country']} | {event['event']}\n"
        else:
            text += "- Takvim API anahtari yok ya da veri gelmedi.\n"

        await _safe_reply(
            update.message,
            text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
    except Exception as exc:
        logger.exception("news error")
        await update.message.reply_text(f"Haber hatasi: {exc}")


async def session_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(session_text())


async def session_filter_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    repo.set_setting("session_filter_enabled", "1")
    await update.message.reply_text("Seans filtresi acildi.")


async def session_filter_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    repo.set_setting("session_filter_enabled", "0")
    await update.message.reply_text("Seans filtresi kapandi.")


async def plan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        symbol, timeframe = parse_symbol_and_tf(context.args)
        result, events = await build_signal_result(symbol, timeframe)
        result = _apply_ultra_selective_gate(result, events)
        plan_text = (
            f"{format_signal(result, events)}\n"
            "<b>Gunluk plan</b>\n"
            "1. Sadece A kalite veya guclu B kalite setup al.\n"
            "2. Haber saatine 15 dk kala yeni pozisyon acma.\n"
            "3. Sweep veya sniper entry yoksa acele etme.\n"
            "4. Gunluk max 2 kayip sonrasi dur.\n"
            "5. Max risk %1."
        )
        await update.message.reply_text(
            plan_text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
    except Exception as exc:
        logger.exception("plan error")
        await update.message.reply_text(f"Hata: {exc}")


async def risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        balance = float(context.args[0])
        risk_pct = float(context.args[1])
        entry = float(context.args[2])
        stop = float(context.args[3])
        text = engine.risk_text(balance, risk_pct, entry, stop)
        await update.message.reply_text(text)
    except Exception:
        await update.message.reply_text("Kullanim: /risk 1000 1 3030 3018")


async def watch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbol, timeframe = parse_symbol_and_tf(context.args)
    repo.add_watch(update.effective_chat.id, symbol.upper(), timeframe)
    await update.message.reply_text(f"Izleme eklendi: {symbol.upper()} {timeframe}")


async def unwatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbol = (context.args[0] if context.args else "XAUUSD").upper()
    timeframe = context.args[1].lower() if len(context.args) > 1 else None
    removed = repo.remove_watch(update.effective_chat.id, symbol, timeframe)
    await update.message.reply_text("Silindi." if removed else "Listede yok.")


async def watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    items = repo.get_watches(update.effective_chat.id)
    if not items:
        await update.message.reply_text("Izleme listesi bos.")
        return
    text = "Izleme listesi:\n" + "\n".join(f"- {x['symbol']} {x['timeframe']}" for x in items)
    await update.message.reply_text(text)


async def subscribe_daily(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    repo.subscribe_daily(update.effective_chat.id)
    await update.message.reply_text("Gunluk plan aboneligi acildi.")


async def unsubscribe_daily(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    repo.unsubscribe_daily(update.effective_chat.id)
    await update.message.reply_text("Gunluk plan aboneligi kapatildi.")


async def logwin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        symbol = context.args[0].upper()
        timeframe = context.args[1].lower()
        rr = float(context.args[2])
        repo.add_trade(update.effective_chat.id, symbol, timeframe, "win", rr)
        await update.message.reply_text(f"Kazanan islem kaydedildi: {symbol} {timeframe} | RR: {rr}")
    except Exception:
        await update.message.reply_text("Kullanim: /logwin XAUUSD 5min 2.1")


async def logloss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        symbol = context.args[0].upper()
        timeframe = context.args[1].lower()
        rr = float(context.args[2])
        repo.add_trade(update.effective_chat.id, symbol, timeframe, "loss", rr)
        await update.message.reply_text(f"Zarar eden islem kaydedildi: {symbol} {timeframe} | RR: {rr}")
    except Exception:
        await update.message.reply_text("Kullanim: /logloss XAUUSD 5min -1")


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    data = repo.get_trade_stats(update.effective_chat.id)
    text = (
        "<b>Genel Performans</b>\n"
        f"Toplam islem: {data['total']}\n"
        f"Kazanc: {data['wins']}\n"
        f"Zarar: {data['losses']}\n"
        f"Win rate: %{data['winrate']}\n"
        f"Net RR: {data['net_rr']}\n"
    )

    if data["by_symbol"]:
        text += "\n<b>Sembol bazli</b>\n"
        for symbol, row in data["by_symbol"].items():
            text += (
                f"- {symbol} | Toplam: {row['total']} | "
                f"W: {row['wins']} | L: {row['losses']} | Net RR: {round(row['net_rr'], 2)}\n"
            )

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def todaystats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    data = repo.get_today_trade_stats(update.effective_chat.id)
    text = (
        "<b>Bugunku Performans</b>\n"
        f"Toplam islem: {data['total']}\n"
        f"Kazanc: {data['wins']}\n"
        f"Zarar: {data['losses']}\n"
        f"Win rate: %{data['winrate']}\n"
        f"Net RR: {data['net_rr']}\n"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def backtest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        symbol, timeframe = parse_symbol_and_tf(context.args)
        df = await market.fetch_candles(symbol=symbol, interval=timeframe, outputsize=settings.backtest_output_size)
        higher_df = await market.fetch_candles(
            symbol=symbol,
            interval=higher_timeframe_for(timeframe),
            outputsize=settings.backtest_output_size,
        )
        bt = backtest_service.run(symbol=symbol, timeframe=timeframe, df=df, higher_df=higher_df)
        text = (
            f"<b>Geriye donuk test: {symbol} {timeframe}</b>\n"
            f"Test edilen sinyal: {bt.tested_signals}\n"
            f"Kazanc: {bt.wins}\n"
            f"Zarar: {bt.losses}\n"
            f"Sonucsuz: {bt.no_result}\n"
            f"Kazanma orani: %{bt.winrate}\n"
            f"Ortalama RR: {bt.avg_rr}\n"
            f"Beklenen deger: {bt.expectancy}"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)
    except Exception as exc:
        logger.exception("backtest error")
        await update.message.reply_text(f"Geriye donuk test hatasi: {exc}")


async def chart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        symbol, timeframe = parse_symbol_and_tf(context.args)
        await update.message.reply_text("Grafik hazirlanıyor, biraz bekleyin...")
        import asyncio as _asyncio
        df, higher_df, dxy_bias = await _asyncio.gather(
            market.fetch_candles(symbol=symbol, interval=timeframe, outputsize=settings.candle_output_size),
            market.fetch_candles(symbol=symbol, interval=higher_timeframe_for(timeframe), outputsize=settings.candle_output_size),
            _fetch_dxy_bias(),
        )
        result = engine.analyze(
            symbol=symbol.upper(),
            df=df,
            timeframe=timeframe,
            higher_tf_df=higher_df,
            dxy_bias=dxy_bias,
        )
        from app.services.chart_service import generate_signal_chart
        chart_bytes = generate_signal_chart(df, result, bars=100)
        caption = format_signal(result, [])
        await update.message.reply_photo(
            photo=chart_bytes,
            caption=caption[:1024],
            parse_mode=ParseMode.HTML,
            reply_markup=signal_keyboard(symbol, timeframe),
        )
    except Exception as exc:
        logger.exception("chart error")
        await update.message.reply_text(f"Grafik olusturma hatasi: {exc}")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = str(query.data)
    parts = data.split("_", 2)
    if len(parts) < 3:
        return
    action, symbol, timeframe = parts[0], parts[1], parts[2]

    if action == "chart":
        try:
            import asyncio as _asyncio
            df, higher_df, dxy_bias = await _asyncio.gather(
                market.fetch_candles(symbol=symbol, interval=timeframe, outputsize=settings.candle_output_size),
                market.fetch_candles(symbol=symbol, interval=higher_timeframe_for(timeframe), outputsize=settings.candle_output_size),
                _fetch_dxy_bias(),
            )
            result = engine.analyze(
                symbol=symbol.upper(), df=df, timeframe=timeframe,
                higher_tf_df=higher_df, dxy_bias=dxy_bias,
            )
            from app.services.chart_service import generate_signal_chart
            chart_bytes = generate_signal_chart(df, result, bars=100)
            caption = format_signal(result, [])
            await query.message.reply_photo(
                photo=chart_bytes,
                caption=caption[:1024],
                parse_mode=ParseMode.HTML,
                reply_markup=signal_keyboard(symbol, timeframe),
            )
        except Exception as exc:
            await query.message.reply_text(f"Grafik hatasi: {exc}")

    elif action == "watch":
        chat_id = query.message.chat_id if query.message else None
        if chat_id:
            repo.add_watch(chat_id, symbol.upper(), timeframe)
            await query.message.reply_text(f"Izleme eklendi: {symbol.upper()} {timeframe}")

    elif action == "refresh":
        try:
            status = get_session_status(settings.default_timezone)
            if session_filter_enabled() and not status.is_open:
                await query.message.reply_text(
                    f"Seans disi. Simdi: {status.session_name}\nSonraki acilis: {status.next_open_text}"
                )
                return
            result, events = await build_signal_result(symbol, timeframe)
            result = _apply_ultra_selective_gate(result, events)
            await query.message.reply_text(
                format_signal(result, events),
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_markup=signal_keyboard(symbol, timeframe),
            )
        except Exception as exc:
            await query.message.reply_text(f"Yenileme hatasi: {exc}")

    elif action == "bt":
        try:
            df = await market.fetch_candles(symbol=symbol, interval=timeframe, outputsize=settings.backtest_output_size)
            higher_df = await market.fetch_candles(
                symbol=symbol, interval=higher_timeframe_for(timeframe), outputsize=settings.backtest_output_size
            )
            bt = backtest_service.run(symbol=symbol, timeframe=timeframe, df=df, higher_df=higher_df)
            text = (
                f"<b>Geriye donuk test: {symbol} {timeframe}</b>\n"
                f"Test edilen sinyal: {bt.tested_signals}\n"
                f"Kazanc: {bt.wins}\n"
                f"Zarar: {bt.losses}\n"
                f"Kazanma orani: %{bt.winrate}\n"
                f"Ortalama RR: {bt.avg_rr}\n"
                f"Beklenen deger: {bt.expectancy}"
            )
            await query.message.reply_text(text, parse_mode=ParseMode.HTML)
        except Exception as exc:
            await query.message.reply_text(f"Backtest hatasi: {exc}")

    elif action in ("tp1", "tp2", "slhit", "be"):
        # Sonuç geri bildirimi — kullanıcı butonla bildirir
        chat_id = query.message.chat_id if query.message else None
        if chat_id:
            result_map = {
                "tp1": ("win", 1.5, "TP1 alindi (%50 kapatildi)"),
                "tp2": ("win", 2.5, "TP2 alindi (tam kapatildi)"),
                "slhit": ("loss", -1.0, "SL yedi"),
                "be": ("win", 0.0, "Breakeven kapatildi"),
            }
            result_type, rr, note = result_map[action]
            repo.add_trade(chat_id, symbol.upper(), timeframe, result_type, rr)
            # Son pending sinyali resolve et
            pending = repo.get_pending_signal_logs(limit=5)
            for row in pending:
                if str(row["symbol"]).upper() == symbol.upper():
                    outcome = "tp_hit" if result_type == "win" and rr > 0 else ("sl_hit" if result_type == "loss" else "breakeven")
                    repo.resolve_signal_outcome(int(row["id"]), outcome, rr, note)
                    break
            await query.message.reply_text(f"✅ Kaydedildi: {symbol} {timeframe} → {note} (R/R: {rr:+.1f})")


async def weekly_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    report = repo.get_weekly_report()
    text = (
        "<b>Haftalik Rapor (7 gun)</b>\n"
        f"Toplam islem: {report['total_trades']}\n"
        f"Kazanc: {report['wins']}\n"
        f"Zarar: {report['losses']}\n"
        f"Kazanma orani: %{report['winrate']}\n"
        f"Net RR: {report['net_rr']}\n"
        f"TP ulasan sinyal: {report['tp_hit']}\n"
        f"SL ulasan sinyal: {report['sl_hit']}\n"
        f"Bekleyen sinyal: {report['pending']}"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


_TF_MINUTES: dict[str, int] = {
    "1min": 1, "5min": 5, "15min": 15, "30min": 30,
    "1h": 60, "4h": 240, "1day": 1440, "1week": 10080,
}

# Sinyal zaman dilimine göre maksimum bekleme süresi (saat)
_TF_EXPIRY_HOURS: dict[str, int] = {
    "1min": 4, "5min": 12, "15min": 24, "30min": 36,
    "1h": 72, "4h": 168, "1day": 720, "1week": 2160,
}


def _candles_needed(created_at_str: str, tf: str) -> int:
    """Sinyal oluşturulduğundan bu yana kaç mum geçti."""
    try:
        created = datetime.fromisoformat(created_at_str)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        elapsed_min = (datetime.now(timezone.utc) - created).total_seconds() / 60
        tf_min = _TF_MINUTES.get(tf, 5)
        return min(500, max(20, int(elapsed_min / tf_min) + 5))
    except Exception:
        return 50


def _is_expired(created_at_str: str, tf: str) -> bool:
    """Sinyal, zaman dilimi bazlı süre dolmuş mu?"""
    try:
        created = datetime.fromisoformat(created_at_str)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        elapsed_h = (datetime.now(timezone.utc) - created).total_seconds() / 3600
        return elapsed_h > _TF_EXPIRY_HOURS.get(tf, 72)
    except Exception:
        return False


def _check_candle_outcome(
    signal: str,
    tp: float,
    sl: float,
    rr_ratio: float,
    candle_open: float,
    candle_high: float,
    candle_low: float,
) -> tuple[str, float, str] | None:
    """
    Tek bir mum için TP/SL dokunuşunu kontrol eder.
    Aynı mumda her ikisi de tetiklenirse mumun open fiyatına göre
    hangisinin önce geldiğini tahmin eder (pessimistic).
    Dönüş: (outcome, realized_rr, note) veya None
    """
    if signal == "LONG":
        tp_hit = candle_high >= tp
        sl_hit = candle_low <= sl
    elif signal == "SHORT":
        tp_hit = candle_low <= tp
        sl_hit = candle_high >= sl
    else:
        return None

    if not tp_hit and not sl_hit:
        return None

    if tp_hit and sl_hit:
        # Her ikisi de aynı mumda: open → TP mi SL mi yakın?
        dist_tp = abs(candle_open - tp)
        dist_sl = abs(candle_open - sl)
        if dist_tp <= dist_sl:
            return ("tp_hit", rr_ratio, "Mum hem TP hem SL dokundu; TP daha yakindi")
        else:
            return ("sl_hit", -1.0, "Mum hem TP hem SL dokundu; SL daha yakindi (pessimistic)")

    if tp_hit:
        return ("tp_hit", rr_ratio, "Mum high/low TP seviyesine ulasti")
    return ("sl_hit", -1.0, "Mum high/low SL seviyesine ulasti")


async def sim_resolve_job(context: CallbackContext) -> None:
    """Simulasyon islemlerini kontrol et ve TP/SL hit olanları kapat."""
    try:
        open_trades = sim_service.get_open_trades()
        if not open_trades:
            return
        symbols = list({str(t["symbol"]) for t in open_trades})
        candle_data: dict[str, dict] = {}
        for symbol in symbols:
            try:
                df = await market.fetch_candles(symbol, interval="5min", outputsize=5)
                if len(df) > 0:
                    last = df.iloc[-1]
                    candle_data[symbol] = {
                        "high": float(last["high"]),
                        "low": float(last["low"]),
                        "open": float(last["open"]),
                        "close": float(last["close"]),
                    }
            except Exception:
                pass
        if candle_data:
            resolved = sim_service.check_and_resolve_trades(candle_data)
            if resolved:
                logger.info("sim resolve: %d trades closed", len(resolved))
    except Exception as exc:
        logger.warning("sim resolve error: %s", exc)


async def resolve_signal_outcomes_job(context: CallbackContext) -> None:
    pending = repo.get_pending_signal_logs(limit=80)

    # Sembolleri grupla → her sembol için tek API çağrısı
    from collections import defaultdict
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for row in pending:
        by_symbol[str(row["symbol"])].append(row)

    for symbol, rows in by_symbol.items():
        # Bu sembol için en fazla mum ihtiyacı olan satırı baz al
        max_candles = max(
            _candles_needed(str(r["created_at"]), str(r["timeframe"])) for r in rows
        )
        # Ortak timeframe: en küçük TF'i kullan (daha yüksek çözünürlük)
        min_tf = min(rows, key=lambda r: _TF_MINUTES.get(str(r["timeframe"]), 5))
        tf = str(min_tf["timeframe"])

        try:
            df = await market.fetch_candles(symbol, interval=tf, outputsize=min(500, max_candles))
        except Exception as exc:
            logger.warning("outcome: candle fetch failed %s: %s", symbol, exc)
            continue

        for row in rows:
            signal_log_id = int(row["id"])
            signal = str(row["signal"])
            tp = float(row["take_profit"])
            sl = float(row["stop_loss"])
            rr_ratio = float(row.get("rr_ratio", 0.0))
            created_at_str = str(row["created_at"])
            row_tf = str(row["timeframe"])

            # Süre dolmuş mu?
            if _is_expired(created_at_str, row_tf):
                repo.resolve_signal_outcome(
                    signal_log_id, "expired", 0.0,
                    f"Sinyal {_TF_EXPIRY_HOURS.get(row_tf, 72)}s icinde sonuclanmadi"
                )
                logger.info("outcome: expired signal_id=%s %s %s", signal_log_id, symbol, signal)
                continue

            # Sinyalin oluşturulduğu zamandan sonraki mumları filtrele
            try:
                created_dt = datetime.fromisoformat(created_at_str)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
            except Exception:
                created_dt = datetime.now(timezone.utc) - timedelta(hours=24)

            candles_after = df[df["datetime"] >= created_dt]
            if candles_after.empty:
                continue

            is_hyp = str(row.get("outcome", "pending")) == "hyp_pending"

            # Hipotetik sinyallerde yönü TP/SL'e göre tahmin et
            effective_signal = signal
            if is_hyp and signal not in ("LONG", "SHORT"):
                # Giriş bölgesi ortasını bul
                ep = float(row.get("current_price") or 0)
                if ep > 0:
                    effective_signal = "LONG" if tp > ep else "SHORT"
                else:
                    continue

            # Her mumu sırayla kontrol et — ilk tetiklenen kaydedilir
            resolved = False
            for _, candle in candles_after.iterrows():
                result = _check_candle_outcome(
                    signal=effective_signal,
                    tp=tp,
                    sl=sl,
                    rr_ratio=rr_ratio,
                    candle_open=float(candle["open"]),
                    candle_high=float(candle["high"]),
                    candle_low=float(candle["low"]),
                )
                if result:
                    raw_outcome, realized_rr, note = result
                    # Hipotetik sinyaller için farklı outcome değeri
                    if is_hyp:
                        final_outcome = "hyp_tp" if raw_outcome == "tp_hit" else "hyp_sl"
                        note = f"[Hipotez - islem alinmadi] {note}"
                    else:
                        final_outcome = raw_outcome
                    repo.resolve_signal_outcome(signal_log_id, final_outcome, realized_rr, note)
                    logger.info(
                        "outcome: %s signal_id=%s %s %s tp=%.5f sl=%.5f rr=%.2f",
                        final_outcome, signal_log_id, symbol, effective_signal, tp, sl, realized_rr,
                    )
                    resolved = True
                    break

            if not resolved:
                logger.debug("outcome: still pending signal_id=%s %s", signal_log_id, symbol)


async def alert_scan_job(context: CallbackContext) -> None:
    status = get_session_status(settings.default_timezone)
    if session_filter_enabled() and not status.is_open:
        logger.info("alert scan skipped: session closed (%s)", status.session_name)
        return

    # ICT Killzone filtresi: sadece yüksek olasılıklı zaman dilimlerinde işlem
    if session_filter_enabled() and not status.in_killzone:
        logger.info("alert scan skipped: not in ICT killzone (current: %s)", status.session_name)
        return

    events = await calendar_service.get_upcoming_high_impact_events(hours_ahead=2, limit=3)
    locked_events = news_lock_events(events, settings.news_lock_minutes)
    min_quality = repo.get_setting("min_quality_for_alert", "A")
    min_score = int(repo.get_setting("min_score_for_alert", "80"))
    min_rr = float(repo.get_setting("min_rr_for_alert", "2.0"))
    if settings.ultra_selective_mode:
        min_quality = "A"
        min_score = max(min_score, 85)
        min_rr = max(min_rr, 2.2)

    # DXY, COT ve sentiment bias'larını tek seferde çek (tüm semboller için paylaş)
    import asyncio as _asyncio2
    dxy_bias, sentiment_score_val = await _asyncio2.gather(
        _fetch_dxy_bias(),
        _fetch_sentiment(),
    )

    for chat_id_str, items in repo.iter_all_watches().items():
        chat_id = int(chat_id_str)

        # Günlük kayıp limiti: bugün 2'den fazla SL yediyse o chat için dur
        today_stats = repo.get_today_trade_stats(chat_id)
        daily_sl_count = int(today_stats.get("losses", 0))
        if daily_sl_count >= 2:
            logger.info("alert scan skipped for chat %s: daily loss limit reached (%s SL)", chat_id, daily_sl_count)
            continue

        for item in items:
            try:
                # Sadece yüksek WR'li kombinasyonlardan sinyal üret
                best_mode = get_best_strategy(item["symbol"], item["timeframe"])
                if best_mode is None:
                    continue  # Bu combo düşük WR, atla

                df = await market.fetch_candles(
                    item["symbol"],
                    interval=item["timeframe"],
                    outputsize=settings.candle_output_size,
                )
                higher_tf = higher_timeframe_for(item["timeframe"])
                higher_df = await market.fetch_candles(
                    item["symbol"],
                    interval=higher_tf,
                    outputsize=settings.candle_output_size,
                )

                cot_bias_val = await _fetch_cot_bias(item["symbol"])
                result = engine.analyze(
                    symbol=item["symbol"],
                    df=df,
                    timeframe=item["timeframe"],
                    higher_tf_df=higher_df,
                    high_impact_events=locked_events,
                    dxy_bias=dxy_bias,
                    cot_bias=cot_bias_val,
                    sentiment_score=sentiment_score_val,
                    strategy_mode=best_mode,
                )
                # ML filtresi
                ml_res = ml_filter.predict(result, df=df)
                from dataclasses import replace as _dc_replace
                result = _dc_replace(result, ml_probability=ml_res.probability)
                if ml_res.is_trained and not ml_res.should_trade and result.signal != "NO TRADE":
                    result = _dc_replace(
                        result,
                        signal="NO TRADE",
                        no_trade_reasons=result.no_trade_reasons + [f"ML filtre: P(win)={ml_res.probability:.2f} < esik"],
                        reason=f"ML filtre engelledi (P={ml_res.probability:.2f}) | {result.reason}",
                    )
                result = _apply_ultra_selective_gate(result, events)
                log_signal(
                    source="auto_alert_scan",
                    chat_id=chat_id,
                    symbol=item["symbol"],
                    timeframe=item["timeframe"],
                    result=result,
                    events=events,
                )

                # Yeni basit alert gate: score zaten her seyi iceriyor
                if (
                    result.signal != "NO TRADE"
                    and quality_meets_min(result.quality, min_quality)
                    and result.setup_score >= min_score
                    and result.rr_ratio >= min_rr
                ):
                    caption = (
                        "<b>🎯 Sniper Scalp Alarm</b>\n"
                        f"<b>{result.symbol} {result.timeframe}</b> ({best_mode})\n"
                        f"Sinyal: <b>{signal_label(result.signal)}</b>\n"
                        f"Kalite: <b>{result.quality}</b>  |  Skor: <b>{result.setup_score}/100</b>  |  R/R: <b>{result.rr_ratio}</b>\n"
                        f"Fiyat: <code>{result.current_price:.5f}</code>\n"
                        f"Giris: <code>{result.entry_zone[0]:.5f} - {result.entry_zone[1]:.5f}</code>\n"
                        f"SL: <code>{result.stop_loss:.5f}</code>  |  TP1: <code>{result.take_profit:.5f}</code>  |  TP2: <code>{result.take_profit_2:.5f}</code>\n"
                        f"Sweep: {result.sweep_signal}  |  Sniper: {result.sniper_entry}\n"
                        f"Rejim: {result.regime}  |  Trend: {result.trend}\n"
                    )
                    if result.choch and result.choch.get("detected"):
                        caption += f"⚡ CHoCH: {result.choch.get('type','')} @ <code>{result.choch.get('price',0):.5f}</code>\n"
                    if result.displacement and result.displacement.get("detected"):
                        disp_dir = "▲" if result.displacement.get("direction") == "bullish" else "▼"
                        caption += f"💥 Displacement: {disp_dir}\n"
                    if result.ote_zone and result.ote_zone.get("valid"):
                        caption += f"🎯 OTE: <code>{result.ote_zone['ote_low']:.5f} - {result.ote_zone['ote_high']:.5f}</code>\n"
                    if result.premium_discount and result.premium_discount.get("zone"):
                        pd_icon = "🔴" if result.premium_discount["zone"] == "PREMIUM" else "🟢"
                        caption += f"{pd_icon} P/D: {result.premium_discount['zone']}\n"
                    caption += f"Sebep: {result.reason}"

                    # Grafik oluştur ve fotoğraf olarak gönder
                    try:
                        from app.services.chart_service import generate_signal_chart
                        chart_bytes = generate_signal_chart(df, result)
                        await context.bot.send_photo(
                            chat_id=chat_id,
                            photo=chart_bytes,
                            caption=caption[:1024],
                            parse_mode=ParseMode.HTML,
                            reply_markup=signal_keyboard(item["symbol"], item["timeframe"]),
                        )
                    except Exception as chart_exc:
                        logger.warning("chart generation failed, sending text: %s", chart_exc)
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=caption,
                            parse_mode=ParseMode.HTML,
                            reply_markup=signal_keyboard(item["symbol"], item["timeframe"]),
                        )

                    # Auto-open simulation trade
                    try:
                        sim_trade_id = sim_service.open_trade(
                            symbol=result.symbol,
                            timeframe=item["timeframe"],
                            direction=result.signal,
                            entry_price=result.current_price,
                            stop_loss=result.stop_loss,
                            take_profit=result.take_profit,
                            strategy_mode=best_mode,
                        )
                        if sim_trade_id:
                            logger.info("sim trade opened: %s %s %s (id=%s)",
                                       result.symbol, result.signal, item["timeframe"], sim_trade_id)
                    except Exception as sim_exc:
                        logger.warning("sim trade open failed: %s", sim_exc)
            except Exception as exc:
                logger.warning("alert scan failed for %s: %s", item, exc)


async def daily_plan_job(context: CallbackContext) -> None:
    events = await calendar_service.get_upcoming_high_impact_events(hours_ahead=2, limit=3)
    locked_events = news_lock_events(events, settings.news_lock_minutes)
    for chat_id in repo.get_daily_subscribers():
        chunks: list[str] = ["<b>Gunluk forex scalp plani</b>"]
        for symbol in settings.default_pairs:
            try:
                df = await market.fetch_candles(symbol=symbol, interval="15min", outputsize=settings.candle_output_size)
                higher_df = await market.fetch_candles(
                    symbol=symbol,
                    interval=higher_timeframe_for("15min"),
                    outputsize=settings.candle_output_size,
                )

                result = engine.analyze(
                    symbol=symbol,
                    df=df,
                    timeframe="15min",
                    higher_tf_df=higher_df,
                    high_impact_events=locked_events,
                )
                result = _apply_ultra_selective_gate(result, events)
                log_signal(
                    source="daily_plan",
                    chat_id=chat_id,
                    symbol=symbol,
                    timeframe="15min",
                    result=result,
                    events=events,
                )
                chunks.append(
                    f"\n<b>{symbol}</b> | {signal_label(result.signal)} | Kalite: {result.quality} | "
                    f"Skor: {result.setup_score}/100 | Rejim: {result.regime} | Fiyat: {result.current_price:.5f} | "
                    f"R/R: {result.rr_ratio} | Sweep: {result.sweep_signal} | Sniper: {result.sniper_entry}"
                )
            except Exception as exc:
                chunks.append(f"\n<b>{symbol}</b> | veri alinamadi: {exc}")

        full_text = "".join(chunks)
        # Telegram 4096 karakter limitine dikkat
        if len(full_text) <= _MAX_TG_MSG:
            await context.bot.send_message(
                chat_id=chat_id,
                text=full_text,
                parse_mode=ParseMode.HTML,
            )
        else:
            for i in range(0, len(full_text), _MAX_TG_MSG):
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=full_text[i:i + _MAX_TG_MSG],
                    parse_mode=ParseMode.HTML,
                )


async def trade_monitor_job(context: CallbackContext) -> None:
    """Açık sinyalleri izle — TP1/BE/SL yaklaşınca bildirim gönder."""
    pending = repo.get_pending_signal_logs(limit=30)
    if not pending:
        return

    from collections import defaultdict
    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for row in pending:
        if str(row["signal"]) in ("LONG", "SHORT"):
            by_symbol[str(row["symbol"])].append(row)

    for symbol, rows in by_symbol.items():
        try:
            df = await market.fetch_candles(symbol, interval="1min", outputsize=5)
            if df.empty:
                continue
            current_price = float(df.iloc[-1]["close"])
        except Exception:
            continue

        for row in rows:
            chat_id = row.get("chat_id")
            if not chat_id:
                continue
            tp = float(row["take_profit"])
            sl = float(row["stop_loss"])
            signal = str(row["signal"])
            entry_mid = (float(row.get("entry_low", 0)) + float(row.get("entry_high", 0))) / 2
            if entry_mid == 0:
                entry_mid = float(row.get("current_price", current_price))

            risk = abs(entry_mid - sl)
            if risk < 1e-9:
                continue

            # TP1 seviyesi (1.5R)
            tp1 = entry_mid + risk * 1.5 if signal == "LONG" else entry_mid - risk * 1.5
            # BE seviyesi
            be_level = entry_mid + risk * 0.5 if signal == "LONG" else entry_mid - risk * 0.5

            try:
                # TP1'e yaklaşma (0.3 ATR içinde)
                if signal == "LONG" and current_price >= tp1 * 0.998:
                    await context.bot.send_message(
                        chat_id=int(chat_id),
                        text=f"🎯 <b>{symbol}</b> TP1 seviyesine ulaştı!\n"
                             f"Fiyat: <code>{current_price:.5f}</code>\n"
                             f"TP1: <code>{tp1:.5f}</code>\n"
                             f"Pozisyonun %50'sini kapatmayı düşünün!",
                        parse_mode=ParseMode.HTML,
                    )
                elif signal == "SHORT" and current_price <= tp1 * 1.002:
                    await context.bot.send_message(
                        chat_id=int(chat_id),
                        text=f"🎯 <b>{symbol}</b> TP1 seviyesine ulaştı!\n"
                             f"Fiyat: <code>{current_price:.5f}</code>\n"
                             f"TP1: <code>{tp1:.5f}</code>\n"
                             f"Pozisyonun %50'sini kapatmayı düşünün!",
                        parse_mode=ParseMode.HTML,
                    )

                # SL'e yaklaşma uyarısı (risk mesafesinin %20'si kaldı)
                if signal == "LONG" and current_price <= sl + risk * 0.2 and current_price > sl:
                    await context.bot.send_message(
                        chat_id=int(chat_id),
                        text=f"⚠️ <b>{symbol}</b> SL'e yaklaşıyor!\n"
                             f"Fiyat: <code>{current_price:.5f}</code>\n"
                             f"SL: <code>{sl:.5f}</code>",
                        parse_mode=ParseMode.HTML,
                    )
                elif signal == "SHORT" and current_price >= sl - risk * 0.2 and current_price < sl:
                    await context.bot.send_message(
                        chat_id=int(chat_id),
                        text=f"⚠️ <b>{symbol}</b> SL'e yaklaşıyor!\n"
                             f"Fiyat: <code>{current_price:.5f}</code>\n"
                             f"SL: <code>{sl:.5f}</code>",
                        parse_mode=ParseMode.HTML,
                    )
            except Exception as exc:
                logger.debug("trade monitor notify failed: %s", exc)


async def weekly_retrain_job(context: CallbackContext) -> None:
    """Haftalık ML model yeniden eğitimi — her Pazar 03:00."""
    logger.info("Haftalık ML yeniden eğitim başlıyor...")
    try:
        labeled = repo.get_labeled_signal_logs(limit=2000)
        n_labeled = len(labeled)
        if n_labeled < 50:
            logger.info("Yeterli veri yok (%d/50), yeniden eğitim atlandı.", n_labeled)
            return
        success = ml_filter.train(repo, min_samples=50)
        if success:
            logger.info("ML model yeniden eğitildi (%d sample)", n_labeled)
            # Admin'e bildir
            for chat_id in repo.get_daily_subscribers():
                try:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"🤖 ML model haftalık eğitimi tamamlandı.\n"
                             f"Kullanılan veri: {n_labeled} sinyal\n"
                             f"Model güncellendi.",
                    )
                except Exception:
                    pass
        else:
            logger.info("ML yeniden eğitim başarısız — min_samples karşılanmadı.")
    except Exception as exc:
        logger.warning("Haftalık ML eğitim hatası: %s", exc)


def build_application() -> Application:
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN eksik.")

    tzinfo = ZoneInfo(settings.default_timezone)
    defaults = Defaults(tzinfo=tzinfo)
    application = Application.builder().token(settings.telegram_bot_token).defaults(defaults).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(CommandHandler("signal", signal))
    application.add_handler(CommandHandler("levels", levels))
    application.add_handler(CommandHandler("news", news))
    application.add_handler(CommandHandler("session", session_cmd))
    application.add_handler(CommandHandler("session_filter_on", session_filter_on))
    application.add_handler(CommandHandler("session_filter_off", session_filter_off))
    application.add_handler(CommandHandler("plan", plan))
    application.add_handler(CommandHandler("risk", risk))
    application.add_handler(CommandHandler("watch", watch))
    application.add_handler(CommandHandler("unwatch", unwatch))
    application.add_handler(CommandHandler("watchlist", watchlist))
    application.add_handler(CommandHandler("subscribe_daily", subscribe_daily))
    application.add_handler(CommandHandler("unsubscribe_daily", unsubscribe_daily))
    application.add_handler(CommandHandler("logwin", logwin))
    application.add_handler(CommandHandler("logloss", logloss))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("todaystats", todaystats))
    application.add_handler(CommandHandler("backtest", backtest))
    application.add_handler(CommandHandler("weeklyreport", weekly_report))
    application.add_handler(CommandHandler("chart", chart))
    application.add_handler(CallbackQueryHandler(button_callback))

    application.job_queue.run_repeating(
        alert_scan_job,
        interval=settings.alert_scan_minutes * 60,
        first=20,
    )
    application.job_queue.run_repeating(
        resolve_signal_outcomes_job,
        interval=max(settings.alert_scan_minutes, 2) * 60,
        first=35,
    )
    application.job_queue.run_repeating(
        trade_monitor_job,
        interval=60,  # Her 60 saniyede bir kontrol
        first=45,
    )
    application.job_queue.run_repeating(
        sim_resolve_job,
        interval=120,  # Her 2 dakikada sim islemleri kontrol et
        first=55,
    )
    # Haftalık ML yeniden eğitim — Pazar 03:00
    retrain_time = time(hour=3, minute=0, tzinfo=tzinfo)
    application.job_queue.run_daily(
        weekly_retrain_job,
        time=retrain_time,
        days=(6,),  # 6 = Pazar
    )
    daily_time = time(hour=settings.daily_plan_hour, minute=0, tzinfo=tzinfo)
    application.job_queue.run_daily(
        daily_plan_job,
        time=daily_time,
    )
    return application
