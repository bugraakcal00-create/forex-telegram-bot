from __future__ import annotations

import logging
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CallbackContext, CommandHandler, ContextTypes, Defaults

from app.config import settings
from app.services.analysis_engine import AnalysisEngine, AnalysisResult
from app.services.backtest_service import BacktestService
from app.services.calendar_service import CalendarService
from app.services.market_data import MarketDataClient, MarketDataError
from app.services.news_service import NewsService
from app.services.session_service import get_session_status
from app.storage.sqlite_store import BotRepository

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

market = MarketDataClient(api_key=settings.twelvedata_api_key)
news_service = NewsService(api_key=settings.newsapi_api_key)
calendar_service = CalendarService(api_key=settings.fmp_api_key)
engine = AnalysisEngine()
backtest_service = BacktestService(engine=engine)
repo = BotRepository(db_path=Path(settings.db_path))

QUALITY_RANK = {"A": 4, "B": 3, "C": 2, "D": 1}


def parse_symbol_and_tf(args: list[str]) -> tuple[str, str]:
    symbol = args[0].upper() if args else "XAUUSD"
    timeframe = args[1].lower() if len(args) > 1 else "5min"
    return symbol, timeframe


def session_filter_enabled() -> bool:
    return repo.get_setting("session_filter_enabled", "1") == "1"


def quality_meets_min(current_quality: str, min_quality: str) -> bool:
    return QUALITY_RANK.get(current_quality.upper(), 0) >= QUALITY_RANK.get(min_quality.upper(), 0)


def session_text() -> str:
    status = get_session_status(settings.default_timezone)
    return (
        f"Sunucu zamani: {status.now_text} ({status.timezone})\n"
        "Londra: 10:00 - 19:00\n"
        "New York: 15:30 - 24:00\n"
        f"Durum: {status.session_name}\n"
        f"Session filtresi: {'ACIK' if session_filter_enabled() else 'KAPALI'}\n"
        f"Ultra selective mode: {'ACIK' if settings.ultra_selective_mode else 'KAPALI'}\n"
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


async def build_signal_result(symbol: str, timeframe: str) -> tuple[AnalysisResult, list[dict[str, str]]]:
    events = await calendar_service.get_upcoming_high_impact_events(hours_ahead=2, limit=3)
    df = await market.fetch_candles(symbol=symbol, interval=timeframe, outputsize=300)
    higher_tf = higher_timeframe_for(timeframe)
    higher_df = await market.fetch_candles(symbol=symbol, interval=higher_tf, outputsize=300)

    result = engine.analyze(
        symbol=symbol.upper(),
        df=df,
        timeframe=timeframe,
        higher_tf_df=higher_df,
        high_impact_events=events,
    )
    return result, events


def format_signal(result: AnalysisResult, events: list[dict[str, str]]) -> str:
    support_text = ", ".join(f"{x:.5f}" for x in result.support) if result.support else "Yok"
    resistance_text = ", ".join(f"{x:.5f}" for x in result.resistance) if result.resistance else "Yok"

    msg = (
        f"<b>{result.symbol} - {result.timeframe}</b>\n"
        f"Sinyal: <b>{result.signal}</b>\n"
        f"Kalite: <b>{result.quality}</b>\n"
        f"Setup Score: <b>{result.setup_score}/100</b>\n"
        f"Ana trend: {result.trend}\n"
        f"Ust TF trend: {result.higher_tf_trend}\n"
        f"Fiyat: {result.current_price:.5f}\n"
        f"RSI: {result.rsi:.2f}\n"
        f"ATR: {result.atr:.5f}\n"
        f"Destekler: {support_text}\n"
        f"Direncler: {resistance_text}\n"
        f"Giris bolgesi: {result.entry_zone[0]:.5f} - {result.entry_zone[1]:.5f}\n"
        f"Stop: {result.stop_loss:.5f}\n"
        f"TP1: {result.take_profit:.5f}\n"
        f"TP2: {result.take_profit_2:.5f}\n"
        f"R/R: {result.rr_ratio}\n"
        f"Sweep: {result.sweep_signal}\n"
        f"Sniper Entry: {result.sniper_entry}\n"
        f"Neden: {result.reason}\n"
    )

    if result.no_trade_reasons:
        msg += "\n<b>No-trade filtreleri</b>\n"
        for reason in result.no_trade_reasons:
            msg += f"- {reason}\n"

    if events:
        msg += "\n<b>Yaklasan yuksek etkili veriler</b>\n"
        for event in events:
            msg += f"- {event['date']} | {event['country']} | {event['event']}\n"

    return msg


def _apply_ultra_selective_gate(result: AnalysisResult, events: list[dict[str, str]]) -> AnalysisResult:
    if not settings.ultra_selective_mode:
        return result

    reasons: list[str] = []
    if result.signal == "NO TRADE":
        reasons.append("Ana kurulum yok")
    if result.quality != "A":
        reasons.append("Kalite A degil")
    if result.setup_score < 85:
        reasons.append("Setup score 85 alti")
    if result.rr_ratio < 2.2:
        reasons.append("R/R 2.2 alti")
    if result.sweep_signal == "Yok":
        reasons.append("Sweep onayi yok")
    if result.sniper_entry == "Yok":
        reasons.append("Sniper onayi yok")
    if events:
        reasons.append("Yuksek etkili haber riski")

    if not reasons:
        return result

    merged_reasons = list(dict.fromkeys(result.no_trade_reasons + reasons))
    gated_reason = "ULTRA_SELECTIVE filtre: " + " | ".join(merged_reasons)
    return AnalysisResult(
        symbol=result.symbol,
        trend=result.trend,
        higher_tf_trend=result.higher_tf_trend,
        current_price=result.current_price,
        support=result.support,
        resistance=result.resistance,
        entry_zone=result.entry_zone,
        stop_loss=result.stop_loss,
        take_profit=result.take_profit,
        take_profit_2=result.take_profit_2,
        rr_ratio=result.rr_ratio,
        signal="NO TRADE",
        timeframe=result.timeframe,
        reason=gated_reason,
        atr=result.atr,
        rsi=result.rsi,
        setup_score=result.setup_score,
        quality=result.quality,
        sweep_signal=result.sweep_signal,
        sniper_entry=result.sniper_entry,
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
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
        "/backtest XAUUSD 5min"
    )
    await update.message.reply_text(text)


async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        symbol, timeframe = parse_symbol_and_tf(context.args)
        status = get_session_status(settings.default_timezone)
        if session_filter_enabled() and not status.is_open:
            await update.message.reply_text(
                f"Session disi. Simdi: {status.session_name}\nSonraki acilis: {status.next_open_text}"
            )
            return

        result, events = await build_signal_result(symbol, timeframe)
        result = _apply_ultra_selective_gate(result, events)
        log_signal(
            source="manual_signal",
            chat_id=update.effective_chat.id if update.effective_chat else None,
            symbol=symbol,
            timeframe=timeframe,
            result=result,
            events=events,
        )
        await update.message.reply_text(
            format_signal(result, events),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
    except MarketDataError as exc:
        await update.message.reply_text(f"Hata: {exc}")
    except Exception as exc:
        logger.exception("signal error")
        await update.message.reply_text(f"Beklenmeyen hata: {exc}")


async def levels(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        symbol, timeframe = parse_symbol_and_tf(context.args)
        df = await market.fetch_candles(symbol=symbol, interval=timeframe, outputsize=250)
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

    await update.message.reply_text(
        text,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )


async def session_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(session_text())


async def session_filter_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    repo.set_setting("session_filter_enabled", "1")
    await update.message.reply_text("Session filtresi acildi.")


async def session_filter_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    repo.set_setting("session_filter_enabled", "0")
    await update.message.reply_text("Session filtresi kapandi.")


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
        df = await market.fetch_candles(symbol=symbol, interval=timeframe, outputsize=700)
        higher_df = await market.fetch_candles(
            symbol=symbol,
            interval=higher_timeframe_for(timeframe),
            outputsize=700,
        )
        bt = backtest_service.run(symbol=symbol, timeframe=timeframe, df=df, higher_df=higher_df)
        text = (
            f"<b>Backtest: {symbol} {timeframe}</b>\n"
            f"Tested signals: {bt.tested_signals}\n"
            f"Wins: {bt.wins}\n"
            f"Losses: {bt.losses}\n"
            f"No result: {bt.no_result}\n"
            f"Win rate: %{bt.winrate}\n"
            f"Avg RR: {bt.avg_rr}\n"
            f"Expectancy: {bt.expectancy}"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)
    except Exception as exc:
        logger.exception("backtest error")
        await update.message.reply_text(f"Backtest hatasi: {exc}")


async def alert_scan_job(context: CallbackContext) -> None:
    status = get_session_status(settings.default_timezone)
    if session_filter_enabled() and not status.is_open:
        logger.info("alert scan skipped: session closed (%s)", status.session_name)
        return

    events = await calendar_service.get_upcoming_high_impact_events(hours_ahead=2, limit=3)
    min_quality = repo.get_setting("min_quality_for_alert", "A")
    min_score = int(repo.get_setting("min_score_for_alert", "80"))
    min_rr = float(repo.get_setting("min_rr_for_alert", "2.0"))
    if settings.ultra_selective_mode:
        min_quality = "A"
        min_score = max(min_score, 85)
        min_rr = max(min_rr, 2.2)

    for chat_id_str, items in repo.iter_all_watches().items():
        chat_id = int(chat_id_str)
        for item in items:
            try:
                df = await market.fetch_candles(
                    item["symbol"],
                    interval=item["timeframe"],
                    outputsize=250,
                )
                higher_tf = higher_timeframe_for(item["timeframe"])
                higher_df = await market.fetch_candles(
                    item["symbol"],
                    interval=higher_tf,
                    outputsize=250,
                )

                result = engine.analyze(
                    symbol=item["symbol"],
                    df=df,
                    timeframe=item["timeframe"],
                    higher_tf_df=higher_df,
                    high_impact_events=events,
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

                near_support = bool(result.support) and abs(result.current_price - result.support[-1]) <= result.atr * 0.45
                near_resistance = bool(result.resistance) and abs(result.current_price - result.resistance[0]) <= result.atr * 0.45

                if (
                    result.signal != "NO TRADE"
                    and quality_meets_min(result.quality, min_quality)
                    and result.setup_score >= min_score
                    and result.rr_ratio >= min_rr
                    and (near_support or near_resistance)
                    and result.sweep_signal != "Yok"
                    and result.sniper_entry != "Yok"
                ):
                    text = (
                        "<b>Sniper Scalp Alarm</b>\n"
                        f"{result.symbol} {result.timeframe}\n"
                        f"Sinyal: <b>{result.signal}</b>\n"
                        f"Kalite: <b>{result.quality}</b>\n"
                        f"Score: <b>{result.setup_score}/100</b>\n"
                        f"Fiyat: {result.current_price:.5f}\n"
                        f"Giris: {result.entry_zone[0]:.5f} - {result.entry_zone[1]:.5f}\n"
                        f"SL: {result.stop_loss:.5f}\n"
                        f"TP1: {result.take_profit:.5f}\n"
                        f"TP2: {result.take_profit_2:.5f}\n"
                        f"R/R: {result.rr_ratio}\n"
                        f"Sweep: {result.sweep_signal}\n"
                        f"Sniper: {result.sniper_entry}\n"
                        f"Sebep: {result.reason}"
                    )
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=text,
                        parse_mode=ParseMode.HTML,
                    )
            except Exception as exc:
                logger.warning("alert scan failed for %s: %s", item, exc)


async def daily_plan_job(context: CallbackContext) -> None:
    events = await calendar_service.get_upcoming_high_impact_events(hours_ahead=2, limit=3)
    for chat_id in repo.get_daily_subscribers():
        chunks: list[str] = ["<b>Gunluk forex scalp plani</b>"]
        for symbol in settings.default_pairs:
            try:
                df = await market.fetch_candles(symbol=symbol, interval="15min", outputsize=250)
                higher_df = await market.fetch_candles(
                    symbol=symbol,
                    interval=higher_timeframe_for("15min"),
                    outputsize=250,
                )

                result = engine.analyze(
                    symbol=symbol,
                    df=df,
                    timeframe="15min",
                    higher_tf_df=higher_df,
                    high_impact_events=events,
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
                    f"\n<b>{symbol}</b> | {result.signal} | Kalite: {result.quality} | "
                    f"Score: {result.setup_score}/100 | Fiyat: {result.current_price:.5f} | "
                    f"R/R: {result.rr_ratio} | Sweep: {result.sweep_signal} | Sniper: {result.sniper_entry}"
                )
            except Exception as exc:
                chunks.append(f"\n<b>{symbol}</b> | veri alinamadi: {exc}")

        await context.bot.send_message(
            chat_id=chat_id,
            text="".join(chunks),
            parse_mode=ParseMode.HTML,
        )


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

    application.job_queue.run_repeating(
        alert_scan_job,
        interval=settings.alert_scan_minutes * 60,
        first=20,
    )
    daily_time = time(hour=settings.daily_plan_hour, minute=0, tzinfo=tzinfo)
    application.job_queue.run_daily(
        daily_plan_job,
        time=daily_time,
    )
    return application
