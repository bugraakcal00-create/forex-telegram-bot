from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackContext,
    CommandHandler,
    ContextTypes,
)

from app.config import settings
from app.services.analysis_engine import AnalysisEngine, AnalysisResult
from app.services.calendar_service import CalendarService
from app.services.market_data import MarketDataClient, MarketDataError
from app.services.news_service import NewsService
from app.storage.trade_journal import TradeJournal
from app.storage.watch_store import WatchStore

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

market = MarketDataClient(api_key=settings.twelvedata_api_key)
news_service = NewsService(api_key=settings.newsapi_api_key)
calendar_service = CalendarService(api_key=settings.fmp_api_key)
engine = AnalysisEngine()
store = WatchStore(Path("data/watchlists.json"))
journal = TradeJournal(Path("data/trade_journal.json"))


def parse_symbol_and_tf(args: list[str]) -> tuple[str, str]:
    symbol = args[0].upper() if args else "XAUUSD"
    timeframe = args[1].lower() if len(args) > 1 else "15min"
    return symbol, timeframe


def session_text() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"Sunucu zamanı: {now}\n"
        "Londra yaklaşık: 10:00 - 19:00 TSİ\n"
        "New York yaklaşık: 15:30 - 24:00 TSİ\n"
        "Overlap: 15:30 - 19:00 TSİ\n"
        "Not: Yaz/kış saati değişimlerinde 1 saat oynama olabilir."
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


async def build_signal_message(symbol: str, timeframe: str) -> str:
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
    return format_signal(result, events)


def format_signal(result: AnalysisResult, events: list[dict[str, str]]) -> str:
    support_text = ", ".join(f"{x:.5f}" for x in result.support) if result.support else "Yok"
    resistance_text = ", ".join(f"{x:.5f}" for x in result.resistance) if result.resistance else "Yok"

    msg = (
        f"<b>{result.symbol} - {result.timeframe}</b>\n"
        f"Sinyal: <b>{result.signal}</b>\n"
        f"Kalite: <b>{result.quality}</b>\n"
        f"Setup Score: <b>{result.setup_score}/100</b>\n"
        f"Ana trend: {result.trend}\n"
        f"Üst TF trend: {result.higher_tf_trend}\n"
        f"Fiyat: {result.current_price:.5f}\n"
        f"RSI: {result.rsi:.2f}\n"
        f"ATR: {result.atr:.5f}\n"
        f"Destekler: {support_text}\n"
        f"Dirençler: {resistance_text}\n"
        f"Giriş bölgesi: {result.entry_zone[0]:.5f} - {result.entry_zone[1]:.5f}\n"
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
        msg += "\n<b>Yaklaşan yüksek etkili veriler</b>\n"
        for event in events:
            msg += f"- {event['date']} | {event['country']} | {event['event']}\n"

    return msg


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Forex scalp bot hazır.\n\n"
        "Komutlar:\n"
        "/signal XAUUSD 5min\n"
        "/signal XAUUSD 15min\n"
        "/levels XAUUSD 15min\n"
        "/news\n"
        "/session\n"
        "/plan XAUUSD 5min\n"
        "/risk 1000 1 3030 3018\n"
        "/watch XAUUSD 5min\n"
        "/watch XAUUSD 15min\n"
        "/unwatch XAUUSD\n"
        "/watchlist\n"
        "/subscribe_daily\n"
        "/unsubscribe_daily\n"
        "/logwin XAUUSD 5min 2.1\n"
        "/logloss XAUUSD 5min -1\n"
        "/stats\n"
        "/todaystats"
    )
    await update.message.reply_text(text)


async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        symbol, timeframe = parse_symbol_and_tf(context.args)
        message = await build_signal_message(symbol, timeframe)
        await update.message.reply_text(
            message,
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
            f"Dirençler: {', '.join(f'{x:.5f}' for x in result.resistance) if result.resistance else 'Yok'}"
        )
        await update.message.reply_text(text)
    except Exception as exc:
        logger.exception("levels error")
        await update.message.reply_text(f"Hata: {exc}")


async def news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    articles = await news_service.get_forex_news()
    events = await calendar_service.get_upcoming_high_impact_events()

    text = "<b>Forex Haber Özeti</b>\n"
    if articles:
        for item in articles:
            text += f"- <a href='{item['url']}'>{item['title']}</a> | {item['source']}\n"
    else:
        text += "- Haber API anahtarı yok ya da veri gelmedi.\n"

    text += "\n<b>Yaklaşan yüksek etkili veriler</b>\n"
    if events:
        for event in events:
            text += f"- {event['date']} | {event['country']} | {event['event']}\n"
    else:
        text += "- Takvim API anahtarı yok ya da veri gelmedi.\n"

    await update.message.reply_text(
        text,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )


async def session_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(session_text())


async def plan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        symbol, timeframe = parse_symbol_and_tf(context.args)
        signal_text = await build_signal_message(symbol, timeframe)
        plan_text = (
            f"{signal_text}\n"
            "<b>Günlük plan</b>\n"
            "1. Sadece A kalite veya güçlü B kalite setup al.\n"
            "2. Haber saatine 15 dk kala yeni pozisyon açma.\n"
            "3. Sweep veya sniper entry yoksa acele etme.\n"
            "4. Günlük max 2 kayıp sonrası dur.\n"
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
        await update.message.reply_text("Kullanım: /risk 1000 1 3030 3018")


async def watch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbol, timeframe = parse_symbol_and_tf(context.args)
    store.add_watch(update.effective_chat.id, symbol.upper(), timeframe)
    await update.message.reply_text(f"İzleme eklendi: {symbol.upper()} {timeframe}")


async def unwatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbol = (context.args[0] if context.args else "XAUUSD").upper()
    removed = store.remove_watch(update.effective_chat.id, symbol)
    await update.message.reply_text("Silindi." if removed else "Listede yok.")


async def watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    items = store.get_watches(update.effective_chat.id)
    if not items:
        await update.message.reply_text("İzleme listesi boş.")
        return
    text = "İzleme listesi:\n" + "\n".join(f"- {x['symbol']} {x['timeframe']}" for x in items)
    await update.message.reply_text(text)


async def subscribe_daily(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    store.subscribe_daily(update.effective_chat.id)
    await update.message.reply_text("Günlük plan aboneliği açıldı.")


async def unsubscribe_daily(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    store.unsubscribe_daily(update.effective_chat.id)
    await update.message.reply_text("Günlük plan aboneliği kapatıldı.")


async def logwin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        symbol = context.args[0].upper()
        timeframe = context.args[1].lower()
        rr = float(context.args[2])
        journal.add_trade(update.effective_chat.id, symbol, timeframe, "win", rr)
        await update.message.reply_text(f"Kazanan işlem kaydedildi: {symbol} {timeframe} | RR: {rr}")
    except Exception:
        await update.message.reply_text("Kullanım: /logwin XAUUSD 5min 2.1")


async def logloss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        symbol = context.args[0].upper()
        timeframe = context.args[1].lower()
        rr = float(context.args[2])
        journal.add_trade(update.effective_chat.id, symbol, timeframe, "loss", rr)
        await update.message.reply_text(f"Zarar eden işlem kaydedildi: {symbol} {timeframe} | RR: {rr}")
    except Exception:
        await update.message.reply_text("Kullanım: /logloss XAUUSD 5min -1")


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    data = journal.get_stats(update.effective_chat.id)
    text = (
        "<b>Genel Performans</b>\n"
        f"Toplam işlem: {data['total']}\n"
        f"Kazanç: {data['wins']}\n"
        f"Zarar: {data['losses']}\n"
        f"Win rate: %{data['winrate']}\n"
        f"Net RR: {data['net_rr']}\n"
    )

    if data["by_symbol"]:
        text += "\n<b>Sembol bazlı</b>\n"
        for symbol, row in data["by_symbol"].items():
            text += (
                f"- {symbol} | Toplam: {row['total']} | "
                f"W: {row['wins']} | L: {row['losses']} | Net RR: {round(row['net_rr'], 2)}\n"
            )

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def todaystats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    data = journal.get_today_stats(update.effective_chat.id)
    text = (
        "<b>Bugünkü Performans</b>\n"
        f"Toplam işlem: {data['total']}\n"
        f"Kazanç: {data['wins']}\n"
        f"Zarar: {data['losses']}\n"
        f"Win rate: %{data['winrate']}\n"
        f"Net RR: {data['net_rr']}\n"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def alert_scan_job(context: CallbackContext) -> None:
    events = await calendar_service.get_upcoming_high_impact_events(hours_ahead=2, limit=3)

    for chat_id_str, items in store.iter_all().items():
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

                near_support = bool(result.support) and abs(result.current_price - result.support[-1]) <= result.atr * 0.45
                near_resistance = bool(result.resistance) and abs(result.current_price - result.resistance[0]) <= result.atr * 0.45

                if (
                    result.signal != "NO TRADE"
                    and result.quality == "A"
                    and result.setup_score >= 80
                    and result.rr_ratio >= 2.0
                    and (near_support or near_resistance)
                    and result.sweep_signal != "Yok"
                    and result.sniper_entry != "Yok"
                ):
                    text = (
                        f"🔔 <b>Sniper Scalp Alarm</b>\n"
                        f"{result.symbol} {result.timeframe}\n"
                        f"Sinyal: <b>{result.signal}</b>\n"
                        f"Kalite: <b>{result.quality}</b>\n"
                        f"Score: <b>{result.setup_score}/100</b>\n"
                        f"Fiyat: {result.current_price:.5f}\n"
                        f"Giriş: {result.entry_zone[0]:.5f} - {result.entry_zone[1]:.5f}\n"
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

    for chat_id in store.get_daily_subscribers():
        chunks: list[str] = ["<b>Günlük forex scalp planı</b>"]
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

                chunks.append(
                    f"\n<b>{symbol}</b> | {result.signal} | Kalite: {result.quality} | "
                    f"Score: {result.setup_score}/100 | Fiyat: {result.current_price:.5f} | "
                    f"R/R: {result.rr_ratio} | Sweep: {result.sweep_signal} | Sniper: {result.sniper_entry}"
                )
            except Exception as exc:
                chunks.append(f"\n<b>{symbol}</b> | veri alınamadı: {exc}")

        await context.bot.send_message(
            chat_id=chat_id,
            text="".join(chunks),
            parse_mode=ParseMode.HTML,
        )


def build_application() -> Application:
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN eksik.")

    application = Application.builder().token(settings.telegram_bot_token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(CommandHandler("signal", signal))
    application.add_handler(CommandHandler("levels", levels))
    application.add_handler(CommandHandler("news", news))
    application.add_handler(CommandHandler("session", session_cmd))
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

    application.job_queue.run_repeating(
        alert_scan_job,
        interval=settings.alert_scan_minutes * 60,
        first=20,
    )
    application.job_queue.run_daily(
        daily_plan_job,
        time=datetime.strptime(f"{settings.daily_plan_hour:02d}:00", "%H:%M").time(),
    )
    return application