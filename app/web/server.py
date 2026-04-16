from __future__ import annotations

import asyncio
import base64
import dataclasses
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from secrets import compare_digest
from typing import Any

from fastapi import Depends, FastAPI, Form, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.services.analysis_engine import AnalysisEngine
from app.services.calendar_service import CalendarService
from app.services.chart_service import generate_signal_chart, generate_backtest_chart
from app.services.market_data import MarketDataClient
from app.services.news_impact import get_event_analysis
from app.services.news_service import NewsService
from app.services.risk_manager import risk_manager
from app.services.sentiment_service import analyze_news_list
from app.services.cot_service import get_cot_bias_sync
from app.services.simulation_service import SimulationService
from app.storage.sqlite_store import BotRepository

import json as _json

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.filters["from_json"] = lambda s: _json.loads(s) if s else []
repo = BotRepository(db_path=Path(settings.db_path))
sim_service = SimulationService(db_path=Path(settings.db_path))
news_service = NewsService(api_key=settings.newsapi_api_key)
calendar_service = CalendarService(api_key=settings.fmp_api_key)
security = HTTPBasic(auto_error=False)

# Analiz servisleri (web tabanlı anlık analiz için)
_market = MarketDataClient(api_key=settings.twelvedata_api_key)
_engine = AnalysisEngine()

# Simple in-memory cache for external API calls
_news_cache: dict[str, Any] = {}
_CACHE_TTL = 300  # 5 dakika
_MAX_NEWS_CACHE = 50


def _cached(key: str, ttl: int = _CACHE_TTL) -> Any | None:
    entry = _news_cache.get(key)
    if entry and (time.time() - entry["ts"]) < ttl:
        return entry["data"]
    return None


def _set_cache(key: str, data: Any) -> None:
    # Cache eviction
    if len(_news_cache) >= _MAX_NEWS_CACHE:
        oldest = sorted(_news_cache, key=lambda k: _news_cache[k]["ts"])[:_MAX_NEWS_CACHE // 4]
        for k in oldest:
            _news_cache.pop(k, None)
    _news_cache[key] = {"data": data, "ts": time.time()}


def require_auth(credentials: HTTPBasicCredentials | None = Depends(security)) -> None:
    if not settings.web_auth_enabled:
        return
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    valid_user = compare_digest(credentials.username, settings.web_admin_user)
    valid_password = compare_digest(credentials.password, settings.web_admin_password)
    if valid_user and valid_password:
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Basic"},
    )


logger = logging.getLogger(__name__)

from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Forex Bot Panel",
    description="Forex sinyal botu web paneli ve API",
    version="2.1.0",
    dependencies=[Depends(require_auth)],
    docs_url="/docs",
    redoc_url="/redoc",
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ── WebSocket Manager ────────────────────────────────────────────────────────

class _WSManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict) -> None:
        for ws in self.active[:]:
            try:
                await ws.send_json(message)
            except Exception:
                self.active.remove(ws)

ws_manager = _WSManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ── Outcome Resolve Engine (web tarafı) ──────────────────────────────────

_TF_MINUTES: dict[str, int] = {
    "1min": 1, "5min": 5, "15min": 15, "30min": 30,
    "1h": 60, "4h": 240, "1day": 1440, "1week": 10080,
}
_TF_EXPIRY_HOURS: dict[str, int] = {
    "1min": 4, "5min": 12, "15min": 24, "30min": 36,
    "1h": 72, "4h": 168, "1day": 720, "1week": 2160,
}


def _is_expired(created_at_str: str, tf: str) -> bool:
    try:
        created = datetime.fromisoformat(created_at_str)
        if created.tzinfo is None:
            from zoneinfo import ZoneInfo
            created = created.replace(tzinfo=ZoneInfo("UTC"))
        from zoneinfo import ZoneInfo as _ZI
        elapsed_h = (datetime.now(_ZI("UTC")) - created).total_seconds() / 3600
        return elapsed_h > _TF_EXPIRY_HOURS.get(tf, 72)
    except Exception:
        return False


def _candles_needed(created_at_str: str, tf: str) -> int:
    try:
        created = datetime.fromisoformat(created_at_str)
        if created.tzinfo is None:
            from zoneinfo import ZoneInfo
            created = created.replace(tzinfo=ZoneInfo("UTC"))
        from zoneinfo import ZoneInfo as _ZI2
        elapsed_min = (datetime.now(_ZI2("UTC")) - created).total_seconds() / 60
        tf_min = _TF_MINUTES.get(tf, 5)
        return min(500, max(20, int(elapsed_min / tf_min) + 5))
    except Exception:
        return 50


def _check_candle_outcome(
    signal: str, tp: float, sl: float, rr_ratio: float,
    candle_open: float, candle_high: float, candle_low: float,
) -> tuple[str, float, str] | None:
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
        if abs(candle_open - tp) <= abs(candle_open - sl):
            return ("tp_hit", rr_ratio, "Mum hem TP hem SL dokundu; TP daha yakindi")
        return ("sl_hit", -1.0, "Mum hem TP hem SL dokundu; SL daha yakindi")
    if tp_hit:
        return ("tp_hit", rr_ratio, "Mum high/low TP seviyesine ulasti")
    return ("sl_hit", -1.0, "Mum high/low SL seviyesine ulasti")


async def _resolve_pending_outcomes() -> dict[str, int]:
    """Tüm bekleyen sinyalleri mum bazlı kontrol et ve çöz."""
    pending = repo.get_pending_signal_logs(limit=200)
    stats = {"resolved": 0, "expired": 0, "still_pending": 0, "errors": 0}

    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for row in pending:
        by_symbol[str(row["symbol"])].append(row)

    for symbol, rows in by_symbol.items():
        max_candles = max(_candles_needed(str(r["created_at"]), str(r["timeframe"])) for r in rows)
        min_tf = min(rows, key=lambda r: _TF_MINUTES.get(str(r["timeframe"]), 5))
        tf = str(min_tf["timeframe"])

        try:
            df = await _market.fetch_candles(symbol, interval=tf, outputsize=min(500, max_candles))
        except Exception as exc:
            logger.warning("resolve: candle fetch failed %s: %s", symbol, exc)
            stats["errors"] += len(rows)
            continue

        for row in rows:
            signal_log_id = int(row["id"])
            signal = str(row["signal"])
            tp = float(row["take_profit"])
            sl = float(row["stop_loss"])
            rr_ratio = float(row.get("rr_ratio", 0.0))
            created_at_str = str(row["created_at"])
            row_tf = str(row["timeframe"])

            if _is_expired(created_at_str, row_tf):
                repo.resolve_signal_outcome(
                    signal_log_id, "expired", 0.0,
                    f"Sinyal {_TF_EXPIRY_HOURS.get(row_tf, 72)}s icinde sonuclanmadi",
                )
                stats["expired"] += 1
                continue

            is_hyp = str(row.get("outcome", "pending")) == "hyp_pending"
            effective_signal = signal
            if is_hyp and signal not in ("LONG", "SHORT"):
                ep = float(row.get("current_price") or 0)
                if ep > 0:
                    effective_signal = "LONG" if tp > ep else "SHORT"
                else:
                    stats["still_pending"] += 1
                    continue

            try:
                created_dt = datetime.fromisoformat(created_at_str)
                if created_dt.tzinfo is not None:
                    created_dt = created_dt.replace(tzinfo=None)  # naive yap, df ile uyumlu
            except Exception:
                created_dt = datetime.now() - timedelta(hours=24)

            # df["datetime"] naive olabilir, karsilastirma icin ayni tip olmali
            try:
                candles_after = df[df["datetime"] >= created_dt]
            except TypeError:
                # Tip uyumsuzlugu — naive'e zorla
                candles_after = df[df["datetime"].dt.tz_localize(None) >= created_dt] if hasattr(df["datetime"].dt, "tz_localize") else df.tail(20)
            if candles_after.empty:
                stats["still_pending"] += 1
                continue

            resolved = False
            for _, candle in candles_after.iterrows():
                result = _check_candle_outcome(
                    signal=effective_signal, tp=tp, sl=sl, rr_ratio=rr_ratio,
                    candle_open=float(candle["open"]),
                    candle_high=float(candle["high"]),
                    candle_low=float(candle["low"]),
                )
                if result:
                    raw_outcome, realized_rr, note = result
                    if is_hyp:
                        final_outcome = "hyp_tp" if raw_outcome == "tp_hit" else "hyp_sl"
                        note = f"[Hipotez] {note}"
                    else:
                        final_outcome = raw_outcome
                    repo.resolve_signal_outcome(signal_log_id, final_outcome, realized_rr, note)
                    stats["resolved"] += 1
                    resolved = True
                    break

            if not resolved:
                stats["still_pending"] += 1

    return stats


# Background task — her 2 dakikada bir çalıştır
_resolve_task: asyncio.Task | None = None


async def _periodic_resolve():
    while True:
        try:
            await asyncio.sleep(120)  # 2 dakika
            stats = await _resolve_pending_outcomes()
            if stats["resolved"] or stats["expired"]:
                logger.info("auto-resolve: %s", stats)
                summary = repo.get_dashboard_summary()
                await ws_manager.broadcast({"type": "summary_update", "data": summary})
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("auto-resolve error: %s", exc)


@app.on_event("startup")
async def _start_resolve_loop():
    global _resolve_task
    _resolve_task = asyncio.create_task(_periodic_resolve())


@app.on_event("shutdown")
async def _stop_resolve_loop():
    if _resolve_task:
        _resolve_task.cancel()


# ── Dashboard ──────────────────────────────────────────────────────────────
def _filter_quality(signals: list[dict], min_quality: str = "C") -> list[dict]:
    """D kalite sinyalleri filtrele. Sadece A/B/C gosterilir."""
    allowed = {"A", "B", "C"}
    return [s for s in signals if str(s.get("quality", "D")).upper() in allowed]


@app.get("/")
def dashboard(request: Request) -> object:
    recent = _filter_quality(repo.get_recent_signal_logs(limit=40))[:20]
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "summary": repo.get_dashboard_summary(),
            "settings": repo.get_all_settings(),
            "recent_signals": recent,
            "watch_groups": repo.iter_all_watches(),
            "daily_subscribers": repo.get_daily_subscribers(),
            "trade_series": repo.get_trade_series(days=14),
            "quality_dist": repo.get_signal_quality_distribution(limit=400),
            "top_symbols": repo.get_top_symbols(limit=6),
            "reason_dist": repo.get_no_trade_reason_distribution(limit=350),
            "weekly_report": repo.get_weekly_report(),
            "equity_series": repo.get_equity_series(days=90),
        },
    )


# ── Signals page ───────────────────────────────────────────────────────────
@app.get("/signals")
def signals_page(request: Request, signal_filter: str = "ALL", limit: int = 100) -> object:
    raw = repo.get_signal_logs_detail(limit=limit * 2, signal_filter=signal_filter)
    signals = _filter_quality(raw)[:limit]
    return templates.TemplateResponse(
        request,
        "signals.html",
        {
            "signals": signals,
            "signal_filter": signal_filter,
            "limit": limit,
        },
    )


# ── News & Calendar page ───────────────────────────────────────────────────
@app.get("/news")
async def news_page(request: Request) -> object:
    cached_news = _cached("news")
    if not cached_news:
        try:
            cached_news = await news_service.get_forex_news(limit=12)
        except Exception:
            cached_news = []
        if cached_news:  # sadece dolu sonucu cache'le
            _set_cache("news", cached_news)

    cached_cal = _cached("calendar")
    if not cached_cal:
        try:
            cached_cal = await calendar_service.get_upcoming_high_impact_events(hours_ahead=48, limit=25)
        except Exception:
            cached_cal = []
        if cached_cal:
            _set_cache("calendar", cached_cal)

    # Her takvim olayına Türkçe etki analizi ekle
    enriched_cal = []
    for event in (cached_cal or []):
        entry = dict(event)
        entry["impact_analysis"] = get_event_analysis(str(event.get("event", "")))
        enriched_cal.append(entry)

    return templates.TemplateResponse(
        request,
        "news.html",
        {
            "news": cached_news,
            "calendar": enriched_cal,
        },
    )


# ── Settings page ──────────────────────────────────────────────────────────
@app.get("/settings")
def settings_page(request: Request) -> object:
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "settings": repo.get_all_settings(),
            "watch_groups": repo.iter_all_watches(),
            "daily_subscribers": repo.get_daily_subscribers(),
        },
    )


# ── JSON API ───────────────────────────────────────────────────────────────
@app.get("/api/summary", tags=["API"], summary="Dashboard ozet istatistikleri")
def api_summary() -> JSONResponse:
    return JSONResponse(repo.get_dashboard_summary())


@app.get("/api/news", tags=["API"], summary="Forex haberleri")
async def api_news() -> JSONResponse:
    cached = _cached("news")
    if not cached:
        try:
            cached = await news_service.get_forex_news(limit=12)
        except Exception:
            cached = []
        if cached:
            _set_cache("news", cached)
    return JSONResponse(cached)


@app.get("/api/calendar", tags=["API"], summary="Ekonomik takvim")
async def api_calendar() -> JSONResponse:
    cached = _cached("calendar")
    if cached is None:
        try:
            cached = await calendar_service.get_upcoming_high_impact_events(hours_ahead=48, limit=25)
        except Exception:
            cached = []
        _set_cache("calendar", cached)
    return JSONResponse(cached)


@app.get("/api/signals", tags=["API"], summary="Son sinyaller (A/B/C kalite)")
def api_signals(limit: int = 50) -> JSONResponse:
    raw = repo.get_recent_signal_logs(limit=limit * 2)
    return JSONResponse(_filter_quality(raw)[:limit])


# ── Form actions ───────────────────────────────────────────────────────────
@app.post("/settings/session-filter")
def update_session_filter(enabled: str = Form(...)) -> RedirectResponse:
    repo.set_setting("session_filter_enabled", "1" if enabled == "1" else "0")
    return RedirectResponse("/settings", status_code=303)


@app.post("/settings/alerts")
def update_alert_settings(
    min_quality: str = Form(...),
    min_score: int = Form(...),
    min_rr: float = Form(...),
) -> RedirectResponse:
    repo.set_setting("min_quality_for_alert", min_quality.upper())
    repo.set_setting("min_score_for_alert", str(max(1, min(100, int(min_score)))))
    repo.set_setting("min_rr_for_alert", str(max(0.1, float(min_rr))))
    return RedirectResponse("/settings", status_code=303)


@app.post("/settings/preset")
def apply_preset(mode: str = Form(...)) -> RedirectResponse:
    mode = mode.strip().lower()
    if mode == "conservative":
        repo.set_setting("min_quality_for_alert", "A")
        repo.set_setting("min_score_for_alert", "88")
        repo.set_setting("min_rr_for_alert", "2.4")
    elif mode == "balanced":
        repo.set_setting("min_quality_for_alert", "A")
        repo.set_setting("min_score_for_alert", "82")
        repo.set_setting("min_rr_for_alert", "2.1")
    elif mode == "aggressive":
        repo.set_setting("min_quality_for_alert", "B")
        repo.set_setting("min_score_for_alert", "72")
        repo.set_setting("min_rr_for_alert", "1.8")
    return RedirectResponse("/settings", status_code=303)


@app.post("/watch/add")
def add_watch(
    chat_id: int = Form(...),
    symbol: str = Form(...),
    timeframe: str = Form(...),
) -> RedirectResponse:
    repo.add_watch(chat_id=chat_id, symbol=symbol.upper(), timeframe=timeframe.lower())
    return RedirectResponse("/settings", status_code=303)


@app.post("/watch/remove")
def remove_watch(
    chat_id: int = Form(...),
    symbol: str = Form(...),
    timeframe: str = Form(default=""),
) -> RedirectResponse:
    repo.remove_watch(
        chat_id=chat_id,
        symbol=symbol.upper(),
        timeframe=timeframe.lower() if timeframe else None,
    )
    return RedirectResponse("/settings", status_code=303)


@app.post("/watch/setup-all")
def setup_all_watches(chat_id: int = Form(...)) -> RedirectResponse:
    """Add all symbols with all timeframes to watchlist."""
    _symbols = ["XAUUSD", "BTCUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"]
    _tfs = ["5min", "15min", "30min", "1h", "4h"]
    for sym in _symbols:
        for tf in _tfs:
            repo.add_watch(chat_id=chat_id, symbol=sym, timeframe=tf)
    return RedirectResponse("/settings", status_code=303)


@app.post("/trade/add")
def add_trade(
    chat_id: int = Form(...),
    symbol: str = Form(...),
    timeframe: str = Form(...),
    result: str = Form(...),
    rr: float = Form(...),
) -> RedirectResponse:
    repo.add_trade(
        chat_id=chat_id,
        symbol=symbol.upper(),
        timeframe=timeframe.lower(),
        result=result.lower(),
        rr=rr,
    )
    return RedirectResponse("/settings", status_code=303)


# ── Kontrol Paneli ─────────────────────────────────────────────────────────

@app.get("/control")
def control_page(request: Request) -> object:
    return templates.TemplateResponse(
        request,
        "control.html",
        {
            "settings": repo.get_all_settings(),
            "watch_groups": repo.iter_all_watches(),
            "record_counts": repo.get_record_counts(),
            "config": {
                "session_filter_enabled": repo.get_setting("session_filter_enabled", "1") == "1",
                "ultra_selective_mode": settings.ultra_selective_mode,
                "news_lock_minutes": settings.news_lock_minutes,
                "min_quality": repo.get_setting("min_quality_for_alert", "A"),
                "min_score": repo.get_setting("min_score_for_alert", "80"),
                "min_rr": repo.get_setting("min_rr_for_alert", "2.0"),
            },
        },
    )


# ── Anlık Analiz API ──────────────────────────────────────────────────────

# Zaman dilimi eşleme yardımcısı
_HIGHER_TF: dict[str, str] = {
    "1min": "5min",
    "5min": "15min",
    "15min": "1h",
    "30min": "1h",
    "1h": "4h",
    "4h": "1day",
    "1day": "1week",
}


@app.post("/api/analyze", tags=["API"], summary="Anlik teknik analiz")
async def api_analyze(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Geçersiz JSON gövdesi."}, status_code=400)

    symbol: str = str(body.get("symbol", "XAUUSD")).upper()
    timeframe: str = str(body.get("timeframe", "5min")).lower()
    higher_tf: str = _HIGHER_TF.get(timeframe, "1h")

    try:
        df, higher_df = await asyncio.gather(
            _market.fetch_candles(symbol, interval=timeframe, outputsize=500),
            _market.fetch_candles(symbol, interval=higher_tf, outputsize=500),
        )
    except Exception as exc:
        return JSONResponse(
            {"error": f"Piyasa verisi alınamadı: {exc}"},
            status_code=503,
        )

    try:
        result = _engine.analyze(
            symbol=symbol,
            df=df,
            timeframe=timeframe,
            higher_tf_df=higher_df,
        )
    except Exception as exc:
        return JSONResponse(
            {"error": f"Analiz hatası: {exc}"},
            status_code=500,
        )

    # dataclass → dict dönüşümü
    raw = dataclasses.asdict(result)

    # tuple alanları JSON-uyumlu listeye çevir
    def _clean(obj: Any) -> Any:
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(i) for i in obj]
        return obj

    return JSONResponse(_clean(raw))


# ── Veri Sıfırlama ────────────────────────────────────────────────────────

@app.post("/control/reset-signals")
def reset_signals() -> RedirectResponse:
    repo.reset_signal_logs()
    return RedirectResponse("/control", status_code=303)


@app.post("/control/reset-trades")
def reset_trades_route() -> RedirectResponse:
    repo.reset_trades()
    return RedirectResponse("/control", status_code=303)


@app.post("/control/reset-all")
def reset_all_route() -> RedirectResponse:
    repo.reset_all_data()
    return RedirectResponse("/control", status_code=303)


# ── Sonuç Çözümleme (Manuel) ─────────────────────────────────────────────

@app.post("/api/resolve-outcomes", tags=["API"], summary="Bekleyen sinyalleri cozumle")
async def api_resolve_outcomes() -> JSONResponse:
    """Tüm bekleyen sinyalleri şimdi çözümle."""
    stats = await _resolve_pending_outcomes()
    return JSONResponse(stats)


@app.post("/signals/resolve-now")
async def resolve_now_redirect() -> RedirectResponse:
    await _resolve_pending_outcomes()
    return RedirectResponse("/signals", status_code=303)


# ── Forecast (Tahmin) Page ────────────────────────────────────────────────

_FORECAST_SYMBOLS = ["XAUUSD", "BTCUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"]
_FORECAST_TIMEFRAMES = ["1min", "5min", "15min", "30min", "1h", "4h", "1day"]


async def _run_forecast(symbol: str, timeframe: str) -> dict:
    """Fetch data, run analysis, generate chart + risk info. Returns a result dict."""
    higher_tf = _HIGHER_TF.get(timeframe, "1h")

    # Parallel market data + COT + news
    try:
        df, higher_df, news_items = await asyncio.gather(
            _market.fetch_candles(symbol, interval=timeframe, outputsize=500),
            _market.fetch_candles(symbol, interval=higher_tf, outputsize=500),
            news_service.get_forex_news(limit=10),
        )
    except Exception as exc:
        return {"error": f"Veri alınamadı: {exc}"}

    # COT bias (sync, wrap in executor)
    try:
        loop = asyncio.get_running_loop()
        cot_bias = await loop.run_in_executor(None, get_cot_bias_sync, symbol)
    except Exception:
        cot_bias = "NEUTRAL"

    # Sentiment
    sentiment_score = 0.0
    try:
        sr = analyze_news_list(news_items)
        sentiment_score = float(sr.gold_score) if hasattr(sr, "gold_score") else 0.0
    except Exception:
        pass

    # DXY bias from cache if available
    dxy_bias = _cached("dxy_bias") or "NEUTRAL"

    # Analysis
    try:
        result = _engine.analyze(
            symbol=symbol,
            df=df,
            timeframe=timeframe,
            higher_tf_df=higher_df,
            cot_bias=cot_bias,
            sentiment_score=sentiment_score,
        )
    except Exception as exc:
        return {"error": f"Analiz hatası: {exc}"}

    # Generate chart as base64 PNG
    chart_b64 = ""
    try:
        png_bytes = generate_signal_chart(df, result)
        if png_bytes:
            chart_b64 = base64.b64encode(png_bytes).decode("utf-8")
    except Exception:
        pass

    # Partial TP plan
    partial_tp = None
    try:
        if result.signal in ("LONG", "SHORT") and result.entry_zone and result.stop_loss:
            entry_mid = (result.entry_zone[0] + result.entry_zone[1]) / 2
            partial_tp = risk_manager.partial_tp_structure(
                entry=entry_mid,
                stop_loss=result.stop_loss,
                direction=result.signal.lower(),
            )
    except Exception:
        pass

    # Monte Carlo — derive win rate from aggregated series
    mc = None
    try:
        series = repo.get_trade_series(days=90)
        total_wins = sum(d.get("wins", 0) for d in series)
        total_losses = sum(d.get("losses", 0) for d in series)
        total_trades = total_wins + total_losses
        if total_trades >= 20:
            wr = total_wins / total_trades
            net_rr = sum(d.get("net_rr", 0.0) for d in series)
            avg_win_r = max(1.0, net_rr / max(1, total_wins))
            mc_res = risk_manager.monte_carlo(wr, avg_win_r, 1.0, risk_per_trade=0.01)
            mc = dataclasses.asdict(mc_res)
    except Exception:
        pass

    raw = dataclasses.asdict(result)

    def _clean(obj: Any) -> Any:
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(i) for i in obj]
        return obj

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "higher_tf": higher_tf,
        "chart_b64": chart_b64,
        "result": _clean(raw),
        "partial_tp": dataclasses.asdict(partial_tp) if partial_tp else None,
        "monte_carlo": mc,
        "cot_bias": cot_bias,
        "sentiment_score": round(sentiment_score, 3),
        "error": None,
    }


@app.get("/forecast")
async def forecast_page(
    request: Request,
    symbol: str = "XAUUSD",
    timeframe: str = "5min",
) -> object:
    symbol = symbol.upper()
    if symbol not in _FORECAST_SYMBOLS:
        symbol = "XAUUSD"
    if timeframe not in _FORECAST_TIMEFRAMES:
        timeframe = "5min"

    try:
        data = await _run_forecast(symbol, timeframe)
    except Exception as exc:
        data = {"error": f"Tahmin hatası: {exc}", "result": None, "chart_b64": "",
                "partial_tp": None, "monte_carlo": None, "cot_bias": "NEUTRAL",
                "sentiment_score": 0.0, "symbol": symbol, "timeframe": timeframe,
                "higher_tf": "1h"}

    return templates.TemplateResponse(
        request,
        "forecast.html",
        {
            "symbols": _FORECAST_SYMBOLS,
            "timeframes": _FORECAST_TIMEFRAMES,
            "selected_symbol": symbol,
            "selected_timeframe": timeframe,
            **data,
        },
    )


@app.post("/api/forecast-chart", tags=["API"], summary="Tahmin chart olustur")
async def api_forecast_chart(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Geçersiz JSON"}, status_code=400)

    symbol = str(body.get("symbol", "XAUUSD")).upper()
    timeframe = str(body.get("timeframe", "5min")).lower()

    if symbol not in _FORECAST_SYMBOLS:
        symbol = "XAUUSD"
    if timeframe not in _FORECAST_TIMEFRAMES:
        timeframe = "5min"

    data = await _run_forecast(symbol, timeframe)
    return JSONResponse(data)


# ── Backtest Sayfasi ─────────────────────────────────────────────────────────

@app.get("/backtest")
def backtest_page(request: Request) -> object:
    results = repo.get_backtest_history(limit=30)
    return templates.TemplateResponse(
        request,
        "backtest.html",
        {
            "results": results,
            "symbols": _FORECAST_SYMBOLS,
            "timeframes": _FORECAST_TIMEFRAMES,
        },
    )


@app.get("/backtest/{backtest_id}")
async def backtest_detail_page(request: Request, backtest_id: int) -> object:
    bt = repo.get_backtest_detail(backtest_id)
    if not bt:
        raise HTTPException(status_code=404, detail="Backtest bulunamadi")

    import json as _j
    equity_curve = _j.loads(bt.get("equity_curve") or "[]")
    monthly_breakdown = _j.loads(bt.get("monthly_breakdown") or "{}")
    trade_log = _j.loads(bt.get("trade_log") or "[]")

    # Chart uret
    chart_b64 = ""
    try:
        from app.services.backtest_service import BacktestResult
        bt_result = BacktestResult(
            tested_signals=bt["tested_signals"], wins=bt["wins"], losses=bt["losses"],
            no_result=bt["no_result"], winrate=bt["winrate"], avg_rr=bt["avg_rr"],
            expectancy=bt["expectancy"], sharpe_ratio=bt.get("sharpe_ratio") or 0,
            profit_factor=bt.get("profit_factor") or 0,
            max_drawdown_pct=bt.get("max_drawdown") or 0,
            max_consecutive_losses=bt.get("max_consecutive_losses") or 0,
            monthly_returns=monthly_breakdown, equity_curve=equity_curve, trade_log=trade_log,
        )
        png = generate_backtest_chart(bt_result)
        chart_b64 = base64.b64encode(png).decode("utf-8")
    except Exception:
        pass

    return templates.TemplateResponse(
        request,
        "backtest_detail.html",
        {
            "bt": bt,
            "equity_curve": equity_curve,
            "monthly_breakdown": monthly_breakdown,
            "trade_log": trade_log,
            "chart_b64": chart_b64,
        },
    )


@app.post("/api/run-backtest", tags=["API"], summary="Yeni backtest calistir")
async def api_run_backtest(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Gecersiz JSON"}, status_code=400)

    symbol = str(body.get("symbol", "XAUUSD")).upper()
    timeframe = str(body.get("timeframe", "5min")).lower()
    higher_tf = _HIGHER_TF.get(timeframe, "1h")

    try:
        df, higher_df = await asyncio.gather(
            _market.fetch_candles(symbol, interval=timeframe, outputsize=int(settings.backtest_output_size)),
            _market.fetch_candles(symbol, interval=higher_tf, outputsize=int(settings.backtest_output_size)),
        )
    except Exception as exc:
        return JSONResponse({"error": f"Veri alinamadi: {exc}"}, status_code=503)

    try:
        from app.services.backtest_service import BacktestService
        bt_svc = BacktestService(engine=_engine)
        strategy_mode = str(body.get("strategy_mode", "default"))
        result = bt_svc.run(symbol=symbol, timeframe=timeframe, df=df, higher_df=higher_df, strategy_mode=strategy_mode)
    except Exception as exc:
        return JSONResponse({"error": f"Backtest hatasi: {exc}"}, status_code=500)

    # DB'ye kaydet
    import json as _j
    try:
        bt_id = repo.save_backtest_result(
            symbol=symbol, timeframe=timeframe,
            tested_signals=result.tested_signals, wins=result.wins,
            losses=result.losses, no_result=result.no_result,
            winrate=result.winrate, avg_rr=result.avg_rr, expectancy=result.expectancy,
            sharpe_ratio=result.sharpe_ratio, profit_factor=result.profit_factor,
            max_drawdown=result.max_drawdown_pct,
            max_consecutive_losses=result.max_consecutive_losses,
            monthly_breakdown=_j.dumps(result.monthly_returns),
            equity_curve=_j.dumps(result.equity_curve),
            trade_log=_j.dumps(result.trade_log[-200:]),
        )
    except Exception:
        bt_id = None

    return JSONResponse({
        "id": bt_id,
        "tested_signals": result.tested_signals,
        "wins": result.wins, "losses": result.losses,
        "winrate": result.winrate, "avg_rr": result.avg_rr,
        "expectancy": result.expectancy,
        "sharpe_ratio": result.sharpe_ratio,
        "profit_factor": result.profit_factor,
        "max_drawdown_pct": result.max_drawdown_pct,
    })


# ── Multi-Timeframe Sayfasi ──────────────────────────────────────────────────

_MULTI_TF_TIMEFRAMES = ["5min", "15min", "1h", "4h"]


@app.get("/multi-tf")
async def multi_tf_page(request: Request, symbol: str = "XAUUSD") -> object:
    symbol = symbol.upper()
    if symbol not in _FORECAST_SYMBOLS:
        symbol = "XAUUSD"

    analyses = []

    async def _analyze_tf(tf: str) -> dict:
        try:
            data = await _run_forecast(symbol, tf)
            return {"timeframe": tf, **data}
        except Exception as exc:
            return {"timeframe": tf, "error": str(exc), "result": None, "chart_b64": ""}

    results = await asyncio.gather(*[_analyze_tf(tf) for tf in _MULTI_TF_TIMEFRAMES])

    return templates.TemplateResponse(
        request,
        "multi_tf.html",
        {
            "selected_symbol": symbol,
            "symbols": _FORECAST_SYMBOLS,
            "analyses": list(results),
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION (Paper Trading) — $100 baslangic, %1 risk
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/simulation")
def simulation_page(request: Request) -> object:
    account = sim_service.get_account_summary()
    equity_curve = sim_service.get_equity_curve(limit=500)
    trade_history = sim_service.get_trade_history(limit=50)
    strategy_stats = sim_service.get_strategy_stats()

    # Best strategies from optimizer (if available)
    best_strategies = {}
    best_file = Path("data/best_strategies.json")
    if best_file.exists():
        try:
            best_strategies = _json.loads(best_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Protections status (daily loss cap, DD, trades today)
    protections_status = []
    try:
        from app.services.protections import (
            check_stoploss_guard, check_max_daily_loss, check_max_drawdown,
            check_max_daily_trades, check_cooldown_after_sl,
        )
        from app.bot import repo as _bot_repo  # paylasilan repo
        # Tum izlenen chat'ler icin durum — ilk chat ornegi yeterli
        watches = _bot_repo.iter_all_watches()
        if watches:
            chat_id = int(next(iter(watches.keys())))
            checks = [
                ("Daily SL Count", check_stoploss_guard(_bot_repo, chat_id)),
                ("Daily Loss Cap", check_max_daily_loss(_bot_repo, chat_id)),
                ("Max Drawdown", check_max_drawdown(_bot_repo, chat_id)),
                ("Daily Trade Cap", check_max_daily_trades(_bot_repo, chat_id)),
                ("Cooldown After SL", check_cooldown_after_sl(_bot_repo, chat_id)),
            ]
            for name, c in checks:
                protections_status.append({
                    "name": name,
                    "allowed": c.allowed,
                    "reason": c.reason if not c.allowed else "OK",
                    "protection": c.protection,
                })
    except Exception as exc:
        logger.debug("protections status fetch failed: %s", exc)

    return templates.TemplateResponse(
        request,
        "simulation.html",
        {
            "account": account,
            "equity_curve": equity_curve,
            "trade_history": trade_history,
            "strategy_stats": strategy_stats,
            "best_strategies": best_strategies,
            "protections_status": protections_status,
            "symbols": _FORECAST_SYMBOLS,
            "timeframes": _FORECAST_TIMEFRAMES,
        },
    )


@app.get("/api/simulation/summary", tags=["Simulation"])
def api_sim_summary() -> JSONResponse:
    return JSONResponse(sim_service.get_account_summary())


@app.get("/api/intermarket/snapshot", tags=["Intermarket"])
async def api_intermarket_snapshot() -> JSONResponse:
    """XAU intermarket snapshot: DXY + US10Y real yield + XAU/XAG ratio."""
    try:
        from app.services.intermarket_service import build_snapshot, confluence_score
        from app.bot import market as _market, _fetch_dxy_bias
        dxy = await _fetch_dxy_bias()
        snap = await build_snapshot(_market, dxy_bias=dxy)
        long_score = confluence_score(snap, "LONG")
        short_score = confluence_score(snap, "SHORT")
        return JSONResponse({
            "dxy_bias": snap.dxy_bias,
            "real_yield_pct": snap.real_yield_pct,
            "real_yield_delta_5d_bps": snap.real_yield_delta_5d,
            "real_yield_pressure": snap.real_yield_pressure,
            "xau_xag_ratio": snap.xau_xag_ratio,
            "xau_xag_zscore": snap.xau_xag_zscore,
            "xau_xag_signal": snap.xau_xag_signal,
            "long_confluence_score": long_score,
            "short_confluence_score": short_score,
            "long_ok": long_score >= 0,
            "short_ok": short_score >= 0,
        })
    except Exception as exc:
        logger.warning("intermarket snapshot API failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/simulation/equity", tags=["Simulation"])
def api_sim_equity() -> JSONResponse:
    return JSONResponse(sim_service.get_equity_curve(limit=500))


@app.get("/api/simulation/trades", tags=["Simulation"])
def api_sim_trades(limit: int = 100) -> JSONResponse:
    return JSONResponse(sim_service.get_all_trades(limit=limit))


@app.post("/api/simulation/open-trade", tags=["Simulation"])
async def api_sim_open_trade(request: Request) -> JSONResponse:
    """Manually open a simulation trade or auto-open from signal."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Gecersiz JSON"}, status_code=400)

    symbol = str(body.get("symbol", "")).upper()
    timeframe = str(body.get("timeframe", "5min")).lower()
    direction = str(body.get("direction", "")).upper()

    if direction not in ("LONG", "SHORT") or not symbol:
        return JSONResponse({"error": "symbol ve direction gerekli"}, status_code=400)

    # If entry/sl/tp provided, use them directly
    entry = body.get("entry_price")
    sl = body.get("stop_loss")
    tp = body.get("take_profit")

    if entry and sl and tp:
        trade_id = sim_service.open_trade(
            symbol=symbol, timeframe=timeframe, direction=direction,
            entry_price=float(entry), stop_loss=float(sl), take_profit=float(tp),
            strategy_mode=str(body.get("strategy_mode", "default")),
        )
        if trade_id:
            return JSONResponse({"trade_id": trade_id, "status": "opened"})
        return JSONResponse({"error": "Islem acilamadi (bakiye yetersiz veya zaten acik pozisyon)"}, status_code=400)

    # Otherwise, run analysis and open based on signal
    higher_tf = _HIGHER_TF.get(timeframe, "1h")
    strategy_mode = str(body.get("strategy_mode", "default"))

    try:
        df, higher_df = await asyncio.gather(
            _market.fetch_candles(symbol, interval=timeframe, outputsize=500),
            _market.fetch_candles(symbol, interval=higher_tf, outputsize=500),
        )
    except Exception as exc:
        return JSONResponse({"error": f"Veri alinamadi: {exc}"}, status_code=503)

    from app.services.analysis_engine import AnalysisEngine as _AE
    result = _engine.analyze(
        symbol=symbol, df=df, timeframe=timeframe,
        higher_tf_df=higher_df, strategy_mode=strategy_mode,
    )

    if result.signal not in ("LONG", "SHORT"):
        return JSONResponse({"error": "Sinyal yok (NO TRADE)", "reason": result.reason})

    trade_id = sim_service.open_trade(
        symbol=symbol, timeframe=timeframe, direction=result.signal,
        entry_price=result.current_price, stop_loss=result.stop_loss,
        take_profit=result.take_profit, strategy_mode=strategy_mode,
    )

    if trade_id:
        return JSONResponse({
            "trade_id": trade_id, "status": "opened",
            "signal": result.signal, "entry": result.current_price,
            "sl": result.stop_loss, "tp": result.take_profit,
            "score": result.setup_score, "quality": result.quality,
        })
    return JSONResponse({"error": "Islem acilamadi"}, status_code=400)


@app.post("/api/simulation/resolve", tags=["Simulation"])
async def api_sim_resolve() -> JSONResponse:
    """Check all open sim trades against current prices and resolve TP/SL hits."""
    open_trades = sim_service.get_open_trades()
    if not open_trades:
        return JSONResponse({"resolved": 0, "message": "Acik pozisyon yok"})

    # Get unique symbols
    symbols = list({str(t["symbol"]) for t in open_trades})

    # Fetch current candles for each symbol
    candle_data: dict[str, dict] = {}
    for symbol in symbols:
        try:
            # Find the most granular timeframe for this symbol
            trades_for_sym = [t for t in open_trades if str(t["symbol"]) == symbol]
            tf = min(trades_for_sym, key=lambda t: _TF_MINUTES.get(str(t["timeframe"]), 5))["timeframe"]
            df = await _market.fetch_candles(symbol, interval=str(tf), outputsize=5)
            if len(df) > 0:
                last = df.iloc[-1]
                candle_data[symbol] = {
                    "high": float(last["high"]),
                    "low": float(last["low"]),
                    "open": float(last["open"]),
                    "close": float(last["close"]),
                }
        except Exception as exc:
            logger.warning("sim resolve: candle fetch failed for %s: %s", symbol, exc)

    resolved = sim_service.check_and_resolve_trades(candle_data)
    return JSONResponse({
        "resolved": len(resolved),
        "details": resolved,
        "balance": sim_service.get_or_create_account().get("balance", 100),
    })


@app.post("/simulation/reset")
def sim_reset() -> RedirectResponse:
    sim_service.reset_account(initial_balance=100.0, risk_pct=1.0)
    return RedirectResponse("/simulation", status_code=303)


# ── Simulation auto-resolve background task ──────────────────────────────
_sim_resolve_task: asyncio.Task | None = None


async def _periodic_sim_resolve():
    """Auto-resolve sim trades every 2 minutes."""
    while True:
        try:
            await asyncio.sleep(120)
            open_trades = sim_service.get_open_trades()
            if not open_trades:
                continue

            symbols = list({str(t["symbol"]) for t in open_trades})
            candle_data: dict[str, dict] = {}
            for symbol in symbols:
                try:
                    tf = "5min"
                    df = await _market.fetch_candles(symbol, interval=tf, outputsize=5)
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
                    logger.info("sim auto-resolve: %d trades resolved", len(resolved))
                    await ws_manager.broadcast({
                        "type": "sim_update",
                        "data": sim_service.get_account_summary(),
                    })
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("sim auto-resolve error: %s", exc)


@app.on_event("startup")
async def _start_sim_resolve_loop():
    global _sim_resolve_task
    _sim_resolve_task = asyncio.create_task(_periodic_sim_resolve())
