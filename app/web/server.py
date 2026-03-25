from __future__ import annotations

from pathlib import Path
from secrets import compare_digest

from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.storage.sqlite_store import BotRepository

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
repo = BotRepository(db_path=Path(settings.db_path))
security = HTTPBasic()


def require_auth(credentials: HTTPBasicCredentials = Depends(security)) -> None:
    if not settings.web_auth_enabled:
        return

    valid_user = compare_digest(credentials.username, settings.web_admin_user)
    valid_password = compare_digest(credentials.password, settings.web_admin_password)
    if valid_user and valid_password:
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Basic"},
    )


app = FastAPI(title="Forex Bot Yerel Panel", dependencies=[Depends(require_auth)])


@app.get("/")
def dashboard(request: Request) -> object:
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "summary": repo.get_dashboard_summary(),
            "settings": repo.get_all_settings(),
            "recent_signals": repo.get_recent_signal_logs(limit=30),
            "watch_groups": repo.iter_all_watches(),
            "daily_subscribers": repo.get_daily_subscribers(),
            "trade_series": repo.get_trade_series(days=14),
            "quality_dist": repo.get_signal_quality_distribution(limit=400),
            "top_symbols": repo.get_top_symbols(limit=6),
            "reason_dist": repo.get_no_trade_reason_distribution(limit=350),
            "weekly_report": repo.get_weekly_report(),
        },
    )


@app.post("/settings/session-filter")
def update_session_filter(enabled: str = Form(...)) -> RedirectResponse:
    repo.set_setting("session_filter_enabled", "1" if enabled == "1" else "0")
    return RedirectResponse("/", status_code=303)


@app.post("/settings/alerts")
def update_alert_settings(
    min_quality: str = Form(...),
    min_score: int = Form(...),
    min_rr: float = Form(...),
) -> RedirectResponse:
    repo.set_setting("min_quality_for_alert", min_quality.upper())
    repo.set_setting("min_score_for_alert", str(max(1, min(100, int(min_score)))))
    repo.set_setting("min_rr_for_alert", str(max(0.1, float(min_rr))))
    return RedirectResponse("/", status_code=303)


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
    return RedirectResponse("/", status_code=303)


@app.post("/watch/add")
def add_watch(
    chat_id: int = Form(...),
    symbol: str = Form(...),
    timeframe: str = Form(...),
) -> RedirectResponse:
    repo.add_watch(chat_id=chat_id, symbol=symbol.upper(), timeframe=timeframe.lower())
    return RedirectResponse("/", status_code=303)


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
    return RedirectResponse("/", status_code=303)


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
    return RedirectResponse("/", status_code=303)
