# Forex Telegram Scalp Bot

Local-first Telegram bot for XAUUSD scalp workflow.

## Core Features
- `/signal` technical analysis with multi-timeframe checks
- support/resistance clustering
- ATR, RSI, ADX, RR filters
- liquidity sweep + sniper entry detection
- ultra selective mode for high quality setups
- watchlist based auto alert scan
- trade journal and stats
- local web panel for runtime controls

## Tech Stack
- Python 3.11+
- python-telegram-bot
- httpx
- pandas / numpy
- FastAPI + Jinja2
- SQLite

## Project Structure
- `main.py`: Telegram bot entrypoint
- `web_main.py`: local web panel entrypoint
- `app/bot.py`: handlers, jobs, signal orchestration
- `app/services/*`: analysis and external API clients
- `app/storage/sqlite_store.py`: persistence layer
- `app/web/*`: local management panel

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Fill `.env` with your secrets:
```env
TELEGRAM_BOT_TOKEN=...
TWELVEDATA_API_KEY=...
NEWSAPI_API_KEY=...
FMP_API_KEY=...
DEFAULT_TIMEZONE=Europe/Istanbul
DEFAULT_PAIRS=XAU/USD
ALERT_SCAN_MINUTES=15
DAILY_PLAN_HOUR=8
DB_PATH=data/bot.db
WEB_HOST=127.0.0.1
WEB_PORT=8080
ULTRA_SELECTIVE_MODE=1
WEB_AUTH_ENABLED=1
WEB_ADMIN_USER=admin
WEB_ADMIN_PASSWORD=change_me_now
```

## Run
Telegram bot:
```bash
python main.py
```

Local web panel:
```bash
python web_main.py
```

Open:
`http://127.0.0.1:8080`

## Telegram Commands
- `/start`
- `/signal XAUUSD 5min`
- `/levels XAUUSD 15min`
- `/news`
- `/session`
- `/session_filter_on`
- `/session_filter_off`
- `/plan XAUUSD 5min`
- `/risk 1000 1 3030 3018`
- `/watch XAUUSD 5min`
- `/unwatch XAUUSD 5min`
- `/watchlist`
- `/subscribe_daily`
- `/unsubscribe_daily`
- `/logwin XAUUSD 5min 2.1`
- `/logloss XAUUSD 5min -1`
- `/stats`
- `/todaystats`
- `/backtest XAUUSD 5min`

## Notes
- Do not run local bot and hosted bot at the same time (polling conflict).
- SQLite file is local. If deployed to ephemeral filesystem, data may reset.
- This bot is an assistant tool, not profit guarantee.
