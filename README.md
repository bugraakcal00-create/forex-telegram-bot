# Forex Telegram Scalp Bot

Bu proje, XAUUSD odakli scalp surecini Telegram uzerinden yonetmek icin gelistirilmis yerel-oncelikli bir bottur.

## Ana Ozellikler
- `/signal` ile teknik analiz
- coklu zaman dilimi kontrolu
- destek/direnc kumeleme
- ATR, RSI, ADX, RR filtreleri
- likidite sweep + sniper giris tespiti
- ultra secici mod
- izleme listesi ile otomatik alarm tarama
- islem gunlugu ve performans istatistikleri
- yerel web panel ile canli yonetim

## Teknolojiler
- Python 3.11+
- python-telegram-bot
- httpx
- pandas / numpy
- FastAPI + Jinja2
- SQLite

## Proje Yapisi
- `main.py`: Telegram bot giris noktasi
- `web_main.py`: yerel web panel giris noktasi
- `app/bot.py`: komutlar, job'lar, sinyal akisi
- `app/services/*`: analiz ve dis API istemcileri
- `app/storage/sqlite_store.py`: kalicilik katmani
- `app/web/*`: yerel yonetim paneli

## Kurulum
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

`.env` dosyasini doldur:
```env
TELEGRAM_BOT_TOKEN=...
TWELVEDATA_API_KEY=...
NEWSAPI_API_KEY=...
FMP_API_KEY=...
DEFAULT_TIMEZONE=Europe/Istanbul
DEFAULT_PAIRS=XAU/USD
ALERT_SCAN_MINUTES=15
DAILY_PLAN_HOUR=8
NEWS_LOCK_MINUTES=20
DB_PATH=data/bot.db
WEB_HOST=127.0.0.1
WEB_PORT=8080
ULTRA_SELECTIVE_MODE=1
WEB_AUTH_ENABLED=1
WEB_ADMIN_USER=admin
WEB_ADMIN_PASSWORD=change_me_now
```

## Calistirma
Bot:
```bash
python main.py
```

Web panel:
```bash
python web_main.py
```

Tarayici:
`http://127.0.0.1:8080`

## Telegram Komutlari
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
- `/weeklyreport`

## Notlar
- Yerel ve host edilen botu ayni anda calistirma (polling cakismasi olur).
- SQLite yerel dosyadir. Gecici dosya sisteminde veri sifirlanabilir.
- Bot yatirim garantisi vermez; karar destek aracidir.
