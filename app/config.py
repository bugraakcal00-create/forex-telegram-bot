from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    twelvedata_api_key: str = os.getenv("TWELVEDATA_API_KEY", "")
    newsapi_api_key: str = os.getenv("NEWSAPI_API_KEY", "")
    fmp_api_key: str = os.getenv("FMP_API_KEY", "")
    default_timezone: str = os.getenv("DEFAULT_TIMEZONE", "Europe/Istanbul")
    default_pairs: tuple[str, ...] = tuple(
        p.strip().upper() for p in os.getenv("DEFAULT_PAIRS", "XAU/USD,EUR/USD,GBP/USD").split(",") if p.strip()
    )
    alert_scan_minutes: int = int(os.getenv("ALERT_SCAN_MINUTES", "5"))
    daily_plan_hour: int = int(os.getenv("DAILY_PLAN_HOUR", "8"))
    news_lock_minutes: int = int(os.getenv("NEWS_LOCK_MINUTES", "20"))
    db_path: str = os.getenv("DB_PATH", "data/bot.db")
    candle_output_size: int = int(os.getenv("CANDLE_OUTPUT_SIZE", "500"))
    backtest_output_size: int = int(os.getenv("BACKTEST_OUTPUT_SIZE", "2000"))
    web_host: str = os.getenv("WEB_HOST", "127.0.0.1")
    web_port: int = int(os.getenv("WEB_PORT", "8080"))
    ultra_selective_mode: bool = os.getenv("ULTRA_SELECTIVE_MODE", "1").strip().lower() in {"1", "true", "yes", "on"}
    web_auth_enabled: bool = os.getenv("WEB_AUTH_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
    web_admin_user: str = os.getenv("WEB_ADMIN_USER", "admin")
    web_admin_password: str = os.getenv("WEB_ADMIN_PASSWORD", "change_me_now")


settings = Settings()
