"""
Merkezi loglama yapilandirmasi — loguru tabanli.

stdlib logging cagrilarini (python-telegram-bot, uvicorn vb.)
otomatik olarak loguru'ya yonlendirir.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from loguru import logger

_LOG_DIR = Path(__file__).parent.parent / "data" / "logs"


class _InterceptHandler(logging.Handler):
    """stdlib logging -> loguru koprusu."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)
        frame, depth = logging.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging() -> None:
    """Loglama sistemini yapilandirir. Uygulama basinda bir kez cagrilmali."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # loguru varsayilan sink'i kaldir, yeniden yapilandir
    logger.remove()

    # Console — renkli, INFO+
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan> — <level>{message}</level>",
        colorize=True,
    )

    # Genel log dosyasi — 10 MB rotation, 30 gun saklama
    logger.add(
        str(_LOG_DIR / "bot_{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} — {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
    )

    # Hata log dosyasi — sadece ERROR+
    logger.add(
        str(_LOG_DIR / "errors_{time:YYYY-MM-DD}.log"),
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} — {message}",
        rotation="5 MB",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
    )

    # stdlib logging -> loguru yonlendirme
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    # Gurultulu kutuphaneleri sustur
    for noisy in ("httpx", "httpcore", "telegram.ext", "apscheduler"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info("Loglama sistemi baslatildi (loglar: {})", _LOG_DIR)
