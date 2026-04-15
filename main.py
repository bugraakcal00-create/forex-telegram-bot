import asyncio
import time
import logging

from app.bot import build_application

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Telegram conflict durumunda retry mekanizmasi
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            app = build_application()
            app.run_polling(drop_pending_updates=True)
            break
        except Exception as e:
            if "Conflict" in str(e) and attempt < max_retries:
                wait = 30 * attempt
                logger.warning("Telegram conflict (attempt %d/%d), waiting %ds...", attempt, max_retries, wait)
                time.sleep(wait)
            else:
                raise
