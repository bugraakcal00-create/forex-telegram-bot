import logging
import time
import sys

logger = logging.getLogger(__name__)

MAX_RETRIES = 10
BASE_WAIT = 30  # saniye


def main():
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            from app.bot import build_application
            app = build_application()
            logger.info("Bot starting (attempt %d/%d)...", attempt, MAX_RETRIES)
            app.run_polling(drop_pending_updates=True)
            break  # Normal cikis
        except Exception as e:
            err_str = str(e)
            if "Conflict" in err_str:
                wait = BASE_WAIT * attempt
                logger.warning(
                    "Telegram conflict (attempt %d/%d). Waiting %ds before retry...",
                    attempt, MAX_RETRIES, wait,
                )
                print(f"[!] Telegram conflict. Waiting {wait}s... (attempt {attempt}/{MAX_RETRIES})", flush=True)
                time.sleep(wait)
            else:
                logger.error("Bot crashed: %s", e)
                raise
    else:
        logger.error("Max retries (%d) reached. Exiting.", MAX_RETRIES)
        sys.exit(1)


if __name__ == "__main__":
    main()
