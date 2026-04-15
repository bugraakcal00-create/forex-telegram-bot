"""
Tum sembolleri tum timeframe'lerde watchlist'e ekler.
Chat ID'yi arguman olarak verir veya .env'den okur.

Kullanim:
    python setup_watchlist.py 123456789
    python setup_watchlist.py  (TELEGRAM_CHAT_ID env var kullanir)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from app.storage.sqlite_store import BotRepository
from app.config import settings

SYMBOLS = ["XAUUSD", "BTCUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"]
TIMEFRAMES = ["5min", "15min", "30min", "1h", "4h"]


def main():
    # Chat ID: arguman veya env var
    if len(sys.argv) > 1:
        chat_id = int(sys.argv[1])
    else:
        chat_id_str = os.getenv("TELEGRAM_CHAT_ID", "")
        if not chat_id_str:
            print("Kullanim: python setup_watchlist.py <CHAT_ID>")
            print("Veya TELEGRAM_CHAT_ID env var ayarlayin")
            sys.exit(1)
        chat_id = int(chat_id_str)

    repo = BotRepository(db_path=Path(settings.db_path))

    print(f"Chat ID: {chat_id}")
    print(f"Semboller: {', '.join(SYMBOLS)}")
    print(f"Timeframe'ler: {', '.join(TIMEFRAMES)}")
    print(f"Toplam: {len(SYMBOLS) * len(TIMEFRAMES)} watchlist girisi")
    print()

    count = 0
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            repo.add_watch(chat_id=chat_id, symbol=symbol, timeframe=tf)
            count += 1
            print(f"  + {symbol} / {tf}")

    print(f"\nToplam {count} watchlist girisi eklendi.")

    # Mevcut durumu goster
    watches = repo.get_watches(chat_id)
    print(f"\nMevcut watchlist ({len(watches)} adet):")
    for w in watches:
        print(f"  {w['symbol']:8s} {w['timeframe']}")


if __name__ == "__main__":
    main()
