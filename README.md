# Forex Telegram Bot - Aşama 1 + Aşama 2

Bu proje senin istediğin iki aşamayı tek bir başlangıç paketi içinde toplar.

## Aşama 1
- `/signal XAUUSD 15min` ile analiz
- destek / direnç çıkarma
- giriş bölgesi, SL, TP, R/R hesaplama
- haber özeti
- ekonomik veri filtresi
- risk hesabı
- seans bilgisi
- günlük trade planı komutu

## Aşama 2
- izleme listesi
- belirli pariteleri otomatik tarama
- fiyat kritik bölgeye gelince Telegram alarmı
- günlük plan aboneliği
- otomatik sabah özeti

## Kullanılan servisler
- Telegram Bot API
- TwelveData: mum verisi / fiyat
- NewsAPI: haberler
- Financial Modeling Prep: ekonomik takvim

## Kurulum
```bash
python -m venv .venv
source .venv/bin/activate  # Windows için: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

`.env` içine kendi anahtarlarını gir:
```env
TELEGRAM_BOT_TOKEN=...
TWELVEDATA_API_KEY=...
NEWSAPI_API_KEY=...
FMP_API_KEY=...
DEFAULT_TIMEZONE=Europe/Istanbul
DEFAULT_PAIRS=XAU/USD,EUR/USD,GBP/USD
ALERT_SCAN_MINUTES=5
DAILY_PLAN_HOUR=8
```

## Çalıştırma
```bash
python main.py
```

## BotFather ile Telegram bot token alma
1. Telegram'da `@BotFather` aç.
2. `/newbot` yaz.
3. Bot adı ve kullanıcı adı ver.
4. Sana verilen token'ı `.env` dosyasına koy.

## Komutlar
```text
/start
/help
/signal XAUUSD 15min
/levels XAUUSD 15min
/news
/session
/plan XAUUSD 15min
/risk 1000 1 3030 3018
/watch XAUUSD 15min
/unwatch XAUUSD
/watchlist
/subscribe_daily
/unsubscribe_daily
```

## Notlar
- Lot hesabı broker sözleşmesine göre ayrıca uyarlanmalı.
- Ekonomik takvim ve haberler API anahtarı yoksa boş döner.
- Bu bot yatırım garantisi vermez; trade asistanı mantığında tasarlandı.
- İlk geliştirme için XAUUSD / EURUSD / GBPUSD odaklıdır.

## Geliştirme fikirleri
- MT5 bağlantısı
- işlem geçmişi kaydı
- kullanıcı bazlı risk profili
- daha iyi market structure algoritması
- screenshot gönderimi / grafik görseli
- AI haber sınıflandırması
