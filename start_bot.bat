@echo off
chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
set "PROJECT_DIR=c:\Users\MSI\Desktop\forex_telegram_bot"

cd /d "%PROJECT_DIR%"

echo.
echo  ============================================================
echo  [96m   _____ ___  ____  _______  __  ____   ___ _____ [0m
echo  [96m  |  ___/ _ \|  _ \| ____\ \/ / | __ ) / _ \_   _|[0m
echo  [96m  | |_ | | | | |_) |  _|  \  /  |  _ \| | | || |  [0m
echo  [96m  |  _|| |_| |  _ <| |___ /  \  | |_) | |_| || |  [0m
echo  [96m  |_|   \___/|_| \_\_____/_/\_\ |____/ \___/ |_|  [0m
echo  [96m                                                   [0m
echo  ============================================================
echo.
echo  [93m  Forex Trading Bot - Launcher[0m
echo  [90m  ------------------------------------------------------------[0m
echo.
echo  [92m  [1] Telegram Bot        [0m  -^> Sinyal ve komut servisi
echo  [92m  [2] Web Panel (:%WEB_PORT%) [0m  -^> Dashboard ve kontrol paneli
echo.
echo  [90m  Proje: %PROJECT_DIR%[0m
echo  [90m  Python: venv\Scripts\python.exe[0m
echo.
echo  [93m  Servisler baslatiliyor...[0m
echo.

:: Start Telegram Bot in a new window
start "ForexBot - Telegram" cmd /k "cd /d "%PROJECT_DIR%" && call venv\Scripts\activate && set PYTHONIOENCODING=utf-8 && chcp 65001 >nul 2>&1 && echo [92m[ForexBot] Telegram Bot baslatiliyor...[0m && python -m app.bot"

:: Small delay to avoid port conflicts
timeout /t 2 /nobreak >nul

:: Start Web Server in a new window
start "ForexBot - Web Panel" cmd /k "cd /d "%PROJECT_DIR%" && call venv\Scripts\activate && set PYTHONIOENCODING=utf-8 && chcp 65001 >nul 2>&1 && echo [92m[ForexBot] Web Panel baslatiliyor (port 8081)...[0m && uvicorn app.web.server:app --host 0.0.0.0 --port 8081 --reload"

echo  [92m  [OK] Telegram Bot penceresi acildi.[0m
echo  [92m  [OK] Web Panel penceresi acildi.[0m
echo.
echo  [96m  Web Panel: http://127.0.0.1:8081[0m
echo.
echo  [90m  Durdurmak icin: stop_bot.bat[0m
echo  [90m  Bu pencereyi kapatabilirsiniz.[0m
echo.
timeout /t 5 /nobreak >nul
