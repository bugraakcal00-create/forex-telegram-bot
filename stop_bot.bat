@echo off
chcp 65001 >nul 2>&1

echo.
echo  ============================================================
echo  [93m  Forex Trading Bot - Durdurma[0m
echo  ============================================================
echo.

:: Kill Python processes running from our project
echo  [91m  [*] Telegram Bot durduruluyor...[0m
taskkill /fi "WINDOWTITLE eq ForexBot - Telegram" /f >nul 2>&1
taskkill /fi "WINDOWTITLE eq ForexBot - Telegram*" /f >nul 2>&1

echo  [91m  [*] Web Panel durduruluyor...[0m
taskkill /fi "WINDOWTITLE eq ForexBot - Web Panel" /f >nul 2>&1
taskkill /fi "WINDOWTITLE eq ForexBot - Web Panel*" /f >nul 2>&1

:: Also kill any uvicorn on port 8080 as a fallback
for /f "tokens=5" %%p in ('netstat -aon ^| findstr :8080 ^| findstr LISTENING 2^>nul') do (
    taskkill /pid %%p /f >nul 2>&1
)

echo.
echo  [92m  [OK] Tum servisler durduruldu.[0m
echo.
timeout /t 3 /nobreak >nul
