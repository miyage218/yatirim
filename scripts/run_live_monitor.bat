@echo off
setlocal

rem Repo kökünden çalıştığından emin ol (bu .bat scripts\ içinde duruyor)
cd /d "%~dp0\.."

if not exist "scripts\.env" (
    echo HATA: scripts\.env bulunamadi.
    echo Once scripts\.env.example dosyasini scripts\.env olarak kopyalayip
    echo TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID degerlerini girin.
    pause
    exit /b 1
)

if not exist "artifacts\model.joblib" (
    echo HATA: artifacts\model.joblib bulunamadi.
    echo Once modeli egitin: python -m yatirim.ml.run_pipeline --data gercek_fiyatlar.csv
    pause
    exit /b 1
)

echo Canli izleme baslatiliyor - hafta ici 10:00-18:10 arasi 15 dakikada bir kontrol edecek.
echo Durdurmak icin bu pencerede Ctrl+C'ye basin.
python scripts\live_monitor.py

pause
