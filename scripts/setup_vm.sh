#!/usr/bin/env bash
# Oracle Cloud (Ubuntu, Always Free Ampere A1) gibi bir Linux VM'de bu
# projeyi kurmak için tek seferlik kurulum script'i.
#
# Kullanım (VM'e SSH ile bağlandıktan sonra, repo klonlandıktan sonra):
#   cd yatirim
#   bash scripts/setup_vm.sh
#
# Bu script:
#   - sistem paketlerini (python3, pip, tesseract-ocr + Türkçe dil paketi) kurar
#   - Python sanal ortamı (.venv) oluşturup requirements-ml.txt'i kurar
#   - scripts/.env yoksa scripts/.env.example'dan bir taslak oluşturur
#     (gerçek token'ları SİZİN elle doldurmanız gerekir — script bunu
#     sizin yerinize YAPMAZ, sırlar asla otomatik yazılmaz)
#
# Bu script'i ROOT OLARAK DEĞİL, normal kullanıcı olarak (sudo gerektiren
# adımlar kendi içinde sudo çağırır) çalıştırın.

set -euo pipefail

echo "[1/4] Sistem paketleri kuruluyor (sudo şifresi isteyebilir)..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip tesseract-ocr tesseract-ocr-tur

echo "[2/4] Python sanal ortamı (.venv) oluşturuluyor..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-ml.txt

echo "[3/4] scripts/.env kontrol ediliyor..."
if [ ! -f scripts/.env ]; then
    cp scripts/.env.example scripts/.env
    echo "  scripts/.env oluşturuldu. ŞİMDİ ELLE DÜZENLEYİN:"
    echo "    nano scripts/.env"
    echo "  TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID değerlerini girin."
else
    echo "  scripts/.env zaten var, dokunulmadı."
fi

echo "[4/4] Kurulum tamamlandı."
echo ""
echo "Sıradaki adımlar:"
echo "  1. scripts/.env dosyasını doldurun (henüz doldurmadıysanız): nano scripts/.env"
echo "  2. Gerçek fiyat verisini toplayın (yoksa):"
echo "       source .venv/bin/activate"
echo "       python scripts/collect_market_data.py --years 3 --out gercek_fiyatlar.csv"
echo "  3. Modeli eğitin:"
echo "       python -m yatirim.ml.run_pipeline --data gercek_fiyatlar.csv"
echo "  4. Test edin:"
echo "       python scripts/daily_signal_report.py --once"
echo "  5. Sürekli/7-24 çalışması için systemd servisini kurun:"
echo "       bash scripts/install_systemd_service.sh"
