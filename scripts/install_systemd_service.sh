#!/usr/bin/env bash
# daily_signal_report.py'yi 7/24, VM yeniden başlasa bile otomatik
# başlayacak ve çökerse kendini yeniden başlatacak şekilde bir systemd
# servisi olarak kurar.
#
# Önce scripts/setup_vm.sh çalıştırılmış, scripts/.env doldurulmuş ve
# model eğitilmiş olmalı (bkz. scripts/setup_vm.sh çıktısındaki adımlar).
#
# Kullanım (repo kökünden):
#   bash scripts/install_systemd_service.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
    echo "HATA: $VENV_PYTHON bulunamadı. Önce 'bash scripts/setup_vm.sh' çalıştırın." >&2
    exit 1
fi

if [ ! -f "$REPO_ROOT/scripts/.env" ]; then
    echo "HATA: scripts/.env yok. Önce onu doldurun (bkz. scripts/.env.example)." >&2
    exit 1
fi

if [ ! -f "$REPO_ROOT/artifacts/model.joblib" ]; then
    echo "UYARI: artifacts/model.joblib yok — servis başlar başlamaz hata verecektir." >&2
    echo "        Önce: $VENV_PYTHON -m yatirim.ml.run_pipeline --data gercek_fiyatlar.csv" >&2
fi

SERVICE_NAME="daily-signal-report"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

sed \
    -e "s|__USER__|$(whoami)|g" \
    -e "s|__WORKDIR__|$REPO_ROOT|g" \
    -e "s|__PYTHON__|$VENV_PYTHON|g" \
    "$REPO_ROOT/scripts/daily-signal-report.service.template" \
    | sudo tee "$SERVICE_PATH" > /dev/null

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

echo ""
echo "Servis kuruldu ve başlatıldı: $SERVICE_NAME"
echo "Durumu görmek için:   sudo systemctl status $SERVICE_NAME"
echo "Canlı log izlemek için: sudo journalctl -u $SERVICE_NAME -f"
echo "Durdurmak için:        sudo systemctl stop $SERVICE_NAME"
echo "Yeniden başlatmak için: sudo systemctl restart $SERVICE_NAME"
