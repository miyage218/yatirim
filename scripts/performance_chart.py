"""Aylık model vs TÜFE vs BIST100 kümülatif getiri karşılaştırma grafiği.

Model performansı `positions.py`'deki gerçek AL/SAT pozisyon takibinden
(sabit gün sayısı değil, gerçek episod getirileri), BIST100 performansı
XU100.IS endeksinden, TÜFE ise `tufe_tracker.py` ile elle girilen aylık
verilerden hesaplanır. Sonuç, Telegram'a resim (PNG) olarak gönderilir.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import yfinance as yf  # noqa: E402

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MONTHLY_PATH = ARTIFACTS_DIR / "aylik_performans.csv"
START_DATE_PATH = ARTIFACTS_DIR / "baslangic_tarihi.json"
CHART_PATH = ARTIFACTS_DIR / "performans_grafik.png"

MONTHLY_COLUMNS = ["ay", "model_kumulatif_yuzde", "tufe_kumulatif_yuzde", "bist100_kumulatif_yuzde"]


def _symbol_to_ticker(sembol: str) -> str:
    if sembol == "USDTRY":
        return "TRY=X"
    if sembol == "EURTRY":
        return "EURTRY=X"
    if sembol.startswith("BIST:"):
        return sembol.split(":", 1)[1] + ".IS"
    return sembol


def fetch_latest_prices(symbols: list[str]) -> dict[str, float]:
    """Verilen sembol listesi için en güncel kapanış fiyatlarını çeker.

    Yalnızca aylık anlık görüntü alınırken (nadiren) çağrılır; günlük
    sinyal döngüsünden bağımsızdır.
    """
    prices: dict[str, float] = {}
    for sembol in symbols:
        ticker = _symbol_to_ticker(sembol)
        try:
            hist = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  [UYARI] {sembol} güncel fiyat alınamadı: {exc}", file=sys.stderr)
            continue
        if hist is None or hist.empty:
            continue
        hist = hist.dropna(subset=["Close"])
        if hist.empty:
            continue
        prices[sembol] = float(hist["Close"].iloc[-1])
    return prices


def _get_or_set_start_date(today_iso: str) -> str:
    if START_DATE_PATH.exists():
        return json.loads(START_DATE_PATH.read_text(encoding="utf-8"))["tarih"]
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    START_DATE_PATH.write_text(json.dumps({"tarih": today_iso}), encoding="utf-8")
    return today_iso


def bist100_cumulative_return(today_iso: str) -> float | None:
    """İzlemeye başladığımız günden bugüne BIST100 (XU100) kümülatif
    getirisi (%). İlk çağrıda başlangıç tarihi olarak bugünü kaydeder.
    """
    start = _get_or_set_start_date(today_iso)
    try:
        hist = yf.Ticker("XU100.IS").history(start=start, end=today_iso, interval="1d", auto_adjust=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  [UYARI] BIST100 verisi alınamadı: {exc}", file=sys.stderr)
        return None
    if hist is None:
        return None
    hist = hist.dropna(subset=["Close"])
    if len(hist) < 2:
        return None
    return float(hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100


def record_monthly_snapshot(
    ay: str,
    model_return_pct: float | None,
    tufe_cumulative_pct: float,
    bist100_return_pct: float | None,
) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(MONTHLY_PATH) if MONTHLY_PATH.exists() else pd.DataFrame(columns=MONTHLY_COLUMNS)
    df = df[df["ay"].astype(str) != ay]
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "ay": ay,
                        "model_kumulatif_yuzde": model_return_pct,
                        "tufe_kumulatif_yuzde": tufe_cumulative_pct,
                        "bist100_kumulatif_yuzde": bist100_return_pct,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    df = df.sort_values("ay").reset_index(drop=True)
    df.to_csv(MONTHLY_PATH, index=False)


def generate_chart() -> Path | None:
    if not MONTHLY_PATH.exists():
        return None
    df = pd.read_csv(MONTHLY_PATH).sort_values("ay")
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["ay"], df["model_kumulatif_yuzde"], marker="o", label="Model (AL/SAT sinyalleri)", color="#2563eb")
    ax.plot(df["ay"], df["tufe_kumulatif_yuzde"], marker="o", label="TÜFE (kümülatif)", color="#dc2626")
    ax.plot(df["ay"], df["bist100_kumulatif_yuzde"], marker="o", label="BIST100 (XU100)", color="#16a34a")
    ax.axhline(0, color="#9ca3af", linewidth=0.8)
    ax.set_ylabel("Kümülatif getiri (%)")
    ax.set_title("Model vs TÜFE vs BIST100 — Kümülatif Getiri")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150)
    plt.close(fig)
    return CHART_PATH


def send_telegram_photo(token: str, chat_id: str, photo_path: Path, caption: str = "") -> None:
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with open(photo_path, "rb") as f:
        resp = requests.post(
            url, data={"chat_id": chat_id, "caption": caption}, files={"photo": f}, timeout=30
        )
    if resp.status_code != 200:
        print(f"  [UYARI] Grafik gönderilemedi: {resp.status_code} {resp.text}", file=sys.stderr)
