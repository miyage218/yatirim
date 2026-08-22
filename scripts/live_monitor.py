#!/usr/bin/env python3
"""BIST açık saatlerinde 15 dakikada bir çalışıp güncel fiyatı kontrol
eden, eşik üstü AL sinyali yakaladığında Telegram'a bildirim gönderen
canlı izleme döngüsü.

NOT: Bu script'i Claude Code ortamında DEĞİL, internete açık kendi
makinenizde çalıştırın (bkz. scripts/run_live_monitor.bat).

Ön koşul: `python -m yatirim.ml.run_pipeline --data <csv>` daha önce
çalıştırılmış olmalı — bu script eğitilmiş modeli `artifacts/model.joblib`
dosyasından yükler, kendi başına eğitim yapmaz.

Nasıl çalışır (dürüst özet):
  - Model GÜNLÜK bar'lar üzerinde eğitildi (10 işlem günü ufuklu tahmin).
    Gün içinde 15 dakikada bir "yeni bir günlük kapanış" oluşmaz; bu
    script her turda güncel/anlık fiyatı, bugünün "oluşmakta olan"
    günlük bar'ı gibi kullanarak (açılıştan bu yana en yüksek/en düşük/
    son fiyat) özellikleri yeniden hesaplar ve modeli bu YAKLAŞIK bar
    üzerinde çalıştırır. Bu, günlük kapanış onaylı bir sinyal DEĞİL,
    gün içi bir ön izlemedir — kapanışta değişebilir.
  - Aynı sembol için aynı gün içinde birden fazla bildirim atılmaz
    (artifacts/live_state.json içinde son bildirim tarihi tutulur).

Kurulum:
    pip install yfinance pandas scikit-learn joblib requests

Telegram kimlik bilgileri (bu script hiçbir token'ı repoya/koda YAZMAZ,
sadece ortam değişkeninden veya yanındaki .env dosyasından okur):
    TELEGRAM_BOT_TOKEN=123456:ABC-...
    TELEGRAM_CHAT_ID=123456789

Kullanım:
    python scripts/live_monitor.py
    python scripts/live_monitor.py --once            # tek tur, döngüsüz (test için)
    python scripts/live_monitor.py --symbols-file scripts/watchlist.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

try:
    import requests
except ImportError:
    print("requests kurulu değil. Önce çalıştırın: pip install requests", file=sys.stderr)
    raise

try:
    import yfinance as yf
except ImportError:
    print("yfinance kurulu değil. Önce çalıştırın: pip install yfinance pandas", file=sys.stderr)
    raise

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from yatirim.ml.features import FEATURE_COLUMNS, build_feature_table  # noqa: E402

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
DEFAULT_SYMBOLS_FILE = REPO_ROOT / "scripts" / "watchlist_live.txt"
STATE_PATH = ARTIFACTS_DIR / "live_state.json"

ISTANBUL_TZ = ZoneInfo("Europe/Istanbul")
MARKET_OPEN = dtime(10, 0)
MARKET_CLOSE = dtime(18, 10)
CHECK_INTERVAL_SECONDS = 15 * 60
HISTORY_WARMUP_DAYS = "1y"  # MA200 için yeterli geçmiş bar
BUY_THRESHOLD = 0.55


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _telegram_credentials() -> tuple[str, str]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID tanımlı değil. "
            "scripts/.env dosyasına ya da ortam değişkenlerine ekleyin "
            "(scripts/.env.example'a bakın)."
        )
    return token, chat_id


def send_telegram_message(text: str) -> None:
    token, chat_id = _telegram_credentials()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(
        url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=15
    )
    if resp.status_code != 200:
        print(f"  [UYARI] Telegram bildirimi gönderilemedi: {resp.status_code} {resp.text}", file=sys.stderr)


def _read_symbols(symbols_file: Path) -> list[str]:
    symbols = []
    for line in symbols_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            symbols.append(line)
    return symbols


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_state(state: dict) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _fetch_intraday_bar(ticker: str) -> dict | None:
    """Bugünün oluşmakta olan bar'ını (açılış/en yüksek/en düşük/son
    fiyat/hacim) 5 dakikalık mumlardan türetir. Veri yoksa None döner.
    """
    try:
        intraday = yf.Ticker(ticker).history(period="1d", interval="5m", auto_adjust=False)
    except Exception as exc:  # noqa: BLE001 - ağ hatası, bu sembolü atla
        print(f"  [UYARI] {ticker} gün içi veri alınamadı: {exc}", file=sys.stderr)
        return None
    if intraday is None or intraday.empty:
        return None
    return {
        "acilis": float(intraday["Open"].iloc[0]),
        "yuksek": float(intraday["High"].max()),
        "dusuk": float(intraday["Low"].min()),
        "kapanis": float(intraday["Close"].iloc[-1]),
        "hacim": float(intraday["Volume"].sum()),
    }


def _fetch_history(ticker: str) -> pd.DataFrame | None:
    try:
        hist = yf.Ticker(ticker).history(period=HISTORY_WARMUP_DAYS, interval="1d", auto_adjust=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  [UYARI] {ticker} geçmiş veri alınamadı: {exc}", file=sys.stderr)
        return None
    if hist is None or hist.empty:
        return None
    hist.index = pd.DatetimeIndex(hist.index).tz_localize(None).normalize()
    return hist


def _build_watchlist(symbols_file: Path) -> list[tuple[str, str, str]]:
    """(ticker, sembol, varlik_tipi) üçlüleri döner."""
    watchlist = [("TRY=X", "USDTRY", "DOVIZ"), ("EURTRY=X", "EURTRY", "DOVIZ")]
    for s in _read_symbols(symbols_file):
        watchlist.append((f"{s}.IS", f"BIST:{s}", "BORSA"))
    return watchlist


def run_check(model, symbols_file: Path, buy_threshold: float, state: dict) -> int:
    """Bir tur kontrol çalıştırır, yeni AL sinyalleri için Telegram
    bildirimi gönderir. Kaç yeni bildirim atıldığını döner.
    """
    today = datetime.now(ISTANBUL_TZ).date().isoformat()
    watchlist = _build_watchlist(symbols_file)
    n_notified = 0

    for ticker, sembol, varlik_tipi in watchlist:
        if state.get(sembol) == today:
            continue  # bugün zaten bu sembol için bildirim atıldı

        hist = _fetch_history(ticker)
        bar = _fetch_intraday_bar(ticker)
        if hist is None or bar is None:
            continue

        today_row = pd.DataFrame(
            [
                {
                    "tarih": pd.Timestamp(today),
                    "sembol": sembol,
                    "varlik_tipi": varlik_tipi,
                    "acilis": bar["acilis"],
                    "yuksek": bar["yuksek"],
                    "dusuk": bar["dusuk"],
                    "kapanis": bar["kapanis"],
                    "hacim": bar["hacim"],
                }
            ]
        )
        hist_long = pd.DataFrame(
            {
                "tarih": hist.index,
                "sembol": sembol,
                "varlik_tipi": varlik_tipi,
                "acilis": hist["Open"].to_numpy(),
                "yuksek": hist["High"].to_numpy(),
                "dusuk": hist["Low"].to_numpy(),
                "kapanis": hist["Close"].to_numpy(),
                "hacim": hist["Volume"].to_numpy(),
            }
        )
        hist_long = hist_long[hist_long["tarih"] < pd.Timestamp(today)].tail(199)
        combined = pd.concat([hist_long, today_row], ignore_index=True)

        features = build_feature_table(combined)
        last_row = features.tail(1)
        if last_row[FEATURE_COLUMNS].isna().any(axis=None):
            continue  # ısınma dönemi tamamlanmamış (yetersiz geçmiş)

        proba = float(model.predict_proba(last_row[FEATURE_COLUMNS])[:, 1][0])
        print(f"  {sembol}: kapanis(anlik)={bar['kapanis']:.2f} model_guveni={proba:.3f}")

        if proba >= buy_threshold:
            message = (
                f"📈 <b>{sembol}</b> AL sinyali\n"
                f"Anlık fiyat: {bar['kapanis']:.2f}\n"
                f"Model güveni: %{proba * 100:.0f}\n"
                f"(Gün içi ön izleme — kapanışta değişebilir, işlem "
                f"maliyeti/backtest için repodaki DURUM ANALİZİ raporuna bakın)"
            )
            send_telegram_message(message)
            state[sembol] = today
            n_notified += 1

    return n_notified


def _is_market_hours(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols-file", type=Path, default=DEFAULT_SYMBOLS_FILE)
    parser.add_argument("--buy-threshold", type=float, default=BUY_THRESHOLD)
    parser.add_argument("--once", action="store_true", help="Tek tur çalışıp çık (döngü yok)")
    args = parser.parse_args()

    _load_env_file(Path(__file__).parent / ".env")
    _telegram_credentials()  # erken doğrulama: eksikse hemen hata ver

    if not MODEL_PATH.exists():
        print(
            f"HATA: {MODEL_PATH} yok. Önce çalıştırın: "
            "python -m yatirim.ml.run_pipeline --data <gercek_fiyatlar.csv>",
            file=sys.stderr,
        )
        return 1

    model = joblib.load(MODEL_PATH)
    state = _load_state()

    print(f"Canlı izleme başladı. Watchlist: {args.symbols_file}, eşik: {args.buy_threshold}")
    while True:
        now = datetime.now(ISTANBUL_TZ)
        if args.once or _is_market_hours(now):
            print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Kontrol turu başlıyor...")
            try:
                n = run_check(model, args.symbols_file, args.buy_threshold, state)
                _save_state(state)
                print(f"Tur bitti: {n} yeni bildirim gönderildi.")
            except Exception as exc:  # noqa: BLE001 - döngü çökmemeli
                print(f"  [HATA] Tur başarısız: {exc}", file=sys.stderr)
        else:
            print(f"[{now.strftime('%H:%M:%S')}] Piyasa kapalı (10:00-18:10, hafta içi), bekleniyor...")

        if args.once:
            return 0
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
