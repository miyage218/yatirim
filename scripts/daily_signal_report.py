#!/usr/bin/env python3
"""Her gün BIST kapanışından sonra (varsayılan 18:15) bir kez çalışıp,
izleme listesindeki her sembol için AL / SAT / TUT sinyalini tek bir
Telegram mesajında raporlayan günlük sinyal raporu.

NOT: Bu script'i Claude Code ortamında DEĞİL, internete açık kendi
makinenizde çalıştırın (bkz. scripts/run_daily_signal_report.bat).

Ön koşul: `python -m yatirim.ml.run_pipeline --data <csv>` daha önce
çalıştırılmış olmalı — bu script eğitilmiş modeli `artifacts/model.joblib`
dosyasından yükler, kendi başına eğitim yapmaz.

Nasıl çalışır (dürüst özet):
  - 18:15, BIST kapanışından (18:10) sonra olduğu için script her
    sembolün GÜNÜN KESİNLEŞMİŞ günlük kapanışını kullanır (gün içi
    tahmini bar yok). Yahoo Finance'in günlük veriyi ne zaman
    güncellediği garanti değildir; bir sembolün en güncel bar'ı hâlâ
    dünküyse rapor bunu açıkça belirtir, o sembolü "veri henüz güncel
    değil" diye işaretler.
  - Model GÜNLÜK bar'lar üzerinde, 10 işlem günü ileriye dönük yön
    tahmini için eğitildi. AL/SAT eşikleri model olasılığına (P(yükseliş))
    simetrik uygulanır: P >= buy_threshold -> AL, P <= sell_threshold ->
    SAT, arası -> TUT.
  - İşlem maliyeti dahil net performans metrikleri için
    `python -m yatirim.ml.run_pipeline` çıktısındaki DURUM ANALİZİ
    bölümüne bakın; bu script o metrikleri tekrar hesaplamaz, sadece
    canlı sinyali üretir.

Kurulum:
    pip install yfinance pandas scikit-learn joblib requests

Telegram kimlik bilgileri (bu script hiçbir token'ı repoya/koda YAZMAZ,
sadece ortam değişkeninden veya yanındaki .env dosyasından okur):
    TELEGRAM_BOT_TOKEN=123456:ABC-...
    TELEGRAM_CHAT_ID=123456789

Kullanım:
    python scripts/daily_signal_report.py            # her gün 18:15'i bekleyip çalışır (döngü)
    python scripts/daily_signal_report.py --once      # hemen tek sefer çalışıp çık (test için)
    python scripts/daily_signal_report.py --run-time 18:30
    python scripts/daily_signal_report.py --symbols-file scripts/bist100_symbols.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, time as dtime
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

ISTANBUL_TZ = ZoneInfo("Europe/Istanbul")
DEFAULT_RUN_TIME = dtime(18, 15)
HISTORY_PERIOD = "1y"  # MA200 ısınması + son kapanış için yeterli geçmiş
BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45
TELEGRAM_MESSAGE_LIMIT = 3500  # Telegram'ın 4096 sınırının altında güvenli pay


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


def _build_watchlist(symbols_file: Path) -> list[tuple[str, str, str]]:
    """(ticker, sembol, varlik_tipi) üçlüleri döner."""
    watchlist = [("TRY=X", "USDTRY", "DOVIZ"), ("EURTRY=X", "EURTRY", "DOVIZ")]
    for s in _read_symbols(symbols_file):
        watchlist.append((f"{s}.IS", f"BIST:{s}", "BORSA"))
    return watchlist


def _fetch_daily_history(ticker: str) -> pd.DataFrame | None:
    try:
        hist = yf.Ticker(ticker).history(period=HISTORY_PERIOD, interval="1d", auto_adjust=False)
    except Exception as exc:  # noqa: BLE001 - ağ hatası, bu sembolü atla
        print(f"  [UYARI] {ticker} veri alınamadı: {exc}", file=sys.stderr)
        return None
    if hist is None or hist.empty:
        return None
    hist.index = pd.DatetimeIndex(hist.index).tz_localize(None).normalize()
    return hist


def _classify(proba: float, buy_threshold: float, sell_threshold: float) -> str:
    if proba >= buy_threshold:
        return "AL"
    if proba <= sell_threshold:
        return "SAT"
    return "TUT"


def build_daily_signals(
    model, symbols_file: Path, buy_threshold: float, sell_threshold: float
) -> pd.DataFrame:
    """Watchlist'teki her sembol için en güncel günlük kapanışa dayalı
    AL/SAT/TUT sinyalini içeren bir DataFrame döner.
    """
    today = datetime.now(ISTANBUL_TZ).date()
    rows = []
    for ticker, sembol, varlik_tipi in _build_watchlist(symbols_file):
        hist = _fetch_daily_history(ticker)
        if hist is None:
            print(f"  [UYARI] {sembol}: veri yok, atlanıyor.")
            continue

        long_df = pd.DataFrame(
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
        features = build_feature_table(long_df)
        last_row = features.tail(1)
        if last_row.empty or last_row[FEATURE_COLUMNS].isna().any(axis=None):
            print(f"  [UYARI] {sembol}: yetersiz geçmiş (ısınma dönemi), atlanıyor.")
            continue

        bar_date = last_row["tarih"].iloc[0].date()
        proba = float(model.predict_proba(last_row[FEATURE_COLUMNS])[:, 1][0])
        rows.append(
            {
                "sembol": sembol,
                "varlik_tipi": varlik_tipi,
                "veri_tarihi": bar_date,
                "veri_guncel_mi": bar_date == today,
                "kapanis": float(last_row["kapanis"].iloc[0]),
                "model_guveni": proba,
                "sinyal": _classify(proba, buy_threshold, sell_threshold),
            }
        )

    return pd.DataFrame(rows).sort_values("model_guveni", ascending=False).reset_index(drop=True)


def _format_report_message(signals: pd.DataFrame, run_time: datetime) -> list[str]:
    header = f"📊 <b>Günlük BIST/Altın/Döviz Sinyal Raporu</b>\n{run_time.strftime('%Y-%m-%d %H:%M')}\n"
    stale = signals[~signals["veri_guncel_mi"]]
    sections = []
    for label, emoji in (("AL", "📈"), ("SAT", "📉"), ("TUT", "⏸")):
        subset = signals[signals["sinyal"] == label]
        if subset.empty:
            continue
        lines = [f"{emoji} <b>{label}</b> ({len(subset)})"]
        for row in subset.itertuples(index=False):
            lines.append(f"  {row.sembol}: {row.kapanis:.2f} (P=%{row.model_guveni * 100:.0f})")
        sections.append("\n".join(lines))

    if stale.shape[0]:
        stale_syms = ", ".join(stale["sembol"].tolist())
        sections.append(f"⚠️ Veri güncel değil (son kapanış bugüne ait değil): {stale_syms}")

    body = header + "\n\n".join(sections)

    messages = []
    while len(body) > TELEGRAM_MESSAGE_LIMIT:
        cut = body.rfind("\n", 0, TELEGRAM_MESSAGE_LIMIT)
        cut = cut if cut > 0 else TELEGRAM_MESSAGE_LIMIT
        messages.append(body[:cut])
        body = body[cut:]
    messages.append(body)
    return messages


def run_daily_report(model, symbols_file: Path, buy_threshold: float, sell_threshold: float) -> pd.DataFrame:
    now = datetime.now(ISTANBUL_TZ)
    signals = build_daily_signals(model, symbols_file, buy_threshold, sell_threshold)
    if signals.empty:
        send_telegram_message("⚠️ Günlük sinyal raporu: hiçbir sembol için veri alınamadı.")
        return signals

    for message in _format_report_message(signals, now):
        send_telegram_message(message)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    signals.to_csv(ARTIFACTS_DIR / "gunluk_sinyal_raporu.csv", index=False)
    return signals


def _next_run_datetime(now: datetime, run_time: dtime) -> datetime:
    candidate = now.replace(hour=run_time.hour, minute=run_time.minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:  # hafta sonu -> pazartesiye kaydır
        candidate += timedelta(days=1)
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols-file", type=Path, default=DEFAULT_SYMBOLS_FILE)
    parser.add_argument("--buy-threshold", type=float, default=BUY_THRESHOLD)
    parser.add_argument("--sell-threshold", type=float, default=SELL_THRESHOLD)
    parser.add_argument(
        "--run-time", type=str, default="18:15", help="Günlük çalışma saati, HH:MM (Europe/Istanbul, varsayılan 18:15)"
    )
    parser.add_argument("--once", action="store_true", help="Hemen tek sefer çalışıp çık (döngü yok, test için)")
    args = parser.parse_args()

    _load_env_file(Path(__file__).parent / ".env")
    _telegram_credentials()  # erken doğrulama: eksikse hemen hata ver

    hour, minute = (int(x) for x in args.run_time.split(":"))
    run_time = dtime(hour, minute)

    if not MODEL_PATH.exists():
        print(
            f"HATA: {MODEL_PATH} yok. Önce çalıştırın: "
            "python -m yatirim.ml.run_pipeline --data <gercek_fiyatlar.csv>",
            file=sys.stderr,
        )
        return 1

    model = joblib.load(MODEL_PATH)

    if args.once:
        print("Tek seferlik çalıştırma...")
        signals = run_daily_report(model, args.symbols_file, args.buy_threshold, args.sell_threshold)
        print(signals.to_string(index=False) if not signals.empty else "Sinyal üretilemedi.")
        return 0

    print(f"Günlük sinyal raporu başladı. Her gün (hafta içi) {args.run_time} (Europe/Istanbul) çalışacak.")
    while True:
        now = datetime.now(ISTANBUL_TZ)
        next_run = _next_run_datetime(now, run_time)
        wait_seconds = (next_run - now).total_seconds()
        print(f"Sıradaki çalışma: {next_run.strftime('%Y-%m-%d %H:%M')} ({wait_seconds / 3600:.1f} saat sonra)")
        time.sleep(wait_seconds)

        print(f"\n[{datetime.now(ISTANBUL_TZ).strftime('%Y-%m-%d %H:%M:%S')}] Günlük rapor çalıştırılıyor...")
        try:
            signals = run_daily_report(model, args.symbols_file, args.buy_threshold, args.sell_threshold)
            print(f"Rapor gönderildi: {len(signals)} sembol.")
        except Exception as exc:  # noqa: BLE001 - döngü çökmemeli
            print(f"  [HATA] Rapor başarısız: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
