#!/usr/bin/env python3
"""Her gün (varsayılan saat 16:00, BIST hâlâ açıkken) bir kez çalışıp,
izleme listesindeki her sembol için AL / SAT / TUT sinyalini tek bir
Telegram mesajında raporlayan günlük sinyal raporu.

NOT: Bu script'i Claude Code ortamında DEĞİL, internete açık kendi
makinenizde çalıştırın (bkz. scripts/run_daily_signal_report.bat).

Ön koşul: `python -m yatirim.ml.run_pipeline --data <csv>` daha önce
çalıştırılmış olmalı — bu script eğitilmiş modeli `artifacts/model.joblib`
dosyasından yükler, kendi başına eğitim yapmaz.

Nasıl çalışır (dürüst özet):
  - Varsayılan çalışma saati (16:00) BIST kapanışından (18:10) ÖNCEDİR —
    SAT sinyali geldiğinde piyasa hâlâ açıkken aynı gün işlem
    yapılabilsin diye kasıtlı olarak böyle seçildi. Bu saatte piyasa
    henüz kapanmadığı için günün KESİNLEŞMİŞ kapanışı yoktur; script
    bugünün şu ana kadar oluşan gün içi bar'ını (5 dakikalık mumlardan
    açılış/en yüksek/en düşük/son fiyat) kullanır ve rapor bunu "piyasa
    henüz açık" notuyla açıkça belirtir — bu KAPANIŞ ONAYLI DEĞİLDİR,
    gün sonuna kadar değişebilir. `--run-time` ile 18:10'dan SONRAKİ bir
    saat verilirse (ör. 18:30), script otomatik olarak günün kesinleşmiş
    kapanışını kullanır (gün içi tahmin gerekmez).
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

Ayrıca Telegram bot menüsüne iki komut eklenir:
  - "/son_rapor": en son gönderilen raporu, ne zaman üretildiği
    bilgisiyle birlikte, yeniden hesaplamadan tekrar gönderir.
  - "/grafik": model (gerçek AL/SAT pozisyon takibinden), TÜFE ve
    BIST100'ün kümülatif getirisini karşılaştıran grafiği gönderir.
Ayrıca her ayın ilk hafta sonunda, o ay için TÜFE henüz girilmediyse
Telegram'dan "TÜFE oranı nedir?" diye sorulur; düz bir sayıyla
("3.2" gibi) cevap verildiğinde kaydedilir ve grafik güncellenir.
Bekleme döngüsü sırasında bu komutlar/cevaplar long-polling ile dinlenir.

Kullanım:
    python scripts/daily_signal_report.py            # her gün 16:00'ı bekler, arada /son_rapor komutunu dinler
    python scripts/daily_signal_report.py --once      # hemen tek sefer rapor üretip çık (test için, komut dinlemez)
    python scripts/daily_signal_report.py --run-time 18:30
    python scripts/daily_signal_report.py --symbols-file scripts/bist100_symbols.txt
"""

from __future__ import annotations

import argparse
import json
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

import performance_chart  # noqa: E402
import positions  # noqa: E402
import tufe_tracker  # noqa: E402

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
DEFAULT_SYMBOLS_FILE = REPO_ROOT / "scripts" / "watchlist_live.txt"
LAST_REPORT_PATH = ARTIFACTS_DIR / "last_report.json"

ISTANBUL_TZ = ZoneInfo("Europe/Istanbul")
DEFAULT_RUN_TIME = dtime(16, 0)
MARKET_CLOSE = dtime(18, 10)  # BIST kapanışı — bu saatten önce çalışılırsa gün içi yaklaşık bar kullanılır
HISTORY_PERIOD = "2y"  # MA200 ısınması + son kapanış için güvenli pay
BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45
TELEGRAM_MESSAGE_LIMIT = 3500  # Telegram'ın 4096 sınırının altında güvenli pay

REPORT_COMMAND = "son_rapor"
CHART_COMMAND = "grafik"
COMMAND_POLL_TIMEOUT_SECONDS = 25  # getUpdates long-poll süresi
GETUPDATES_ERROR_BACKOFF_SECONDS = 10  # hata durumunda sıkı döngüye girmemek için bekleme


def _load_env_file(path: Path) -> None:
    """`scripts/.env`'i ortam değişkenlerine yükler.

    Kasıtlı olarak MEVCUT ortam değişkenlerinin üzerine yazar (setdefault
    DEĞİL): sistemde/kullanıcıda daha önceden (başka bir uygulama için)
    ayarlanmış aynı isimli bir TELEGRAM_BOT_TOKEN varsa, `.env` dosyası
    hâlâ öncelikli olmalı — aksi halde script sessizce yanlış bot'u
    kullanır ve bunu fark etmek zordur. Bir çakışma varsa uyarı basılır.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        existing = os.environ.get(key)
        if existing is not None and existing != value:
            print(
                f"  [UYARI] Ortam değişkeni {key} zaten sistemde tanımlıydı, "
                f"scripts/.env'deki değerle değiştiriliyor (sistemdeki eski "
                f"değer kullanılmayacak).",
                file=sys.stderr,
            )
        os.environ[key] = value


def print_bot_identity(token: str, chat_id: str) -> None:
    """Hangi Telegram bot'una ve chat ID'ye bağlanıldığını ekrana basar.

    Aynı bilgisayarda birden fazla proje/bot varsa, `.env`'deki token'ın
    gerçekten beklenen bot'a ait olup olmadığını göz kararı doğrulamak
    için kullanılır (bkz. README "getUpdates Conflict" notu).
    """
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
    except requests.RequestException as exc:
        print(f"  [UYARI] Bot kimliği doğrulanamadı (ağ hatası): {exc}", file=sys.stderr)
        return
    if not data.get("ok"):
        print(f"  [UYARI] Bot kimliği doğrulanamadı: {data}", file=sys.stderr)
        return
    bot = data["result"]
    print(f"Kullanılan bot: @{bot.get('username')} ({bot.get('first_name')}) -> chat_id={chat_id}")


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


def register_bot_commands() -> None:
    """Bot menüsüne (sohbette '/' tuşuna basınca çıkan liste) /son_rapor
    ve /grafik komutlarını ekler. Bu bir kereye mahsus değildir, script
    her başladığında tekrar çağrılır (Telegram API'de idempotent).
    """
    token, _ = _telegram_credentials()
    url = f"https://api.telegram.org/bot{token}/setMyCommands"
    commands = [
        {"command": REPORT_COMMAND, "description": "En son sinyal raporunu tekrar gönder"},
        {"command": CHART_COMMAND, "description": "Model vs TÜFE vs BIST100 grafiğini gönder"},
    ]
    try:
        resp = requests.post(url, json={"commands": commands}, timeout=15)
        if resp.status_code != 200:
            print(f"  [UYARI] Bot komut menüsü kaydedilemedi: {resp.status_code} {resp.text}", file=sys.stderr)
    except requests.RequestException as exc:
        print(f"  [UYARI] Bot komut menüsü kaydedilemedi: {exc}", file=sys.stderr)


def _save_last_report(messages: list[str], generated_at: datetime) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    LAST_REPORT_PATH.write_text(
        json.dumps(
            {"olusturulma_zamani": generated_at.isoformat(), "mesajlar": messages},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_last_report() -> dict | None:
    if not LAST_REPORT_PATH.exists():
        return None
    return json.loads(LAST_REPORT_PATH.read_text(encoding="utf-8"))


def reply_with_last_report() -> None:
    """/son_rapor komutuna cevap: en son ÜRETİLMİŞ raporu, ne zaman
    üretildiği bilgisiyle birlikte AYNEN tekrar gönderir. Modeli tekrar
    çalıştırmaz/yeniden hesaplamaz — sadece son sonucu tekrarlar.
    """
    last = _load_last_report()
    if last is None:
        send_telegram_message("Henüz gönderilmiş bir rapor yok. İlk günlük rapor 16:00'ta gönderilecek.")
        return

    generated_at = datetime.fromisoformat(last["olusturulma_zamani"])
    header = f"🕐 <b>En son rapor tarihi: {generated_at.strftime('%Y-%m-%d %H:%M')}</b>\n(yeniden hesaplanmadı, son çalıştırmadan tekrar gönderildi)\n\n"
    messages = last.get("mesajlar") or []
    if not messages:
        send_telegram_message(header + "(mesaj içeriği bulunamadı)")
        return
    send_telegram_message(header + messages[0])
    for message in messages[1:]:
        send_telegram_message(message)


def _get_updates(token: str, offset: int | None, timeout: int) -> list[dict]:
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    params: dict = {"timeout": timeout}
    if offset is not None:
        params["offset"] = offset
    try:
        resp = requests.get(url, params=params, timeout=timeout + 10)
    except requests.RequestException as exc:
        print(f"  [UYARI] getUpdates başarısız: {exc}", file=sys.stderr)
        time.sleep(GETUPDATES_ERROR_BACKOFF_SECONDS)
        return []
    if resp.status_code != 200:
        print(f"  [UYARI] getUpdates başarısız: {resp.status_code} {resp.text}", file=sys.stderr)
        if resp.status_code == 409:
            print(
                "  [UYARI] Bu genelde AYNI bot token'ıyla başka bir process de "
                "getUpdates çağırıyor demektir (Telegram tek seferde tek dinleyiciye "
                "izin verir). Bu proje için @BotFather'dan ayrı bir bot oluşturmanız "
                "önerilir.",
                file=sys.stderr,
            )
        time.sleep(GETUPDATES_ERROR_BACKOFF_SECONDS)
        return []
    return resp.json().get("result", [])


def _send_performance_chart(token: str, chat_id: str) -> None:
    chart_path = performance_chart.generate_chart()
    if chart_path is None:
        send_telegram_message(
            "Henüz grafik için yeterli veri yok — en az bir aylık TÜFE girişi gerekiyor "
            "(her ayın ilk hafta sonunda otomatik sorulur)."
        )
        return
    performance_chart.send_telegram_photo(token, chat_id, chart_path, caption="Model vs TÜFE vs BIST100")


def _finalize_month_snapshot(ay: str, token: str, chat_id: str) -> None:
    """Bir ayın TÜFE'si girildiğinde: o ana kadarki model performansını
    (gerçek AL/SAT pozisyonlarından) ve BIST100 kümülatif getirisini de
    hesaplayıp aylık karşılaştırma tablosuna ekler ve grafiği gönderir.
    """
    open_df = positions.load_open_positions()
    latest_prices = (
        performance_chart.fetch_latest_prices(open_df["sembol"].tolist()) if not open_df.empty else {}
    )
    model_return = positions.overall_model_return(latest_prices)

    tufe_cum_df = tufe_tracker.cumulative_tufe_series()
    tufe_cumulative = float(tufe_cum_df.iloc[-1]["kumulatif_yuzde"]) if not tufe_cum_df.empty else 0.0

    today_iso = datetime.now(ISTANBUL_TZ).date().isoformat()
    bist100_cumulative = performance_chart.bist100_cumulative_return(today_iso)

    performance_chart.record_monthly_snapshot(ay, model_return, tufe_cumulative, bist100_cumulative)
    _send_performance_chart(token, chat_id)


def _maybe_send_tufe_reminder(today, chat_id: str) -> None:
    ay = tufe_tracker.should_prompt_for_tufe(today)
    if ay is None:
        return
    send_telegram_message(
        f"📅 {ay} ayı için TÜFE (aylık) oranı nedir? Lütfen düz bir sayı olarak "
        f"cevapla (örn: 3.2)."
    )
    tufe_tracker.mark_prompted(ay)


def _handle_updates(updates: list[dict], chat_id: str, token: str) -> int | None:
    """Gelen komutları/cevapları işler, işlenen son update_id'yi döner
    (offset'i ilerletmek için)."""
    last_update_id = None
    for update in updates:
        last_update_id = update["update_id"]
        message = update.get("message") or update.get("channel_post")
        if not message:
            continue
        text = (message.get("text") or "").strip()
        incoming_chat_id = str(message.get("chat", {}).get("id", ""))
        if incoming_chat_id != str(chat_id):
            continue  # yalnızca .env'deki yetkili chat_id'den gelen komutlar işlenir

        if text.startswith(f"/{REPORT_COMMAND}"):
            print(f"  Komut alındı: {text} -> son rapor gönderiliyor")
            reply_with_last_report()
        elif text.startswith(f"/{CHART_COMMAND}"):
            print(f"  Komut alındı: {text} -> performans grafiği gönderiliyor")
            _send_performance_chart(token, chat_id)
        elif text == "/start":
            send_telegram_message(
                f"Merhaba! Her gün 16:00'ta otomatik AL/SAT/TUT raporu gönderilir. "
                f"İstediğin an en son raporu tekrar görmek için /{REPORT_COMMAND}, "
                f"performans grafiğini görmek için /{CHART_COMMAND} komutunu kullanabilirsin."
            )
        else:
            pending_ay = tufe_tracker.pending_month()
            if pending_ay is None:
                continue
            value = tufe_tracker.try_parse_reply(text)
            if value is None:
                send_telegram_message("Anlayamadım, lütfen sadece sayı olarak yaz (örn: 3.2).")
                continue
            print(f"  TÜFE cevabı alındı: {pending_ay} -> %{value}")
            tufe_tracker.save_tufe(pending_ay, value)
            send_telegram_message(f"✅ {pending_ay} için TÜFE %{value} olarak kaydedildi.")
            _finalize_month_snapshot(pending_ay, token, chat_id)
    return last_update_id


def wait_until_with_command_listening(next_run: datetime, token: str, chat_id: str, offset_state: dict) -> None:
    """`next_run`'a kadar bekler; beklerken Telegram'dan gelen komutları/
    cevapları long-polling ile dinleyip anında cevaplar, ayrıca her
    hafta sonu TÜFE hatırlatmasının gerekip gerekmediğini kontrol eder.
    """
    while True:
        now = datetime.now(ISTANBUL_TZ)
        try:
            _maybe_send_tufe_reminder(now.date(), chat_id)
        except Exception as exc:  # noqa: BLE001 - döngü asla bunun için çökmemeli
            print(f"  [HATA] TÜFE hatırlatma kontrolü başarısız: {exc}", file=sys.stderr)

        remaining = (next_run - now).total_seconds()
        if remaining <= 0:
            return
        poll_timeout = int(min(COMMAND_POLL_TIMEOUT_SECONDS, max(1, remaining)))
        updates = _get_updates(token, offset_state.get("offset"), poll_timeout)
        try:
            last_id = _handle_updates(updates, chat_id, token)
        except Exception as exc:  # noqa: BLE001 - bir komutun başarısız olması döngüyü öldürmemeli
            print(f"  [HATA] Komut işlenemedi: {exc}", file=sys.stderr)
            try:
                send_telegram_message(f"⚠️ Komut işlenirken bir hata oluştu: {exc}")
            except Exception:  # noqa: BLE001 - bildirim de başarısız olursa sessizce devam
                pass
            last_id = updates[-1]["update_id"] if updates else None
        if last_id is not None:
            offset_state["offset"] = last_id + 1


def _discard_pending_updates(token: str, offset_state: dict) -> None:
    """Script kapalıyken birikmiş eski komutları (varsa) yeniden işlemeden
    atlamak için başlangıçta bir kere çağrılır."""
    updates = _get_updates(token, offset_state.get("offset"), timeout=0)
    if updates:
        offset_state["offset"] = updates[-1]["update_id"] + 1


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


def _fetch_intraday_bar(ticker: str) -> dict | None:
    """Piyasa kapanmadan (MARKET_CLOSE'dan önce) çalıştırıldığında,
    bugünün şu ana kadar oluşan bar'ını (açılış/en yüksek/en düşük/son
    fiyat/hacim) dakikalık mumlardan türetir. Veri yoksa None döner.

    Yahoo Finance BIST gibi yerel borsalar için gün içi veriyi bazen
    `period="1d"` ile boş döndürüyor (özellikle açılıştan hemen sonra);
    bu durumda daha geniş bir pencere (`period="5d"`, bugüne filtrelenir)
    ile tekrar denenir.
    """
    today = datetime.now(ISTANBUL_TZ).date()
    for period, interval in (("1d", "5m"), ("5d", "15m")):
        try:
            intraday = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        except Exception as exc:  # noqa: BLE001 - ağ hatası, bu denemeyi atla
            print(f"  [UYARI] {ticker} gün içi veri alınamadı ({period}/{interval}): {exc}", file=sys.stderr)
            continue
        if intraday is None or intraday.empty:
            continue
        intraday.index = pd.DatetimeIndex(intraday.index).tz_localize(None)
        intraday = intraday[intraday.index.date == today]
        intraday = intraday.dropna(subset=["Close"])
        if not intraday.empty:
            break
    else:
        return None

    return {
        "acilis": float(intraday["Open"].iloc[0]),
        "yuksek": float(intraday["High"].max()),
        "dusuk": float(intraday["Low"].min()),
        "kapanis": float(intraday["Close"].iloc[-1]),
        "hacim": float(intraday["Volume"].sum()),
    }


def _classify(proba: float, buy_threshold: float, sell_threshold: float) -> str:
    if proba >= buy_threshold:
        return "AL"
    if proba <= sell_threshold:
        return "SAT"
    return "TUT"


def build_daily_signals(
    model, symbols_file: Path, buy_threshold: float, sell_threshold: float
) -> pd.DataFrame:
    """Watchlist'teki her sembol için AL/SAT/TUT sinyalini içeren bir
    DataFrame döner.

    MARKET_CLOSE'dan (18:10) SONRA çalıştırılırsa günün kesinleşmiş
    kapanışı kullanılır. ÖNCESİNDE çalıştırılırsa (ör. 16:00 varsayılan
    rapor saati — piyasa hâlâ açıkken SAT sinyaline aynı gün işlem
    yapılabilsin diye), bugünün şu ana kadar oluşan gün içi bar'ı (5
    dakikalık mumlardan türetilmiş açılış/en yüksek/en düşük/son fiyat)
    kullanılır; bu KAPANIŞ ONAYLI DEĞİLDİR, gün sonuna kadar değişebilir.
    """
    now = datetime.now(ISTANBUL_TZ)
    today = now.date()
    market_closed = now.time() >= MARKET_CLOSE

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
        # Yahoo Finance bazen gün sonuna en yakın (henüz kapanmamış/eksik)
        # bar'ı NaN kapanış ile döndürür; bu tek satır tail(1) ile seçilip
        # tüm özellik hesaplamasını (getiri, MA, RSI, MACD) NaN'a çevirir.
        # Gerçek kapanışı olmayan satırları en baştan eleyerek önlüyoruz.
        long_df = long_df.dropna(subset=["kapanis"])

        if not market_closed:
            # Piyasa henüz açık: bugüne ait (varsa) satırı çıkar, yerine
            # gün içi yaklaşık bar'ı ekle — böylece "bugün" her zaman
            # kesinleşmemiş ama GÜNCEL bir fiyatı yansıtır.
            long_df = long_df[long_df["tarih"].dt.date < today]
            bar = _fetch_intraday_bar(ticker)
            if bar is not None:
                today_row = pd.DataFrame(
                    [{"tarih": pd.Timestamp(today), "sembol": sembol, "varlik_tipi": varlik_tipi, **bar}]
                )
                long_df = pd.concat([long_df, today_row], ignore_index=True)
            # bar None ise: gün içi veri alınamadı, en son kapanışla devam
            # edilir (aşağıda veri_guncel_mi=False olarak işaretlenecek).

        features = build_feature_table(long_df)
        last_row = features.tail(1)
        if last_row.empty or last_row[FEATURE_COLUMNS].isna().any(axis=None):
            if last_row.empty:
                nan_cols = "tüm sütunlar (satır yok)"
            else:
                nan_cols = last_row[FEATURE_COLUMNS].columns[
                    last_row[FEATURE_COLUMNS].isna().any()
                ].tolist()
            print(
                f"  [UYARI] {sembol}: yetersiz geçmiş ({len(long_df)} geçerli bar), "
                f"NaN sütunlar: {nan_cols}, atlanıyor."
            )
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

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("model_guveni", ascending=False).reset_index(drop=True)


def _format_report_message(signals: pd.DataFrame, run_time: datetime) -> list[str]:
    header = f"📊 <b>Günlük BIST/Altın/Döviz Sinyal Raporu</b>\n{run_time.strftime('%Y-%m-%d %H:%M')}\n"
    if run_time.time() < MARKET_CLOSE:
        header += (
            "⏳ Piyasa henüz açık — fiyatlar gün içi anlık değerlerdir, "
            "kapanış onaylı değildir.\n"
        )
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

    messages = _format_report_message(signals, now)
    for message in messages:
        send_telegram_message(message)
    _save_last_report(messages, now)

    positions.update_positions(signals)

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
        "--run-time", type=str, default="16:00", help="Günlük çalışma saati, HH:MM (Europe/Istanbul, varsayılan 16:00)"
    )
    parser.add_argument("--once", action="store_true", help="Hemen tek sefer çalışıp çık (döngü yok, test için)")
    args = parser.parse_args()

    _load_env_file(Path(__file__).parent / ".env")
    token, chat_id = _telegram_credentials()  # erken doğrulama: eksikse hemen hata ver
    print_bot_identity(token, chat_id)

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

    register_bot_commands()
    offset_state: dict = {"offset": None}
    _discard_pending_updates(token, offset_state)

    print(
        f"Günlük sinyal raporu başladı. Her gün (hafta içi) {args.run_time} (Europe/Istanbul) çalışacak. "
        f"Bu arada Telegram'da /{REPORT_COMMAND} komutu dinleniyor."
    )
    while True:
        now = datetime.now(ISTANBUL_TZ)
        next_run = _next_run_datetime(now, run_time)
        print(f"Sıradaki çalışma: {next_run.strftime('%Y-%m-%d %H:%M')} ({(next_run - now).total_seconds() / 3600:.1f} saat sonra)")
        wait_until_with_command_listening(next_run, token, chat_id, offset_state)

        print(f"\n[{datetime.now(ISTANBUL_TZ).strftime('%Y-%m-%d %H:%M:%S')}] Günlük rapor çalıştırılıyor...")
        try:
            signals = run_daily_report(model, args.symbols_file, args.buy_threshold, args.sell_threshold)
            print(f"Rapor gönderildi: {len(signals)} sembol.")
        except Exception as exc:  # noqa: BLE001 - döngü çökmemeli
            print(f"  [HATA] Rapor başarısız: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
