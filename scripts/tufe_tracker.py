"""Aylık TÜFE verisinin TAMAMEN MANUEL, Telegram hatırlatmalı takibi.

Bu ortamdan (ve pratikte otomatik kazımadan) TCMB/TÜİK verisine güvenilir
erişim kurulamadığı için TÜFE rakamı otomatik çekilmiyor. Bunun yerine:
her ayın ilk hafta sonunda (cumartesi/pazar, ayın ilk 7 günü içinde),
o ay için henüz rakam girilmediyse bir Telegram hatırlatması gönderilir;
kullanıcı düz bir sayıyla ("3.2" gibi) cevap verince kaydedilir.

Bu modül Telegram'a mesaj GÖNDERMEZ (döngüsel import'u önlemek için) —
sadece "hatırlatma zamanı mı", "bekleyen bir soru var mı" ve "cevabı
ayrıştır/kaydet" mantığını sağlar. Gönderme işini çağıran taraf
(daily_signal_report.py) yapar.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
TUFE_PATH = ARTIFACTS_DIR / "tufe.csv"
PENDING_PATH = ARTIFACTS_DIR / "tufe_bekliyor.json"

_NUMBER_PATTERN = re.compile(r"-?\d+([.,]\d+)?")


def month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def is_first_weekend(d: date) -> bool:
    """Ayın ilk 7 günü içindeki cumartesi/pazar günleri."""
    return d.day <= 7 and d.weekday() >= 5


def _load_tufe() -> pd.DataFrame:
    if not TUFE_PATH.exists():
        return pd.DataFrame(columns=["ay", "tufe_aylik_yuzde"])
    return pd.read_csv(TUFE_PATH)


def has_tufe_for_month(ay: str) -> bool:
    df = _load_tufe()
    return ay in df["ay"].astype(str).values


def _load_pending() -> dict | None:
    if not PENDING_PATH.exists():
        return None
    return json.loads(PENDING_PATH.read_text(encoding="utf-8"))


def _save_pending(ay: str) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    PENDING_PATH.write_text(json.dumps({"ay": ay}), encoding="utf-8")


def _clear_pending() -> None:
    if PENDING_PATH.exists():
        PENDING_PATH.unlink()


def should_prompt_for_tufe(today: date) -> str | None:
    """Bugün hatırlatma gönderilmeli mi? Öyleyse ay anahtarını (ör.
    '2026-08') döner, değilse None. Zaten bekleyen bir soru varsa (aynı ay
    için) tekrar sormaz.
    """
    if not is_first_weekend(today):
        return None
    ay = month_key(today)
    if has_tufe_for_month(ay):
        return None
    pending = _load_pending()
    if pending is not None and pending.get("ay") == ay:
        return None
    return ay


def mark_prompted(ay: str) -> None:
    _save_pending(ay)


def pending_month() -> str | None:
    pending = _load_pending()
    return pending.get("ay") if pending else None


def try_parse_reply(text: str) -> float | None:
    """'%3.2', '3,2', '3.2' gibi metinlerden yüzde değeri çıkarır."""
    match = _NUMBER_PATTERN.search(text)
    if not match:
        return None
    return float(match.group(0).replace(",", "."))


def save_tufe(ay: str, value: float) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    df = _load_tufe()
    df = df[df["ay"].astype(str) != ay]
    df = pd.concat([df, pd.DataFrame([{"ay": ay, "tufe_aylik_yuzde": value}])], ignore_index=True)
    df = df.sort_values("ay").reset_index(drop=True)
    df.to_csv(TUFE_PATH, index=False)
    _clear_pending()


def cumulative_tufe_series() -> pd.DataFrame:
    """Aylara göre kümülatif (bileşik) TÜFE getirisini döner: ay, kumulatif_yuzde."""
    df = _load_tufe().sort_values("ay")
    if df.empty:
        return pd.DataFrame(columns=["ay", "kumulatif_yuzde"])
    cumulative = (1 + df["tufe_aylik_yuzde"] / 100).cumprod() - 1
    return pd.DataFrame({"ay": df["ay"].values, "kumulatif_yuzde": cumulative.values * 100})
