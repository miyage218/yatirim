"""Telegram'a atılan portföy ekran görüntüsünü (yerel OCR ile) okuyup,
tespit edilen hisseler için günün AL/SAT/TUT sinyaline göre öneri üretir.

Kurulum (Windows):
    pip install pytesseract pillow
    Tesseract OCR motorunu AYRICA kurun (pip paketi sadece bir python
    sarmalayıcısı, OCR motorunun kendisini kurmaz):
    https://github.com/UB-Mannheim/tesseract/wiki adresinden .exe indirip
    kurun. Kurulumdan sonra varsayılan yol genelde
    "C:\\Program Files\\Tesseract-OCR\\tesseract.exe" olur; farklı bir yere
    kurduysanız scripts/.env dosyasına şunu ekleyin:
        TESSERACT_CMD=C:\\tam\\yol\\tesseract.exe
    Kurulumda Türkçe dil paketini ("tur") de işaretlemeyi unutmayın.

Dürüst sınırlama: Yerel/ücretsiz OCR, uygulama ekran görüntülerindeki
hisse kodlarını/adetlerini HER ZAMAN doğru okumayabilir (bulanık
görüntü, farklı yazı tipi/tema, kırpılmış ekran gibi durumlarda). Bu
yüzden bot, önerisinden önce NE OKUDUĞUNU açıkça listeler — kullanıcı
yanlış okunan bir şey görürse o kısmı görmezden gelebilir.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd
import requests

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
SIGNALS_CSV_PATH = ARTIFACTS_DIR / "gunluk_sinyal_raporu.csv"
DOWNLOADED_PHOTO_PATH = ARTIFACTS_DIR / "son_portfoy_foto.jpg"

_TICKER_PATTERN = re.compile(r"\b([A-ZÇĞİÖŞÜ]{3,6})(?:\.[A-Z])?\b")
_ADET_PATTERN = re.compile(r"Sat[ıi]labilir\s*Adet[^\d\n]{0,60}(\d{1,3}(?:[.,]\d+)?)", re.IGNORECASE)


def _load_known_symbols() -> set[str]:
    path = Path(__file__).parent / "bist100_symbols.txt"
    symbols = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            symbols.add(line.upper())
    return symbols


def download_telegram_photo(token: str, file_id: str) -> Path | None:
    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{token}/getFile", params={"file_id": file_id}, timeout=15
        )
        data = resp.json()
        if not data.get("ok"):
            print(f"  [UYARI] getFile başarısız: {data}", file=sys.stderr)
            return None
        file_path = data["result"]["file_path"]
        file_url = f"https://api.telegram.org/file/bot{token}/{file_path}"
        img_resp = requests.get(file_url, timeout=30)
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        DOWNLOADED_PHOTO_PATH.write_bytes(img_resp.content)
        return DOWNLOADED_PHOTO_PATH
    except requests.RequestException as exc:
        print(f"  [UYARI] Fotoğraf indirilemedi: {exc}", file=sys.stderr)
        return None


def extract_text_from_image(image_path: Path) -> str | None:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        print(
            "  [UYARI] pytesseract/Pillow kurulu değil. "
            "pip install pytesseract pillow ve Tesseract OCR motorunu kurun "
            "(bkz. scripts/portfolio_photo.py başındaki not).",
            file=sys.stderr,
        )
        return None

    tesseract_cmd = os.environ.get("TESSERACT_CMD")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image, lang="tur+eng")
    except Exception as exc:  # noqa: BLE001 - Tesseract motoru kurulu/PATH'te değilse burada patlar
        print(f"  [UYARI] OCR başarısız (Tesseract motoru kurulu mu?): {exc}", file=sys.stderr)
        return None


def parse_portfolio_from_text(text: str) -> list[dict]:
    """OCR metninden (sembol, adet) çiftlerini best-effort çıkarır.

    Yalnızca scripts/bist100_symbols.txt'teki bilinen BIST kodlarını
    aday olarak kabul eder — bu, rastgele OCR gürültüsünün gerçek bir
    hisse kodu gibi yanlış algılanmasını büyük ölçüde engeller.
    """
    known = _load_known_symbols()
    lines = text.splitlines()
    holdings: list[dict] = []
    seen: set[str] = set()

    for i, line in enumerate(lines):
        for match in _TICKER_PATTERN.finditer(line.upper()):
            ticker = match.group(1)
            if ticker not in known or ticker in seen:
                continue
            # Uygulama ekranında her hisse bloğu birkaç satır sürüyor
            # (Son Fiyat, Bugünkü Değer, Satılabilir Adet, ...) — ticker
            # satırından sonraki birkaç satır içinde adedi ara.
            window = "\n".join(lines[i : i + 9])
            adet_match = _ADET_PATTERN.search(window)
            if adet_match is None:
                continue
            try:
                adet = float(adet_match.group(1).replace(",", "."))
            except ValueError:
                continue
            holdings.append({"sembol": f"BIST:{ticker}", "adet": adet})
            seen.add(ticker)

    return holdings


def build_portfolio_advice(holdings: list[dict]) -> str:
    if not holdings:
        return (
            "⚠️ Görüntüden hiçbir hisse/adet net şekilde okunamadı. Ekran "
            "görüntüsünün net, kırpılmamış olduğundan ve hisse kartlarının "
            "(Son Fiyat / Satılabilir Adet alanları dahil) tam göründüğünden "
            "emin olup tekrar dener misin?"
        )

    if not SIGNALS_CSV_PATH.exists():
        return "⚠️ Henüz bir günlük sinyal raporu üretilmemiş, önce raporun oluşmasını bekle."

    signals = pd.read_csv(SIGNALS_CSV_PATH)
    signals_by_symbol = {row.sembol: row for row in signals.itertuples()}
    held_symbols = {h["sembol"] for h in holdings}

    lines = ["📷 <b>Görüntüden okunanlar (kontrol et):</b>"]
    for h in holdings:
        lines.append(f"  {h['sembol'].split(':')[-1]}: {h['adet']:.0f} adet")

    lines.append("")
    lines.append("📊 <b>Son rapora göre öneri:</b>")
    emoji_by_signal = {"AL": "📈", "SAT": "📉", "TUT": "⏸"}

    for h in holdings:
        row = signals_by_symbol.get(h["sembol"])
        ticker = h["sembol"].split(":")[-1]
        if row is None:
            lines.append(f"  {ticker}: son raporda bu sembol yok, öneri veremiyorum.")
            continue
        emoji = emoji_by_signal.get(row.sinyal, "")
        if row.sinyal == "SAT":
            lines.append(
                f"  {emoji} {ticker}: SAT — model bu hissede düşüş öngörüyor (P=%{row.model_guveni * 100:.0f})."
            )
        elif row.sinyal == "AL":
            lines.append(
                f"  {emoji} {ticker}: TUT (zaten elinde, model de yükseliş bekliyor — P=%{row.model_guveni * 100:.0f})."
            )
        else:
            lines.append(f"  {emoji} {ticker}: TUT (model kararsız — P=%{row.model_guveni * 100:.0f}).")

    extra_al = signals[(signals["sinyal"] == "AL") & (~signals["sembol"].isin(held_symbols))]
    if not extra_al.empty:
        lines.append("")
        lines.append("💡 Elinde olmayıp modelin AL dediği diğer hisseler:")
        for row in extra_al.itertuples():
            ticker = row.sembol.split(":")[-1] if row.sembol.startswith("BIST:") else row.sembol
            lines.append(f"  {ticker}: P=%{row.model_guveni * 100:.0f}")

    lines.append("")
    lines.append(
        "⚠️ Bu öneri, yerel (ücretsiz) OCR ile okunan bir ekran görüntüsüne "
        "dayanıyor — yanlış okuma ihtimaline karşı yukarıdaki 'okunanlar' "
        "listesini mutlaka kontrol et."
    )
    return "\n".join(lines)
