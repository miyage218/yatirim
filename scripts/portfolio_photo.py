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

# Gerçek Tesseract çıktısında (mobil uygulama arayüzünün sütunlu
# yapısı yüzünden) etiketler ("Satılabilir Adet" gibi) tüm hisseler
# için önce tek blok halinde, DEĞERLER ise ayrı bir blok halinde,
# etiketlerle aynı sırada gelebiliyor — yani "Satılabilir Adet"
# kelimesinin hemen yanında sayı olmuyor. Bu yüzden adet sayılarını
# "TL yazmayan, işaretsiz (+/-/%siz), yanında TL kelimesi olmayan
# bağımsız sayı" olarak tanıyıp, tespit edilen ticker sırasıyla
# pozisyonel eşliyoruz (Son Fiyat/Bugünkü Değer hep "<sayı> TL",
# Kar/Zarar ve Getiri hep işaretli/% içerir; sadece Adet çıplak bir
# sayıdır).
_BARE_NUMBER_TOKEN = re.compile(r"^\d{1,4}(?:[.,]\d{1,2})?$")
# Bu kelimelerden biri bir sayının hemen ardından geliyorsa, o sayı adet
# değil başka bir şeydir (TL tutarı, "X dakika önce" zaman damgası, ...).
_NOT_ADET_FOLLOWERS = {"TL", "DAKIKA", "SAAT", "SANIYE", "GÜN", "GUN"}


def _extract_bare_adet_tokens(text: str, known_symbols: set[str]) -> list[float]:
    adet_values: list[float] = []
    for line in text.splitlines():
        # Bir hisse kodunun (ör. "TUPRS.E") göründüğü başlık satırında
        # genelde fiyat/yüzde değişimi de bulunuyor — bunlar adet değil,
        # bu yüzden ticker içeren satırları tamamen atlıyoruz.
        if any(m.group(1) in known_symbols for m in _TICKER_PATTERN.finditer(line.upper())):
            continue
        tokens = line.split()
        for idx, tok in enumerate(tokens):
            if not _BARE_NUMBER_TOKEN.fullmatch(tok):
                continue
            next_tok = tokens[idx + 1] if idx + 1 < len(tokens) else ""
            if next_tok.upper() in _NOT_ADET_FOLLOWERS:
                continue
            try:
                adet_values.append(float(tok.replace(",", ".")))
            except ValueError:
                continue
    return adet_values


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
    seen: set[str] = set()
    tickers: list[str] = []
    for line in text.splitlines():
        for match in _TICKER_PATTERN.finditer(line.upper()):
            ticker = match.group(1)
            if ticker in known and ticker not in seen:
                tickers.append(ticker)
                seen.add(ticker)

    if not tickers:
        return []

    adet_values = _extract_bare_adet_tokens(text, known)
    if not adet_values:
        return []

    # Ticker'lar ve adetler ekranda aynı sırada göründüğü için
    # pozisyonel olarak eşleniyor. Sayıları tutarsız çıkarsa (adet
    # sayısı ticker sayısından azsa) sadece eşleşen kadarını kullan.
    holdings: list[dict] = []
    for ticker, adet in zip(tickers, adet_values):
        holdings.append({"sembol": f"BIST:{ticker}", "adet": adet})

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
