#!/usr/bin/env python3
"""Yahoo Finance'ten (yfinance) son N yıllık BIST 100 hisseleri + gram
altın + USD/EUR verisini indirip yatirim/ml pipeline'ının beklediği
şemada tek bir CSV üretir.

NOT: Bu script'i Claude Code ortamında DEĞİL, internete açık kendi
makinenizde (veya CI/sunucunuzda) çalıştırın — bu depo, güvenlik amaçlı
kısıtlı bir ağ ortamında geliştirildiği için buradan Yahoo Finance'e
erişilemiyor.

Kurulum:
    pip install yfinance pandas

Kullanım:
    python scripts/collect_market_data.py --years 3 --out artifacts/gercek_fiyatlar.csv

Çıktı şeması (yatirim/ml/synthetic_data.generate_synthetic_market_data ile
birebir aynı, doğrudan yatirim.ml.features.build_feature_table'a verilebilir):
    tarih, sembol, varlik_tipi, acilis, yuksek, dusuk, kapanis, hacim
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print(
        "yfinance kurulu değil. Önce çalıştırın: pip install yfinance pandas",
        file=sys.stderr,
    )
    raise

TROY_OUNCE_IN_GRAMS = 31.1034768
DEFAULT_SYMBOLS_FILE = Path(__file__).parent / "bist100_symbols.txt"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 3


def _read_symbols(symbols_file: Path) -> list[str]:
    symbols = []
    for line in symbols_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            symbols.append(line)
    return symbols


def _download_with_retry(ticker: str, period: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
            if df is not None and not df.empty:
                return df
            last_error = RuntimeError(f"{ticker}: boş veri döndü")
        except Exception as exc:  # noqa: BLE001 - üçüncü parti ağ hatası, tekrar denenecek
            last_error = exc
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)
    print(f"  [UYARI] {ticker} indirilemedi, atlanıyor: {last_error}", file=sys.stderr)
    return pd.DataFrame()


def _normalized_naive_dates(index: pd.Index) -> pd.DatetimeIndex:
    """Tz-aware/naive farkı olmadan takvim gününe indirger.

    GC=F (ons altın, ABD borsası saati) ve TRY=X/EURTRY=X (FX, farklı
    saat dilimi) farklı tz'lerle döner; ham tz-aware index'ler üzerinden
    join yapılırsa 'aynı gün' aslında eşleşmez ve sonuç sessizce boş
    kalır. Bu yüzden her iki taraf da join'den önce tz'siz takvim gününe
    indirgenmelidir.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    return idx.normalize()


def _to_long_format(df: pd.DataFrame, sembol: str, varlik_tipi: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "tarih": _normalized_naive_dates(df.index),
            "sembol": sembol,
            "varlik_tipi": varlik_tipi,
            "acilis": df["Open"].to_numpy(),
            "yuksek": df["High"].to_numpy(),
            "dusuk": df["Low"].to_numpy(),
            "kapanis": df["Close"].to_numpy(),
            "hacim": df["Volume"].to_numpy(),
        }
    )
    return out.dropna(subset=["kapanis"])


def collect(years: int, symbols_file: Path) -> pd.DataFrame:
    period = f"{years}y"
    frames: list[pd.DataFrame] = []

    print(f"USD/TRY ve EUR/TRY indiriliyor (period={period})...")
    usdtry = _download_with_retry("TRY=X", period)
    eurtry = _download_with_retry("EURTRY=X", period)
    if usdtry.empty or eurtry.empty:
        raise RuntimeError("USD/TRY veya EUR/TRY verisi alınamadı, devam edilemiyor.")
    frames.append(_to_long_format(usdtry, "USDTRY", "DOVIZ"))
    frames.append(_to_long_format(eurtry, "EURTRY", "DOVIZ"))

    print("Ons altın (USD) indiriliyor...")
    ons_altin = _download_with_retry("GC=F", period)
    if ons_altin.empty:
        raise RuntimeError("Ons altın verisi alınamadı, devam edilemiyor.")

    ons_altin_gunluk = ons_altin[["Open", "High", "Low", "Close", "Volume"]].copy()
    ons_altin_gunluk.index = _normalized_naive_dates(ons_altin.index)
    usdtry_gunluk_close = usdtry["Close"].copy()
    usdtry_gunluk_close.index = _normalized_naive_dates(usdtry.index)

    gram_altin = ons_altin_gunluk.join(
        usdtry_gunluk_close.rename("usdtry_close"), how="inner"
    )
    if gram_altin.empty:
        raise RuntimeError(
            "Ons altın ve USD/TRY serileri hiçbir takvim gününde örtüşmedi, "
            "gram altın hesaplanamadı."
        )
    for col in ("Open", "High", "Low", "Close"):
        gram_altin[col] = gram_altin[col] * gram_altin["usdtry_close"] / TROY_OUNCE_IN_GRAMS
    frames.append(_to_long_format(gram_altin, "GRAM_ALTIN", "ALTIN"))

    symbols = _read_symbols(symbols_file)
    print(f"{len(symbols)} BIST hissesi indiriliyor (period={period})...")
    for i, symbol in enumerate(symbols, start=1):
        ticker = f"{symbol}.IS"
        print(f"  [{i}/{len(symbols)}] {ticker}")
        df = _download_with_retry(ticker, period)
        if df.empty:
            continue
        frames.append(_to_long_format(df, f"BIST:{symbol}", "BORSA"))

    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["sembol", "tarih"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, default=3, help="Kaç yıllık veri (varsayılan 3)")
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=DEFAULT_SYMBOLS_FILE,
        help="Sembol listesi dosyası (varsayılan: scripts/bist100_symbols.txt)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/gercek_fiyatlar.csv"),
        help="Çıktı CSV yolu",
    )
    args = parser.parse_args()

    data = collect(args.years, args.symbols_file)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)

    n_symbols = data["sembol"].nunique()
    print(f"\nBitti: {len(data)} satır, {n_symbols} sembol -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
