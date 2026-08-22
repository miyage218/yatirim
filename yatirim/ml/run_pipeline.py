"""Uçtan uca demo: veri hazırla -> özellik çıkar -> model eğit
-> geçmiş dönem üzerinde durum analizi (backtest) göster -> canlı sinyalleri
öneri formatında bas.

Varsayılan olarak sentetik veri üretir (bu ortamdan gerçek piyasa
verisine ağ erişimi yok). Gerçek veriyle çalıştırmak için önce
`scripts/collect_market_data.py`'yi internete açık bir makinede
çalıştırıp çıkan CSV'yi `--data` ile verin:

Kullanım:
    python -m yatirim.ml.run_pipeline
    python -m yatirim.ml.run_pipeline --data artifacts/gercek_fiyatlar.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .backtest import latest_signals, run_backtest
from .data_loader import load_price_csv
from .features import build_feature_table
from .integration import signals_to_recommendations
from .model import train_model
from .synthetic_data import generate_synthetic_market_data

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
TEST_WINDOW_DAYS = 252  # backtest/demo dönemi: son ~1 yıl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="scripts/collect_market_data.py çıktısı gerçek fiyat CSV'si. "
        "Verilmezse sentetik veri üretilir.",
    )
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    if args.data is not None:
        print(f"[1/4] Gerçek fiyat verisi yükleniyor: {args.data}")
        raw_prices = load_price_csv(args.data)
    else:
        print("[1/4] 3 yıllık sentetik BIST/altın/döviz verisi üretiliyor...")
        raw_prices = generate_synthetic_market_data(years=3, seed=42)
        raw_prices.to_csv(ARTIFACTS_DIR / "sentetik_fiyatlar.csv", index=False)

    print("[2/4] Teknik göstergeler ve etiketler hesaplanıyor...")
    feature_df = build_feature_table(raw_prices)

    test_start = feature_df["tarih"].max() - pd.Timedelta(days=TEST_WINDOW_DAYS * 7 / 5)
    print(f"[3/4] Model eğitiliyor (test/demo dönemi: {test_start.date()} sonrası)...")
    result = train_model(feature_df, test_start=test_start)
    joblib.dump(result.model, ARTIFACTS_DIR / "model.joblib")

    print(
        f"    Eğitim doğruluğu: %{result.train_accuracy * 100:.1f} | "
        f"Test doğruluğu: %{result.test_accuracy * 100:.1f} | "
        f"Test AUC: {result.test_auc:.3f}"
    )

    print("[4/4] Geçmiş dönem durum analizi (backtest demosu) çalıştırılıyor...")
    signal_table, summary = run_backtest(result)
    signal_table.to_csv(ARTIFACTS_DIR / "backtest_sinyalleri.csv", index=False)

    print()
    print("=" * 72)
    print("DURUM ANALİZİ DEMOSU (geçmiş dönem, gerçekleşmiş sonuçlarla)")
    print("=" * 72)
    if summary.sinyal_sayisi == 0:
        print("Bu dönemde eşik üzerinde AL sinyali üretilmedi.")
    else:
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        print()
        print("Örnek sinyaller (ilk 5):")
        with pd.option_context("display.max_columns", None, "display.width", 120):
            print(signal_table.head(5).to_string(index=False))

    print()
    print("=" * 72)
    print("GÜNCEL MODEL SİNYALLERİ (son toplanan veriye göre, CANLI DEĞİL)")
    print("=" * 72)
    latest = latest_signals(result, feature_df)
    n_al = (latest["sinyal"] == "AL").sum()
    print(f"{n_al} AL sinyali / {len(latest)} enstrüman. En güvenli 20 sinyal:")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(latest.head(20).to_string(index=False))
    latest.to_csv(ARTIFACTS_DIR / "guncel_sinyaller.csv", index=False)

    recommendations = signals_to_recommendations(latest, summary)
    print()
    print("Öneri (Recommendation) formatına dönüştürülmüş örnek:")
    print(json.dumps(recommendations[0].to_dict(), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
