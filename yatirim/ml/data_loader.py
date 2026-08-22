"""`scripts/collect_market_data.py` ile toplanan gerçek fiyat CSV'sini yükler."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "tarih",
    "sembol",
    "varlik_tipi",
    "acilis",
    "yuksek",
    "dusuk",
    "kapanis",
    "hacim",
]


def load_price_csv(path: str | Path) -> pd.DataFrame:
    """`yatirim.ml.synthetic_data.generate_synthetic_market_data` ile aynı
    şemadaki bir CSV'yi okuyup doğrular; `yatirim.ml.features.build_feature_table`'a
    doğrudan verilebilecek bir DataFrame döndürür.
    """
    df = pd.read_csv(path, parse_dates=["tarih"])

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV beklenen şemayı karşılamıyor, eksik sütunlar: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(f"{path} boş, veri yok.")

    return df.sort_values(["sembol", "tarih"]).reset_index(drop=True)
