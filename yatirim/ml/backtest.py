"""Geçmiş veri üzerinde sinyal/pozisyon demosu (backtest)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, FORWARD_HORIZON_DAYS
from .model import TrainResult

DEFAULT_BUY_THRESHOLD = 0.55


@dataclass
class BacktestSummary:
    sinyal_sayisi: int
    isabet_orani: float
    ortalama_getiri: float
    ortalama_getiri_isabetli: float
    ortalama_getiri_isabetsiz: float
    kumulatif_getiri: float

    def to_dict(self) -> dict:
        return {
            "sinyal_sayisi": self.sinyal_sayisi,
            "isabet_orani_yuzde": round(self.isabet_orani * 100, 1),
            "ortalama_getiri_yuzde": round(self.ortalama_getiri * 100, 2),
            "isabetli_sinyal_ort_getiri_yuzde": round(self.ortalama_getiri_isabetli * 100, 2),
            "isabetsiz_sinyal_ort_getiri_yuzde": round(self.ortalama_getiri_isabetsiz * 100, 2),
            "esit_agirlik_kumulatif_getiri_yuzde": round(self.kumulatif_getiri * 100, 1),
        }


def run_backtest(
    result: TrainResult, buy_threshold: float = DEFAULT_BUY_THRESHOLD
) -> tuple[pd.DataFrame, BacktestSummary]:
    """`test_df` üzerinde modelin AL sinyallerini ve gerçekleşen sonuçları üretir.

    Her sinyal, sinyal gününün kapanışında pozisyon açıldığını ve
    FORWARD_HORIZON_DAYS işlem günü sonra kapatıldığını varsayar
    (`ileri_getiri_10g`, bu ufuktaki gerçekleşen getiridir — özellik
    tablosunda etiketle birlikte zaten hesaplanmıştır, gelecek sızıntısı
    yoktur çünkü backtest yalnızca zaten geçmişte kalmış tarihleri raporlar).
    """
    test_df = result.test_df
    proba = result.model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]

    signals = test_df.assign(model_proba=proba)
    signals = signals[signals["model_proba"] >= buy_threshold].copy()
    signals["isabet"] = signals["ileri_getiri_10g"] > 0
    signals = signals.sort_values("tarih")

    signal_table = signals[
        [
            "tarih",
            "sembol",
            "kapanis",
            "model_proba",
            "ileri_getiri_10g",
            "isabet",
        ]
    ].rename(
        columns={
            "tarih": "sinyal_tarihi",
            "kapanis": "giris_fiyati",
            "model_proba": "model_guveni",
            "ileri_getiri_10g": f"gerceklesen_getiri_{FORWARD_HORIZON_DAYS}g",
        }
    )

    if signals.empty:
        summary = BacktestSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return signal_table, summary

    returns = signals["ileri_getiri_10g"]
    hits = signals["isabet"]

    summary = BacktestSummary(
        sinyal_sayisi=len(signals),
        isabet_orani=hits.mean(),
        ortalama_getiri=returns.mean(),
        ortalama_getiri_isabetli=returns[hits].mean() if hits.any() else 0.0,
        ortalama_getiri_isabetsiz=returns[~hits].mean() if (~hits).any() else 0.0,
        kumulatif_getiri=float(np.prod(1 + returns / len(returns)) - 1) if len(returns) else 0.0,
    )
    return signal_table, summary


def latest_signals(
    result: TrainResult, feature_df: pd.DataFrame, buy_threshold: float = DEFAULT_BUY_THRESHOLD
) -> pd.DataFrame:
    """Her sembol için en güncel (etiketsiz) satırda canlı model sinyali üretir."""
    usable = feature_df.dropna(subset=FEATURE_COLUMNS).copy()
    latest = usable.sort_values("tarih").groupby("sembol", as_index=False).tail(1)

    proba = result.model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]
    latest = latest.assign(model_guveni=proba)
    latest["sinyal"] = np.where(latest["model_guveni"] >= buy_threshold, "AL", "TUT")
    return latest[
        ["tarih", "sembol", "varlik_tipi", "kapanis", "model_guveni", "sinyal"]
    ].sort_values("model_guveni", ascending=False).reset_index(drop=True)
