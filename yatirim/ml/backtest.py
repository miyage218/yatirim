"""Geçmiş veri üzerinde sinyal/pozisyon demosu (backtest), işlem maliyeti dahil."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, FORWARD_HORIZON_DAYS
from .model import TrainResult

DEFAULT_BUY_THRESHOLD = 0.55

# Varlık tipine göre TEK YÖN (alış ya da satış) işlem maliyeti tahmini:
# aracı kurum komisyonu + borsa payı + BSMV + tipik alış-satış makas
# (spread) etkisi bir arada. Round-trip (giriş+çıkış) maliyeti bunun 2
# katıdır. Gerçek maliyetler aracı kurum/enstrümana göre değişir; bu
# değerler muhafazakar birer varsayılan tahmindir, `cost_by_asset_type`
# parametresiyle ezilebilir.
DEFAULT_COST_BY_ASSET_TYPE = {
    "BORSA": 0.0015,  # ~binde 1.5 (komisyon + BSMV + borsa payı)
    "ALTIN": 0.0020,  # gram altın alış-satış makası
    "DOVIZ": 0.0008,  # döviz alış-satış makası
}
_FALLBACK_ONE_WAY_COST = 0.0015


def _round_trip_cost(varlik_tipi: pd.Series, cost_by_asset_type: dict[str, float]) -> pd.Series:
    one_way = varlik_tipi.map(cost_by_asset_type).fillna(_FALLBACK_ONE_WAY_COST)
    return one_way * 2


@dataclass
class BacktestSummary:
    sinyal_sayisi: int
    isabet_orani_brut: float
    isabet_orani_net: float
    ortalama_getiri_brut: float
    ortalama_getiri_net: float
    ortalama_getiri_isabetli_net: float
    ortalama_getiri_isabetsiz_net: float
    kumulatif_getiri_brut: float
    kumulatif_getiri_net: float
    ortalama_islem_maliyeti: float

    def to_dict(self) -> dict:
        return {
            "sinyal_sayisi": self.sinyal_sayisi,
            "isabet_orani_brut_yuzde": round(self.isabet_orani_brut * 100, 1),
            "isabet_orani_net_yuzde": round(self.isabet_orani_net * 100, 1),
            "ortalama_getiri_brut_yuzde": round(self.ortalama_getiri_brut * 100, 2),
            "ortalama_getiri_net_yuzde": round(self.ortalama_getiri_net * 100, 2),
            "isabetli_sinyal_net_getiri_yuzde": round(self.ortalama_getiri_isabetli_net * 100, 2),
            "isabetsiz_sinyal_net_getiri_yuzde": round(self.ortalama_getiri_isabetsiz_net * 100, 2),
            "esit_agirlik_kumulatif_getiri_brut_yuzde": round(self.kumulatif_getiri_brut * 100, 1),
            "esit_agirlik_kumulatif_getiri_net_yuzde": round(self.kumulatif_getiri_net * 100, 1),
            "ortalama_islem_maliyeti_yuzde": round(self.ortalama_islem_maliyeti * 100, 2),
        }


def run_backtest(
    result: TrainResult,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    cost_by_asset_type: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, BacktestSummary]:
    """`test_df` üzerinde modelin AL sinyallerini, işlem maliyeti öncesi
    (brüt) ve sonrası (net) gerçekleşen sonuçlarla raporlar.

    Her sinyal, sinyal gününün kapanışında pozisyon açıldığını ve
    FORWARD_HORIZON_DAYS işlem günü sonra kapatıldığını varsayar
    (`ileri_getiri_10g`, bu ufuktaki gerçekleşen brüt getiridir — özellik
    tablosunda etiketle birlikte zaten hesaplanmıştır, gelecek sızıntısı
    yoktur çünkü backtest yalnızca zaten geçmişte kalmış tarihleri
    raporlar). Net getiri, brüt getiriden varlık tipine göre tahmini
    round-trip işlem maliyeti (`DEFAULT_COST_BY_ASSET_TYPE`) düşülerek
    hesaplanır.
    """
    cost_by_asset_type = cost_by_asset_type or DEFAULT_COST_BY_ASSET_TYPE
    test_df = result.test_df
    proba = result.model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]

    signals = test_df.assign(model_proba=proba)
    signals = signals[signals["model_proba"] >= buy_threshold].copy()
    signals["islem_maliyeti"] = _round_trip_cost(signals["varlik_tipi"], cost_by_asset_type)
    signals["net_getiri"] = signals["ileri_getiri_10g"] - signals["islem_maliyeti"]
    signals["isabet_brut"] = signals["ileri_getiri_10g"] > 0
    signals["isabet_net"] = signals["net_getiri"] > 0
    signals = signals.sort_values("tarih")

    signal_table = signals[
        [
            "tarih",
            "sembol",
            "varlik_tipi",
            "kapanis",
            "model_proba",
            "ileri_getiri_10g",
            "islem_maliyeti",
            "net_getiri",
            "isabet_brut",
            "isabet_net",
        ]
    ].rename(
        columns={
            "tarih": "sinyal_tarihi",
            "kapanis": "giris_fiyati",
            "model_proba": "model_guveni",
            "ileri_getiri_10g": f"brut_getiri_{FORWARD_HORIZON_DAYS}g",
            "islem_maliyeti": "islem_maliyeti_yuzde",
        }
    )

    if signals.empty:
        summary = BacktestSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return signal_table, summary

    brut = signals["ileri_getiri_10g"]
    net = signals["net_getiri"]
    hits_brut = signals["isabet_brut"]
    hits_net = signals["isabet_net"]

    summary = BacktestSummary(
        sinyal_sayisi=len(signals),
        isabet_orani_brut=hits_brut.mean(),
        isabet_orani_net=hits_net.mean(),
        ortalama_getiri_brut=brut.mean(),
        ortalama_getiri_net=net.mean(),
        ortalama_getiri_isabetli_net=net[hits_net].mean() if hits_net.any() else 0.0,
        ortalama_getiri_isabetsiz_net=net[~hits_net].mean() if (~hits_net).any() else 0.0,
        kumulatif_getiri_brut=float(np.prod(1 + brut / len(brut)) - 1),
        kumulatif_getiri_net=float(np.prod(1 + net / len(net)) - 1),
        ortalama_islem_maliyeti=signals["islem_maliyeti"].mean(),
    )
    return signal_table, summary


def latest_signals(
    result: TrainResult, feature_df: pd.DataFrame, buy_threshold: float = DEFAULT_BUY_THRESHOLD
) -> pd.DataFrame:
    """Her sembol için en güncel (etiketsiz) satırda canlı model sinyali üretir.

    Not: Bu, "gerçek zamanlı akan veri" değildir — `feature_df`'in en son
    satırı, elindeki fiyat CSV'sinin toplandığı ana ait "en güncel"
    veridir. Gerçek anlamda canlı izleme için veri toplama ve bu
    fonksiyonun periyodik olarak (ör. her gün piyasa kapanışından sonra)
    yeniden çalıştırılması gerekir; bkz. README "Canlı izleme" bölümü.
    """
    usable = feature_df.dropna(subset=FEATURE_COLUMNS).copy()
    latest = usable.sort_values("tarih").groupby("sembol", as_index=False).tail(1)

    proba = result.model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]
    latest = latest.assign(model_guveni=proba)
    latest["sinyal"] = np.where(latest["model_guveni"] >= buy_threshold, "AL", "TUT")
    return latest[
        ["tarih", "sembol", "varlik_tipi", "kapanis", "model_guveni", "sinyal"]
    ].sort_values("model_guveni", ascending=False).reset_index(drop=True)
