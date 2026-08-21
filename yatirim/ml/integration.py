"""Model sinyallerini portföy önerisi (Recommendation) formatına çevirir."""

from __future__ import annotations

import pandas as pd

from ..models import Recommendation
from .backtest import BacktestSummary


def signals_to_recommendations(
    latest: pd.DataFrame, backtest_summary: BacktestSummary
) -> list[Recommendation]:
    """En güncel model sinyallerini, backtest performansına atıfla gerekçelendirilmiş
    Recommendation nesnelerine dönüştürür. Sadece 'AL' sinyali üreten satırlar
    işlem önerisine dönüşür; diğerleri TUT olarak raporlanır.
    """
    recs = []
    for row in latest.itertuples(index=False):
        if row.sinyal == "AL":
            hedef_fiyat = round(row.kapanis * 1.02, 2)
            stop_loss = round(row.kapanis * 0.97, 2)
            gerekce = (
                f"Model {row.model_guveni * 100:.0f}% olasılıkla 10 işlem günü içinde "
                f"yükseliş öngörüyor (backtest isabet oranı "
                f"%{backtest_summary.isabet_orani * 100:.0f}, "
                f"ortalama sinyal getirisi %{backtest_summary.ortalama_getiri * 100:.1f})."
            )
        else:
            hedef_fiyat = round(row.kapanis, 2)
            stop_loss = round(row.kapanis * 0.97, 2)
            gerekce = (
                f"Model güveni (%{row.model_guveni * 100:.0f}) alım eşiğinin altında, "
                "pozisyon değişikliği önerilmiyor."
            )

        recs.append(
            Recommendation(
                islem_tipi=row.sinyal,
                varlik_kodu=row.sembol,
                varlik_tipi=row.varlik_tipi,
                agirlik_yuzdesi=0.0,
                hedef_fiyat=hedef_fiyat,
                stop_loss=stop_loss,
                gerekce=gerekce,
            )
        )
    return recs
