from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MacroData:
    tufe_yillik: float
    tcmb_politika_faizi: float
    beklenti_enflasyon: float


@dataclass
class TechnicalData:
    bist100: float
    gram_altin: float
    ons_altin: float
    usdtry: float
    eurtry: float
    ma20: float
    ma50: float
    ma200: float
    rsi: float
    macd: float


@dataclass
class Position:
    varlik_kodu: str
    varlik_tipi: str
    agirlik_yuzdesi: float
    ortalama_maliyet: float


@dataclass
class PortfolioState:
    portfoy_degeri: float
    gunluk_getiri_yuzde: float
    yillik_getiri_yuzde: float
    borsa_yuzde: float
    altin_yuzde: float
    doviz_yuzde: float
    likit_repo_yuzde: float
    positions: list[Position] = field(default_factory=list)


@dataclass
class DailyInput:
    tarih: str
    macro: MacroData
    technical: TechnicalData
    portfolio: PortfolioState
    sentiment_notlari: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    islem_tipi: str
    varlik_kodu: str
    varlik_tipi: str
    agirlik_yuzdesi: float
    hedef_fiyat: float
    stop_loss: float
    gerekce: str

    def to_dict(self) -> dict:
        return {
            "islem_tipi": self.islem_tipi,
            "varlik_kodu": self.varlik_kodu,
            "varlik_tipi": self.varlik_tipi,
            "ağırlık_yuzdesi": self.agirlik_yuzdesi,
            "hedef_fiyat": self.hedef_fiyat,
            "stop_loss": self.stop_loss,
            "gerekce": self.gerekce,
        }


@dataclass
class DailyReport:
    tarih: str
    piyasa_ozeti: str
    hedef_reel_getiri_durumu: str
    oneriler: list[Recommendation]
    yeni_portfoy_dagilimi: dict
    otomatik_emir_onayi: bool

    def to_dict(self) -> dict:
        return {
            "tarih": self.tarih,
            "piyasa_ozeti": self.piyasa_ozeti,
            "hedef_reel_getiri_durumu": self.hedef_reel_getiri_durumu,
            "oneriler": [o.to_dict() for o in self.oneriler],
            "yeni_portfoy_dagilimi": self.yeni_portfoy_dagilimi,
            "otomatik_emir_onayi": self.otomatik_emir_onayi,
        }
