from __future__ import annotations

from .models import MacroData, PortfolioState, TechnicalData

MAX_DAILY_DRAWDOWN_PCT = -2.5
MIN_TRADE_DELTA_PCT = 1.5
MIN_LIQUID_FLOOR_PCT = 10.0
MAX_LIQUID_CEILING_PCT = 20.0

REGIME_YUKSEK_ENFLASYON = "YUKSEK_ENFLASYON"
REGIME_BUYUME_RALLI = "BUYUME_RALLI"
REGIME_NORMAL = "NORMAL"

# BIST 30 evreni: sadece bu semboller hisse önerisine konu olabilir.
BIST30 = {
    "AKBNK", "ARCLK", "ASELS", "BIMAS", "EKGYO", "ENKAI", "EREGL", "FROTO",
    "GARAN", "GUBRF", "HALKB", "ISCTR", "KCHOL", "KOZAA", "KOZAL", "KRDMD",
    "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "TAVHL", "TCELL", "THYAO",
    "TOASO", "TUPRS", "VAKBN", "VESTL", "YKBNK", "ASTOR",
}

TARGET_ALLOCATIONS = {
    REGIME_YUKSEK_ENFLASYON: {
        "borsa_yuzde": 20,
        "altin_yuzde": 30,
        "doviz_yuzde": 20,
        "likit_repo_yuzde": 30,
    },
    REGIME_BUYUME_RALLI: {
        "borsa_yuzde": 55,
        "altin_yuzde": 15,
        "doviz_yuzde": 10,
        "likit_repo_yuzde": 20,
    },
    REGIME_NORMAL: {
        "borsa_yuzde": 35,
        "altin_yuzde": 20,
        "doviz_yuzde": 15,
        "likit_repo_yuzde": 30,
    },
}

STOP_LOSS_ALLOCATION = {
    "borsa_yuzde": 0,
    "altin_yuzde": 30,
    "doviz_yuzde": 20,
    "likit_repo_yuzde": 50,
}


def classify_regime(macro: MacroData, technical: TechnicalData) -> str:
    real_policy_rate = macro.tcmb_politika_faizi - macro.beklenti_enflasyon
    enflasyon_yukseliyor = macro.beklenti_enflasyon > macro.tufe_yillik

    if real_policy_rate < 0 or enflasyon_yukseliyor:
        return REGIME_YUKSEK_ENFLASYON

    bullish_trend = technical.bist100 > technical.ma50 > technical.ma200
    healthy_momentum = 50 <= technical.rsi <= 70 and technical.macd > 0

    if bullish_trend and healthy_momentum:
        return REGIME_BUYUME_RALLI

    return REGIME_NORMAL


def target_allocation_for_regime(regime: str) -> dict:
    allocation = dict(TARGET_ALLOCATIONS[regime])
    if allocation["likit_repo_yuzde"] < MIN_LIQUID_FLOOR_PCT:
        allocation["likit_repo_yuzde"] = MIN_LIQUID_FLOOR_PCT
    return allocation


def drawdown_breached(portfolio: PortfolioState) -> bool:
    return portfolio.gunluk_getiri_yuzde <= MAX_DAILY_DRAWDOWN_PCT


def exceeds_trade_threshold(current_pct: float, target_pct: float) -> bool:
    return abs(target_pct - current_pct) >= MIN_TRADE_DELTA_PCT


def real_return_status(macro: MacroData, portfolio: PortfolioState) -> str:
    hedef = macro.tufe_yillik + 5.0
    fark = portfolio.yillik_getiri_yuzde - hedef
    if fark >= 0:
        durum = "hedefin üzerinde"
    else:
        durum = "hedefin altında"
    return (
        f"Yıllıklandırılmış portföy getirisi %{portfolio.yillik_getiri_yuzde:.1f}, "
        f"hedef (TÜFE+%5) %{hedef:.1f} — {durum} (fark: %{fark:+.1f})."
    )


def is_bist30_symbol(kod: str) -> bool:
    ticker = kod.split(":")[-1].upper()
    return ticker in BIST30
