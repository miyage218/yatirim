from __future__ import annotations

from .models import DailyInput, DailyReport, Recommendation
from . import strategy

_ASSET_CLASS_CODES = {
    "BORSA": "BIST30_SEPETI",
    "ALTIN": "GRAM_ALTIN",
    "DOVIZ": "USDTRY",
    "LIKIT": "BPP",
}

_ALLOCATION_KEY_TO_ASSET_TYPE = {
    "borsa_yuzde": "BORSA",
    "altin_yuzde": "ALTIN",
    "doviz_yuzde": "DOVIZ",
    "likit_repo_yuzde": "LIKIT",
}

_REGIME_LABELS = {
    strategy.REGIME_YUKSEK_ENFLASYON: "Yüksek enflasyon/belirsizlik",
    strategy.REGIME_BUYUME_RALLI: "Büyüme/ralli",
    strategy.REGIME_NORMAL: "Normal seyir",
}


def _reference_price(asset_type: str, technical) -> float:
    if asset_type == "BORSA":
        return technical.bist100
    if asset_type == "ALTIN":
        return technical.gram_altin
    if asset_type == "DOVIZ":
        return technical.usdtry
    return 1.0


def _price_and_stop(current_price: float, islem_tipi: str) -> tuple[float, float]:
    if islem_tipi == "AL":
        return round(current_price * 1.03, 2), round(current_price * 0.97, 2)
    if islem_tipi == "SAT":
        return round(current_price * 0.97, 2), round(current_price * 1.03, 2)
    return round(current_price, 2), round(current_price * 0.97, 2)


def _build_recommendation(
    asset_type: str, current_pct: float, target_pct: float, technical, gerekce: str
) -> Recommendation:
    if asset_type == "LIKIT":
        islem_tipi = "TUT"
    elif not strategy.exceeds_trade_threshold(current_pct, target_pct):
        islem_tipi = "TUT"
    elif target_pct > current_pct:
        islem_tipi = "AL"
    else:
        islem_tipi = "SAT"

    price = _reference_price(asset_type, technical)
    hedef_fiyat, stop_loss = _price_and_stop(price, islem_tipi)

    return Recommendation(
        islem_tipi=islem_tipi,
        varlik_kodu=_ASSET_CLASS_CODES[asset_type],
        varlik_tipi=asset_type,
        agirlik_yuzdesi=target_pct,
        hedef_fiyat=hedef_fiyat,
        stop_loss=stop_loss,
        gerekce=gerekce,
    )


def build_daily_report(data: DailyInput) -> DailyReport:
    portfolio = data.portfolio
    macro = data.macro
    technical = data.technical

    if strategy.drawdown_breached(portfolio):
        target_allocation = dict(strategy.STOP_LOSS_ALLOCATION)
        piyasa_ozeti = (
            f"Günlük portföy kaybı %{portfolio.gunluk_getiri_yuzde:.1f} ile "
            f"%{strategy.MAX_DAILY_DRAWDOWN_PCT} limitini aştı, güvenli limana "
            "geçiş tetiklendi. Hisse pozisyonları kapatılıp likidite ve altın "
            "ağırlığı artırılıyor."
        )
        gerekce = (
            f"Günlük drawdown limiti (%{strategy.MAX_DAILY_DRAWDOWN_PCT}) aşıldı, "
            "otomatik stop-loss / güvenli limana geçiş senaryosu uygulanıyor."
        )
        oneriler = [
            _build_recommendation(
                asset_type, getattr(portfolio, key), target_allocation[key],
                technical, gerekce,
            )
            for key, asset_type in _ALLOCATION_KEY_TO_ASSET_TYPE.items()
        ]
        otomatik_emir_onayi = True
    else:
        regime = strategy.classify_regime(macro, technical)
        target_allocation = strategy.target_allocation_for_regime(regime)
        piyasa_ozeti = (
            f"{_REGIME_LABELS[regime]} rejimi: TÜFE %{macro.tufe_yillik:.1f}, "
            f"enflasyon beklentisi %{macro.beklenti_enflasyon:.1f}, TCMB "
            f"politika faizi %{macro.tcmb_politika_faizi:.1f}. BIST100 "
            f"{'MA50/MA200 üzerinde' if technical.bist100 > technical.ma50 else 'MA50 altında'}, "
            f"RSI {technical.rsi:.0f}."
        )
        gerekce = f"{_REGIME_LABELS[regime]} rejiminde hedef varlık dağılımına yaklaşma."
        oneriler = [
            _build_recommendation(
                asset_type, getattr(portfolio, key), target_allocation[key],
                technical, gerekce,
            )
            for key, asset_type in _ALLOCATION_KEY_TO_ASSET_TYPE.items()
        ]
        otomatik_emir_onayi = any(o.islem_tipi != "TUT" for o in oneriler)

    return DailyReport(
        tarih=data.tarih,
        piyasa_ozeti=piyasa_ozeti,
        hedef_reel_getiri_durumu=strategy.real_return_status(macro, portfolio),
        oneriler=oneriler,
        yeni_portfoy_dagilimi=target_allocation,
        otomatik_emir_onayi=otomatik_emir_onayi,
    )
