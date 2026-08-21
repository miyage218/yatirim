"""3 yıllık sentetik BIST/altın/döviz fiyat verisi üretimi.

Gerçek veri kaynaklarına (Yahoo Finance, BIST, TCMB) bu ortamdan ağ erişimi
yok. Bu modül, ML pipeline'ını uçtan uca gösterebilmek için rejim geçişli
(bull/bear/sideways) ve volatilite kümelenmesi olan gerçekçi istatistiksel
özelliklerde sentetik fiyat serileri üretir. Gerçek veri elde edildiğinde
bu modülün ürettiği DataFrame ile aynı şemaya sahip bir DataFrame
sağlanarak `yatirim.ml.features` ve sonrası doğrudan kullanılabilir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# sembol -> (varlik_tipi, baslangic_fiyati, yillik_baz_volatilite, yillik_baz_drift)
INSTRUMENTS: dict[str, tuple[str, float, float, float]] = {
    "BIST:THYAO": ("BORSA", 260.0, 0.42, 0.20),
    "BIST:GARAN": ("BORSA", 95.0, 0.35, 0.15),
    "BIST:ASELS": ("BORSA", 65.0, 0.38, 0.18),
    "BIST:BIMAS": ("BORSA", 550.0, 0.30, 0.12),
    "BIST:EREGL": ("BORSA", 45.0, 0.36, 0.10),
    "BIST:KCHOL": ("BORSA", 165.0, 0.32, 0.14),
    "BIST:SASA": ("BORSA", 5.2, 0.50, 0.05),
    "BIST:TUPRS": ("BORSA", 165.0, 0.34, 0.13),
    "GRAM_ALTIN": ("ALTIN", 4300.0, 0.22, 0.30),
    "USDTRY": ("DOVIZ", 34.0, 0.18, 0.25),
    "EURTRY": ("DOVIZ", 37.0, 0.19, 0.22),
}

_REGIMES = {
    "bull": (1.6, 0.85),
    "bear": (-1.8, 1.25),
    "sideways": (0.0, 0.70),
}
_REGIME_NAMES = list(_REGIMES.keys())
_REGIME_STAY_PROB = 0.985


def _simulate_regime_path(n_days: int, rng: np.random.Generator) -> list[str]:
    path = []
    state = rng.choice(_REGIME_NAMES)
    for _ in range(n_days):
        if rng.random() > _REGIME_STAY_PROB:
            state = rng.choice([r for r in _REGIME_NAMES if r != state])
        path.append(state)
    return path


def _simulate_price_path(
    n_days: int, start_price: float, annual_vol: float, annual_drift: float, seed: int
) -> tuple[np.ndarray, list[str]]:
    """GBM tabanlı, rejime göre sürüklenme/volatilite ölçeklenen ve hafif
    volatilite kümelenmesi (mean-reverting vol-of-vol çarpanı) içeren
    fiyat yolu üretir. Volatilite çarpanı kendi karesiyle beslenmez, bu
    yüzden patlayan/çöken (explosive) bir geri besleme oluşmaz.
    """
    rng = np.random.default_rng(seed)
    regimes = _simulate_regime_path(n_days, rng)

    daily_vol_base = annual_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    vol_cluster_mult = 1.0
    lam = 0.90

    log_returns = np.empty(n_days)
    for i, regime in enumerate(regimes):
        drift_mult, regime_vol_mult = _REGIMES[regime]
        vol_cluster_mult = lam * vol_cluster_mult + (1 - lam) * 1.0
        vol_cluster_mult = float(np.clip(vol_cluster_mult + rng.normal(0, 0.05), 0.6, 1.8))

        daily_vol = daily_vol_base * regime_vol_mult * vol_cluster_mult
        daily_drift = (annual_drift * drift_mult) / TRADING_DAYS_PER_YEAR - 0.5 * daily_vol**2
        shock = rng.standard_normal()
        log_returns[i] = daily_drift + daily_vol * shock

    prices = start_price * np.exp(np.cumsum(log_returns))
    return prices, regimes


def generate_synthetic_market_data(years: int = 3, seed: int = 42) -> pd.DataFrame:
    """Uzun formatlı sentetik piyasa verisi üretir.

    Dönen sütunlar: tarih, sembol, varlik_tipi, acilis, yuksek, dusuk,
    kapanis, hacim, rejim (üretimde kullanılan gizli rejim etiketi -
    yalnızca demo/analiz amaçlı, model eğitiminde kullanılmaz).
    """
    n_days = years * TRADING_DAYS_PER_YEAR
    all_dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    frames = []
    for offset, (symbol, (tip, start_price, vol, drift)) in enumerate(INSTRUMENTS.items()):
        prices, regimes = _simulate_price_path(n_days, start_price, vol, drift, seed + offset)
        rng = np.random.default_rng(seed + offset + 1000)
        intraday_noise = rng.uniform(0.003, 0.012, size=n_days)
        high = prices * (1 + intraday_noise)
        low = prices * (1 - intraday_noise)
        open_ = np.roll(prices, 1)
        open_[0] = prices[0]
        volume = rng.integers(1_000_000, 50_000_000, size=n_days)

        frames.append(
            pd.DataFrame(
                {
                    "tarih": all_dates,
                    "sembol": symbol,
                    "varlik_tipi": tip,
                    "acilis": open_,
                    "yuksek": high,
                    "dusuk": low,
                    "kapanis": prices,
                    "hacim": volume,
                    "rejim": regimes,
                }
            )
        )

    return pd.concat(frames, ignore_index=True).sort_values(["sembol", "tarih"]).reset_index(drop=True)
