"""Teknik gösterge özellik mühendisliği ve etiketleme."""

from __future__ import annotations

import numpy as np
import pandas as pd

FORWARD_HORIZON_DAYS = 10
UP_LABEL_THRESHOLD = 0.0  # forward getiri > 0 ise "yükseliş" etiketi

FEATURE_COLUMNS = [
    "getiri_1g",
    "getiri_5g",
    "getiri_10g",
    "getiri_20g",
    "ma20_uzaklik",
    "ma50_uzaklik",
    "ma200_uzaklik",
    "ma20_ma50_spread",
    "rsi14",
    "macd",
    "macd_sinyal",
    "macd_histogram",
    "volatilite_20g",
    "hacim_z_skoru",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _build_symbol_features(group: pd.DataFrame, sembol: str) -> pd.DataFrame:
    group = group.sort_values("tarih").reset_index(drop=True)
    close = group["kapanis"]

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    macd_line, macd_signal, macd_hist = _macd(close)
    volume_z = (
        group["hacim"] - group["hacim"].rolling(20).mean()
    ) / group["hacim"].rolling(20).std()

    out = pd.DataFrame(
        {
            "tarih": group["tarih"],
            "sembol": sembol,
            "varlik_tipi": group["varlik_tipi"],
            "kapanis": close,
            "getiri_1g": close.pct_change(1),
            "getiri_5g": close.pct_change(5),
            "getiri_10g": close.pct_change(10),
            "getiri_20g": close.pct_change(20),
            "ma20_uzaklik": close / ma20 - 1,
            "ma50_uzaklik": close / ma50 - 1,
            "ma200_uzaklik": close / ma200 - 1,
            "ma20_ma50_spread": ma20 / ma50 - 1,
            "rsi14": _rsi(close),
            "macd": macd_line,
            "macd_sinyal": macd_signal,
            "macd_histogram": macd_hist,
            "volatilite_20g": close.pct_change().rolling(20).std() * np.sqrt(252),
            "hacim_z_skoru": volume_z,
        }
    )

    forward_return = close.shift(-FORWARD_HORIZON_DAYS) / close - 1
    out["ileri_getiri_10g"] = forward_return
    label = pd.Series(pd.NA, index=out.index, dtype="Int64")
    label[forward_return.notna()] = (forward_return[forward_return.notna()] > UP_LABEL_THRESHOLD).astype(int)
    out["etiket_yukselis"] = label
    return out


def build_feature_table(raw_prices: pd.DataFrame) -> pd.DataFrame:
    """Ham fiyat verisinden özellik + etiket tablosu üretir.

    İlk ~200 satır (MA200 ısınma dönemi) ve son FORWARD_HORIZON_DAYS satır
    (ileri getiri hesaplanamadığı için) NaN üretir; bu satırlar burada
    atılmaz, `dropna(subset=FEATURE_COLUMNS)` çağıran taraf sorumludur ki
    en güncel (etiketsiz) satır canlı sinyal üretimi için saklanabilsin.
    """
    frames = [
        _build_symbol_features(group, sembol)
        for sembol, group in raw_prices.groupby("sembol", sort=False)
    ]
    return pd.concat(frames, ignore_index=True)
