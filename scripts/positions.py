"""AL/SAT sinyallerinden gerçek pozisyon takibi.

Sabit bir gün sayısı (1 gün, 10 gün vb.) sonrasına bakmak yerine, bu
modül modelin verdiği sinyallere göre gerçekçi bir pozisyon durum
makinesi işletir:

- Bir sembol için açık pozisyon YOKSA ve sinyal AL ise -> pozisyon açılır
  (giriş tarihi/fiyatı o günün kapanışı olarak kaydedilir).
- Açık pozisyon VARSA ve sinyal SAT ise -> pozisyon kapatılır, gerçekleşen
  getiri = (çıkış fiyatı / giriş fiyatı - 1) hesaplanıp geçmişe eklenir.
- TUT veya zaten açıkken tekrar AL gelmesi -> durum değişmez (pozisyon
  olduğu gibi açık kalır, giriş fiyatı SIFIRLANMAZ).
- Hiç pozisyon yokken SAT gelmesi -> yok sayılır (elde olmayan satılamaz).

"O hissenin yüzdelik kazanç ortalaması" = o sembol için kapanmış tüm
episodların getirisinin ortalaması; pozisyon hâlâ açıksa (henüz SAT
gelmediyse) o güne kadarki gerçekleşmemiş (unrealized) getiri de bu
ortalamaya bir gözlem olarak dahil edilir.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
OPEN_POSITIONS_PATH = ARTIFACTS_DIR / "acik_pozisyonlar.csv"
CLOSED_POSITIONS_PATH = ARTIFACTS_DIR / "kapanan_pozisyonlar.csv"

OPEN_COLUMNS = ["sembol", "varlik_tipi", "giris_tarihi", "giris_fiyati"]
CLOSED_COLUMNS = [
    "sembol",
    "varlik_tipi",
    "giris_tarihi",
    "giris_fiyati",
    "cikis_tarihi",
    "cikis_fiyati",
    "getiri_yuzde",
]


def _load_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path, parse_dates=[c for c in columns if "tarih" in c])


def load_open_positions() -> pd.DataFrame:
    return _load_csv(OPEN_POSITIONS_PATH, OPEN_COLUMNS)


def load_closed_positions() -> pd.DataFrame:
    return _load_csv(CLOSED_POSITIONS_PATH, CLOSED_COLUMNS)


def update_positions(signals: pd.DataFrame) -> None:
    """Günün sinyallerine göre açık/kapalı pozisyon defterlerini günceller.

    `signals` en az şu sütunları içermeli: sembol, varlik_tipi, kapanis,
    sinyal, veri_tarihi (her satırın kendi tarihi kullanılır, semboller
    arası veri güncellik farkına saygı gösterilir).
    """
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    open_df = load_open_positions().set_index("sembol") if not load_open_positions().empty else pd.DataFrame(
        columns=OPEN_COLUMNS
    ).set_index("sembol")
    closed_rows = []

    for row in signals.itertuples(index=False):
        sembol = row.sembol
        has_open = sembol in open_df.index

        if row.sinyal == "AL" and not has_open:
            open_df.loc[sembol] = {
                "varlik_tipi": row.varlik_tipi,
                "giris_tarihi": row.veri_tarihi,
                "giris_fiyati": row.kapanis,
            }
        elif row.sinyal == "SAT" and has_open:
            giris = open_df.loc[sembol]
            getiri = row.kapanis / giris["giris_fiyati"] - 1
            closed_rows.append(
                {
                    "sembol": sembol,
                    "varlik_tipi": row.varlik_tipi,
                    "giris_tarihi": giris["giris_tarihi"],
                    "giris_fiyati": giris["giris_fiyati"],
                    "cikis_tarihi": row.veri_tarihi,
                    "cikis_fiyati": row.kapanis,
                    "getiri_yuzde": getiri,
                }
            )
            open_df = open_df.drop(index=sembol)
        # TUT, ya da açıkken tekrar AL: durum değişmez.

    open_df.reset_index().to_csv(OPEN_POSITIONS_PATH, index=False)

    if closed_rows:
        closed_df = load_closed_positions()
        closed_df = pd.concat([closed_df, pd.DataFrame(closed_rows)], ignore_index=True)
        closed_df.to_csv(CLOSED_POSITIONS_PATH, index=False)


def average_return_per_symbol(latest_prices: dict[str, float]) -> pd.DataFrame:
    """Her sembol için, kapanmış episodların + (varsa) açık pozisyonun
    o günkü gerçekleşmemiş getirisinin ortalamasını döner.

    `latest_prices`: {sembol: en güncel kapanış fiyatı} — açık pozisyonların
    unrealized getirisini hesaplamak için gerekli.
    """
    closed_df = load_closed_positions()
    open_df = load_open_positions()

    returns_by_symbol: dict[str, list[float]] = {}
    for row in closed_df.itertuples(index=False):
        returns_by_symbol.setdefault(row.sembol, []).append(row.getiri_yuzde)

    for row in open_df.itertuples(index=False):
        current_price = latest_prices.get(row.sembol)
        if current_price is None:
            continue
        unrealized = current_price / row.giris_fiyati - 1
        returns_by_symbol.setdefault(row.sembol, []).append(unrealized)

    result = pd.DataFrame(
        [
            {
                "sembol": sembol,
                "episod_sayisi": len(returns),
                "ortalama_getiri_yuzde": sum(returns) / len(returns),
            }
            for sembol, returns in returns_by_symbol.items()
        ]
    )
    if result.empty:
        return result
    return result.sort_values("ortalama_getiri_yuzde", ascending=False).reset_index(drop=True)


def overall_model_return(latest_prices: dict[str, float]) -> float | None:
    """Tüm sembollerin ortalama getirilerinin ortalaması: tek bir 'model
    performansı' sayısı (aylık karşılaştırma grafiği için).
    """
    per_symbol = average_return_per_symbol(latest_prices)
    if per_symbol.empty:
        return None
    return float(per_symbol["ortalama_getiri_yuzde"].mean())
