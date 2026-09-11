import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import portfolio_photo  # noqa: E402


SAMPLE_OCR_TEXT = """
SAHOL.E  ▼ %1,77
SABANCI HOLDING

Son Fiyat                          94,35 TL
Bugünkü Değer                     943,50 TL
Satılabilir Adet                    10,00
Potansiyel Kar/Zarar               +34,60 TL
Potansiyel Getiri                   +%3,81

ISCTR.E  ▼ %2,37
IS BANKASI (C)

Son Fiyat                          13,59 TL
Bugünkü Değer                     883,35 TL
Satılabilir Adet                    65,00
Potansiyel Kar/Zarar               -58,50 TL
Potansiyel Getiri                   -%6,21

TUPRS.E  ▲ %0,66
TUPRAS

Son Fiyat                         417,00 TL
Bugünkü Değer                     834,00 TL
Satılabilir Adet                     2,00
"""


def test_parse_portfolio_from_text_extracts_known_tickers_and_quantities():
    holdings = portfolio_photo.parse_portfolio_from_text(SAMPLE_OCR_TEXT)
    by_symbol = {h["sembol"]: h["adet"] for h in holdings}
    assert by_symbol == {
        "BIST:SAHOL": 10.0,
        "BIST:ISCTR": 65.0,
        "BIST:TUPRS": 2.0,
    }


def test_parse_portfolio_from_text_ignores_unknown_words():
    # "IS" (2 harf) ve "BANKASI" (7 harf) bilinen ticker uzunluğunun
    # (3-6) dışında kaldığı için yanlış pozitif üretmemeli.
    holdings = portfolio_photo.parse_portfolio_from_text("IS BANKASI (C)\nSatılabilir Adet 5,00")
    assert holdings == []


def test_parse_portfolio_from_text_empty_on_garbage():
    assert portfolio_photo.parse_portfolio_from_text("asdkjaslkdj 12345 !!!") == []


def test_build_portfolio_advice_no_holdings_returns_warning():
    advice = portfolio_photo.build_portfolio_advice([])
    assert "okunamadı" in advice


def test_build_portfolio_advice_missing_signals_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(portfolio_photo, "SIGNALS_CSV_PATH", tmp_path / "yok.csv")
    advice = portfolio_photo.build_portfolio_advice([{"sembol": "BIST:THYAO", "adet": 5.0}])
    assert "günlük sinyal raporu" in advice


# Gerçek bir İş Bankası "Mobil Borsa" ekran görüntüsünden Tesseract'ın
# ürettiği ham çıktı (2026-09-11'de VM üzerinde alındı). Etiketler
# ("Satılabilir Adet" gibi) tüm hisseler için önce tek blok halinde,
# değerler ise ayrı bir blok halinde geliyor — SAMPLE_OCR_TEXT'teki
# düzenli satır-satır yapıdan çok farklı. parse_portfolio_from_text
# her iki düzeni de doğru okuyabilmeli.
REAL_TESSERACT_OCR_TEXT = (
    "16:127 all 4G ED\n\nMobil Borsa\n\nSon Güncelleme: 2 dakika önce\n\na\n\n"
    "SAHOL.E V 1/77\nSABANCI HOLDING\n\nSon Fiyat\n\nBugünkü Değer\nSatılabilir Adet\n"
    "Potansiyel Kar/Zarar\n\nPotansiyel Getiri\n\nISCTR.E V 72.557\nIS BANKASI (C)\n\n"
    "Son Fiyat\n\nBugünkü Değer\nSatılabilir Adet\nPotansiyel Kar/Zarar\n\n"
    "Potansiyel Getiri\n\nTUPRS.E 4 70,66\nTUPRAS\n\nSon Fiyat\nBugünkü Değer\n"
    "Satılabilir Adet\n\nPotansiyel Kar/Zarar\n\nTo aolia) Emirlerim O Portföyüm VIOP\n\n \n\n"
    "o0\nKN\n943,50 TL\n10,00\n+34,60 TL\n*73,81\n13,59 TL\n883,35 TL\n65,00\n-58,50 TL\n"
    "-%6,21\n417,00 TL\n834,00 TL\n2,00\n+28,50TL\n\n0)\n\nAnaliz\n\x0c"
)


def test_parse_portfolio_from_text_handles_real_tesseract_layout():
    holdings = portfolio_photo.parse_portfolio_from_text(REAL_TESSERACT_OCR_TEXT)
    by_symbol = {h["sembol"]: h["adet"] for h in holdings}
    assert by_symbol == {
        "BIST:SAHOL": 10.0,
        "BIST:ISCTR": 65.0,
        "BIST:TUPRS": 2.0,
    }


def test_build_portfolio_advice_classifies_al_sat_tut(tmp_path, monkeypatch):
    signals_path = tmp_path / "gunluk_sinyal_raporu.csv"
    pd.DataFrame(
        [
            {"sembol": "BIST:SAHOL", "varlik_tipi": "BORSA", "kapanis": 94.35, "model_guveni": 0.42, "sinyal": "SAT"},
            {"sembol": "BIST:ISCTR", "varlik_tipi": "BORSA", "kapanis": 13.59, "model_guveni": 0.61, "sinyal": "AL"},
            {"sembol": "BIST:PETKM", "varlik_tipi": "BORSA", "kapanis": 20.0, "model_guveni": 0.59, "sinyal": "AL"},
        ]
    ).to_csv(signals_path, index=False)
    monkeypatch.setattr(portfolio_photo, "SIGNALS_CSV_PATH", signals_path)

    holdings = [{"sembol": "BIST:SAHOL", "adet": 10.0}, {"sembol": "BIST:ISCTR", "adet": 65.0}]
    advice = portfolio_photo.build_portfolio_advice(holdings)

    assert "SAHOL: SAT" in advice
    assert "ISCTR: TUT" in advice
    assert "PETKM" in advice  # elde olmayıp AL denen hisse ayrıca listelenmeli
