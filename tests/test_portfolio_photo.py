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
