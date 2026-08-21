from yatirim.models import MacroData, PortfolioState, Position, TechnicalData
from yatirim import strategy
from yatirim.advisor import build_daily_report
from yatirim.models import DailyInput


def _macro(tufe=60.0, faiz=45.0, beklenti=55.0):
    return MacroData(tufe_yillik=tufe, tcmb_politika_faizi=faiz, beklenti_enflasyon=beklenti)


def _technical(bist=11000.0, ma50=10500.0, ma200=10000.0, rsi=60.0, macd=10.0):
    return TechnicalData(
        bist100=bist,
        gram_altin=4300.0,
        ons_altin=2600.0,
        usdtry=34.0,
        eurtry=37.0,
        ma20=10900.0,
        ma50=ma50,
        ma200=ma200,
        rsi=rsi,
        macd=macd,
    )


def _portfolio(gunluk=0.5, yillik=50.0, borsa=30.0, altin=20.0, doviz=15.0, likit=35.0):
    return PortfolioState(
        portfoy_degeri=1_000_000.0,
        gunluk_getiri_yuzde=gunluk,
        yillik_getiri_yuzde=yillik,
        borsa_yuzde=borsa,
        altin_yuzde=altin,
        doviz_yuzde=doviz,
        likit_repo_yuzde=likit,
        positions=[],
    )


def test_negative_real_rate_triggers_yuksek_enflasyon_regime():
    macro = _macro(tufe=60.0, faiz=40.0, beklenti=55.0)
    technical = _technical()
    assert strategy.classify_regime(macro, technical) == strategy.REGIME_YUKSEK_ENFLASYON


def test_bullish_technicals_with_healthy_real_rate_trigger_buyume_ralli():
    macro = _macro(tufe=60.0, faiz=65.0, beklenti=55.0)
    technical = _technical(bist=11500.0, ma50=11000.0, ma200=10000.0, rsi=60.0, macd=15.0)
    assert strategy.classify_regime(macro, technical) == strategy.REGIME_BUYUME_RALLI


def test_liquid_floor_never_below_minimum():
    for regime, allocation in strategy.TARGET_ALLOCATIONS.items():
        resolved = strategy.target_allocation_for_regime(regime)
        assert resolved["likit_repo_yuzde"] >= strategy.MIN_LIQUID_FLOOR_PCT


def test_drawdown_breach_detected_at_limit():
    assert strategy.drawdown_breached(_portfolio(gunluk=-2.5))
    assert strategy.drawdown_breached(_portfolio(gunluk=-3.1))
    assert not strategy.drawdown_breached(_portfolio(gunluk=-2.4))


def test_small_allocation_delta_is_hold():
    assert not strategy.exceeds_trade_threshold(current_pct=30.0, target_pct=31.0)
    assert strategy.exceeds_trade_threshold(current_pct=30.0, target_pct=35.0)


def test_stop_loss_scenario_forces_full_safe_haven_shift():
    data = DailyInput(
        tarih="2026-08-21",
        macro=_macro(),
        technical=_technical(),
        portfolio=_portfolio(gunluk=-3.0, borsa=40.0),
    )
    report = build_daily_report(data)

    assert report.otomatik_emir_onayi is True
    assert report.yeni_portfoy_dagilimi["borsa_yuzde"] == 0
    borsa_rec = next(o for o in report.oneriler if o.varlik_tipi == "BORSA")
    assert borsa_rec.islem_tipi == "SAT"


def test_report_holds_when_allocation_already_near_target():
    macro = _macro(tufe=60.0, faiz=65.0, beklenti=50.0)
    technical = _technical(bist=9000.0, ma50=9500.0, ma200=9800.0, rsi=45.0, macd=-5.0)
    regime = strategy.classify_regime(macro, technical)
    assert regime == strategy.REGIME_NORMAL
    target = strategy.target_allocation_for_regime(regime)

    portfolio = _portfolio(
        gunluk=0.2,
        borsa=target["borsa_yuzde"],
        altin=target["altin_yuzde"],
        doviz=target["doviz_yuzde"],
        likit=target["likit_repo_yuzde"],
    )
    data = DailyInput(tarih="2026-08-21", macro=macro, technical=technical, portfolio=portfolio)
    report = build_daily_report(data)

    assert all(o.islem_tipi == "TUT" for o in report.oneriler)
    assert report.otomatik_emir_onayi is False


def test_only_bist30_symbols_are_recognized():
    assert strategy.is_bist30_symbol("BIST:THYAO")
    assert not strategy.is_bist30_symbol("BIST:OBSCURESTOCK")


def test_position_dataclass_roundtrip():
    position = Position(
        varlik_kodu="BIST:THYAO",
        varlik_tipi="BORSA",
        agirlik_yuzdesi=10.0,
        ortalama_maliyet=260.0,
    )
    assert position.varlik_kodu == "BIST:THYAO"
