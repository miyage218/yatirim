import numpy as np
import pandas as pd

from yatirim.ml.backtest import latest_signals, run_backtest
from yatirim.ml.features import FEATURE_COLUMNS, build_feature_table
from yatirim.ml.integration import signals_to_recommendations
from yatirim.ml.model import train_model
from yatirim.ml.synthetic_data import generate_synthetic_market_data


def _small_dataset():
    return generate_synthetic_market_data(years=2, seed=7)


def test_synthetic_prices_stay_within_sane_bounds():
    raw = _small_dataset()
    assert not raw["kapanis"].isna().any()
    assert (raw["kapanis"] > 0).all()
    for _, group in raw.groupby("sembol"):
        start, end = group["kapanis"].iloc[0], group["kapanis"].iloc[-1]
        assert 0.1 <= end / start <= 10.0


def test_feature_table_has_expected_columns_and_no_leakage_on_last_row():
    raw = _small_dataset()
    features = build_feature_table(raw)
    for col in FEATURE_COLUMNS:
        assert col in features.columns

    last_rows = features.sort_values("tarih").groupby("sembol").tail(1)
    assert last_rows["etiket_yukselis"].isna().all()


def test_model_trains_and_backtest_reports_are_consistent():
    raw = _small_dataset()
    features = build_feature_table(raw)
    test_start = features["tarih"].max() - pd.Timedelta(days=180)

    result = train_model(features, test_start=test_start)
    assert 0.0 <= result.test_accuracy <= 1.0
    assert 0.0 <= result.test_auc <= 1.0

    signal_table, summary = run_backtest(result, buy_threshold=0.5)
    assert summary.sinyal_sayisi == len(signal_table)
    if summary.sinyal_sayisi:
        assert 0.0 <= summary.isabet_orani <= 1.0

    latest = latest_signals(result, features)
    assert set(latest["sinyal"]).issubset({"AL", "TUT"})
    assert latest["tarih"].nunique() == 1

    recs = signals_to_recommendations(latest, summary)
    assert len(recs) == len(latest)
    assert all(r.islem_tipi in ("AL", "TUT") for r in recs)


def test_train_raises_on_empty_split():
    raw = _small_dataset()
    features = build_feature_table(raw)
    with_labels = features.dropna(subset=FEATURE_COLUMNS + ["etiket_yukselis"])
    future_date = with_labels["tarih"].max() + pd.Timedelta(days=1)
    try:
        train_model(features, test_start=future_date)
        assert False, "boş test kümesinde ValueError beklenirdi"
    except ValueError:
        pass
