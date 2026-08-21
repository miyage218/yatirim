"""Model eğitimi: geçmiş 3 yıllık veriden yükseliş/düşüş sınıflandırıcısı."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from .features import FEATURE_COLUMNS


@dataclass
class TrainResult:
    model: GradientBoostingClassifier
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_accuracy: float
    test_accuracy: float
    test_auc: float


def _labeled_rows(feature_df: pd.DataFrame) -> pd.DataFrame:
    return feature_df.dropna(subset=FEATURE_COLUMNS + ["etiket_yukselis"]).copy()


def train_model(
    feature_df: pd.DataFrame,
    test_start: str | pd.Timestamp,
    random_state: int = 42,
) -> TrainResult:
    """Zaman bazlı (walk-forward) train/test ayrımıyla model eğitir.

    `test_start` tarihinden önceki tüm etiketli satırlar eğitim, sonrası
    (ve etiketlenebilir olanlar) test/backtest kümesidir. Sembol bazında
    değil takvim bazında bölünür; böylece test döneminde hiçbir enstrüman
    için eğitimde görülmemiş bir tarih aralığı sızmaz.
    """
    labeled = _labeled_rows(feature_df)
    test_start = pd.Timestamp(test_start)

    train_df = labeled[labeled["tarih"] < test_start]
    test_df = labeled[labeled["tarih"] >= test_start]

    if train_df.empty or test_df.empty:
        raise ValueError("train veya test kümesi boş; test_start tarihini kontrol edin.")

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=random_state,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["etiket_yukselis"].astype(int))

    train_pred = model.predict(train_df[FEATURE_COLUMNS])
    test_pred = model.predict(test_df[FEATURE_COLUMNS])
    test_proba = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]

    return TrainResult(
        model=model,
        train_df=train_df,
        test_df=test_df,
        train_accuracy=accuracy_score(train_df["etiket_yukselis"].astype(int), train_pred),
        test_accuracy=accuracy_score(test_df["etiket_yukselis"].astype(int), test_pred),
        test_auc=roc_auc_score(test_df["etiket_yukselis"].astype(int), test_proba),
    )
