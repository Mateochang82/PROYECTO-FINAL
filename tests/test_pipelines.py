"""
Test de pipelines: garantizar que el pipeline mínimo se entrena y predice
sobre datos sintéticos sin error.

Más pipelines específicos se verifican en los notebooks de modelado.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression

from src.features import build_features, feature_columns
from src.targets import add_targets
from src.splits import temporal_split
from src.pipelines import (
    make_baseline_regression_pipeline,
    make_baseline_classification_pipeline,
)


def _prepare(synthetic_ohlcv):
    df = build_features(synthetic_ohlcv)
    # construir targets PRIMERO, después dropna (el target tiene shift(-1) → último NaN)
    n = len(df)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[: int(n * 0.7)] = True
    df = add_targets(df, horizon=1, train_mask=train_mask)
    df = df.dropna().reset_index(drop=True)
    return df


def test_regression_pipeline_runs(synthetic_ohlcv):
    df = _prepare(synthetic_ohlcv)
    train, val, test = temporal_split(df)
    feats = feature_columns(df)

    pipe = make_baseline_regression_pipeline(Ridge(alpha=1.0))
    pipe.fit(train[feats], train["target_vol"])
    preds = pipe.predict(val[feats])

    assert len(preds) == len(val)
    assert np.isfinite(preds).all(), "Predicciones contienen NaN o inf"


def test_classification_pipeline_runs(synthetic_ohlcv):
    df = _prepare(synthetic_ohlcv)
    train, val, test = temporal_split(df)
    feats = feature_columns(df)

    pipe = make_baseline_classification_pipeline(
        LogisticRegression(max_iter=1000)
    )
    pipe.fit(train[feats], train["target_regime"].astype(int))
    preds = pipe.predict(val[feats])
    probas = pipe.predict_proba(val[feats])

    assert len(preds) == len(val)
    assert probas.shape == (len(val), 2)
    assert set(np.unique(preds)).issubset({0, 1})
