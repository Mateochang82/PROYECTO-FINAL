"""
Tests del particionamiento temporal.

Garantizan:
- Ningún solapamiento de fechas entre train/val/test.
- Las fracciones se respetan dentro de tolerancia.
- TimeSeriesSplit nunca pone validación antes del entrenamiento.
"""
from src.splits import temporal_split, get_tscv


def test_temporal_split_no_overlap(synthetic_ohlcv):
    train, val, test = temporal_split(synthetic_ohlcv)
    assert train["date"].max() < val["date"].min(), \
        "Train debe terminar estrictamente antes de val"
    assert val["date"].max() < test["date"].min(), \
        "Val debe terminar estrictamente antes de test"


def test_temporal_split_fractions(synthetic_ohlcv):
    train, val, test = temporal_split(synthetic_ohlcv)
    n = len(synthetic_ohlcv)
    assert abs(len(train) / n - 0.70) < 0.01
    assert abs(len(val) / n - 0.15) < 0.01
    assert abs(len(test) / n - 0.15) < 0.02      # un poco más por redondeo


def test_temporal_split_disjoint(synthetic_ohlcv):
    train, val, test = temporal_split(synthetic_ohlcv)
    total = len(train) + len(val) + len(test)
    assert total == len(synthetic_ohlcv), "Las particiones deben cubrir todo el dataset"


def test_tscv_no_shuffle(synthetic_ohlcv):
    """En cada split, todos los índices de train deben ser anteriores a los de val."""
    tscv = get_tscv(n_splits=3)
    X = synthetic_ohlcv[["close"]].values
    for train_idx, val_idx in tscv.split(X):
        assert train_idx.max() < val_idx.min(), \
            "TimeSeriesSplit violó orden temporal"
