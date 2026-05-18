"""
Tests de causalidad de features.

Estrategia: si una feature en el tiempo t depende SOLO de información hasta t,
entonces modificar los valores del dataset DESPUÉS de t no debe alterar la
feature en t. Verificamos esto sobre todas las features generadas.
"""
import numpy as np
import pandas as pd
from src.features import build_features, feature_columns


def test_features_no_future_info(synthetic_ohlcv):
    """
    Mutamos los valores del último 30% del dataset y verificamos que las
    features del primer 70% queden idénticas.
    """
    df = synthetic_ohlcv.copy()
    feat_orig = build_features(df)

    cutoff = int(len(df) * 0.70)
    df_mut = df.copy()
    df_mut.loc[cutoff:, "close"] = df_mut.loc[cutoff:, "close"] * 1.5
    df_mut.loc[cutoff:, "high"] = df_mut.loc[cutoff:, "high"] * 1.5
    df_mut.loc[cutoff:, "low"] = df_mut.loc[cutoff:, "low"] * 1.5
    df_mut.loc[cutoff:, "volume"] = df_mut.loc[cutoff:, "volume"] * 2.0
    feat_mut = build_features(df_mut)

    feats = feature_columns(feat_orig)
    for col in feats:
        a = feat_orig.loc[:cutoff - 1, col].values
        b = feat_mut.loc[:cutoff - 1, col].values
        mask = ~(pd.isna(a) | pd.isna(b))
        assert np.allclose(a[mask], b[mask], rtol=1e-9, atol=1e-12), \
            f"Feature '{col}' usa información futura: cambia al mutar el futuro"


def test_no_target_columns_in_features(synthetic_ohlcv):
    """Los targets no deben estar en la lista de features."""
    df = build_features(synthetic_ohlcv)
    feats = feature_columns(df)
    for forbidden in ["target_vol", "target_regime", "vol_realized"]:
        assert forbidden not in feats, f"Columna prohibida en features: {forbidden}"


def test_features_present(synthetic_ohlcv):
    """Verifica que el constructor genera al menos las familias clave."""
    df = build_features(synthetic_ohlcv)
    expected = {
        "ret_lag_1", "vol_5", "vol_d", "vol_w", "vol_m",
        "momentum_5", "range_hl", "true_range", "atr_14",
        "log_volume", "sma_22", "px_dist_sma_22",
        "skew_22", "kurt_22", "dow", "month",
    }
    missing = expected - set(df.columns)
    assert not missing, f"Faltan features esperadas: {missing}"
