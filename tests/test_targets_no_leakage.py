"""
Tests anti-leakage sobre targets.

Garantizan:
- ``target_vol[t]`` proviene exactamente de ``vol_realized[t+horizon]``.
- Los targets no aparecen en la lista de features.
- El umbral del régimen se calcula solo con train: mutar el test/val
  no debe cambiar el umbral si se pasó ``train_mask``.
"""
import numpy as np
import pandas as pd
from src.features import build_features, feature_columns
from src.targets import add_targets


def test_target_is_future_value(synthetic_ohlcv):
    """target_vol[t] == vol_realized[t+1] cuando horizon=1."""
    df = build_features(synthetic_ohlcv)
    df = add_targets(df, horizon=1)
    n = len(df)
    for i in range(0, n - 1):
        tv = df.loc[i, "target_vol"]
        vr_next = df.loc[i + 1, "vol_realized"]
        if pd.notna(tv) and pd.notna(vr_next):
            assert tv == vr_next, \
                f"target_vol[{i}] no coincide con vol_realized[{i+1}]"


def test_no_target_in_features(synthetic_ohlcv):
    df = build_features(synthetic_ohlcv)
    df = add_targets(df, horizon=1)
    feats = feature_columns(df)
    assert "target_vol" not in feats
    assert "target_regime" not in feats
    assert "vol_realized" not in feats


def test_regime_threshold_from_train_only(synthetic_ohlcv):
    """
    Mutar el futuro del dataset NO debe afectar el umbral cuando se pasa train_mask.
    """
    n = len(synthetic_ohlcv)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[: int(n * 0.7)] = True

    # Caso A: dataset original
    df_a = build_features(synthetic_ohlcv)
    df_a = add_targets(df_a, horizon=1, train_mask=train_mask)
    thr_a = df_a.attrs["regime_threshold"]

    # Caso B: dataset con valores futuros multiplicados x5 fuera de train
    syn_b = synthetic_ohlcv.copy()
    syn_b.loc[int(n * 0.7):, "close"] *= 5.0
    syn_b.loc[int(n * 0.7):, "high"] *= 5.0
    syn_b.loc[int(n * 0.7):, "low"] *= 5.0
    df_b = build_features(syn_b)
    df_b = add_targets(df_b, horizon=1, train_mask=train_mask)
    thr_b = df_b.attrs["regime_threshold"]

    assert abs(thr_a - thr_b) < 1e-9, (
        f"Umbral del régimen depende del futuro: thr_a={thr_a}, thr_b={thr_b}"
    )


def test_regime_threshold_source_is_traceable(synthetic_ohlcv):
    df = build_features(synthetic_ohlcv)
    n = len(df)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[: int(n * 0.7)] = True
    df = add_targets(df, horizon=1, train_mask=train_mask)
    assert df.attrs["regime_threshold_source"] == "train_quantile"
