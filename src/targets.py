"""
Construcción de targets.

- Regresión:     ``target_vol = vol_realized.shift(-h)``
                  donde ``vol_realized_t = std(log_return_{t-w+1 ... t})``.
- Clasificación: ``target_regime = (target_vol > umbral).astype(int)``
                  donde el umbral es la **mediana** de ``target_vol`` calculada
                  exclusivamente sobre el conjunto de entrenamiento.

Causalidad: ``vol_realized`` solo usa información hasta ``t``. El target se
construye con ``shift(-h)``, es decir, ``vol_realized_{t+h}``, que es lo que
queremos predecir desde el estado del mundo en ``t``.

Reglas absolutas:
- El umbral del régimen se calcula SOLO con train.
- Los targets NO se incluyen como features (ver ``features.feature_columns``).
- Las filas con NaN en target deben filtrarse antes del entrenamiento.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import HORIZON, VOL_WINDOW, REGIME_THRESHOLD_QUANTILE


# =====================================================================
#  Bases — log-retornos y volatilidad realizada causal
# =====================================================================
def add_log_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Añade log-retornos. Causal: ``log_return_t = log(close_t / close_{t-1})``.
    """
    df = df.copy()
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    return df


def add_realized_volatility(df: pd.DataFrame, window: int = VOL_WINDOW) -> pd.DataFrame:
    """
    Volatilidad realizada causal:
        ``vol_realized_t = std(log_return_{t-w+1 ... t})``

    Esta columna es la **variable base**. El target se deriva con ``shift(-h)``.
    """
    df = df.copy()
    if "log_return" not in df.columns:
        df = add_log_returns(df)
    df["vol_realized"] = (
        df["log_return"].rolling(window=window, min_periods=window).std()
    )
    return df


# =====================================================================
#  Targets
# =====================================================================
def add_targets(
    df: pd.DataFrame,
    horizon: int = HORIZON,
    train_mask: np.ndarray | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Construye los dos targets del proyecto:

    - ``target_vol``   = ``vol_realized.shift(-horizon)``   (regresión)
    - ``target_regime`` = ``(target_vol > threshold).astype(Int64)``  (clasif.)

    El umbral se determina así, en orden de prioridad:
    1. ``threshold`` argumento explícito (si se pasa).
    2. Mediana de ``target_vol`` en ``train_mask`` (anti-leakage).
    3. Mediana de ``target_vol`` global (solo modo exploratorio,
       imprime advertencia implícita: ``df.attrs['regime_threshold_source']``).

    Parameters
    ----------
    df : DataFrame con ``vol_realized``. Si no la tiene, se calcula.
    horizon : pasos adelante (default 1).
    train_mask : máscara booleana del subconjunto train. Recomendado SIEMPRE.
    threshold : umbral explícito. Si se pasa, se ignora ``train_mask``.

    Returns
    -------
    DataFrame con columnas ``target_vol`` y ``target_regime`` añadidas.
    Los atributos ``df.attrs`` incluyen ``regime_threshold`` y
    ``regime_threshold_source`` con trazabilidad.
    """
    df = df.copy()
    if "vol_realized" not in df.columns:
        df = add_realized_volatility(df)

    df["target_vol"] = df["vol_realized"].shift(-horizon)

    # --- Determinar umbral ---
    if threshold is not None:
        thr = float(threshold)
        source = "explicit"
    elif train_mask is not None:
        vol_train = df.loc[train_mask, "target_vol"].dropna()
        thr = float(vol_train.quantile(REGIME_THRESHOLD_QUANTILE))
        source = "train_quantile"
    else:
        vol_all = df["target_vol"].dropna()
        thr = float(vol_all.quantile(REGIME_THRESHOLD_QUANTILE))
        source = "global_quantile_NOT_RECOMMENDED"

    df["target_regime"] = (df["target_vol"] > thr).astype("Int64")
    df.attrs["regime_threshold"] = thr
    df.attrs["regime_threshold_source"] = source
    df.attrs["regime_horizon"] = horizon
    return df
