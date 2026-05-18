"""
Construcción de features causales temporales.

REGLA ABSOLUTA: ninguna feature usa información de t+1 en adelante.

Implementación:
- Todas las operaciones ``rolling`` se calculan sobre ``[t-w+1, t]``.
- ``vol_d``, ``vol_w``, ``vol_m`` para HAR-RV se calculan con ``.shift(1)``
  para garantizar que en ``t`` solo se use información hasta ``t-1``.
- ``true_range`` usa ``close.shift(1)`` (cierre del día anterior), causal.
- Indicadores de calendario son determinísticos de la fecha en ``t``.

Verificación: ``tests/test_features_causal.py`` confirma que mutar el
futuro del dataset no altera las features del pasado.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .targets import add_log_returns, add_realized_volatility


# =====================================================================
#  Helpers
# =====================================================================
def _safe_log(x):
    """log(x) seguro: devuelve NaN si x ≤ 0."""
    return np.log(np.where(x > 0, x, np.nan))


# =====================================================================
#  Constructor principal
# =====================================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye ~30 features causales sobre el DataFrame OHLCV.

    Familias incluidas:
    1) Retornos rezagados (1, 2, 3, 5, 10).
    2) Volatilidad rolling a varias ventanas (5, 10, 22).
    3) Componentes HAR-RV: vol_d, vol_w, vol_m (con shift(1)).
    4) Momentum acumulado (5, 10, 22).
    5) Rangos de precio: range_hl, true_range, ATR(14).
    6) Volumen: log_volume, lag, MA, ratio.
    7) Medias móviles del precio y distancia normalizada (10, 22, 50).
    8) Asimetría y curtosis rolling de retornos (22).
    9) Calendario: día de la semana, mes.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # --- Bases ---
    if "log_return" not in df.columns:
        df = add_log_returns(df)
    if "vol_realized" not in df.columns:
        df = add_realized_volatility(df)

    # 1) Retornos rezagados
    for lag in [1, 2, 3, 5, 10]:
        df[f"ret_lag_{lag}"] = df["log_return"].shift(lag)

    # 2) Volatilidad rolling a distintas ventanas (todas causales hasta t)
    for w in [5, 10, 22]:
        df[f"vol_{w}"] = df["log_return"].rolling(w, min_periods=w).std()

    # 3) Componentes HAR-RV (Corsi 2009): diario, semanal, mensual.
    #    El shift(1) garantiza que en t solo se use información hasta t-1.
    df["vol_d"] = df["vol_realized"].shift(1)
    df["vol_w"] = df["vol_realized"].rolling(5, min_periods=5).mean().shift(1)
    df["vol_m"] = df["vol_realized"].rolling(22, min_periods=22).mean().shift(1)

    # 4) Momentum (retorno acumulado en ventana)
    for w in [5, 10, 22]:
        df[f"momentum_{w}"] = df["log_return"].rolling(w, min_periods=w).sum()

    # 5) Rangos de precio
    df["range_hl"] = (df["high"] - df["low"]) / df["close"]
    df["true_range"] = np.maximum.reduce([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ]) / df["close"]
    df["atr_14"] = df["true_range"].rolling(14, min_periods=14).mean()

    # 6) Volumen
    df["log_volume"] = _safe_log(df["volume"])
    df["volume_lag_1"] = df["volume"].shift(1)
    df["log_volume_ma_10"] = df["log_volume"].rolling(10, min_periods=10).mean()
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20, min_periods=20).mean()

    # 7) Medias móviles del precio y distancia normalizada
    for w in [10, 22, 50]:
        df[f"sma_{w}"] = df["close"].rolling(w, min_periods=w).mean()
        df[f"px_dist_sma_{w}"] = (df["close"] - df[f"sma_{w}"]) / df[f"sma_{w}"]

    # 8) Asimetría y curtosis rolling (efecto colas / leverage)
    df["skew_22"] = df["log_return"].rolling(22, min_periods=22).skew()
    df["kurt_22"] = df["log_return"].rolling(22, min_periods=22).kurt()

    # 9) Calendario
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    return df


# =====================================================================
#  Selección de columnas
# =====================================================================
def feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Lista de columnas a usar como features.

    Excluye OHLCV crudo (las features ya capturan su info de forma causal),
    la base ``log_return``, ``vol_realized``, y por supuesto los targets.
    """
    exclude = {
        "date", "open", "high", "low", "close", "volume",
        "log_return", "vol_realized",
        "target_vol", "target_regime",
    }
    return [c for c in df.columns if c not in exclude]
