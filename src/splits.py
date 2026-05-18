"""
Particionamiento temporal del dataset.

Reglas absolutas:
- Sin ``shuffle=True``.
- Sin ``train_test_split`` aleatorio.
- Particionamiento cronológico estricto.
- ``TimeSeriesSplit`` para validación cruzada interna.
"""
from __future__ import annotations
from typing import Tuple
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .config import TRAIN_FRAC, VAL_FRAC, TEST_FRAC


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Particiona df en train/val/test cronológicamente.

    Parameters
    ----------
    df : DataFrame ordenado o no por fecha.
    train_frac, val_frac, test_frac : deben sumar 1.
    date_col : nombre de la columna fecha.

    Returns
    -------
    (train, val, test) : tupla de DataFrames disjuntos y ordenados.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, \
        "train_frac + val_frac + test_frac debe ser 1.0"
    assert date_col in df.columns, f"Falta la columna '{date_col}'"

    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = df_sorted.iloc[:n_train].copy().reset_index(drop=True)
    val = df_sorted.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test = df_sorted.iloc[n_train + n_val :].copy().reset_index(drop=True)

    return train, val, test


def get_tscv(n_splits: int = 5, test_size: int | None = None, gap: int = 0) -> TimeSeriesSplit:
    """Devuelve un ``TimeSeriesSplit`` configurado."""
    kwargs = {"n_splits": n_splits, "gap": gap}
    if test_size is not None:
        kwargs["test_size"] = test_size
    return TimeSeriesSplit(**kwargs)


def split_summary(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                  date_col: str = "date") -> pd.DataFrame:
    """Tabla resumen con tamaño y rangos de fechas de cada subconjunto."""
    return pd.DataFrame({
        "subset": ["train", "val", "test"],
        "n_rows": [len(train), len(val), len(test)],
        "date_min": [train[date_col].min(), val[date_col].min(), test[date_col].min()],
        "date_max": [train[date_col].max(), val[date_col].max(), test[date_col].max()],
    })
