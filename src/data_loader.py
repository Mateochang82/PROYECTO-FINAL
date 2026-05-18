"""
Carga del dataset INTC OHLCV.

Tres modos con fallback automático en este orden:

1. **Local (preferido):** lee ``data/raw/intc.us.txt`` (formato Kaggle).
   Es el modo de producción para la entrega académica.

2. **Online:** descarga vía ``yfinance`` y guarda parquet en ``data/raw/``.
   Útil para reproducir el proyecto si no se tiene el archivo Kaggle.

3. **Sintético (CI):** si las dos opciones anteriores fallan (sin archivo
   local y sin acceso a red), genera un OHLCV sintético con clustering
   de volatilidad estilo GARCH y semilla fija ``RANDOM_STATE``. Este modo
   está pensado SOLO para validar que el pipeline corre en entornos
   sandbox (CI, contenedores sin red). NO debe usarse para la entrega.
   Cualquier dataframe generado sintéticamente lleva el atributo
   ``df.attrs['source'] == 'synthetic'`` para trazabilidad.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from .config import (
    RAW_FILE, RAW_DIR, DATE_START, DATE_END, TICKER,
    RANDOM_STATE, ensure_dirs,
)


# =====================================================================
#  Lectores
# =====================================================================
def _load_from_kaggle_txt(path: Path) -> pd.DataFrame:
    """
    Lee el formato Kaggle ``intc.us.txt``:
        Date,Open,High,Low,Close,Volume,OpenInt
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    keep = ["date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df.attrs["source"] = "kaggle"
    return df


def _load_from_yfinance(start: str, end: str) -> pd.DataFrame:
    """Descarga INTC desde Yahoo Finance vía ``yfinance``."""
    import yfinance as yf

    raw = yf.download(
        TICKER, start=start, end=end,
        progress=False, auto_adjust=False, threads=False,
    )
    if raw.empty:
        raise RuntimeError(f"yfinance retornó DataFrame vacío para {TICKER}")

    # yfinance puede devolver MultiIndex en columnas; aplanar
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    keep = ["date", "open", "high", "low", "close", "volume"]
    df = raw[keep].copy()
    df["date"] = pd.to_datetime(df["date"])
    df.attrs["source"] = "yfinance"
    return df


def _load_synthetic_ohlcv(start: str, end: str, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Genera un OHLCV sintético determinista con clustering de volatilidad
    estilo GARCH(1,1). Solo para CI/sandbox; NO usar en la entrega.

    El proceso:
        log_ret_t = mu + sigma_t * z_t,    z_t ~ N(0,1)
        sigma_t^2 = omega + alpha*eps_{t-1}^2 + beta*sigma_{t-1}^2

    Los parámetros son típicos de una acción tecnológica (alpha=0.08,
    beta=0.90, omega calibrado para vol diaria ~1.5%).
    """
    rng = np.random.default_rng(seed)

    # Calendario de días hábiles (lun-vie) en el rango
    dates = pd.bdate_range(start=start, end=end, freq="B")
    n = len(dates)

    # Parámetros GARCH(1,1)
    mu = 0.0003
    omega = 0.000005
    alpha = 0.08
    beta = 0.90
    sigma2 = np.zeros(n)
    eps = np.zeros(n)
    sigma2[0] = omega / (1.0 - alpha - beta)

    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()

    log_returns = mu + eps
    close = 10.0 * np.exp(np.cumsum(log_returns))

    # OHLC sintético consistente: open ≈ close anterior, high/low alrededor del close
    intraday_vol = 0.6 * np.sqrt(sigma2)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.standard_normal(n)) * intraday_vol)
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.standard_normal(n)) * intraday_vol)
    volume = (rng.lognormal(mean=15.0, sigma=0.3, size=n)).astype(np.int64)

    df = pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    df.attrs["source"] = "synthetic"
    return df


# =====================================================================
#  API pública
# =====================================================================
def load_intc(
    prefer_local: bool = True,
    start: str = DATE_START,
    end: str = DATE_END,
    cache_to_raw: bool = True,
    allow_synthetic: bool = True,
) -> pd.DataFrame:
    """
    Carga el dataset INTC OHLCV con fallback de tres niveles.

    Parameters
    ----------
    prefer_local : bool
        Si True y existe ``data/raw/intc.us.txt``, se usa.
    start, end : str
        Rango temporal (cuando se descarga online o se genera sintético).
    cache_to_raw : bool
        Si se descarga online, persistir copia en ``data/raw/intc_yf.parquet``
        para acelerar ejecuciones siguientes.
    allow_synthetic : bool
        Si True (default), cae a un OHLCV sintético determinista cuando
        no hay archivo local y no hay red. Para la entrega académica
        este flag debe quedar implícitamente desactivado porque el
        archivo local existirá. Cualquier dataframe sintético lleva
        ``df.attrs['source'] == 'synthetic'``.

    Returns
    -------
    pd.DataFrame
        Columnas ``date, open, high, low, close, volume``, ordenado
        cronológicamente, índice reseteado. Atributo ``df.attrs['source']``
        ∈ {"kaggle", "yfinance", "synthetic"}.
    """
    ensure_dirs()
    cached = RAW_DIR / "intc_yf.parquet"

    if prefer_local and RAW_FILE.exists():
        df = _load_from_kaggle_txt(RAW_FILE)
    elif cached.exists():
        df = pd.read_parquet(cached)
        df.attrs["source"] = "yfinance_cached"
    else:
        try:
            df = _load_from_yfinance(start, end)
            if cache_to_raw:
                df.to_parquet(cached, index=False)
        except Exception as e:
            if not allow_synthetic:
                raise
            warnings.warn(
                f"[data_loader] Sin archivo Kaggle y sin acceso a yfinance ({e!r}). "
                f"Activando modo SINTÉTICO determinista para CI. "
                f"NO usar para entrega académica.",
                stacklevel=2,
            )
            df = _load_synthetic_ohlcv(start, end)

    # Filtro temporal y limpieza
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    df = df[
        (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    ].reset_index(drop=True)
    # Preservar el atributo source tras los filtros
    if "source" not in df.attrs:
        df.attrs["source"] = "unknown"
    return df
