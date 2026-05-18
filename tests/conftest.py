"""
Fixtures sintéticos para tests anti-leakage.

Generamos un OHLCV de ~500 días con drift suave y clustering de volatilidad
tipo GARCH simple, suficiente para verificar la lógica de features/targets.
"""
import numpy as np
import pandas as pd
import pytest

# Path hack para importar src/ desde tests/
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """
    OHLCV sintético de 500 días con volatilidad clustering simple.
    """
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2010-01-04", periods=n, freq="B")

    # vol con persistencia AR(1) simple
    vol = np.full(n, 0.02)
    for i in range(1, n):
        vol[i] = 0.95 * vol[i - 1] + 0.05 * abs(rng.standard_normal()) * 0.02

    rets = rng.standard_normal(n) * vol
    close = 30 * np.exp(np.cumsum(rets))

    high = close * (1 + np.abs(rng.standard_normal(n)) * 0.01)
    low = close * (1 - np.abs(rng.standard_normal(n)) * 0.01)
    open_ = np.r_[close[0], close[:-1]]
    volume = rng.integers(1_000_000, 50_000_000, size=n).astype(float)

    return pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
