"""
Configuración global del proyecto INTC-Vol-ML.

Centraliza constantes, paths y semillas para garantizar reproducibilidad
y para que cualquier módulo lea el mismo estado.
"""
from pathlib import Path

# =====================================================================
#  Reproducibilidad
# =====================================================================
RANDOM_STATE = 42

# =====================================================================
#  Paths
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
TABLES_DIR = OUTPUTS_DIR / "tables"

# =====================================================================
#  Dataset
# =====================================================================
TICKER = "INTC"
RAW_FILE = RAW_DIR / "intc.us.txt"                  # archivo Kaggle (si existe)
PROCESSED_FILE = PROCESSED_DIR / "intc_clean.parquet"
FEATURES_FILE = PROCESSED_DIR / "features.parquet"

# Rango temporal por defecto (estable para reproducibilidad).
# El dataset Kaggle ``intc.us.txt`` cubre 1972-01-07 → 2017-11-10. Filtramos
# desde 1990 para excluir la microestructura pre-electronic-trading y dejar
# un rango moderno suficiente para split 70/15/15 (~7.000 días).
DATE_START = "1990-01-01"
DATE_END = "2017-12-31"

# =====================================================================
#  Splits temporales (rúbrica Dr. Lihki)
# =====================================================================
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# =====================================================================
#  Definición del problema
# =====================================================================
HORIZON = 1                                          # predicción a 1 día
VOL_WINDOW = 5                                       # ventana causal para vol realizada
REGIME_THRESHOLD_QUANTILE = 0.5                      # mediana en train → régimen binario

# =====================================================================
#  Utilidades
# =====================================================================
def ensure_dirs() -> None:
    """Crea todas las carpetas necesarias si no existen. Idempotente."""
    for d in [
        RAW_DIR, INTERIM_DIR, PROCESSED_DIR,
        FIGURES_DIR, METRICS_DIR, MODELS_DIR,
        PREDICTIONS_DIR, TABLES_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
