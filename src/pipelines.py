"""
Construcción de pipelines scikit-learn / imblearn.

Se completa progresivamente en los notebooks de modelado:
- NB 05: pipelines de regresión.
- NB 06: pipelines de clasificación.
- NB 07: pipelines con balanceo (imblearn).
- NB 08: pipelines parametrizados para optimización.

Convención: TODO preprocesamiento ajustable (imputación, escalado, SMOTE,
selección de variables) DEBE ir como paso del Pipeline para que la
validación cruzada lo reajuste por fold, evitando data leakage.
"""
from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def make_baseline_regression_pipeline(estimator):
    """Pipeline mínimo: imputación mediana + escalado + estimador."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def make_baseline_classification_pipeline(estimator):
    """Pipeline mínimo para clasificación: igual estructura."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])
