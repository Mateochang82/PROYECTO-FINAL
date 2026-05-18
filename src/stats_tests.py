"""
Pruebas estadísticas sobre residuos y para comparación entre modelos.

Cubre:
- Diagnóstico de residuos (NB 10):
    * White (homocedasticidad)
    * Breusch-Pagan (homocedasticidad, alternativa)
    * BDS (independencia / linealidad)
    * Ljung-Box (autocorrelación serial)
    * Jarque-Bera (normalidad)

- Comparación entre modelos (NB 11):
    * Diebold-Mariano para errores de pronóstico de regresión
      (con corrección Newey-West para autocorrelación)
    * Bootstrap CI para una métrica genérica
    * DeLong (Sun & Xu, 2014) para diferencias de AUC entre
      clasificadores binarios

Todas las funciones retornan dataclasses o dicts simples con
los campos relevantes (estadístico, p-valor, IC).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# =====================================================================
#  Residuos
# =====================================================================
@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    extra: dict | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["extra"] is None:
            d.pop("extra")
        return d


def white_test(residuals: np.ndarray, X: np.ndarray | None = None) -> TestResult:
    """
    Test de White para heteroscedasticidad. La hipótesis nula es
    homocedasticidad. Si ``X`` no se pasa, se usa una constante + el
    índice temporal como regresor (versión simple).

    H0: los residuos son homocedásticos (varianza constante).
    """
    from statsmodels.stats.diagnostic import het_white
    import statsmodels.api as sm

    res = np.asarray(residuals).ravel()
    n = len(res)
    if X is None:
        # Sin features: usar índice como proxy. White requiere X con constante.
        X = np.column_stack([np.ones(n), np.arange(n)])
    else:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # Añadir constante si no la tiene
        if not np.allclose(X[:, 0], 1.0):
            X = sm.add_constant(X, has_constant="add")

    lm_stat, lm_pvalue, _, _ = het_white(res, X)
    return TestResult(name="white", statistic=float(lm_stat), p_value=float(lm_pvalue))


def breusch_pagan_test(residuals: np.ndarray, X: np.ndarray | None = None) -> TestResult:
    """
    Test de Breusch-Pagan para heteroscedasticidad. H0: homocedasticidad.
    Alternativa más liviana que White.
    """
    from statsmodels.stats.diagnostic import het_breuschpagan
    import statsmodels.api as sm

    res = np.asarray(residuals).ravel()
    n = len(res)
    if X is None:
        X = np.column_stack([np.ones(n), np.arange(n)])
    else:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if not np.allclose(X[:, 0], 1.0):
            X = sm.add_constant(X, has_constant="add")

    lm_stat, lm_pvalue, _, _ = het_breuschpagan(res, X)
    return TestResult(name="breusch_pagan", statistic=float(lm_stat),
                      p_value=float(lm_pvalue))


def ljung_box_test(residuals: np.ndarray, lags: int = 10) -> TestResult:
    """
    Test de Ljung-Box para autocorrelación serial. H0: no hay
    autocorrelación hasta el lag ``lags``. P-valor bajo indica
    dependencia residual no capturada por el modelo.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    res = np.asarray(residuals).ravel()
    out = acorr_ljungbox(res, lags=[lags], return_df=True)
    return TestResult(
        name=f"ljung_box(lag={lags})",
        statistic=float(out["lb_stat"].iloc[0]),
        p_value=float(out["lb_pvalue"].iloc[0]),
        extra={"lags": lags},
    )


def jarque_bera_test(residuals: np.ndarray) -> TestResult:
    """
    Test de Jarque-Bera para normalidad. H0: los residuos provienen
    de una distribución Normal. Usa asimetría y kurtosis muestrales.
    """
    res = np.asarray(residuals).ravel()
    jb, p = stats.jarque_bera(res)
    return TestResult(name="jarque_bera", statistic=float(jb), p_value=float(p))


def bds_test(
    residuals: np.ndarray,
    max_dim: int = 3,
    epsilon: float | None = None,
) -> TestResult:
    """
    Test BDS de Brock-Dechert-Scheinkman para independencia (no
    linealidad oculta).

    H0: los residuos son i.i.d.
    Si H0 se rechaza, hay estructura residual remanente que el modelo
    no capturó (típicamente no lineal o no estacionaria).

    Se reporta el estadístico para ``dim = max_dim``.
    """
    from statsmodels.tsa.stattools import bds

    res = np.asarray(residuals).ravel()
    if epsilon is None:
        epsilon = 0.5 * float(np.std(res))

    stat, p = bds(res, max_dim=max_dim, epsilon=epsilon)
    # bds devuelve arrays para cada dimensión 2..max_dim
    stat = np.atleast_1d(stat)
    p = np.atleast_1d(p)
    # Tomamos la última dimensión (max_dim)
    return TestResult(
        name=f"bds(dim={max_dim})",
        statistic=float(stat[-1]),
        p_value=float(p[-1]),
        extra={"epsilon": float(epsilon),
                "all_stats": [float(s) for s in stat],
                "all_pvalues": [float(p_) for p_ in p]},
    )


def acf_values(residuals: np.ndarray, n_lags: int = 30) -> np.ndarray:
    """ACF (función de autocorrelación) hasta ``n_lags``."""
    from statsmodels.tsa.stattools import acf
    res = np.asarray(residuals).ravel()
    return acf(res, nlags=n_lags, fft=True)


def residual_summary(residuals: np.ndarray, X: np.ndarray | None = None,
                     n_lags_lb: int = 10, max_dim_bds: int = 3) -> dict:
    """Aplica todas las pruebas y devuelve un dict completo."""
    out = {}
    for fn in [
        lambda r: white_test(r, X=X),
        lambda r: breusch_pagan_test(r, X=X),
        lambda r: ljung_box_test(r, lags=n_lags_lb),
        lambda r: jarque_bera_test(r),
        lambda r: bds_test(r, max_dim=max_dim_bds),
    ]:
        try:
            tr = fn(residuals)
            out[tr.name] = tr.to_dict()
        except Exception as e:
            out[fn.__name__] = {"error": str(e)}
    # Estadísticos descriptivos básicos
    res = np.asarray(residuals).ravel()
    out["descriptive"] = {
        "n": int(len(res)),
        "mean": float(np.mean(res)),
        "std": float(np.std(res, ddof=1)),
        "min": float(np.min(res)),
        "max": float(np.max(res)),
        "skew": float(stats.skew(res)),
        "kurtosis": float(stats.kurtosis(res, fisher=True)),
    }
    return out


# =====================================================================
#  Comparación entre modelos
# =====================================================================
@dataclass
class DMResult:
    statistic: float
    p_value: float
    mean_loss_diff: float
    horizon: int

    def to_dict(self) -> dict:
        return asdict(self)


def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    loss: str = "se",   # 'se' = squared error; 'ae' = absolute error
    horizon: int = 1,
) -> DMResult:
    """
    Test de Diebold-Mariano para comparar precisión predictiva entre
    dos modelos sobre la misma serie y_true.

    H0: ambos modelos tienen igual precisión esperada.
    Estadístico positivo = modelo A peor que B (pérdida media A > B).
    P-valor bajo → diferencia significativa.

    Implementación con corrección Newey-West para varianza de la
    serie de diferencias de pérdida.
    """
    y_true = np.asarray(y_true).ravel()
    pred_a = np.asarray(pred_a).ravel()
    pred_b = np.asarray(pred_b).ravel()

    if loss == "se":
        loss_a = (pred_a - y_true) ** 2
        loss_b = (pred_b - y_true) ** 2
    elif loss == "ae":
        loss_a = np.abs(pred_a - y_true)
        loss_b = np.abs(pred_b - y_true)
    else:
        raise ValueError(f"loss desconocida: {loss!r}")

    d = loss_a - loss_b
    n = len(d)
    mean_d = float(np.mean(d))

    # Varianza con corrección Newey-West usando bandwidth = horizon - 1
    nw_bw = max(horizon - 1, 0)
    var_d = float(np.var(d, ddof=0))
    for k in range(1, nw_bw + 1):
        autocov = float(np.mean((d[k:] - mean_d) * (d[:-k] - mean_d)))
        var_d += 2.0 * autocov

    se_d = np.sqrt(var_d / n)
    if se_d <= 0 or not np.isfinite(se_d):
        return DMResult(statistic=float("nan"), p_value=float("nan"),
                        mean_loss_diff=mean_d, horizon=horizon)

    stat = mean_d / se_d
    # Two-sided p-value usando N(0,1)
    p = 2.0 * (1.0 - stats.norm.cdf(np.abs(stat)))
    return DMResult(statistic=float(stat), p_value=float(p),
                    mean_loss_diff=mean_d, horizon=horizon)


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    IC bootstrap percentil para una métrica genérica
    ``metric_fn(y_true, y_pred)``.

    Retorna (estimación puntual, lo, hi).
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_true)
    point = float(metric_fn(y_true, y_pred))
    samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[b] = metric_fn(y_true[idx], y_pred[idx])
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return point, lo, hi


# =====================================================================
#  DeLong para AUC (Sun & Xu, 2014)
# =====================================================================
@dataclass
class DeLongResult:
    auc_a: float
    auc_b: float
    auc_diff: float
    statistic: float
    p_value: float

    def to_dict(self) -> dict:
        return asdict(self)


def _midrank(x):
    """Ranks con promedio en empates. Usado por DeLong."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1  # rangos 1-based
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _structural_components(predictions, label_1_count):
    """Componentes de varianza estructural según Sun & Xu (2014)."""
    m = label_1_count
    n = predictions.shape[1] - m
    pos = predictions[:, :m]
    neg = predictions[:, m:]
    k = predictions.shape[0]

    tx = np.zeros((k, m), dtype=float)
    ty = np.zeros((k, n), dtype=float)
    tz = np.zeros((k, m + n), dtype=float)

    for r in range(k):
        tx[r, :] = _midrank(pos[r, :])
        ty[r, :] = _midrank(neg[r, :])
        tz[r, :] = _midrank(predictions[r, :])

    aucs = (tz[:, :m].sum(axis=1) / (m * n)) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray) -> DeLongResult:
    """
    Test de DeLong (Sun & Xu, 2014) para comparar dos AUC sobre el
    mismo conjunto de etiquetas y_true ∈ {0, 1}.

    Retorna AUCs de A y B, su diferencia, el estadístico Z y el
    p-valor two-sided bajo N(0, 1).
    """
    y_true  = np.asarray(y_true).ravel().astype(int)
    proba_a = np.asarray(proba_a).ravel().astype(float)
    proba_b = np.asarray(proba_b).ravel().astype(float)

    # Reordenar: positivos primero
    order = np.argsort(-y_true)  # 1's first
    y_sorted = y_true[order]
    proba_a_s = proba_a[order]
    proba_b_s = proba_b[order]
    m = int(np.sum(y_sorted == 1))

    if m == 0 or m == len(y_sorted):
        return DeLongResult(auc_a=float("nan"), auc_b=float("nan"),
                            auc_diff=float("nan"),
                            statistic=float("nan"), p_value=float("nan"))

    predictions = np.vstack([proba_a_s, proba_b_s])
    aucs, delongcov = _structural_components(predictions, m)
    auc_a, auc_b = float(aucs[0]), float(aucs[1])
    L = np.array([1.0, -1.0])
    var = float(L @ delongcov @ L)
    if var <= 0 or not np.isfinite(var):
        return DeLongResult(auc_a=auc_a, auc_b=auc_b, auc_diff=auc_a - auc_b,
                            statistic=float("nan"), p_value=float("nan"))
    z = (auc_a - auc_b) / np.sqrt(var)
    p = 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))
    return DeLongResult(auc_a=auc_a, auc_b=auc_b, auc_diff=auc_a - auc_b,
                        statistic=float(z), p_value=float(p))
