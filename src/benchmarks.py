"""
Benchmarks econométricos para predicción de volatilidad realizada.

Todos los benchmarks predicen ``target_vol_t = vol_realized_{t+1}``,
la volatilidad realizada del día siguiente.

Convención común:
- Se reciben (a) ``full_df`` ordenado cronológicamente con índice 0..N-1,
  y (b) ``test_idx`` con los índices del subconjunto sobre el que predecir.
- Los modelos que requieren entrenamiento previo (ARIMA, GARCH, HAR-RV)
  reciben también ``train_idx`` para evitar leakage.
- Cada función retorna un ``np.ndarray`` de longitud ``len(test_idx)`` con
  las predicciones, alineadas con ``full_df.loc[test_idx, 'target_vol']``.

Walk-forward (filtering): para ARIMA y GARCH, los parámetros se estiman
una sola vez sobre ``train`` y luego el estado del modelo se actualiza
secuencialmente con cada nueva observación del test. Esto representa el
escenario realista de un trader que mantiene una calibración estable pero
incorpora la nueva información a medida que llega. No se re-estiman los
parámetros en cada paso (eso introduce noise de optimización y es
computacionalmente prohibitivo para corridas comparativas).

Referencias:
- Bollerslev (1986), J. of Econometrics, GARCH.
- Corsi (2009), J. of Financial Econometrics, HAR-RV.
- RiskMetrics (J.P. Morgan, 1996), EWMA con lambda=0.94 para datos diarios.
- Box & Jenkins (1970), ARIMA.
- Andersen, Bollerslev, Diebold, Labys (2003), Modeling and Forecasting RV.
"""
from __future__ import annotations
import warnings
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# =====================================================================
#  1. Naive (persistencia)
# =====================================================================
def naive_forecast(
    full_df: pd.DataFrame,
    test_idx: Sequence[int],
) -> np.ndarray:
    """
    Predicción de persistencia: ``pred_{t+1} = vol_realized_t``.

    Es el benchmark mas conservador. Para series con autocorrelacion
    fuerte en la volatilidad (como INTC), es dificil de batir y suele
    ser el punto de referencia de Diebold-Mariano.
    """
    return full_df.loc[test_idx, "vol_realized"].to_numpy()


# =====================================================================
#  2. Rolling mean
# =====================================================================
def rolling_mean_forecast(
    full_df: pd.DataFrame,
    test_idx: Sequence[int],
    window: int = 22,
) -> np.ndarray:
    """
    Promedio movil causal de la volatilidad realizada:
        ``pred_{t+1} = mean(vol_realized_{t-w+1}, ..., vol_realized_t)``

    Robusto a outliers pero pierde reactividad a cambios de regimen.
    Sirve de baseline para mostrar la utilidad del modelado dinamico.
    """
    s = full_df["vol_realized"].rolling(window=window, min_periods=1).mean()
    return s.loc[test_idx].to_numpy()


# =====================================================================
#  3. EWMA (RiskMetrics)
# =====================================================================
def ewma_forecast(
    full_df: pd.DataFrame,
    test_idx: Sequence[int],
    lambda_: float = 0.94,
    warm_up: int = 60,
) -> np.ndarray:
    """
    Exponentially Weighted Moving Average de la varianza condicional,
    siguiendo la convencion RiskMetrics (J.P. Morgan, 1996):

        ``sigma^2_{t+1} = lambda sigma^2_t + (1-lambda) r_t^2``

    La prediccion es ``sigma_{t+1} = sqrt(sigma^2_{t+1})``. Para inicializar
    ``sigma^2_0`` se usa la varianza muestral de los primeros ``warm_up``
    retornos.

    El parametro ``lambda=0.94`` es el valor por defecto de RiskMetrics
    para datos diarios y representa una vida media efectiva de ~11 dias.

    Parameters
    ----------
    full_df : DataFrame con columna ``log_return``.
    test_idx : indices del subconjunto sobre el que predecir.
    lambda_ : factor de decaimiento, en (0, 1). Default 0.94.
    warm_up : numero de retornos iniciales para estimar sigma^2_0.

    Notes
    -----
    EWMA predice la volatilidad instantanea ``sigma_{t+1}``, no la
    volatilidad realizada de una ventana de 5 dias. Para que sea
    comparable con el target, se reporta tal cual; el sesgo de definicion
    es absorbido en la comparacion relativa entre modelos (Diebold-Mariano
    usa diferenciales de perdida, no niveles absolutos).
    """
    r = full_df["log_return"].to_numpy()
    n = len(r)

    sigma2 = np.empty(n)
    # Inicializacion: varianza muestral de los primeros `warm_up` retornos.
    init_window = r[1 : warm_up + 1]
    sigma2[0] = np.nanvar(init_window, ddof=0)

    for t in range(1, n):
        rt = r[t] if not np.isnan(r[t]) else 0.0
        sigma2[t] = lambda_ * sigma2[t - 1] + (1.0 - lambda_) * rt ** 2

    sigma = np.sqrt(sigma2)
    return pd.Series(sigma, index=full_df.index).loc[test_idx].to_numpy()


# =====================================================================
#  4. ARIMA
# =====================================================================
def arima_forecast(
    full_df: pd.DataFrame,
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    order: Tuple[int, int, int] = (1, 0, 1),
    refit_freq: int = 0,
) -> np.ndarray:
    """
    ARIMA sobre la serie ``vol_realized`` con walk-forward filtering.

    El modelo se ajusta una sola vez sobre ``train`` y posteriormente
    se actualiza con ``.append()`` cada vez que llega una nueva
    observacion del test, sin re-estimar parametros (filtering).

    Parameters
    ----------
    full_df : DataFrame con columna ``vol_realized``.
    train_idx, test_idx : indices.
    order : (p, d, q) del ARIMA. Default ``(1, 0, 1)``.
    refit_freq : si > 0, re-estima parametros cada ``refit_freq`` pasos.
        Por defecto 0 (no re-estima); poner ej. 250 para re-estimar
        anualmente. Costoso.

    Returns
    -------
    np.ndarray de predicciones para cada elemento de ``test_idx``.
    """
    from statsmodels.tsa.arima.model import ARIMA

    y_train = full_df.loc[train_idx, "vol_realized"].dropna().to_numpy()
    y_test_actual = full_df.loc[test_idx, "vol_realized"].to_numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(y_train, order=order)
        fit = model.fit(method_kwargs={"warn_convergence": False})

    preds = np.empty(len(test_idx))
    state = fit
    for i, y_t_actual in enumerate(y_test_actual):
        fc = state.forecast(steps=1)
        preds[i] = float(fc[0]) if hasattr(fc, "__len__") else float(fc)
        # Filtering: actualizar el estado con la observacion real.
        if not np.isnan(y_t_actual):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                state = state.append([y_t_actual], refit=False)
        if refit_freq and (i + 1) % refit_freq == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                state = state.apply(state.data.endog, refit=True)

    return np.maximum(preds, 0.0)


# =====================================================================
#  5. GARCH(1,1) - Bollerslev 1986
# =====================================================================
def garch_forecast(
    full_df: pd.DataFrame,
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    p: int = 1,
    q: int = 1,
    rescale_factor: float = 100.0,
) -> np.ndarray:
    """
    GARCH(p, q) con distribucion normal sobre log-retornos, walk-forward.

    El modelo se ajusta una vez sobre ``train``; luego se aplica la
    recursion ``sigma^2_{t+1} = omega + alpha eps^2_t + beta sigma^2_t``
    con los parametros estimados, alimentada por los retornos observados
    del test (filtering, parametros fijos).

    Parameters
    ----------
    full_df : DataFrame con columna ``log_return``.
    train_idx, test_idx : indices del DataFrame.
    p, q : orden GARCH. Default GARCH(1, 1) que es el estandar.
    rescale_factor : ``arch`` recomienda escalar los retornos en 100 para
        estabilidad numerica. La prediccion se des-escala al final.

    Returns
    -------
    np.ndarray con predicciones de ``sigma_{t+1}`` para cada elemento de
    ``test_idx``, en la escala original de los retornos.
    """
    from arch import arch_model

    r = full_df["log_return"].to_numpy() * rescale_factor

    train_end = int(max(train_idx)) + 1  # exclusive
    test_positions = np.array(list(test_idx))

    # Ajuste GARCH sobre train (sin NaNs)
    r_train = r[:train_end]
    r_train = r_train[~np.isnan(r_train)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(
            r_train,
            mean="Zero",
            vol="GARCH",
            p=p, q=q,
            dist="normal",
            rescale=False,
        )
        res = am.fit(disp="off", show_warning=False)

    omega = float(res.params["omega"])
    alpha = float(res.params["alpha[1]"])
    beta = float(res.params["beta[1]"])

    # Estado inicial: ultimo sigma^2 estimado en train y ultimo retorno
    sigma2_prev = float(res.conditional_volatility[-1] ** 2)
    eps_prev = float(r[train_end - 1]) if not np.isnan(r[train_end - 1]) else 0.0

    sigma2_pred = np.empty(len(test_positions))
    for i, t in enumerate(test_positions):
        sigma2_pred[i] = omega + alpha * eps_prev ** 2 + beta * sigma2_prev
        rt = float(r[t]) if not np.isnan(r[t]) else 0.0
        sigma2_prev = sigma2_pred[i]
        eps_prev = rt

    sigma_pred = np.sqrt(sigma2_pred) / rescale_factor
    return sigma_pred


# =====================================================================
#  6. HAR-RV - Corsi 2009
# =====================================================================
def har_rv_forecast(
    full_df: pd.DataFrame,
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    components: Sequence[str] = ("vol_d", "vol_w", "vol_m"),
) -> np.ndarray:
    """
    Heterogeneous Autoregressive model of Realized Volatility (Corsi 2009):

        ``target_vol_t = b0 + b_d * vol_d_t + b_w * vol_w_t + b_m * vol_m_t + e_t``

    donde ``vol_d_t``, ``vol_w_t``, ``vol_m_t`` son la volatilidad
    realizada rezagada sobre ventanas diaria, semanal y mensual
    respectivamente (definidas en ``features.py`` con shift(1) para
    causalidad).

    Es el estandar de oro moderno para forecasting de volatilidad
    realizada. Estima por OLS, asi que es trivial computacionalmente y
    estable. Suele batir a GARCH y EWMA en horizontes cortos.

    Parameters
    ----------
    full_df : DataFrame con columnas ``components`` + ``target_vol``.
    train_idx, test_idx : indices.
    components : nombres de columnas a usar. Default Corsi original
        ``(vol_d, vol_w, vol_m)``.

    Returns
    -------
    np.ndarray con predicciones.
    """
    X_train = full_df.loc[train_idx, list(components)].to_numpy()
    y_train = full_df.loc[train_idx, "target_vol"].to_numpy()
    X_test = full_df.loc[test_idx, list(components)].to_numpy()

    # Filtrar filas con NaN
    stack = np.column_stack([X_train, y_train.reshape(-1, 1)])
    mask = ~np.any(np.isnan(stack), axis=1)
    X_train_clean = X_train[mask]
    y_train_clean = y_train[mask]

    model = LinearRegression()
    model.fit(X_train_clean, y_train_clean)

    col_means = np.nanmean(X_train_clean, axis=0)
    X_test_imputed = np.where(np.isnan(X_test), col_means, X_test)

    preds = model.predict(X_test_imputed)
    return np.maximum(preds, 0.0)


# =====================================================================
#  Utilidad: tabla resumen
# =====================================================================
def summarize_benchmarks(
    predictions: dict,
    y_true: np.ndarray,
) -> pd.DataFrame:
    """
    Calcula RMSE y MAE para un conjunto de predicciones y los devuelve
    como DataFrame ordenado por RMSE ascendente.

    Parameters
    ----------
    predictions : dict[str, np.ndarray]
        Mapa nombre_del_modelo -> vector de predicciones (mismo largo
        que ``y_true``).
    y_true : np.ndarray
        Vector de targets reales.

    Returns
    -------
    pd.DataFrame con columnas ``model, rmse, mae`` ordenado por RMSE.
    """
    rows = []
    mask_true = ~np.isnan(y_true)
    for name, pred in predictions.items():
        valid = mask_true & ~np.isnan(pred)
        e = (pred[valid] - y_true[valid])
        rmse = float(np.sqrt(np.mean(e ** 2)))
        mae = float(np.mean(np.abs(e)))
        rows.append({"model": name, "rmse": rmse, "mae": mae, "n_valid": int(valid.sum())})
    return (
        pd.DataFrame(rows)
        .sort_values("rmse")
        .reset_index(drop=True)
    )
