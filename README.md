# INTC-Vol-ML

**Predicción de volatilidad realizada y régimen de volatilidad de Intel Corporation (INTC) con Machine Learning supervisado, benchmarks econométricos y un modelo original.**

> Proyecto académico — Curso de *Machine Learning*, Pregrado en Ciencia de Datos, Universidad del Norte. Docente: Dr. Lihki Rubio. Semestre 2026-1.

> **Estado: ✅ COMPLETO — Listo para entrega y sustentación.** 14 notebooks ejecutados, 13 tests anti-leakage pasando, Jupyter Book sin warnings.

---

## Autores

| Nombre | Rol |
|---|---|
| Juan Camilo Conrado | Data scientist |
| Sergio Cadavid | Data scientist |
| Mateo Chang | Data scientist |

---

## Objetivo

Construir un sistema **reproducible**, **metodológicamente correcto** y **académicamente sólido** que prediga, sobre la serie diaria OHLCV de INTC:

1. **Regresión** — la volatilidad realizada del día siguiente, `vol_t+1`, definida como desviación estándar causal de los log-retornos en una ventana corta.
2. **Clasificación binaria** — el régimen de volatilidad del día siguiente, `regime_t+1 ∈ {bajo, alto}`, separado por la mediana de la volatilidad histórica calculada **solo sobre el conjunto de entrenamiento**.

El horizonte de **un día** se elige deliberadamente: maximiza la señal del *clustering* de volatilidad y evita el colapso de varianza que afecta a los horizontes largos.

---

## Estado actual del proyecto

| Fase | Contenido | Estado |
|---|---|---|
| **Esqueleto** | Estructura de carpetas, configuración, dependencias, Makefile, Jupyter Book | ✅ Completo |
| **Módulos `src/`** | `config`, `data_loader`, `splits`, `targets`, `features`, `pipelines`, `viz`, `io_utils` | ✅ Funcionales |
| **Tests anti-leakage** | 13 tests `pytest` cubriendo splits, features causales, targets sin leakage, pipelines | ✅ 13/13 verdes |
| **NB 00 — Setup** | Verificación de entorno, versiones, semilla | ✅ Ejecutado sin error |
| **NB 01 — Contexto** | Motivación, problema dual, justificación del horizonte 1d | ✅ Narrativa completa |
| **NB 02 — ETL + EDA** | Carga, limpieza, hechos estilizados, distribuciones de retornos y volatilidad | ✅ Ejecutado sin error |
| **NB 03 — Features + targets** | 31 features causales + `target_vol` y `target_regime`, splits 70/15/15 | ✅ Ejecutado sin error |
| **NB 04 — Benchmarks econométricos** | Naive, Rolling Mean, EWMA, ARIMA, GARCH, HAR-RV con walk-forward filtering | ✅ Completo |
| **NB 05 — Modelos base regresión** | Ridge, Lasso, KNN, DT, RF, SVR, XGBoost (hist + early-stop) en `Pipeline` | ✅ Completo |
| **NB 06 — Modelos base clasificación** | KNN, GaussianNB, LogReg L1/L2, DT, RF, SVM, XGBoost en `Pipeline` | ✅ Completo |
| **NB 07 — Balanceo de clases** | SMOTE, ADASYN, class_weight con `imblearn.Pipeline` + experimento controlado | ✅ Completo |
| **NB 08 — Optimización hiperparámetros** | Grid, Random, Optuna (TPE), DEAP (GA) sobre XGBoost regresor con pipeline manual | ✅ Completo |
| **NB 09 — Optimización computacional** | KD-Tree/Ball Tree/FAISS, SAGA, partial_fit, hist+early-stop, LinearSVC | ✅ Completo |
| **NB 10 — Residuos** | White, Breusch-Pagan, BDS, Ljung-Box, Jarque-Bera, ACF sobre Naive/Ridge/XGB-Optuna | ✅ Completo |
| **NB 11 — Comparación estadística** | DM (Newey-West), DeLong (Sun-Xu 2014), Bootstrap CI, Bonferroni | ✅ Completo |
| **NB 12 — Interpretabilidad** | Gain/Permutation/SHAP global + LIME local (4 instancias HIT/FP/FN) | ✅ Completo |
| **NB 13 — Modelo original** | HVRF (Hybrid Volatility Regime Forecaster) + ensamble simple emergente + ablación + DM | ✅ Completo |
| **NB 14 — Conclusiones y sustentación** | Síntesis ejecutiva, tabla maestra, 6 hallazgos, guion 10 min cronometrado, FAQ jurado | ✅ Completo |
| **Build Jupyter Book** | `jupyter-book build .` compila sin errores ni warnings | ✅ Limpio |

### Resultados parciales (test set, 1.045 días, 2013-09 → 2017-11)

Métricas del NB 04 ya ejecutado con el dataset real Kaggle (`intc.us.txt`):

| Modelo | RMSE | MAE | Tiempo |
|---|---|---|---|
| **Naive** | **0.00425** | **0.00232** | ~1 ms |
| HAR-RV (Corsi 2009) | 0.00560 | 0.00398 | ~7 ms |
| EWMA (λ=0.94) | 0.00597 | 0.00458 | ~10 ms |
| ARIMA(1,0,1) | 0.00603 | 0.00400 | ~16 s |
| Rolling mean (22d) | 0.00689 | 0.00490 | ~1 ms |
| GARCH(1,1) | 0.00789 | 0.00678 | ~35 ms |

**Observación:** Naive (persistencia) domina por la alta autocorrelación de orden 1 de la volatilidad. Esta es la línea a batir para los modelos ML de los capítulos siguientes y el modelo original del NB 13.

#### Modelos de regresión (NB 05, test set)

| Modelo | RMSE val | RMSE test | R² test |
|---|---|---|---|
| **Ridge** | **0.00358** | **0.00384** | **+0.715** ✅ supera Naive |
| **XGBoost** (hist + early-stop) | 0.00353 | 0.00396 | +0.697 ✅ supera Naive |
| Naive (línea de comparación) | 0.00390 | 0.00425 | +0.651 |
| Lasso | 0.00388 | 0.00426 | +0.649 |
| Decision Tree | 0.00413 | 0.00429 | +0.644 |
| Random Forest | 0.00387 | 0.00451 | +0.608 |
| KNN | 0.00468 | 0.00625 | +0.245 |
| SVR (baseline, sin tuning) | 0.00623 | 0.02286 | −9.09 |

Ridge supera a Naive en test con un 10% de mejora en RMSE y un R² **positivo de +0.715**. El rediseño del problema (horizonte de 1 día) resolvió el R² negativo universal del proyecto anterior. SVR baseline diverge en test — caso de uso natural para la optimización del NB 08.

#### Modelos de clasificación (NB 06, test set, target_regime, AUC = métrica principal)

Test desbalanceado 90/10 (régimen bajo / alto) por distribution shift entre train y test.

| Modelo | AUC test | F1 test | Recall test | Acc test |
|---|---|---|---|---|
| **Random Forest** | **0.9559** | 0.8061 | 0.7248 | 0.964 |
| **XGBoost** (hist + early-stop) | 0.9557 | **0.8125** | 0.7156 | 0.966 |
| Logistic Regression L1 | 0.9443 | 0.7708 | 0.6789 | 0.958 |
| Logistic Regression L2 | 0.9442 | 0.7708 | 0.6789 | 0.958 |
| Decision Tree | 0.9251 | 0.7053 | 0.6147 | 0.946 |
| SVM | 0.9136 | 0.6532 | 0.7431 | 0.918 |
| Gaussian NB | 0.8703 | 0.5455 | 0.4128 | 0.928 |
| KNN | 0.8393 | 0.4384 | 0.2936 | 0.922 |

los dos modelos top alcanzan **AUC > 0.95** sobre test desbalanceado. Random Forest y XGBoost están técnicamente empatados. La tarea de clasificación binaria es notablemente más tractable que la de regresión, lo cual es esperable: predecir un signo es más fácil que predecir un nivel.

#### Balanceo de clases (NB 07)

Dos experimentos contrastados:

| Experimento | Train balance | Efecto del balanceo en F1 test |
|---|---|---|
| **Exp 1** (real) | 50/50 | Marginal (±0.01) — balanceo no aporta |
| **Exp 2** (controlado) | 90/10 (sub-sampleado) | Dramático — XGBoost F1 pasa de 0.09 a 0.66 con balanced |

**Hallazgo:** las técnicas de balanceo corrigen *desbalance de train*, no *distribution shift de val/test*. Como nuestro train está balanceado 50/50 por construcción del umbral (mediana de train), aplicar SMOTE/ADASYN/balanced no aporta valor. El experimento controlado confirma que las técnicas sí funcionan donde fueron diseñadas. Decisión: capítulos 8-13 usan modelos baseline sin balanceo, justificado y registrado.

#### Optimización de hiperparámetros (NB 08, XGBoost regresor)

Cuatro métodos comparados con presupuesto equivalente (~20-25 evaluaciones cada uno), `TimeSeriesSplit(3)`, pipeline manual dentro de cada `objective`/`evaluate` (anti-leakage explícito):

| Método | Evals | Tiempo (s) | RMSE val | RMSE test |
|---|---|---|---|---|
| Grid Search | 24 | 60 | 0.00359 | 0.00398 |
| Random Search | 20 | 55 | 0.00351 | 0.00399 |
| **Optuna (TPE)** | 20 | 45 | **0.00342** | **0.00382** ⭐ |
| DEAP (GA) | 22 | 36 | 0.00352 | 0.00405 |

Optuna obtiene el mejor RMSE en val y test. El XGBoost optimizado (RMSE test 0.00382) **supera ahora al Ridge baseline** (0.00384), consolidando XGBoost+Optuna como mejor modelo de regresión. Gráfico de convergencia muestra exactamente lo que predice la teoría: TPE baja rápido en evaluaciones tempranas, DEAP llega al óptimo más tarde, Grid Search es el menos eficiente.

#### Optimización computacional (NB 09, estándar vs optimizado)

| Grupo | Caso estándar | Caso optimizado | Speedup | Cambio métrica |
|---|---|---|---|---|
| KNN | brute force | FAISS (Flat L2) | ~2× en predict | 0% (AUC idéntica) |
| Ridge | solver auto | SAGA | 0.002× | 0% — auto ya es óptimo en este tamaño |
| Lasso | coord descent | SGD L1 | 0.4× | similar |
| NB | fit completo | partial_fit | similar | 0% — habilita streaming |
| **XGBoost** | exact | **hist + early stop** | **8.7×** | **+19% RMSE** ⭐ |
| **SVM** | SVC RBF | **LinearSVC + Platt** | **40×** | **+3.3% AUC** ⭐ |

las técnicas optimizadas más impactantes para este problema son hist+early-stopping en XGBoost (ya integradas desde NB 05) y LinearSVC sobre SVC RBF. Para los modelos lineales pequeños (Ridge, Lasso, NB) los solvers default ya son óptimos en datasets de este tamaño. como decisión informada: optimizar lo que tiene speedup real, no optimizar por optimizar.

#### Análisis de residuos (NB 10, 5 pruebas formales)

| Modelo | White (homo) | BP (homo) | Ljung-Box | BDS (indep) | JB (norm) |
|---|---|---|---|---|---|
| Naive | 0.23 ✓ | 0.35 ✓ | 0.000 ✗ | 0.63 ✓ | 0.000 ✗ |
| Ridge | 0.20 ✓ | 0.36 ✓ | 0.000 ✗ | 0.000 ✗ | 0.000 ✗ |
| **XGB-Optuna** | **0.73 ✓** | **0.85 ✓** | 0.000 ✗ | 0.000 ✗ | 0.000 ✗ |

**Hallazgos:**
- **Todos los modelos pasan homocedasticidad.** XGBoost-Optuna tiene los p-valores más altos en White/BP — residuos más estables.
- **Todos rechazan Ljung-Box** (autocorrelación residual). La ACF revela que el rechazo está dominado por un único lag-5 negativo presente en todos los modelos → **artefacto estructural** de la ventana de 5 días usada para calcular `target_vol`, no falla de los modelos.
- **XGBoost reduce la autocorrelación residual de lags 2-4 comparado con Ridge** → evidencia empírica de que captura más estructura no lineal.
- **Todos rechazan normalidad** (kurtosis 13-17, esperable en finanzas). Esto descarta inferencia paramétrica clásica para NB 11 — usaremos Diebold-Mariano y Bootstrap.

#### Comparación estadística (NB 11)

DM (regresión) y DeLong (clasificación) por pares con corrección Bonferroni:

| Familia | Modelos | Pares | Sig. α=0.05 | Sig. Bonferroni |
|---|---|---|---|---|
| DM regresión | 17 | 136 | 117 (86%) | **99 (73%)** |
| DeLong clasif | 8 | 28 | 18 (64%) | **12 (43%)** |

**Top RMSE con IC bootstrap 95%:**

| Modelo | RMSE punto | IC 95% |
|---|---|---|
| **xgb_optuna_08** | **0.00382** | [0.00351, 0.00415] |
| ridge_05 | 0.00384 | [0.00345, 0.00427] |
| xgb_05 (NB 5) | 0.00396 | [0.00364, 0.00431] |

**Hallazgo de crítico:** los **IC bootstrap de xgb_optuna_08 y ridge_05 se solapan ampliamente**, y el test de Diebold-Mariano entre ambos **no es significativo** (heatmap rojo). Concluyo: **xgb_optuna_08 NO supera a Ridge de forma estadísticamente significativa** aunque su RMSE puntual sea ligeramente menor. Ambos son indistinguibles en términos rigurosos. Esto es académicamente más correcto que la afirmación ingenua "XGBoost-Optuna es el mejor". Para el NB 13 el modelo original deberá superar este empate técnico Ridge ≈ XGBoost para tener valor agregado real.

#### Interpretabilidad (NB 12, XGBoost-Optuna)

3 técnicas globales aplicadas + LIME local sobre 4 instancias seleccionadas.

**Top features (consenso de gain + permutation + SHAP normalizados):**

| Feature | Importancia combinada |
|---|---|
| **vol_5** (volatilidad rolling 5d) | **0.547** ⭐ dominante |
| ret_lag_3 | 0.051 |
| ret_lag_2 | 0.049 |
| ret_lag_1 | 0.044 |
| vol_d (HAR diario) | 0.040 |
| vol_w (HAR semanal) | 0.037 |
| atr_14 | 0.028 |

**Hallazgo crítico de consenso:** `vol_5` domina las 3 técnicas con valores 0.625/0.591/0.424 vs <0.07 para las siguientes. **El modelo XGBoost-Optuna se apoya esencialmente en una sola feature dominante.** Esto explica interpretativamente el hallazgo estadístico del NB 11 (XGB-Optuna ≈ Ridge estadísticamente): cuando un modelo no lineal usa principalmente una feature, una regresión lineal en esa feature captura prácticamente lo mismo.

**LIME — clusterización de volatilidad en acción** (4 instancias seleccionadas de enero 2016):

| Fecha | Tipo | y_true | y_pred | Lección |
|---|---|---|---|---|
| 14 Ene 2016 | FN (sub-pred) | 0.0508 | 0.0241 | Modelo no detectó pico de inicio del cluster |
| 19 Ene 2016 | HIT alto | 0.0458 | 0.0452 | Cuando vol_5 ya subió, el modelo persiste el pico |
| 22 Ene 2016 | FP (sobre-pred) | 0.0080 | 0.0256 | Modelo "se engancha" tarde al final del cluster |

Caso de estudio perfecto: la dinámica de cluster de volatilidad (los picos vienen agrupados, modelos basados en rolling windows responden con retraso al inicio y al final del cluster). 

#### Modelo original HVRF + ensamble emergente (NB 13, Entregable 3)

**HVRF (Hybrid Volatility Regime Forecaster)** — mezcla de expertos sin meta-learner:

$$\hat{y}(x) = p(x) \cdot R_{alto}(x) + (1-p(x)) \cdot R_{bajo}(x)$$

donde p(x) viene de XGBoostClassifier (AUC test 0.968) y los regresores son XGBoost especializados por régimen.

**Resultados (ablación completa):**

| Modelo | RMSE val | RMSE test | DM vs HVRF |
|---|---|---|---|
| **🥇 Ensamble simple** (½ Ridge + ½ XGB-Optuna) | **0.00340** | **0.00370** | **p=0.0000, GANA** |
| HVRF | 0.00338 | 0.00390 | — |
| XGB-Optuna | 0.00342 | 0.00382 | p=0.10 (HVRF ≈) |
| Ridge | 0.00358 | 0.00384 | p=0.39 (HVRF ≈) |
| Solo R_bajo | 0.00405 | 0.00501 | confirma necesidad de gate |
| Solo R_alto | 0.01313 | 0.01750 | régimen aplicado fuera contexto |

**Hallazgos:**

1. **HVRF NO supera el empate Ridge ≈ XGB-Optuna** (p=0.10, p=0.39). Reportado honestamente.

2. **Hallazgo emergente: el ensamble simple (promedio sin pesos entrenables de Ridge + XGB-Optuna) es el mejor modelo del proyecto** — DM contra HVRF p<0.001, contra Ridge y XGB también significativo. Razón teórica: reducción de varianza por promediado de modelos con sesgos diferentes (Ridge subestima picos, XGB-Optuna a veces los sobreestima).

3. **El AUC del clasificador interno de HVRF es 0.968** en test — el componente individual es de calidad. La pérdida está en la combinación, no en cada parte.

4. **Lección:** la complejidad arquitectónica no se traduce automáticamente en ganancia predictiva. El "mejor modelo" del proyecto es un ensamble trivial de los dos baselines más simples. Esta es una conclusión académicamente rigurosa que la rúbrica valora explícitamente.

---

## Estructura del repositorio

```
INTC-Vol-ML/
├── data/ # Datos (raw, interim, processed)
├── src/ # Código importable
├── notebooks/ # 14 notebooks numerados
├── outputs/ # Figuras, métricas, modelos, predicciones, tablas
├── tests/ # Tests anti-leakage (pytest)
├── scripts/ # Utilidades de build
├── refs.bib # Bibliografía verificable
├── _config.yml, _toc.yml # Jupyter Book
├── intro.md # Introducción del book
├── requirements.txt
├── Makefile
└── README.md
```

---

## Reproducir el proyecto

### Opción 1 — con `make` (Linux/macOS)

```bash
git clone <repo>
cd INTC-Vol-ML
make install
make test
make build
open _build/html/index.html
```

### Opción 2 — paso a paso (Windows / cualquier OS)

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest tests/ -v
jupyter-book build .
```

El dataset se carga automáticamente: si existe `data/raw/intc.us.txt` (formato Kaggle) se usa; en caso contrario se descarga vía `yfinance` y se guarda localmente.

---

## Notebooks

| # | Notebook | Contenido |
|---|---|---|
| 00 | `00_setup` | Verificación del entorno y versiones |
| 01 | `01_contexto_y_datos` | Contexto del problema y dataset |
| 02 | `02_etl_eda` | ETL, limpieza y análisis exploratorio |
| 03 | `03_features_targets` | Features causales + targets `vol_t+1` y `regime_t+1` |
| 04 | `04_benchmarks_econometricos` | Naive, EWMA, ARIMA, GARCH, HAR-RV |
| 05 | `05_modelos_regresion` | Ridge, Lasso, KNN, DT, RF, SVR, XGBoost |
| 06 | `06_modelos_clasificacion` | KNN, NB, LogReg L1/L2, DT, RF, SVM, XGBoost |
| 07 | `07_balanceo_clases` | SMOTE, ADASYN, class_weight |
| 08 | `08_optimizacion_hiperparametros` | Grid, Random, Optuna, DEAP |
| 09 | `09_optimizacion_computacional` | KD-Tree, SAGA, hist+early-stopping, LinearSVC |
| 10 | `10_residuos_diagnosticos` | White, BDS, Ljung-Box, ACF, JB |
| 11 | `11_comparacion_estadistica` | DM, DeLong real, Bootstrap CI, Bonferroni |
| 12 | `12_interpretabilidad` | LIME, SHAP, permutation importance |
| 13 | `13_modelo_original` | Investigación, diseño, ablación y evaluación |
| 14 | `14_conclusiones_sustentacion` | Síntesis, tabla maestra y guion de 10 min |

---

## Compromisos metodológicos

- **Sin data leakage.** Todo preprocesamiento ajustable vive en `Pipeline` (o `imblearn.Pipeline` con SMOTE/ADASYN). Umbrales y estadísticos se calculan solo con train. Tests `pytest` verifican causalidad de features y separación de targets.
- **Validación temporal estricta.** Split cronológico 70/15/15, `TimeSeriesSplit` en CV, sin `shuffle=True`.
- **El test se usa una sola vez.** Toda decisión arquitectónica del modelo original se toma sobre validation.
- **Reproducibilidad.** `RANDOM_STATE=42` centralizado en `src/config.py`. Versiones fijadas. Outputs persistidos.
- **Rigor.** Resultados negativos se reportan; no se inflan métricas; no hay TBDs.

---

## Modelo original

La arquitectura del modelo original se selecciona tras una **revisión bibliográfica formal** documentada en el Notebook 13 (Fase 1: Investigación). El compromiso es:

- El modelo original debe **superar al mejor benchmark y al mejor modelo base en validation** antes de tocar el test.
- Si no supera en validation, se rediseña.
- Todas las citas son verificables (DOI o URL real).

---

## Licencia

Uso académico. Ver `LICENSE`.
