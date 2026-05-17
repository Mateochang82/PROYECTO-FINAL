# 1. PROYECTO FINAL. PREDICCIÓN DE VOLATILIDAD DE INTEL (INTC)

**Desarrollado por:**
Juan Camilo Conrado
Sergio Cadavid
Mateo Chang

**Curso de Machine Learning** — Pregrado en Ciencia de Datos
Universidad del Norte
Docente: Dr. Lihki Rubio
Semestre 2026-1

## 1. Contexto

Antes de comenzar con el análisis exploratorio, es necesario conocer el contexto en el que vamos a trabajar para entender la variable que queremos predecir y la serie temporal sobre la que trabajaremos.

El dataset utilizado para este proyecto proviene de **Kaggle** y puede accederse mediante el siguiente enlace: <https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs>. Se trata de la serie histórica diaria OHLCV (Open, High, Low, Close, Volume) del stock **Intel Corporation (INTC)** desde 1990 hasta 2017, con **7 022 días** de cotización.

### 1.1 Predicción de volatilidad

La volatilidad realizada es una medida estadística de la variabilidad de los rendimientos de un activo financiero. En el contexto del trading y de la gestión de riesgo, predecir la volatilidad del día siguiente permite ajustar el tamaño de las posiciones, fijar niveles de stop-loss, dimensionar primas de opciones y diseñar coberturas adecuadas. La volatilidad presenta dos propiedades empíricas bien documentadas que dificultan su modelado: el efecto **clustering** (períodos turbulentos se agrupan en el tiempo) y la **persistencia** decreciente con el horizonte de predicción.

En este proyecto se busca predecir dos magnitudes complementarias para el día $t+1$, condicionadas a la información disponible hasta el día $t$:

- **Regresión** sobre la volatilidad realizada $\sigma_{t+1}$, definida como la desviación estándar causal de los log-retornos en una ventana corta.
- **Clasificación binaria** del régimen de volatilidad $r_{t+1} \in \{0, 1\}$ — *bajo* o *alto* — separado por la mediana de la volatilidad calculada exclusivamente sobre el conjunto de entrenamiento.

### 1.2 Variables disponibles

En esta sección presentamos las variables originales del dataset y las features causales construidas para el modelado:

**Variables originales (OHLCV diario):**

- **Date**: Fecha de la cotización.
- **Open**: Precio de apertura de la sesión.
- **High**: Precio máximo intradía.
- **Low**: Precio mínimo intradía.
- **Close**: Precio de cierre.
- **Volume**: Número de acciones negociadas en la sesión.
- **OpenInt**: Open interest (no usado, siempre cero en acciones).

**Features causales construidas (31 features, todas con shift(1)):**

- **Retornos rezagados**: `ret_lag_1`, `ret_lag_2`, `ret_lag_3`, `ret_lag_5`, `ret_lag_10` — log-retornos de días anteriores.
- **Volatilidad rolling**: `vol_5`, `vol_10`, `vol_22` — desviación estándar de retornos en ventanas de 5, 10 y 22 días.
- **Componentes HAR-RV**: `vol_d`, `vol_w`, `vol_m` — agregación diaria, semanal y mensual de la volatilidad realizada según Corsi (2009).
- **Momentum**: `momentum_5`, `momentum_10`, `momentum_22` — variación porcentual del precio en distintas ventanas.
- **Rango de precio**: `range_hl`, `true_range`, `atr_14` — métricas del rango de cotización diario y su promedio.
- **Volumen**: `volume_ratio`, `volume_lag_1`, `volume_lag_5`, `log_volume` — features derivadas del volumen negociado.
- **Medias móviles**: `sma_5`, `sma_10`, `sma_22` — promedio del precio de cierre.
- **Indicadores técnicos**: `rsi_14`, `bb_width` — RSI y ancho de bandas de Bollinger.
- **Asimetría rolling**: `skew_22`, `kurt_22` — momentos de orden 3 y 4 de los retornos.
- **Calendario**: `day_of_week`, `month` — variables temporales discretas.

**Variables objetivo:**

- `target_vol`: Volatilidad realizada del día $t+1$ (regresión).
- `target_regime`: Régimen del día $t+1$, binario (clasificación).

### 1.3 Objetivo

El objetivo de este proyecto es la **implementación, comparación y selección de modelos clásicos de Machine Learning** para la predicción de la volatilidad realizada y del régimen de volatilidad de INTC, junto con la **propuesta de un modelo original** (HVRF — *Hybrid Volatility Regime Forecaster*). Para cada modelo se evalúa la calidad de la predicción mediante métricas estándar (RMSE, MAE, R², AUC, F1) y se valida estadísticamente mediante pruebas formales (Diebold-Mariano, DeLong, bootstrap con corrección Bonferroni). El proyecto cumple con la rúbrica completa del curso e incorpora análisis de interpretabilidad (LIME, SHAP, permutation importance) sobre los modelos top.

### 1.4 Compromisos metodológicos

- **Sin data leakage.** Todo preprocesamiento ajustable vive dentro de `Pipeline` (o `imblearn.Pipeline` cuando hay SMOTE/ADASYN). Los umbrales y estadísticos se calculan únicamente con el conjunto de entrenamiento. Tests `pytest` verifican que las features no usen información futura y que los targets no aparezcan en X.
- **Validación temporal estricta.** Split cronológico 70/15/15. Validación cruzada con `TimeSeriesSplit`. Ningún `shuffle=True`. Ningún `train_test_split` aleatorio.
- **El test se usa una sola vez.** Toda decisión arquitectónica del modelo original se toma sobre validation. El test final se ejecuta solo con la arquitectura congelada.
- **Reproducibilidad.** `RANDOM_STATE=42` centralizado, versiones fijadas, semillas controladas, outputs persistidos en `outputs/` con esquemas estables.
- **Honestidad estadística.** Resultados negativos se reportan. Modelos que pierden se documentan. Ningún número fabricado.

### 1.5 Tabla de contenido

- [2. Contexto y dataset](notebooks/01_contexto_y_datos.ipynb)
- [3. ETL y análisis exploratorio (EDA)](notebooks/02_etl_eda.ipynb)
- [4. Features causales y targets](notebooks/03_features_targets.ipynb)
- [5. Benchmarks econométricos](notebooks/04_benchmarks_econometricos.ipynb)
- [6. Modelos base de regresión](notebooks/05_modelos_regresion.ipynb)
- [7. Modelos base de clasificación](notebooks/06_modelos_clasificacion.ipynb)
- [8. Balanceo de clases](notebooks/07_balanceo_clases.ipynb)
- [9. Optimización de hiperparámetros](notebooks/08_optimizacion_hiperparametros.ipynb)
- [10. Optimización computacional](notebooks/09_optimizacion_computacional.ipynb)
- [11. Análisis de residuos](notebooks/10_residuos_diagnosticos.ipynb)
- [12. Comparación estadística](notebooks/11_comparacion_estadistica.ipynb)
- [13. Interpretabilidad](notebooks/12_interpretabilidad.ipynb)
- [14. Modelo original HVRF](notebooks/13_modelo_original_hvrf.ipynb)
- [15. Conclusiones y guion de sustentación](notebooks/14_conclusiones_sustentacion.ipynb)
- [16. Referencias bibliográficas](references.md)
