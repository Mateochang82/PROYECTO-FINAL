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

### 1.1 Motivación: por qué predecir volatilidad

La **volatilidad** de un activo financiero — la magnitud típica de sus fluctuaciones de precio — es una de las variables más importantes en finanzas cuantitativas. Modelarla bien permite calibrar precios de opciones, dimensionar posiciones, calcular *Value-at-Risk* y diseñar estrategias de cobertura. A diferencia del retorno medio, que en mercados eficientes es difícil de pronosticar, la volatilidad exhibe **clustering**: períodos de alta volatilidad tienden a ser seguidos por más alta volatilidad, y lo mismo en regímenes de calma. Este hecho estilizado, documentado desde Engle (1982) y formalizado en Bollerslev (1986), da base predictiva al problema.

### 1.2 Por qué Intel Corporation

Trabajamos sobre **Intel Corporation (INTC)**, listado en NASDAQ. La elección se justifica por tres razones:

1. **Cobertura temporal larga.** El histórico OHLCV disponible cubre más de 25 años, ofreciendo regímenes diversos: burbuja .com, recuperación posterior, crisis financiera 2008 y el período de "Gran Calma" 2013-2017.
2. **Liquidez alta.** Reduce el ruido idiosincrático y hace que la volatilidad observada sea genuina y no producto de microestructura.
3. **Sector relevante.** El sector tecnológico tiene episodios de estrés y cambios estructurales claramente identificables, útiles para evaluar robustez de modelos a cambios de régimen.

### 1.3 Definición formal del problema

Para cada día de trading $t$ disponemos del estado del mercado hasta ese momento (precios apertura, máximo, mínimo, cierre y volumen, más todas las transformaciones causales que de ahí se deriven). Queremos predecir, para el día siguiente $t+1$:

- **Regresión** sobre la volatilidad realizada $\sigma_{t+1} = \mathrm{std}(\{r_{t-w+2}, \dots, r_{t+1}\})$, donde $r_s = \log(p_s / p_{s-1})$ y $w$ es la ventana causal (configuración: $w=5$).
- **Clasificación binaria** del régimen de volatilidad $\rho_{t+1} \in \{0, 1\}$ — *bajo* o *alto* — definido por $\rho_{t+1} = \mathbb{1}\{\sigma_{t+1} > \tau\}$, donde $\tau$ es la **mediana de $\sigma$ calculada únicamente sobre el conjunto de entrenamiento**.

### 1.4 Por qué horizonte de un día

El proyecto académico original de este equipo trabajó con horizontes de 7, 14 y 21 días. En esa configuración el $R^2$ sobre test fue negativo en todos los modelos evaluados — desde regresión lineal hasta LSTM — síntoma de que la varianza condicional de la volatilidad agregada a esos horizontes es tan baja en el período de prueba que predecir la media histórica gana sistemáticamente.

Al fijar el horizonte en un día, recuperamos el efecto de clustering de corto plazo, donde la dependencia serial de la volatilidad es fuerte y predecible. Esta es la corrección metodológica central del nuevo planteamiento del proyecto y permite obtener $R^2$ test de +0.74 con los mejores modelos.

### 1.5 Variables disponibles

**Variables originales del dataset OHLCV diario:**

- **Date**: Fecha de la cotización.
- **Open**: Precio de apertura de la sesión.
- **High**: Precio máximo intradía.
- **Low**: Precio mínimo intradía.
- **Close**: Precio de cierre.
- **Volume**: Número de acciones negociadas en la sesión.
- **OpenInt**: Open interest (no usado, siempre cero en acciones).

**Features causales construidas (31 features, todas con shift(1) para evitar leakage):**

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

### 1.6 Objetivo

El objetivo de este proyecto es la **implementación, comparación y selección de modelos clásicos de Machine Learning** para la predicción de la volatilidad realizada y del régimen de volatilidad de INTC, junto con la **propuesta de un modelo original** llamado HVRF (*Hybrid Volatility Regime Forecaster*). Para cada modelo se evalúa la calidad de la predicción mediante métricas estándar (RMSE, MAE, R², AUC, F1) y se valida estadísticamente mediante pruebas formales (Diebold-Mariano, DeLong, bootstrap con corrección Bonferroni). El proyecto cumple con la rúbrica completa del curso e incorpora análisis de interpretabilidad (LIME, SHAP, permutation importance) sobre los modelos top.

### 1.7 Compromisos metodológicos

- **Sin data leakage.** Todo preprocesamiento ajustable vive dentro de `Pipeline` (o `imblearn.Pipeline` cuando hay SMOTE/ADASYN). Los umbrales y estadísticos se calculan únicamente con el conjunto de entrenamiento. Tests `pytest` verifican que las features no usen información futura y que los targets no aparezcan en X.
- **Validación temporal estricta.** Split cronológico 70/15/15. Validación cruzada con `TimeSeriesSplit`. Ningún `shuffle=True`. Ningún `train_test_split` aleatorio.
- **El test se usa una sola vez.** Toda decisión arquitectónica del modelo original se toma sobre validation. El test final se ejecuta solo con la arquitectura congelada.
- **Reproducibilidad.** `RANDOM_STATE=42` centralizado, versiones fijadas, semillas controladas, outputs persistidos en `outputs/` con esquemas estables.
- **Reproducibilidad y rigor.** Resultados negativos se reportan. Modelos que pierden se documentan. Ningún número fabricado.

### 1.8 Tabla de contenido

- [2. ETL y Análisis Exploratorio de Datos (EDA)](notebooks/02_etl_eda.ipynb)
- [3. Features causales y targets](notebooks/03_features_targets.ipynb)
- [4. Benchmarks econométricos](notebooks/04_benchmarks_econometricos.ipynb)
- [5. Modelos base de regresión](notebooks/05_modelos_regresion.ipynb)
- [6. Modelos base de clasificación](notebooks/06_modelos_clasificacion.ipynb)
- [7. Balanceo de clases](notebooks/07_balanceo_clases.ipynb)
- [8. Optimización de hiperparámetros](notebooks/08_optimizacion_hiperparametros.ipynb)
- [9. Optimización computacional](notebooks/09_optimizacion_computacional.ipynb)
- [10. Análisis de residuos](notebooks/10_residuos_diagnosticos.ipynb)
- [11. Comparación estadística](notebooks/11_comparacion_estadistica.ipynb)
- [12. Interpretabilidad](notebooks/12_interpretabilidad.ipynb)
- [13. Modelo original HVRF](notebooks/13_modelo_original_hvrf.ipynb)
- [14. Conclusiones](notebooks/14_conclusiones_sustentacion.ipynb)
- [15. Referencias bibliográficas](references.md)
