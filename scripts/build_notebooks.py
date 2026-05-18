"""
Genera los 14 notebooks del proyecto INTC-Vol-ML.

Los notebooks 00-03 contienen código funcional ejecutable.
Los notebooks 04-14 son stubs estructurados con propósito y secciones marcadas.

Uso:
    python scripts/build_notebooks.py
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def save(nb, name: str):
    nbf.write(nb, NB_DIR / f"{name}.ipynb")
    print(f"  ✓ {name}.ipynb")


# =====================================================================
#  NOTEBOOK 00 — Setup
# =====================================================================
def make_00_setup():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# 0. Verificación del entorno

Este notebook verifica que el entorno está correctamente configurado para
ejecutar el resto del proyecto: versiones de paquetes clave, presencia
de directorios, valor de la semilla aleatoria y carga del módulo `src/`.

Cualquier discrepancia debería resolverse antes de seguir.
"""),

        md("## 0.1 Versiones del entorno"),
        code("""
import sys, platform
import numpy, pandas, sklearn, scipy, matplotlib, seaborn
print(f"Python      : {sys.version.split()[0]}")
print(f"Platform    : {platform.system()} {platform.release()}")
print(f"numpy       : {numpy.__version__}")
print(f"pandas      : {pandas.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"scipy       : {scipy.__version__}")
print(f"matplotlib  : {matplotlib.__version__}")
print(f"seaborn     : {seaborn.__version__}")
try:
    import xgboost; print(f"xgboost     : {xgboost.__version__}")
except ImportError:
    print("xgboost     : NO instalado")
try:
    import imblearn; print(f"imblearn    : {imblearn.__version__}")
except ImportError:
    print("imblearn    : NO instalado")
try:
    import statsmodels; print(f"statsmodels : {statsmodels.__version__}")
except ImportError:
    print("statsmodels : NO instalado")
try:
    import arch; print(f"arch        : {arch.__version__}")
except ImportError:
    print("arch        : NO instalado")
"""),

        md("## 0.2 Carga del paquete `src/` y configuración global"),
        code("""
import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
print("RANDOM_STATE :", config.RANDOM_STATE)
print("PROJECT_ROOT :", config.PROJECT_ROOT)
print("DATE_START   :", config.DATE_START)
print("DATE_END     :", config.DATE_END)
print("HORIZON      :", config.HORIZON, "día(s)")
print("VOL_WINDOW   :", config.VOL_WINDOW, "días")
print("Splits       :", f"{config.TRAIN_FRAC:.0%} / {config.VAL_FRAC:.0%} / {config.TEST_FRAC:.0%}")
"""),

        md("## 0.3 Estructura de directorios"),
        code("""
config.ensure_dirs()
for d in [config.RAW_DIR, config.PROCESSED_DIR, config.FIGURES_DIR,
          config.METRICS_DIR, config.MODELS_DIR, config.PREDICTIONS_DIR,
          config.TABLES_DIR]:
    print(f"{'✓' if d.exists() else '✗'} {d.relative_to(config.PROJECT_ROOT)}")
"""),

        md("""
## 0.4 Verificación rápida del pipeline de datos

Llamamos a `load_intc`. Si existe el archivo local lo usa; en caso contrario,
descarga vía yfinance y lo cachea.
"""),
        code("""
from src.data_loader import load_intc
df = load_intc()
print(f"Filas cargadas : {len(df):,}")
print(f"Rango temporal : {df['date'].min().date()} → {df['date'].max().date()}")
print()
print(df.head())
print()
print(df.dtypes)
"""),

        md("""
## 0.5 Conclusión de la verificación

Si esta celda imprime tabla y rango temporal sin error, el entorno está
listo para los notebooks siguientes. Cualquier paquete faltante puede
instalarse con `pip install -r requirements.txt` desde la raíz del repo.
"""),
    ]
    save(nb, "00_setup")


# =====================================================================
#  NOTEBOOK 01 — Contexto y datos
# =====================================================================
def make_01_contexto():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# 1. Contexto del problema y dataset

## 1.1 Motivación

La **volatilidad** de un activo financiero — la magnitud típica de sus
fluctuaciones de precio — es una de las variables más importantes en
finanzas cuantitativas. Modelarla bien permite calibrar precios de
opciones, dimensionar posiciones, calcular *Value-at-Risk* y diseñar
estrategias de cobertura. A diferencia del retorno medio, que en
mercados eficientes es difícil de pronosticar, la volatilidad exhibe
**clustering**: períodos de alta volatilidad tienden a ser seguidos
por más alta volatilidad, y lo mismo en regímenes de calma. Este hecho
estilizado, documentado desde {cite}`engle1982autoregressive` y
formalizado en {cite}`bollerslev1986generalized`, da base predictiva
al problema.

## 1.2 Elección del activo

Trabajamos sobre **Intel Corporation (INTC)**, listado en NASDAQ. La
elección se justifica por tres razones:

1. **Cobertura temporal larga.** El histórico OHLCV disponible cubre
   más de 30 años, ofreciendo regímenes diversos (burbuja .com,
   crisis 2008, COVID-19, ciclo del semiconductor).
2. **Liquidez alta.** Reduce el ruido idiosincrático y hace que la
   volatilidad observada sea genuina y no producto de microestructura.
3. **Sector relevante.** El sector tecnológico tiene episodios de
   estrés y cambios estructurales claramente identificables, útiles
   para evaluar robustez de modelos a cambios de régimen.

## 1.3 Problema

Para cada día de trading $t$ disponemos del estado del mercado hasta
ese momento (precios apertura, máximo, mínimo, cierre y volumen, más
todas las transformaciones causales que de ahí se deriven). Queremos
predecir, para el día siguiente $t+1$:

- **Tarea de regresión:** la volatilidad realizada
  $\\sigma_{t+1} = \\mathrm{std}(\\{r_{t-w+2}, \\dots, r_{t+1}\\})$,
  donde $r_s = \\log(p_s / p_{s-1})$ y $w$ es la ventana causal
  (configuración: $w=5$).
- **Tarea de clasificación binaria:** el régimen
  $\\rho_{t+1} \\in \\{0, 1\\}$ — bajo o alto — definido por
  $\\rho_{t+1} = \\mathbb{1}\\{\\sigma_{t+1} > \\tau\\}$, donde $\\tau$
  es la **mediana de $\\sigma$ calculada únicamente sobre el conjunto
  de entrenamiento**.

## 1.4 Por qué horizonte de un día

El proyecto académico original de este equipo trabajó con horizontes
de 7, 14 y 21 días. En esa configuración el $R^2$ sobre test fue
negativo en todos los modelos evaluados — desde regresión lineal
hasta LSTM — síntoma de que la varianza condicional de la volatilidad
agregada a esos horizontes es tan baja en el período de prueba que
predecir la media histórica gana sistemáticamente.

Al fijar el horizonte en un día, recuperamos el efecto de clustering
de corto plazo, donde la dependencia serial de la volatilidad es
fuerte y predecible. Esta es la corrección metodológica central del
nuevo planteamiento.

## 1.5 Variables del dataset

El dataset crudo es la serie diaria OHLCV de INTC:

| Columna | Descripción |
|---|---|
| `date` | Fecha de la sesión |
| `open` | Precio de apertura |
| `high` | Máximo intradiario |
| `low` | Mínimo intradiario |
| `close` | Precio de cierre |
| `volume` | Volumen negociado (acciones) |

A partir de estas seis variables construimos, en el capítulo 3, ~30
features causales que capturan retornos rezagados, volatilidad
rolling, componentes HAR-RV (Corsi 2009), momentum, rangos de precio,
indicadores técnicos y variables de calendario.

## 1.6 Conexión con el capítulo 2

En el siguiente capítulo descargamos los datos (con `yfinance` como
fallback automático), aplicamos limpieza básica, verificamos integridad
y producimos el primer análisis exploratorio orientado a justificar
la elección del problema y a documentar los hechos estilizados de
INTC en los que descansará todo el modelado posterior.
"""),
    ]
    save(nb, "01_contexto_y_datos")


# =====================================================================
#  NOTEBOOK 02 — ETL + EDA
# =====================================================================
def make_02_etl_eda():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# 2. ETL y análisis exploratorio (EDA)

## Propósito

Cargar el dataset OHLCV de INTC, aplicar limpieza básica, persistir la
versión procesada y producir el análisis exploratorio que justifique la
elección del problema y documente los hechos estilizados relevantes.

## Conexión con el capítulo anterior

En el capítulo 1 definimos el problema (regresión de $\\sigma_{t+1}$ y
clasificación de $\\rho_{t+1}$, ambos a horizonte de un día) y la fuente
de datos. Aquí materializamos esa fuente como un objeto tabular limpio
y reproducible.
"""),

        md("## 2.1 Carga del dataset"),
        code("""
import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.data_loader import load_intc
from src.viz import set_style, savefig, PALETTE
from src.io_utils import save_parquet

set_style()

df = load_intc()
print(f"Filas cargadas : {len(df):,}")
print(f"Rango temporal : {df['date'].min().date()} → {df['date'].max().date()}")
df.head()
"""),

        md("""
**Interpretación.** El loader devuelve un DataFrame ordenado
cronológicamente, sin valores nulos en `close`, con las seis columnas
OHLCV. El rango temporal cubre más de tres décadas — suficientes para
ver múltiples regímenes de volatilidad.
"""),

        md("## 2.2 Limpieza e integridad"),
        code("""
# 2.2.1 Conteo de NaN
print("Valores NaN por columna:")
print(df.isna().sum())
print()

# 2.2.2 Continuidad de fechas hábiles
date_diff = df["date"].diff().dt.days.dropna()
print(f"Saltos entre fechas (días calendario):")
print(date_diff.describe())
print()

# 2.2.3 Coherencia OHLC: high >= max(open, close), low <= min(open, close)
bad_high = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
bad_low = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
print(f"Filas con high < max(open, close) : {bad_high}")
print(f"Filas con low  > min(open, close) : {bad_low}")
"""),

        md("""
**Interpretación.** Las verificaciones de integridad cubren los tres
problemas clásicos de series OHLCV: nulos, discontinuidades temporales
inesperadas y violaciones de la relación lógica entre máximo, mínimo,
apertura y cierre. La distribución de saltos entre fechas debe
concentrarse en 1 día (sesión a sesión) con valores de 3 cuando hay fin
de semana, y ocasionalmente 4+ cuando hay festivos.
"""),

        md("## 2.3 Persistencia del dataset limpio"),
        code("""
config.ensure_dirs()
save_parquet(df, config.PROCESSED_FILE)
print(f"Guardado en: {config.PROCESSED_FILE.relative_to(config.PROJECT_ROOT)}")
print(f"Tamaño: {config.PROCESSED_FILE.stat().st_size / 1024:.1f} KB")
"""),

        md("## 2.4 Serie de precios"),
        code("""
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["date"], df["close"], color=PALETTE["primary"], linewidth=1)
ax.set_title("INTC — Precio de cierre diario")
ax.set_xlabel("Fecha")
ax.set_ylabel("Cierre (USD)")
savefig(config.FIGURES_DIR / "02_close_price.png", fig)
plt.show()
"""),

        md("""
**Interpretación.** La serie de precios muestra el ciclo clásico de
INTC: crecimiento sostenido durante los 90, pico de la burbuja .com a
principios de los 2000, corrección prolongada, recuperación gradual y
volatilidad reciente del sector semiconductor. Estos cambios de régimen
en precio se traducen, como veremos, en cambios de régimen en
volatilidad.
"""),

        md("## 2.5 Log-retornos"),
        code("""
df["log_return"] = np.log(df["close"] / df["close"].shift(1))

fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(df["date"], df["log_return"], color=PALETTE["primary"], linewidth=0.5)
axes[0].set_title("INTC — Log-retornos diarios")
axes[0].set_ylabel("log(p_t / p_{t-1})")
axes[0].set_xlabel("")

axes[1].hist(df["log_return"].dropna(), bins=120, color=PALETTE["primary"], alpha=0.85)
axes[1].set_title("Distribución de log-retornos")
axes[1].set_xlabel("Log-retorno diario")
axes[1].set_ylabel("Frecuencia")
plt.tight_layout()
savefig(config.FIGURES_DIR / "02_log_returns.png", fig)
plt.show()

print(df["log_return"].describe())
print(f"\\nAsimetría : {df['log_return'].skew():.4f}")
print(f"Curtosis  : {df['log_return'].kurt():.4f}")
"""),

        md("""
**Interpretación.** Tres hechos estilizados quedan visualmente claros:

1. **Heterocedasticidad.** Hay períodos densos (alta volatilidad) y
   períodos calmos (baja volatilidad). Esto es el clustering que motiva
   GARCH/HAR-RV y todo el aparato de modelado de volatilidad.
2. **Colas pesadas.** La curtosis empírica es muy superior a 3 (la de
   una normal), indicando colas pesadas — eventos extremos son más
   frecuentes de lo que una distribución gaussiana predeciría.
3. **Asimetría ligeramente negativa.** Es típico de equities: caídas
   grandes son algo más frecuentes que subidas grandes equivalentes.

Estos hechos justifican usar la volatilidad realizada como target
predecible — su componente persistente domina sobre el ruido.
"""),

        md("## 2.6 Volatilidad realizada"),
        code("""
df["vol_5d"] = df["log_return"].rolling(5, min_periods=5).std()
df["vol_22d"] = df["log_return"].rolling(22, min_periods=22).std()

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["date"], df["vol_5d"], color=PALETTE["primary"], linewidth=0.8,
        label="Vol realizada (ventana 5d)")
ax.plot(df["date"], df["vol_22d"], color=PALETTE["secondary"], linewidth=1.0,
        label="Vol realizada (ventana 22d)", alpha=0.9)
ax.set_title("INTC — Volatilidad realizada causal a dos ventanas")
ax.set_xlabel("Fecha")
ax.set_ylabel("σ (desv. estándar de log-retornos)")
ax.legend()
savefig(config.FIGURES_DIR / "02_realized_volatility.png", fig)
plt.show()
"""),

        md("""
**Interpretación.** La volatilidad realizada exhibe persistencia: los
picos coinciden con eventos macro conocidos (.com bust 2000-2002,
crisis financiera 2008-2009, COVID-19 marzo 2020). La serie de 22 días
suaviza el ruido y revela mejor los regímenes; la de 5 días captura
cambios rápidos. En el modelado base usamos $w=5$ porque mantiene la
señal sin sobre-suavizar el clustering reciente que queremos
pronosticar.
"""),

        md("## 2.7 Autocorrelación de retornos y de volatilidad"),
        code("""
from statsmodels.graphics.tsaplots import plot_acf

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
plot_acf(df["log_return"].dropna(), lags=40, ax=axes[0])
axes[0].set_title("ACF de log-retornos")
plot_acf(df["log_return"].dropna().pow(2), lags=40, ax=axes[1])
axes[1].set_title("ACF de log-retornos al cuadrado (proxy de volatilidad)")
savefig(config.FIGURES_DIR / "02_acf_returns_vs_vol.png", fig)
plt.show()
"""),

        md("""
**Interpretación.** Este es uno de los gráficos más importantes del
proyecto. Los **retornos** no muestran autocorrelación significativa
(consistente con eficiencia débil de mercado), pero los retornos al
cuadrado — proxy de la magnitud — muestran autocorrelación positiva y
persistente en muchos lags. Esto **valida empíricamente** la hipótesis
de que la volatilidad es pronosticable, mientras que el signo del
retorno no lo es. Es la base teórica que justifica que nuestro proyecto
modele $\\sigma_{t+1}$ y no $r_{t+1}$.
"""),

        md("## 2.8 Volumen"),
        code("""
df["log_volume"] = np.log(df["volume"].replace(0, np.nan))

fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(df["date"], df["volume"] / 1e6, color=PALETTE["primary"], linewidth=0.4)
axes[0].set_title("INTC — Volumen diario (millones de acciones)")
axes[0].set_ylabel("Volumen (M)")
axes[1].plot(df["date"], df["log_volume"], color=PALETTE["primary"], linewidth=0.5)
axes[1].set_title("log(Volumen)")
axes[1].set_xlabel("Fecha")
axes[1].set_ylabel("log(volumen)")
plt.tight_layout()
savefig(config.FIGURES_DIR / "02_volume.png", fig)
plt.show()
"""),

        md("""
**Interpretación.** El volumen presenta saltos estructurales claros
(particularmente en torno a desdoblamientos y eventos corporativos) y
una tendencia decreciente en años recientes, consistente con el
cambio en la base accionaria del sector tech. Trabajamos con
`log(volume)` y con un ratio `volume / volume.rolling(20).mean()` como
features, para que el modelo capture sorpresas de volumen relativas y
no niveles absolutos que varían con el tiempo.
"""),

        md("""
## 2.9 Cierre del capítulo y conexión con el capítulo 3

El dataset está limpio, validado, persistido y visualmente
caracterizado. Los hallazgos clave que importan para el modelado son:

- La volatilidad de INTC tiene clustering fuerte y predecible (ACF de
  retornos al cuadrado significativa).
- Las colas son pesadas — los modelos deben ser robustos a *outliers*.
- Múltiples regímenes a lo largo del tiempo (crisis vs calma).
- Volumen con cambios estructurales — usar transformaciones relativas.

En el **capítulo 3** construimos las ~30 features causales y los dos
targets (`target_vol` y `target_regime`), aplicamos el split temporal
70/15/15 y persistimos los conjuntos `train/val/test` para los
notebooks de modelado. Allí también ejecutamos los tests anti-leakage
que blindan la pipeline.
"""),
    ]
    save(nb, "02_etl_eda")


# =====================================================================
#  NOTEBOOK 03 — Features + Targets
# =====================================================================
def make_03_features_targets():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# 3. Features causales y targets

## Propósito

Construir las ~30 features causales y los dos targets (`target_vol` y
`target_regime`), aplicar el particionamiento temporal 70/15/15,
verificar que las pruebas anti-leakage pasan, y persistir los
conjuntos `train`, `val` y `test` listos para los notebooks de
modelado.

## Conexión con el capítulo anterior

En el capítulo 2 caracterizamos los hechos estilizados de INTC. Aquí
codificamos esos hechos en variables predictoras causales y definimos
formalmente lo que el modelo debe predecir.
"""),

        md("## 3.1 Carga del dataset limpio"),
        code("""
import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src import config
from src.features import build_features, feature_columns
from src.targets import add_targets
from src.splits import temporal_split, split_summary
from src.io_utils import save_parquet, load_parquet, save_json

df = load_parquet(config.PROCESSED_FILE)
print(f"Dataset cargado: {len(df):,} filas")
df.head()
"""),

        md("## 3.2 Construcción de features"),
        code("""
df_feat = build_features(df)
feats = feature_columns(df_feat)
print(f"Features construidas : {len(feats)}")
print()
print("Lista completa:")
for i, c in enumerate(feats, 1):
    print(f"  {i:2d}. {c}")
"""),

        md("""
**Interpretación.** Las ~30 features cubren cinco familias funcionales:

1. **Memoria de retornos** (`ret_lag_*`, `momentum_*`): capturan la
   dinámica autoregresiva de los rendimientos.
2. **Memoria de volatilidad** (`vol_*`, `vol_d/w/m`): incluyen los
   componentes HAR-RV de Corsi (2009) — volatilidad diaria, semanal y
   mensual — que constituirán uno de los benchmarks más fuertes del
   capítulo 4.
3. **Rango y actividad** (`range_hl`, `true_range`, `atr_14`):
   complementan la volatilidad con información intradiaria.
4. **Volumen** (`log_volume`, `volume_ratio`, `volume_lag_1`): capturan
   sorpresas de actividad relativas al promedio reciente.
5. **Precio y momentum técnico** (`sma_*`, `px_dist_sma_*`): distancias
   normalizadas a medias móviles.
6. **Forma de la distribución reciente** (`skew_22`, `kurt_22`):
   asimetría y curtosis rolling como proxy de leverage y cola.
7. **Calendario** (`dow`, `month`): efectos estacionales.

Toda feature respeta causalidad estricta: ninguna usa información de
$t+1$ en adelante. Esto se verifica formalmente en
`tests/test_features_causal.py`.
"""),

        md("## 3.3 Construcción de targets"),
        code("""
# Paso 1: construir splits temporales sobre el dataset con features.
# Esto nos da una train_mask que usaremos para calcular el umbral del régimen.
train_raw, val_raw, test_raw = temporal_split(df_feat)
print(split_summary(train_raw, val_raw, test_raw))
"""),

        code("""
# Paso 2: construir train_mask para evitar leakage en el umbral del régimen.
n = len(df_feat)
train_mask = np.zeros(n, dtype=bool)
train_mask[: len(train_raw)] = True

# Paso 3: añadir targets usando solo train para definir el umbral.
df_all = add_targets(df_feat, horizon=config.HORIZON, train_mask=train_mask)

print(f"Umbral del régimen : {df_all.attrs['regime_threshold']:.6f}")
print(f"Origen del umbral   : {df_all.attrs['regime_threshold_source']}")
print(f"Horizonte           : {df_all.attrs['regime_horizon']} día(s)")
"""),

        md("""
**Interpretación.** El umbral se calcula como la **mediana de
`target_vol` en el conjunto de entrenamiento**. La etiqueta
`'train_quantile'` en `regime_threshold_source` confirma trazabilidad.
Cualquier otra fuente (`'global_quantile_NOT_RECOMMENDED'` o `'explicit'`)
sería una señal de alerta metodológica.
"""),

        md("## 3.4 Filtrado de filas con NaN inducidos por features y targets"),
        code("""
# Las features rolling y el target shift(-h) introducen NaN al principio
# y al final respectivamente. Los eliminamos antes del split definitivo.
df_clean = df_all.dropna(subset=feats + ["target_vol", "target_regime"]).reset_index(drop=True)
print(f"Filas antes del dropna : {len(df_all):,}")
print(f"Filas después          : {len(df_clean):,}")
print(f"Diferencia             : {len(df_all) - len(df_clean):,}")
"""),

        md("""
**Interpretación.** Las primeras ~50 filas se pierden por el burn-in
de las features rolling largas (SMA-50, kurtosis-22, etc.) y la
**última fila** se pierde porque su `target_vol` requeriría conocer
el día $t+1$ que no existe. Esta pérdida es esperable y necesaria.
"""),

        md("## 3.5 Split temporal 70 / 15 / 15"),
        code("""
train, val, test = temporal_split(df_clean)
print(split_summary(train, val, test))
"""),

        md("""
**Interpretación.** Las tres particiones son disjuntas y cronológicas.
Cualquier intento futuro de mezclar pasado y futuro sería capturado por
`tests/test_splits_temporal.py`.
"""),

        md("## 3.6 Distribución del target de regresión por subconjunto"),
        code("""
import matplotlib.pyplot as plt
from src.viz import PALETTE, set_style
set_style()

fig, ax = plt.subplots(figsize=(11, 4))
for subset, color, label in [(train, PALETTE["train"], "train"),
                              (val,   PALETTE["val"],   "val"),
                              (test,  PALETTE["test"],  "test")]:
    ax.hist(subset["target_vol"], bins=60, alpha=0.5, color=color,
            label=f"{label} (n={len(subset):,})", density=True)
ax.axvline(df_all.attrs["regime_threshold"], color=PALETTE["secondary"],
           linestyle="--", linewidth=1.5, label="umbral régimen (train)")
ax.set_title("Distribución de target_vol por subconjunto")
ax.set_xlabel("vol_t+1")
ax.set_ylabel("densidad")
ax.legend()
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "03_target_vol_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
"""),

        md("""
**Interpretación.** Las distribuciones de `target_vol` se solapan
parcialmente entre train, val y test, pero hay desplazamientos
visibles — particularmente en test, donde períodos recientes de
volatilidad atípica pueden producir colas distintas. Este *covariate
shift* moderado es esperable en finanzas y es exactamente la razón
por la que evaluamos en test solo al final, para que la decisión de
modelo no se sobreajuste a esas particularidades.
"""),

        md("## 3.7 Balance de clases en target_regime"),
        code("""
for name, subset in [("train", train), ("val", val), ("test", test)]:
    counts = subset["target_regime"].value_counts().sort_index()
    p_pos = counts.get(1, 0) / counts.sum()
    print(f"{name:5s} : 0={counts.get(0,0):>5d}  1={counts.get(1,0):>5d}  "
          f"P(régimen alto)={p_pos:.3f}")
"""),

        md("""
**Interpretación.** Por construcción del umbral (mediana de train), el
conjunto de entrenamiento está casi perfectamente balanceado al 50/50.
Val y test pueden alejarse del balance — esto es información útil:
indica si el régimen "alto" se ha vuelto más o menos común con el
tiempo. Es uno de los hallazgos que el capítulo 7 (balanceo de clases)
explotará explícitamente para evaluar si SMOTE/ADASYN aportan algo en
este problema.
"""),

        md("## 3.8 Persistencia de splits para los notebooks de modelado"),
        code("""
# Guardar features.parquet (todo el dataset con features+targets)
save_parquet(df_clean, config.FEATURES_FILE)

# Guardar splits separados — los notebooks de modelado leen estos archivos.
save_parquet(train, config.PROCESSED_DIR / "train.parquet")
save_parquet(val,   config.PROCESSED_DIR / "val.parquet")
save_parquet(test,  config.PROCESSED_DIR / "test.parquet")

# Guardar lista de features y umbral del régimen para trazabilidad.
save_json({
    "feature_columns": feats,
    "regime_threshold": float(df_all.attrs["regime_threshold"]),
    "regime_threshold_source": df_all.attrs["regime_threshold_source"],
    "horizon": config.HORIZON,
    "vol_window": config.VOL_WINDOW,
    "train_size": len(train),
    "val_size": len(val),
    "test_size": len(test),
}, config.PROCESSED_DIR / "metadata.json")

print("Archivos persistidos:")
for f in sorted(config.PROCESSED_DIR.glob("*")):
    if f.is_file():
        print(f"  {f.name}  ({f.stat().st_size/1024:.1f} KB)")
"""),

        md("## 3.9 Verificación final con tests anti-leakage"),
        code("""
# Re-ejecutamos los tests desde el notebook para evidencia visible.
import subprocess
result = subprocess.run(
    ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
    cwd=str(PROJECT_ROOT), capture_output=True, text=True
)
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-500:])
"""),

        md("""
## 3.10 Cierre del capítulo y conexión con el capítulo 4

Tenemos: features causales verificadas, targets sin leakage, splits
cronológicos disjuntos y archivos persistidos. La base está blindada.

En el **capítulo 4** ejecutamos los **benchmarks econométricos**:
- Naive (último valor)
- Rolling mean
- EWMA (RiskMetrics, J.P. Morgan 1996)
- ARIMA (Box-Jenkins)
- GARCH(1,1) ({cite}`bollerslev1986generalized`)
- **HAR-RV** ({cite}`corsi2009simple`) — el estándar de oro
  contemporáneo para volatilidad realizada

Estos benchmarks marcan la barra que cualquier modelo ML deberá
superar para ser declarado mejor.
"""),
    ]
    save(nb, "03_features_targets")


# =====================================================================
#  NOTEBOOKS 04 — 14: stubs estructurados con propósito y secciones
# =====================================================================
STUBS = [
    ("04_benchmarks_econometricos", """
# 4. Benchmarks econométricos

## Propósito

Establecer una línea de comparación rigurosa con seis benchmarks de
series de tiempo financieras:

1. **Naive** — el último valor observado.
2. **Rolling mean** — media móvil de la volatilidad reciente.
3. **EWMA** — suavizado exponencial (RiskMetrics).
4. **ARIMA** — Box-Jenkins sobre `vol_realized`.
5. **GARCH(1,1)** — {cite}`bollerslev1986generalized`, estándar de oro
   para heterocedasticidad condicional.
6. **HAR-RV** — {cite}`corsi2009simple`, descomposición diaria/semanal/
   mensual.

Cada benchmark se entrena en `train`, se ajusta en `val` (si tiene
hiperparámetros) y se evalúa una sola vez en `test` al final del libro.

## Secciones del notebook

- 4.1 Naive
- 4.2 Rolling mean
- 4.3 EWMA
- 4.4 ARIMA
- 4.5 GARCH(1,1)
- 4.6 HAR-RV (Corsi 2009)
- 4.7 Tabla comparativa (RMSE, MAE)
- 4.8 Visualización: real vs predicho en val
- 4.9 Cierre y conexión con el capítulo 5

> **Estado:** stub. Implementación en el siguiente turno.
"""),

    ("05_modelos_regresion", """
# 5. Modelos base de regresión

## Propósito

Entrenar y evaluar siete modelos ML obligatorios por la rúbrica sobre
`target_vol`, todos encapsulados en `Pipeline` para garantizar
ausencia de leakage al hacer CV temporal:

1. Ridge
2. Lasso
3. KNN Regressor
4. Decision Tree Regressor
5. Random Forest Regressor
6. SVR
7. XGBoost Regressor

## Secciones

- 5.1 Carga de splits y verificación
- 5.2 Pipelines por modelo
- 5.3 Entrenamiento con CV temporal (TimeSeriesSplit)
- 5.4 Métricas en val: RMSE, MAE, R²
- 5.5 Gráficos real vs predicho
- 5.6 Tabla comparativa
- 5.7 Selección de top 2-3 para optimización en el capítulo 8
- 5.8 Cierre y conexión con el capítulo 6

> **Estado:** stub.
"""),

    ("06_modelos_clasificacion", """
# 6. Modelos base de clasificación

## Propósito

Entrenar y evaluar ocho modelos ML obligatorios sobre `target_regime`:

1. KNN Classifier
2. Gaussian Naive Bayes
3. Logistic Regression L1
4. Logistic Regression L2
5. Decision Tree Classifier
6. Random Forest Classifier
7. SVM
8. XGBoost Classifier

## Secciones

- 6.1 Carga y verificación de clases en cada split
- 6.2 Pipelines por modelo
- 6.3 Entrenamiento con CV temporal
- 6.4 Métricas: Accuracy, Precision, Recall, F1, AUC
- 6.5 Matrices de confusión y curvas ROC
- 6.6 Justificación de la métrica principal (AUC) en val
- 6.7 Tabla comparativa
- 6.8 Cierre y conexión con el capítulo 7

> **Estado:** stub.
"""),

    ("07_balanceo_clases", """
# 7. Balanceo de clases

## Propósito

Comparar tres estrategias de balanceo dentro de `imblearn.Pipeline`,
todas aplicadas únicamente sobre los folds de entrenamiento:

1. SMOTE ({cite}`chawla2002smote`)
2. ADASYN
3. `class_weight='balanced'`

## Secciones

- 7.1 Distribución real de clases en train/val/test (recordatorio)
- 7.2 SMOTE dentro de imblearn.Pipeline
- 7.3 ADASYN dentro de imblearn.Pipeline
- 7.4 class_weight='balanced' en modelos lineales y de árbol
- 7.5 Tabla comparativa: con y sin balanceo
- 7.6 Discusión: si el balanceo aporta o no en este problema
- 7.7 Conexión con el capítulo 8

> **Estado:** stub.
"""),

    ("08_optimizacion_hiperparametros", """
# 8. Optimización de hiperparámetros

## Propósito

Comparar empíricamente cuatro métodos de búsqueda sobre los 2-3
modelos más fuertes seleccionados en los capítulos 5 y 6:

1. Grid Search
2. Random Search
3. Optimización Bayesiana con **Optuna**
4. Algoritmos Genéticos con **DEAP**

## Secciones

- 8.1 Espacios de búsqueda justificados (escala log donde aplica)
- 8.2 Grid Search (CV temporal)
- 8.3 Random Search (CV temporal)
- 8.4 Optuna (pipeline manual dentro de cada `objective` — sin leakage)
- 8.5 DEAP (pipeline manual dentro de cada evaluación de individuo)
- 8.6 Tabla comparativa: mejor métrica, # evaluaciones, tiempo
- 8.7 Gráficos de convergencia (Optuna trials, DEAP fitness por generación)
- 8.8 Mejor configuración global y modelos optimizados persistidos
- 8.9 Conexión con el capítulo 9

> **Estado:** stub.
"""),

    ("09_optimizacion_computacional", """
# 9. Optimización computacional

## Propósito

Mostrar tradeoffs de eficiencia para los modelos seleccionados:

- KNN: KD-Tree vs Ball Tree vs brute
- Ridge/Lasso: solver `saga` para datasets grandes
- XGBoost: `tree_method='hist'` + early stopping
- SVM: comparación con `LinearSVC` y `SGDClassifier`

## Secciones

- 9.1 Métricas medidas: tiempo, # iteraciones, RMSE/AUC final
- 9.2 KNN: comparación de algoritmos
- 9.3 Ridge/Lasso: solver SAGA
- 9.4 XGBoost: hist + early_stopping_rounds
- 9.5 SVM: LinearSVC y SGD
- 9.6 Tabla maestra estándar vs optimizado
- 9.7 Discusión costo-beneficio
- 9.8 Conexión con el capítulo 10

> **Estado:** stub.
"""),

    ("10_residuos_diagnosticos", """
# 10. Análisis de residuos

## Propósito

Diagnosticar los supuestos de los modelos de regresión top y detectar
estructura remanente no capturada:

- Test de **White** ({cite}`white1980heteroskedasticity`) —
  homocedasticidad
- Test de **BDS** — independencia
- **ACF** de residuos
- Test de **Ljung-Box** ({cite}`ljung1978measure`) — autocorrelación
- **Histograma** + test de **Jarque-Bera** — normalidad

## Secciones

- 10.1 Cálculo de residuos
- 10.2 Histograma de residuos
- 10.3 ACF de residuos
- 10.4 Ljung-Box
- 10.5 White
- 10.6 BDS
- 10.7 Jarque-Bera
- 10.8 Síntesis: qué le falta a cada modelo
- 10.9 Conexión con el capítulo 11

> **Estado:** stub.
"""),

    ("11_comparacion_estadistica", """
# 11. Comparación estadística

## Propósito

Demostrar formalmente, no solo intuitivamente, qué modelos difieren
significativamente en desempeño:

- **Diebold-Mariano** ({cite}`diebold1995comparing`) — regresión
- **DeLong real** ({cite}`delong1988comparing`, {cite}`sun2014fast`)
  — clasificación
- **Bootstrap percentil** — IC 95% para todas las métricas
- **Bonferroni** — control de FWER en comparaciones múltiples

## Secciones

- 11.1 DM por pares (regresión)
- 11.2 DeLong por pares (clasificación)
- 11.3 Bootstrap CI 95% por modelo
- 11.4 Corrección por comparaciones múltiples
- 11.5 Tabla maestra: pares con diferencia significativa
- 11.6 Interpretación de magnitud de efecto
- 11.7 Conexión con el capítulo 12

> **Estado:** stub.
"""),

    ("12_interpretabilidad", """
# 12. Interpretabilidad

## Propósito

Abrir la caja negra del mejor XGBoost (obligatorio por rúbrica) y de
los otros modelos top con tres técnicas complementarias:

- **Feature importance nativa** de XGBoost
- **Permutation importance** (model-agnostic)
- **LIME** ({cite}`ribeiro2016should`) — interpretación local
- **SHAP** — interpretación local y global

## Secciones

- 12.1 Feature importance nativa
- 12.2 Permutation importance (sobre val)
- 12.3 LIME: 3-5 explicaciones locales en casos representativos
- 12.4 SHAP summary + dependence plots
- 12.5 Conexión entre variables importantes y conocimiento financiero
- 12.6 Limitaciones de la interpretación
- 12.7 Conexión con el capítulo 13

> **Estado:** stub.
"""),

    ("13_modelo_original", """
# 13. Investigación, diseño y evaluación del modelo original

## Propósito

Diseñar, justificar bibliográficamente, implementar y evaluar el
**modelo original** del proyecto. Esta es la **contribución principal**
del trabajo y la pieza más importante para sustentación.

## Fase 1 — Investigación bibliográfica

- 13.1 Revisión de literatura: ≥12 referencias verificadas (DOI/URL)
- 13.2 Familias revisadas: HAR-RV y variantes, híbridos GARCH+ML,
  stacking temporal con OOF predictions, mixture-of-experts,
  boosting de residuales, regime-switching ML.
- 13.3 Tabla comparativa: 4+ arquitecturas candidatas

## Fase 2 — Selección

- 13.4 Justificación de la arquitectura seleccionada
- 13.5 Pseudocódigo y flujo de entrenamiento
- 13.6 Análisis de riesgos de data leakage

## Fase 3 — Implementación

- 13.7 Código en `src/original_model.py`
- 13.8 Entrenamiento con CV temporal

## Fase 4 — Evaluación en validation

- 13.9 Comparación contra mejor benchmark
- 13.10 Comparación contra mejor modelo base
- 13.11 Si **NO supera en val** → rediseñar (loop)
- 13.12 Si supera → congelar arquitectura

## Fase 5 — Ablación

- 13.13 Modelo completo vs sin componente A vs sin componente B
- 13.14 Versión "ingenua" del ensamble
- 13.15 Discusión

## Fase 6 — Evaluación final en test

- 13.16 Test ejecutado una sola vez con el modelo congelado
- 13.17 Comparación final con benchmarks y modelos base en test
- 13.18 Pruebas estadísticas (DM, DeLong, Bootstrap)
- 13.19 Interpretabilidad del modelo original

> **Estado:** stub. Implementación tras los capítulos 4-12.
"""),

    ("14_conclusiones_sustentacion", """
# 14. Conclusiones y guion de sustentación

## Propósito

Cerrar el proyecto con: (i) tabla maestra de resultados, (ii)
conclusiones críticas respaldadas por evidencia, (iii) limitaciones
honestas, (iv) trabajo futuro, y (v) un guion de **10 minutos** para
sustentar el proyecto frente al docente.

## Secciones

- 14.1 Tabla maestra: todos los modelos × todas las métricas
- 14.2 Ranking con intervalos de confianza bootstrap
- 14.3 Conclusión principal sobre el problema de predicción de
  volatilidad
- 14.4 Conclusión sobre la comparación benchmarks vs ML vs original
- 14.5 Limitaciones honestas (datos, horizonte, generalización)
- 14.6 Trabajo futuro
- 14.7 **Guion de sustentación** (minuto a minuto):
  - 0:00-1:00 Contexto y problema
  - 1:00-2:00 Datos, features, splits, anti-leakage
  - 2:00-3:30 Benchmarks econométricos
  - 3:30-5:00 Modelos base
  - 5:00-6:30 Optimización + comparación estadística
  - 6:30-8:00 Modelo original: motivación, arquitectura, resultados
  - 8:00-9:00 Conclusiones e interpretación
  - 9:00-10:00 Limitaciones, trabajo futuro, preguntas
- 14.8 Preguntas probables del docente y respuestas sugeridas
- 14.9 Defensa contra preguntas sobre data leakage
- 14.10 Cierre

> **Estado:** stub.
"""),
]


def make_stub(filename, content):
    nb = nbf.v4.new_notebook()
    nb.cells = [md(content)]
    save(nb, filename)


def main():
    print("Generando notebooks en", NB_DIR)
    make_00_setup()
    make_01_contexto()
    make_02_etl_eda()
    make_03_features_targets()
    for fname, body in STUBS:
        make_stub(fname, body)
    print("\nTodos los notebooks generados.")


if __name__ == "__main__":
    main()
