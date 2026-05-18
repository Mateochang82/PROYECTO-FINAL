# Guion de sustentación y FAQ del jurado

> **DOCUMENTO PRIVADO DEL EQUIPO.** Este archivo es material de
> preparación para la sustentación. NO forma parte del Jupyter Book
> público que ve el evaluador. Su propósito es servir como hoja de
> ruta operativa para los 10 minutos de defensa oral y como banco
> de respuestas pre-elaboradas a las preguntas más probables del
> jurado.
>
> **Equipo:** Juan Camilo Conrado, Sergio Cadavid, Mateo Chang.

---

## 14.6 Guion de sustentación — 10 minutos cronometrados

> *Lo que sigue está pensado como una guía operativa para la
> defensa oral. Cada bloque tiene tiempo asignado, qué decir, qué
> mostrar en pantalla y qué evitar.*

### Bloque 1 — 0:00 a 1:00 (hook + contexto)

**Qué decir:**

> "Buenos días. Vamos a presentar la predicción de volatilidad
> realizada de Intel Corporation con horizonte de un día, usando
> Machine Learning y benchmarks econométricos clásicos. El proyecto
> integra los tres entregables del curso, con énfasis en
> reproducibilidad, anti-leakage y honestidad estadística.
> Trabajamos con 7 022 días de cotización entre 1990 y 2017."

**Qué mostrar:** título + miembros del equipo + diapositiva de
contexto financiero (gráfico de retornos INTC con su volatilidad
realizada — figura de NB 02).

**Evitar:** introducción larga sobre qué es la volatilidad. El
jurado ya lo sabe.

---

### Bloque 2 — 1:00 a 2:00 (decisión metodológica clave)

**Qué decir:**

> "Tomamos una decisión metodológica explícita al iniciar el
> proyecto: cambiamos el horizonte de predicción a un día, en lugar
> de los siete del proyecto anterior. Esto se hizo después de
> identificar que con horizontes largos el problema produce R²
> universalmente negativo. Con el horizonte de un día logramos R²
> positivo de más del 70 % en test."

**Qué mostrar:** comparación R² antiguo vs nuevo (tabla simple en
diapositiva).

**Por qué es importante:** *demuestra criterio metodológico*. Es
una de las cosas que más valoran los profesores.

---

### Bloque 3 — 2:00 a 3:30 (pipeline y arquitectura)

**Qué decir:**

> "El pipeline está organizado en 14 notebooks reproducibles bajo
> Jupyter Book. Las decisiones anti-leakage son explícitas y están
> verificadas por 13 tests automatizados con pytest. Todos los
> targets se calculan con shift causal, todas las features con
> shift(1), y todo preprocesamiento está dentro de pipelines de
> scikit-learn o imblearn según corresponda."

**Qué mostrar:** diagrama del pipeline (puedes hacerlo en
PowerPoint o usar la estructura del README como referencia).

**Evitar:** entrar en código. El jurado lo puede ver después.

---

### Bloque 4 — 3:30 a 5:00 (regresión + clasificación)

**Qué decir:**

> "En regresión entrenamos los siete modelos exigidos por la
> rúbrica. Ridge alcanzó RMSE test 0.00384, superando a Naive en
> 10 %. XGBoost obtuvo prácticamente lo mismo. En clasificación,
> Random Forest y XGBoost alcanzaron AUC > 0.95 sobre un test
> fuertemente desbalanceado 90/10 — un resultado muy fuerte."

**Qué mostrar:**

- Forest plot del NB 11 (`outputs/figures/11_rmse_forest.png`).
- Curvas ROC del NB 06 (`outputs/figures/06_roc_curves_test.png`).

**Evitar:** listar las 7 métricas para cada modelo. Solo dos
métricas clave por categoría.

---

### Bloque 5 — 5:00 a 6:30 (balanceo y optimización)

**Qué decir:**

> "En el capítulo de balanceo hicimos un experimento controlado:
> aplicamos SMOTE, ADASYN y class_weight tanto al train real
> (balanceado 50/50) como a un train artificialmente desbalanceado.
> En el primer caso, ninguna técnica aportó. En el segundo, F1
> subió hasta seis veces. Conclusión: el balanceo corrige
> desbalance de train, no distribution shift de val/test. Para
> optimización, comparamos los cuatro métodos pedidos: Grid,
> Random, Optuna y DEAP. Optuna ganó por punto en val y test."

**Qué mostrar:**

- Figura comparativa Exp1 vs Exp2 (`outputs/figures/07_exp1_vs_exp2_f1.png`).
- Curva de convergencia (`outputs/figures/08_convergence.png`).

---

### Bloque 6 — 6:30 a 8:00 (estadística e interpretabilidad)

**Qué decir:**

> "Aplicamos Diebold-Mariano con corrección Newey-West a las 136
> comparaciones entre pares de modelos de regresión, y DeLong a las
> 28 comparaciones entre clasificadores. Con corrección Bonferroni
> sobreviven 99 y 12 pares respectivamente. El hallazgo central es
> que XGBoost-Optuna y Ridge **no son estadísticamente distintos**
> — el ranking puntual no implica significancia. La interpretabilidad
> con SHAP y LIME explicó causalmente por qué: el XGBoost usa
> esencialmente una sola feature, `vol_5`, lo que hace que un
> modelo lineal capture prácticamente lo mismo."

**Qué mostrar:**

- Heatmap DM (`outputs/figures/11_dm_heatmap.png`).
- Heatmap de importancia combinada (`outputs/figures/12_global_importance_heatmap.png`).
- LIME enero 2016 (`outputs/figures/12_lime_local_explanations.png`).

---

### Bloque 7 — 8:00 a 9:30 (modelo original + hallazgo final)

**Qué decir:**

> "Diseñamos un modelo original llamado HVRF — Hybrid Volatility
> Regime Forecaster — que combina un clasificador de régimen
> (AUC test 0.968) con dos regresores especializados, mezclados de
> forma suave por la probabilidad del clasificador. HVRF no superó
> al empate Ridge ≈ XGBoost-Optuna con diferencia significativa.
> Pero al hacer la ablación descubrimos algo más valioso: **el
> ensamble simple — el promedio sin pesos entrenables de Ridge y
> XGBoost-Optuna — es el mejor modelo del proyecto.** Con RMSE
> test 0.00370, supera a HVRF con DM p < 0.001. La lección
> académica es que la complejidad arquitectónica no compra ventaja
> automáticamente; la diversificación de sesgos sí."

**Qué mostrar:**

- Forest plot final de HVRF (`outputs/figures/13_hvrf_rmse_forest.png`).
- Diagrama conceptual del HVRF (diapositiva manual con la fórmula
  $\hat{y} = p \cdot R_{\uparrow} + (1-p) \cdot R_{\downarrow}$).

---

### Bloque 8 — 9:30 a 10:00 (cierre)

**Qué decir:**

> "En resumen: el proyecto integra todas las técnicas pedidas por
> la rúbrica con anti-leakage estricto, y produce conclusiones
> estadísticamente honestas. El mejor modelo es el ensamble simple
> Ridge + XGBoost-Optuna; el modelo original HVRF, aunque no supera
> el empate técnico, está implementado correctamente y revela una
> lección sobre los límites de la complejidad arquitectónica en
> este problema. Gracias."

**Qué mostrar:** diapositiva final con los seis hallazgos
centrales y una línea de "gracias" + GitHub.

---

### Estimaciones de tiempo realistas

- Si la sustentación es 10 minutos en total con preguntas, **usa 8
  minutos para hablar** y deja 2 para preguntas.
- Si es 15 minutos en total, sigue exactamente esta estructura.
- Si te quedas corto, expande el bloque 7 (modelo original) con un
  ejemplo concreto del LIME enero 2016.


---

## 14.7 FAQ esperada del jurado

Preguntas que es muy probable que les hagan en la sustentación, con
respuestas pre-elaboradas. Memorizar el sentido, no el texto.

---

**P1: "¿Por qué horizonte de 1 día y no 7 como el proyecto
anterior? ¿No se les pidió predecir más lejos?"**

> "Tomamos esa decisión metodológicamente, no por evitar la
> dificultad. En el proyecto anterior con horizonte 7 días el R²
> era universalmente negativo, lo que significa que ningún modelo
> superaba la media histórica. Esto se debe a que la volatilidad
> es predecible a corto plazo por la autocorrelación de la serie
> (efecto cluster), pero esa autocorrelación decae rápidamente.
> Con un horizonte de 1 día capturamos la fracción de varianza
> que es realmente predecible, sin forzar la métrica. La rúbrica
> permite definir el target; lo definimos para que el problema
> tenga señal."

---

**P2: "Dicen que Ridge y XGBoost-Optuna son indistinguibles. ¿Por
qué entonces hicieron toda la optimización del NB 8?"**

> "Porque la rúbrica lo exige y porque sin haberla hecho no
> podríamos *demostrar* el empate. Hacer Optuna, DEAP, Grid y
> Random sobre el mismo problema, con presupuestos comparables,
> nos da la evidencia para reportar que la diferencia no es
> significativa. Si solo hubiéramos entrenado Ridge baseline,
> habríamos asumido —incorrectamente— que XGBoost es mejor."

---

**P3: "¿Qué es exactamente HVRF y por qué lo eligieron como
modelo original?"**

> "HVRF — Hybrid Volatility Regime Forecaster — combina tres
> componentes: un clasificador XGBoost que estima la probabilidad
> de que mañana sea régimen alto; un regresor entrenado solo con
> días de régimen alto; un regresor entrenado solo con días de
> régimen bajo. La predicción final es la mezcla ponderada por la
> probabilidad. Es una mezcla de expertos sin meta-learner. Lo
> elegimos porque los hallazgos previos sugerían que un modelo
> único general no captura los dos regímenes con la misma
> eficacia."

---

**P4: "Si el ensamble simple es mejor que HVRF, ¿por qué no lo
declaran como modelo original?"**

> "Por dos razones. Primera: el ensamble simple no es un *diseño
> original*, es un combinador trivial; sería honesto pero poco
> apropiado para la rúbrica que pide creatividad metodológica.
> Segunda: declararlo como original sería oportunismo retroactivo.
> Lo que hicimos es reportar honestamente que HVRF no aporta
> ventaja y que el ensamble simple emerge como mejor —
> respetando el método pero reconociendo el resultado."

---

**P5: "¿No es trampa usar los hiperparámetros del Optuna del NB 8
para entrenar los componentes del HVRF?"**

> "No, no es trampa. Esos hiperparámetros se obtuvieron usando
> solo train y validation cruzada interna; no tocaron el test.
> Usarlos en HVRF es transferir conocimiento legítimo del
> proyecto. Lo que sí sería leakage es haber optimizado los
> hiperparámetros de HVRF mirando el test, y eso no lo hicimos."

---

**P6: "¿Cómo manejaron el desbalance del test?"**

> "Lo manejamos a través de tres decisiones explícitas: primero,
> usamos AUC como métrica principal de clasificación, que es
> robusta al desbalance. Segundo, evaluamos SMOTE/ADASYN/class
> weight y reportamos honestamente que no aportan, porque el
> desbalance del test es distribution shift, no class imbalance
> tradicional. Tercero, en el bootstrap CI controlamos por el
> número de positivos, y reportamos intervalos de confianza para
> ser transparentes sobre la incertidumbre adicional."

---

**P7: "¿Qué pasaría si el horizonte fuera de 5 o 10 días?"**

> "Esperaríamos que el R² baje sustancialmente, porque la
> autocorrelación de la volatilidad realizada decae rápidamente.
> A 5 días algunos modelos (especialmente HAR-RV y XGBoost con
> features de larga escala) podrían mantener R² positivo, pero
> con valores típicamente entre 0.3 y 0.5. A 10 días es probable
> que la mayoría de modelos caigan por debajo del Naive. No lo
> validamos empíricamente en este proyecto, pero es una
> extensión natural."

---

**P8: "El test va de 2013 a 2017, periodo de baja volatilidad
post-crisis. ¿Cómo defienden la generalización?"**

> "No defendemos generalización a 2020-2025 con este proyecto;
> sería deshonesto. Lo que sí defendemos es que el pipeline está
> diseñado para ser re-aplicado: las features son causales, los
> targets están bien definidos, y los procedimientos
> anti-leakage funcionan para cualquier periodo. Si tuviéramos
> que evaluar contra COVID o el rally de IA 2023, deberíamos
> re-bajar el dataset y re-validar — el código está listo para
> eso."

---

**P9: "¿Por qué no usaron deep learning (LSTM, Transformer)?"**

> "Porque la rúbrica del curso pidió específicamente las
> técnicas que aplicamos: KNN, NB, LogReg, árboles, ensambles,
> SVM, XGBoost. Las técnicas secuenciales como LSTM y
> Transformer son material del siguiente curso. Sí lo
> mencionamos en trabajo futuro porque sería una extensión
> natural — pero no implementarlo aquí es respetar el alcance
> del curso."

---

**P10: "Su mejor RMSE es 0.00370. ¿Eso es operacionalmente útil
para alguien que quiere usar esto en producción?"**

> "Honestamente, depende de para qué. Para predecir el nivel
> exacto de volatilidad, 0.0037 es un error relativo del ~25%
> sobre la volatilidad media — útil pero no preciso. Para
> decisiones direccionales (alto/bajo régimen), la clasificación
> con AUC 0.96 sí es operacionalmente útil. Lo que el modelo
> NO captura son shocks fundamentales (noticias, eventos macro)
> que no están en sus features. Un trader real combinaría este
> modelo con análisis fundamental, no lo usaría como única señal."
