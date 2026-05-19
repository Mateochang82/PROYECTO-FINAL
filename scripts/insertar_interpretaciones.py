"""
Inserta interpretaciones específicas debajo de cada gráfico que las necesite.

Las interpretaciones están basadas en los números REALES obtenidos de los
JSONs de outputs/metrics/. Cada interpretación tiene 4-6 líneas de prosa,
en estilo académico moderado (como el proyecto referencia).
"""
import json
import nbformat as nbf
from pathlib import Path


# Diccionario { archivo : { cell_idx_grafico : texto_interpretacion } }
# El texto se inserta como NUEVA celda markdown justo después del gráfico.
INTERPRETACIONES = {
    "04_benchmarks_econometricos.ipynb": {
        20: """**Interpretación.** El gráfico revela una jerarquía clara entre los seis benchmarks econométricos. Naive (RMSE 0.00425, MAE 0.00232) domina por amplio margen frente a sus competidores: HAR-RV queda segundo (RMSE 0.00560), seguido por EWMA y ARIMA en posiciones cercanas. GARCH(1,1), pese a ser el modelo teóricamente más sofisticado, queda último con RMSE 0.00789. La explicación es directa: la autocorrelación de orden 1 de la volatilidad realizada es lo suficientemente alta como para que predecir $\\hat{\\sigma}_{t+1} = \\sigma_t$ supere a cualquier modelo que intente capturar dependencias más complejas con pocos parámetros.""",
        22: """**Interpretación.** Visualmente confirmamos lo que la tabla anticipaba. Naive sigue la trayectoria real con un retraso de un día, lo cual visualmente parece "exacto" porque la volatilidad cambia poco día a día. EWMA y HAR-RV producen series más suaves pero pierden los picos. Rolling mean (ventana de 22 días) queda fuertemente subestimada en períodos de alta volatilidad como inicios de 2014 y mediados de 2016. GARCH muestra una dinámica oscilante que no logra alinearse con los movimientos reales del target.""",
        24: """**Interpretación.** Las pendientes del error acumulado muestran cuándo cada modelo falla. Naive mantiene la pendiente más plana durante todo el período de test, confirmando su superioridad sostenida. Los modelos basados en promedios (rolling mean, EWMA) tienen pendientes empinadas durante los episodios volátiles de 2014 y 2016, momentos en los que la inercia del promedio les juega en contra. GARCH acumula error de forma casi monotónica, evidencia de que su especificación no se ajusta bien a este período relativamente calmo del test.""",
    },
    "05_modelos_regresion.ipynb": {
        20: """**Interpretación.** La comparación val vs test posiciona a Ridge como el modelo más sólido del capítulo (RMSE val 0.00358, RMSE test 0.00384, R² test +0.715). XGBoost queda muy cerca (RMSE test 0.00396) y supera levemente a la referencia Naive. Lasso, Decision Tree y Random Forest quedan en una banda media. KNN se aleja notoriamente del resto y SVR colapsa con RMSE 0.0229 (R² test negativo de -9), señal de que el kernel RBF no fue capaz de calibrarse al rango de magnitud de la volatilidad sin tuning específico.""",
        22: """**Interpretación.** El panel "real vs predicho" muestra que Ridge y XGBoost siguen los picos del target en los momentos críticos del test, aunque ambos lo subestiman ligeramente en los máximos. Esto es esperable: ambos modelos se entrenan minimizando el error cuadrático medio, lo que les penaliza más las predicciones lejanas pero los empuja hacia la media en los extremos. Ningún modelo logra capturar los picos puntuales de mayo de 2014 ni de noviembre de 2016, donde el target subió bruscamente sin que las features lo anticiparan.""",
        24: """**Interpretación.** El gap val→test mide cuánto degrada el desempeño cuando el modelo enfrenta el período de test (2013-2017) tras haber sido entrenado en 1990-2009. Los modelos con menor gap son Ridge (+0.00026) y XGBoost (+0.00043), lo cual sugiere que generalizan bien. Decision Tree, Random Forest y Lasso tienen gaps moderados, compatibles con un sobreajuste leve. KNN y SVR tienen gaps grandes, evidencia de overfitting estructural al train.""",
    },
    "06_modelos_clasificacion.ipynb": {
        20: """**Interpretación.** Las curvas ROC ordenan los clasificadores de forma clara. Random Forest y XGBoost dominan en la zona de bajos falsos positivos (AUC test 0.9559 y 0.9557 respectivamente), lo cual es deseable en un problema desbalanceado donde la clase positiva es la minoritaria. Las regresiones logísticas L1 y L2 quedan muy parejas entre sí (AUC ≈ 0.944), confirmando que la diferencia entre regularizaciones es marginal para este dataset. Naive Bayes y KNN quedan claramente por debajo, con AUC del orden de 0.87 y 0.84 respectivamente.""",
        22: """**Interpretación.** Sobre el mejor modelo (Random Forest, AUC test 0.9559), la matriz de confusión muestra que de los 109 positivos reales en test se predicen correctamente la mayoría, con tasa de recall de 0.7248. Los falsos positivos son relativamente pocos pese al desbalance, lo cual contribuye al F1 alto de 0.8061. Esta calidad permite usar el clasificador con confianza para alimentar el gate del modelo original HVRF en el capítulo 13.""",
        24: """**Interpretación.** Random Forest y XGBoost mantienen su ventaja tanto en val como en test, con poca variación entre los dos conjuntos. Los modelos lineales (LogReg L1/L2) muestran un pequeño aumento de AUC en test, consistente con que el régimen 2013-2017 es más estable que el de validación. KNN es el único que cae claramente del val al test, indicio de sensibilidad al cambio de distribución temporal.""",
    },
    "07_balanceo_clases.ipynb": {
        17: """**Interpretación.** Las cuatro barras agrupadas por modelo muestran que las técnicas de balanceo (SMOTE, ADASYN, class_weight) producen variaciones marginales sobre el train balanceado real. ADASYN incluso falla en varios modelos porque la clase minoritaria no tiene suficientes muestras "difíciles" para sintetizar. La conclusión es contraintuitiva pero clara: si el train ya está balanceado por construcción del umbral, las técnicas de balanceo no aportan nada.""",
        25: """**Interpretación.** La diferencia entre experimentos es elocuente. En el Exp 2 (train artificialmente desbalanceado 90/10), aplicar SMOTE o ADASYN recupera entre 5 y 50 puntos porcentuales de F1 respecto al baseline. Pero en el Exp 1 (train real, balanceado), esas mismas técnicas son prácticamente irrelevantes. Esto demuestra empíricamente que el balanceo corrige desbalance del train, no distribution shift del test.""",
        27: """**Interpretación.** El gráfico final encierra la lección del capítulo: la elección de aplicar o no aplicar SMOTE/ADASYN depende exclusivamente del balance del train. Para nuestro problema, donde el train está balanceado por construcción, ninguna técnica de balanceo es necesaria. El distribution shift del test (90% bajo, 10% alto) es un fenómeno diferente que requiere herramientas distintas (domain adaptation, recalibración de probabilidades).""",
    },
    "08_optimizacion_hiperparametros.ipynb": {
        20: """**Interpretación.** Las curvas de convergencia muestran cómo cada método explora el espacio de hiperparámetros. Optuna (TPE) y DEAP (algoritmo genético) convergen más rápido y a mejores soluciones que Grid Search y Random Search, evidenciando que los métodos guiados por información histórica son más eficientes. Random Search se desempeña sorprendentemente bien para ser un método sin guía, consistente con la literatura. Grid Search es el menos eficiente porque evalúa configuraciones predefinidas que no necesariamente son las óptimas.""",
        22: """**Interpretación.** El gráfico tiempo vs calidad sitúa a Optuna en la zona ideal: baja RMSE test (0.00382) con tiempo razonable. DEAP también logra buenos resultados pero con tiempo mayor. Grid y Random Search obtienen RMSE test muy similares entre sí (≈0.00398), confirmando que para este problema los métodos guiados no compran ventaja dramática sobre exploración aleatoria, pero sí sobre la búsqueda exhaustiva en grilla.""",
    },
    "09_optimizacion_computacional.ipynb": {
        22: """**Interpretación.** Los speedups son consistentes en su dirección pero modestos en magnitud. Las técnicas optimizadas más impactantes para este dataset son `hist`+early-stopping en XGBoost (8.7× más rápido manteniendo precisión) y LinearSVC sobre SVC con RBF (40× más rápido). Para modelos lineales pequeños como Ridge o Naive Bayes, los solvers default son ya óptimos en un dataset de este tamaño. La lección es priorizar las técnicas donde el speedup es real y no optimizar por optimizar.""",
    },
    "10_residuos_diagnosticos.ipynb": {
        10: """**Interpretación.** Los histogramas muestran residuos centrados en cero pero con colas pesadas (kurtosis entre 13 y 17 para los tres modelos), evidencia clara de no-normalidad. Los Q-Q plots confirman visualmente la desviación: los puntos extremos se alejan sistemáticamente de la diagonal. Esta no-normalidad es esperable en residuos de modelos sobre datos financieros y descarta el uso de inferencia paramétrica clásica basada en distribución t-Student.""",
        12: """**Interpretación.** La ACF revela un hallazgo importante: aparece una autocorrelación significativa en el lag 5 para los tres modelos (Naive, Ridge, XGBoost-Optuna), lo cual es un artefacto estructural del cálculo del target con ventana de 5 días, no una falla de los modelos. Comparativamente, XGBoost-Optuna reduce la autocorrelación residual en los lags 2-4 frente a Ridge, evidencia empírica de que captura estructura no lineal adicional.""",
        14: """**Interpretación.** Los residuos vs predicción no muestran patrones estructurados marcados, lo cual es consistente con las pruebas formales que pasan homocedasticidad (White p=0.20 para Ridge, p=0.73 para XGBoost-Optuna). La dispersión es relativamente uniforme a lo largo del rango de predicción, sin formas de embudo ni curvaturas evidentes. Esto valida que las técnicas no paramétricas (Diebold-Mariano, Bootstrap) son apropiadas para el siguiente capítulo.""",
    },
    "11_comparacion_estadistica.ipynb": {
        8: """**Interpretación.** El heatmap de Diebold-Mariano revela cuáles modelos difieren significativamente entre sí en pérdida cuadrática. De las 136 comparaciones únicas, 117 son significativas a α=0.05 y 99 sobreviven la corrección Bonferroni (α/k = 0.00037). Los XGBoost optimizados forman un bloque con DM significativo contra los benchmarks econométricos. Sin embargo, dentro del top del libro (Ridge, XGB-Optuna, XGB base) las diferencias entre sí no son significativas, lo cual confirma que el ranking puntual no implica significancia estadística.""",
        11: """**Interpretación.** El forest plot revela tres bandas claras de desempeño. La banda superior agrupa a XGB-Optuna (RMSE 0.00382), Ridge (0.00384) y XGB base (0.00396), todos con intervalos de confianza que se solapan ampliamente. La banda media incluye Naive, Lasso, los demás modelos de NB 05 y los benchmarks econométricos. La banda inferior contiene SVR como outlier extremo. El solapamiento de IC en la banda superior confirma estadísticamente lo que el DM ya indicaba: el ranking puntual del mejor modelo no es robusto a la incertidumbre muestral.""",
        14: """**Interpretación.** El heatmap de DeLong sobre los 8 clasificadores muestra 18 pares significativos a α=0.05 y 12 que sobreviven Bonferroni. Random Forest y XGBoost son indistinguibles entre sí (AUC 0.9559 vs 0.9557 con DeLong p>0.05), confirmando un empate técnico en el liderato. Ambos sí difieren significativamente de KNN, Naive Bayes y árboles individuales. Las regresiones logísticas L1 y L2 también son indistinguibles entre sí, consistente con la marginalidad del tipo de regularización en este dataset.""",
        17: """**Interpretación.** Los intervalos de confianza bootstrap para AUC confirman las distinciones del DeLong. Random Forest y XGBoost tienen IC casi idénticos en torno a [0.928, 0.980], reforzando su empate técnico. Las regresiones logísticas mantienen IC más estrechos centrados en 0.944, claramente por debajo del top pero por encima del resto. El IC de KNN (centrado en 0.84) no se solapa con el de los modelos top, evidencia visual de su desempeño inferior. Esta visualización es complementaria al test de DeLong y permite ver gráficamente qué modelos son indistinguibles entre sí.""",
    },
    "12_interpretabilidad.ipynb": {
        6: """**Interpretación.** El feature importance nativo de XGBoost por gain muestra que `vol_5` (volatilidad rolling de 5 días) domina ampliamente con un gain muy superior al resto. Las siguientes features (retornos rezagados, componentes HAR-RV, ATR) aportan contribuciones moderadas. La concentración en `vol_5` sugiere que el modelo no lineal está principalmente reproduciendo la persistencia de corto plazo de la volatilidad realizada.""",
        9: """**Interpretación.** La permutation importance confirma el dominio de `vol_5`: permutar esta feature degrada el RMSE de forma drásticamente mayor que cualquier otra. El segundo grupo lo conforman las features de retornos rezagados (`ret_lag_1`, `ret_lag_2`, `ret_lag_3`) y los componentes HAR-RV (`vol_d`, `vol_w`), con contribuciones modestas pero detectables. Esta convergencia entre gain nativo y permutation importance da confianza en la jerarquía identificada.""",
        12: """**Interpretación.** La importancia global por SHAP refuerza el patrón: `vol_5` aporta la mayor parte de la varianza explicada del modelo. Las top features secundarias (retornos rezagados, vol_d, vol_w, atr_14) muestran contribuciones similares entre sí y mucho menores que vol_5. Esta consistencia entre las tres técnicas (gain, permutación, SHAP) reduce sustancialmente la incertidumbre sobre qué features son los verdaderos drivers del modelo.""",
        19: """**Interpretación.** Las cuatro instancias seleccionadas para LIME forman una secuencia narrativa instructiva del cluster de volatilidad de enero 2016. El 14 de enero (FN, sub-predicción) el modelo no anticipó el pico real de 0.0508 prediciendo solo 0.0241. Para el 19 de enero (HIT alto) el modelo ya había absorbido el cambio de régimen y predijo el pico con precisión. El 22 de enero (FP, sobre-predicción) el modelo continuó esperando alta volatilidad cuando ya había bajado a 0.008. Esta secuencia FN→HIT→FP ilustra empíricamente el efecto de retraso que tienen los modelos basados en rolling windows ante cambios de régimen.""",
        22: """**Interpretación.** El heatmap comparativo confirma el consenso entre las tres técnicas: `vol_5` aparece con peso aproximado de 0.62 (gain), 0.59 (permutación) y 0.42 (SHAP). Las siguientes features quedan todas por debajo de 0.07 normalizado en cualquiera de las tres métricas. Esta convergencia interpretativa explica de forma causal el hallazgo estadístico del Capítulo 11: cuando un modelo no lineal depende fuertemente de una sola feature, una regresión lineal sobre esa feature captura prácticamente la misma señal.""",
    },
    "13_modelo_original_hvrf.ipynb": {
        20: """**Interpretación.** El forest plot ubica al ensamble simple en el primer lugar (RMSE test 0.00370), seguido por XGB-Optuna (0.00382), Ridge (0.00384), HVRF (0.00390) y el resto. Los intervalos de confianza bootstrap de los modelos top se solapan parcialmente, pero el ensamble simple tiene la posición puntual más baja con IC ajustado. HVRF queda en el cuarto puesto del proyecto. Los componentes individuales (Solo R_bajo y Solo R_alto) confirman su valor solo cuando se combinan: aislados se desempeñan claramente peor.""",
        22: """**Interpretación.** La serie temporal del test muestra que tanto Ridge como HVRF siguen el target con fidelidad similar durante la mayor parte del período. En los momentos de pico (mayo 2014, noviembre 2016), ambos modelos subestiman el máximo pero captan correctamente la dirección del cambio. Las trayectorias de Ridge y HVRF son prácticamente indistinguibles a simple vista, consistente con la prueba DM que arroja p=0.39 (no se rechaza igualdad).""",
        24: """**Interpretación.** El diagnóstico interno revela cómo HVRF combina sus componentes. Cuando p(régimen_alto|x) es cercano a 1, la predicción se inclina hacia R_alto; cuando es cercano a 0, hacia R_bajo. La probabilidad oscila en función de las señales que recibe el clasificador (AUC test 0.968), lo cual valida que la gate funciona correctamente. La predicción HVRF final (línea azul) se mantiene cerca del target real con suavidad, pero como muestra el forest plot, esta sofisticación no traduce en ventaja estadística sobre el promedio simple Ridge + XGB.""",
    },
    "14_conclusiones_sustentacion.ipynb": {
        6: """**Interpretación.** La figura final consolida todos los modelos del proyecto en una sola vista. En el panel de regresión (izquierda), el ensamble simple lidera con RMSE 0.00370, seguido por XGB-Optuna y Ridge en una banda muy ajustada. La línea discontinua marca el RMSE de Naive (0.00425), referencia mínima que todos los modelos top superan. En el panel de clasificación (derecha), Random Forest y XGBoost alcanzan AUC test cercano a 0.96, claramente por encima de los demás clasificadores. Esta vista resumen es la evidencia visual que sintetiza todo el trabajo experimental del proyecto.""",
    },
}


def insertar_interpretaciones():
    """Inserta nuevas celdas markdown con las interpretaciones."""
    total_insertadas = 0
    for nb_name, interps in INTERPRETACIONES.items():
        nb_path = Path(f"notebooks/{nb_name}")
        nb = nbf.read(nb_path, as_version=4)
        
        # Necesitamos insertar de mayor a menor para no romper los índices
        for cell_idx in sorted(interps.keys(), reverse=True):
            texto = interps[cell_idx]
            nueva_celda = nbf.v4.new_markdown_cell(texto)
            # Insertar inmediatamente después del gráfico
            nb['cells'].insert(cell_idx + 1, nueva_celda)
            total_insertadas += 1
        
        nbf.write(nb, nb_path)
        print(f"  ✓ {nb_name}: {len(interps)} interpretaciones insertadas")
    
    print(f"\nTOTAL INTERPRETACIONES AGREGADAS: {total_insertadas}")


if __name__ == "__main__":
    insertar_interpretaciones()
