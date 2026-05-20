"""
Genera el PDF maestro de estudio para la sustentación del proyecto
INTC-Vol-ML.

Estructura:
1. Portada
2. Resumen ejecutivo (1 página)
3. Las 7 secciones del plan de presentación con qué decir, números clave,
   visuales y posibles preguntas
4. Banco de preguntas anticipadas (~25 con respuestas mixtas)
5. Cheatsheet de números críticos
6. Glosario rápido
7. Errores comunes a evitar

Usa reportlab Platypus para tipografía y layout profesional.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    KeepTogether, ListFlowable, ListItem
)
from reportlab.platypus.tableofcontents import TableOfContents


# =====================================================================
#  CONFIGURACIÓN VISUAL
# =====================================================================
PRIMARY = HexColor("#0B3B5C")   # azul institucional
ACCENT  = HexColor("#C9A227")   # dorado académico
TEXT    = HexColor("#1a1a1a")
MUTED   = HexColor("#666666")
BG_LIGHT = HexColor("#f5f5f5")

styles = getSampleStyleSheet()

# Estilos personalizados
title_style = ParagraphStyle(
    "TitleX", parent=styles["Title"],
    fontName="Helvetica-Bold", fontSize=22,
    textColor=PRIMARY, alignment=TA_CENTER, spaceAfter=12, leading=26,
)
subtitle_style = ParagraphStyle(
    "SubtitleX", parent=styles["Normal"],
    fontName="Helvetica", fontSize=14,
    textColor=MUTED, alignment=TA_CENTER, spaceAfter=20, leading=18,
)
h1_style = ParagraphStyle(
    "H1X", parent=styles["Heading1"],
    fontName="Helvetica-Bold", fontSize=18,
    textColor=PRIMARY, spaceBefore=18, spaceAfter=12, leading=22,
    keepWithNext=True,
)
h2_style = ParagraphStyle(
    "H2X", parent=styles["Heading2"],
    fontName="Helvetica-Bold", fontSize=14,
    textColor=PRIMARY, spaceBefore=14, spaceAfter=8, leading=18,
    keepWithNext=True,
)
h3_style = ParagraphStyle(
    "H3X", parent=styles["Heading3"],
    fontName="Helvetica-Bold", fontSize=12,
    textColor=TEXT, spaceBefore=10, spaceAfter=6, leading=15,
    keepWithNext=True,
)
body_style = ParagraphStyle(
    "BodyX", parent=styles["BodyText"],
    fontName="Helvetica", fontSize=10.5,
    textColor=TEXT, leading=14, spaceAfter=8, alignment=TA_JUSTIFY,
)
bullet_style = ParagraphStyle(
    "BulletX", parent=body_style,
    leftIndent=18, bulletIndent=6, spaceAfter=4,
)
quote_style = ParagraphStyle(
    "QuoteX", parent=body_style,
    fontName="Helvetica-Oblique", fontSize=10,
    textColor=HexColor("#333333"), leftIndent=22, rightIndent=22,
    spaceBefore=6, spaceAfter=8, leading=13,
    borderColor=ACCENT, borderWidth=0, borderPadding=4,
    backColor=HexColor("#fff8e0"),
)
number_style = ParagraphStyle(
    "NumberX", parent=body_style,
    fontName="Courier-Bold", fontSize=11,
    textColor=PRIMARY,
)
warning_style = ParagraphStyle(
    "WarningX", parent=body_style,
    fontName="Helvetica", fontSize=10,
    textColor=HexColor("#8B0000"), leftIndent=12,
    backColor=HexColor("#fff0f0"), borderPadding=6,
)


# =====================================================================
#  HELPERS PARA CONSTRUIR CONTENIDO
# =====================================================================
def page_break():
    return PageBreak()


def spacer(h=12):
    return Spacer(1, h)


def p(text, style=body_style):
    return Paragraph(text, style)


def h1(text):
    return Paragraph(text, h1_style)


def h2(text):
    return Paragraph(text, h2_style)


def h3(text):
    return Paragraph(text, h3_style)


def quote(text):
    return Paragraph(text, quote_style)


def bullets(items, style=bullet_style):
    return ListFlowable(
        [ListItem(Paragraph(it, style), leftIndent=12) for it in items],
        bulletType='bullet', leftIndent=18, bulletFontSize=8,
        bulletColor=PRIMARY,
    )


def numbered_bullets(items, style=bullet_style):
    return ListFlowable(
        [ListItem(Paragraph(it, style), leftIndent=12) for it in items],
        bulletType='1', leftIndent=18, bulletFontSize=10,
        bulletColor=PRIMARY,
    )


def info_box(title_text, body_html, color=ACCENT):
    """Caja informativa con título y cuerpo."""
    title_para = Paragraph(f"<b>{title_text}</b>", h3_style)
    body_para = Paragraph(body_html, body_style)
    tbl = Table(
        [[title_para], [body_para]],
        colWidths=[6.5 * inch],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), color),
            ("TEXTCOLOR", (0, 0), (0, 0), white),
            ("BACKGROUND", (0, 1), (0, 1), HexColor("#fafafa")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("BOX", (0, 0), (-1, -1), 0.5, color),
        ])
    )
    return tbl


def fact_table(data, col_widths=None):
    """Tabla de hechos clave."""
    if col_widths is None:
        col_widths = [2.2 * inch, 4.3 * inch]
    
    # Convertir contenido a Paragraphs si son strings
    rows = []
    for row in data:
        new_row = []
        for cell in row:
            if isinstance(cell, str):
                new_row.append(Paragraph(cell, body_style))
            else:
                new_row.append(cell)
        rows.append(new_row)
    
    tbl = Table(rows, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("ALIGN", (0, 0), (-1, 0), "LEFT"),
        ("BACKGROUND", (0, 1), (-1, -1), HexColor("#fafafa")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f4f7fa")]),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.25, HexColor("#d0d0d0")),
    ]))
    return tbl


def qa_block(pregunta, respuesta_corta, profundizar=None):
    """Bloque de pregunta y respuesta."""
    elements = []
    
    # Pregunta
    elements.append(Paragraph(
        f"<font color='#C9A227'><b>Pregunta:</b></font> {pregunta}",
        ParagraphStyle("Q", parent=body_style, fontSize=11,
                       textColor=PRIMARY, leftIndent=10, spaceBefore=10,
                       spaceAfter=4, leading=14, fontName="Helvetica-Bold")
    ))
    
    # Respuesta corta
    elements.append(Paragraph(
        f"<font color='#0B3B5C'><b>Respuesta:</b></font> {respuesta_corta}",
        ParagraphStyle("R", parent=body_style, fontSize=10.5,
                       leftIndent=10, spaceAfter=4, leading=14)
    ))
    
    # Si profundizan
    if profundizar:
        elements.append(Paragraph(
            f"<font color='#666666'><i>Si profundizan más:</i></font> {profundizar}",
            ParagraphStyle("PD", parent=body_style, fontSize=10,
                           leftIndent=22, textColor=HexColor("#444444"),
                           spaceAfter=10, leading=13, fontName="Helvetica-Oblique")
        ))
    
    return KeepTogether(elements)


# =====================================================================
#  CONTENIDO DEL DOCUMENTO
# =====================================================================
story = []


# ====================== PORTADA ======================
story.append(spacer(2 * inch))
story.append(p("DOCUMENTO DE ESTUDIO PARA SUSTENTACIÓN",
               ParagraphStyle("MiniTitle", parent=body_style,
                              fontSize=11, textColor=ACCENT,
                              alignment=TA_CENTER, fontName="Helvetica-Bold",
                              spaceAfter=20)))
story.append(p("Proyecto Final", title_style))
story.append(p("PREDICCIÓN DE VOLATILIDAD<br/>DE INTEL CORPORATION (INTC)",
               ParagraphStyle("MainTitle", parent=title_style,
                              fontSize=26, leading=32)))
story.append(spacer(0.4 * inch))
story.append(p("Curso de Machine Learning | Pregrado en Ciencia de Datos",
               subtitle_style))
story.append(p("Universidad del Norte — Dr. Lihki Rubio — Semestre 2026-1",
               subtitle_style))

story.append(spacer(1.5 * inch))

# Equipo
equipo_data = [
    [Paragraph("<b>Equipo</b>", body_style),
     Paragraph("Juan Camilo Conrado<br/>Sergio Cadavid<br/>Mateo Chang", body_style)],
    [Paragraph("<b>Repositorio</b>", body_style),
     Paragraph('<font color="#0B3B5C">github.com/Mateochang82/PROYECTO-FINAL</font>', body_style)],
    [Paragraph("<b>Sitio público</b>", body_style),
     Paragraph('<font color="#0B3B5C">mateochang82.github.io/PROYECTO-FINAL</font>', body_style)],
    [Paragraph("<b>Duración sustentación</b>", body_style),
     Paragraph("15 minutos (todos estudiamos todo)", body_style)],
]
story.append(Table(
    equipo_data, colWidths=[1.7 * inch, 4.3 * inch],
    style=TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), HexColor("#f0f4f8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEABOVE", (0, 0), (-1, 0), 0.5, PRIMARY),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, PRIMARY),
    ])
))

story.append(spacer(1.5 * inch))
story.append(p("Este documento es material privado del equipo para preparar "
               "la defensa oral. No forma parte de la entrega académica.",
               ParagraphStyle("Footer", parent=body_style,
                              fontSize=9, alignment=TA_CENTER,
                              textColor=MUTED, fontName="Helvetica-Oblique")))
story.append(page_break())


# ====================== RESUMEN EJECUTIVO ======================
story.append(h1("Resumen ejecutivo"))
story.append(p(
    "Este proyecto predice la volatilidad realizada del stock Intel "
    "Corporation (INTC) con horizonte de un día, usando 27 años de "
    "datos OHLCV (1990-2017). Se evaluaron más de 40 configuraciones "
    "de modelos clásicos de Machine Learning, benchmarks econométricos "
    "y un modelo original llamado HVRF. Todo el pipeline es "
    "reproducible y está documentado como un Jupyter Book navegable."
))

story.append(h2("Hallazgo principal"))
story.append(p(
    "El mejor modelo del proyecto es el <b>ensamble simple</b> (½ Ridge + "
    "½ XGBoost-Optuna): <b>RMSE test = 0.00370</b>, <b>R² test = +0.74</b>, "
    "supera al benchmark Naive en 13% con diferencia estadísticamente "
    "significativa por Diebold-Mariano (p < 0.001)."
))

story.append(h2("Los cinco hallazgos no triviales"))
story.append(numbered_bullets([
    "<b>El cambio de horizonte de 7 días a 1 día</b> es la corrección "
    "metodológica más importante. Convirtió R² de negativo (proyecto "
    "anterior) a +0.74.",
    "<b>XGBoost-Optuna y Ridge son estadísticamente indistinguibles</b> "
    "(Diebold-Mariano p = 0.39). El ranking puntual no implica "
    "significancia estadística.",
    "<b>Una sola feature (vol_5) explica el 55% del modelo</b> según las "
    "tres técnicas de interpretabilidad (gain, permutación, SHAP). Esto "
    "explica causalmente el hallazgo 2.",
    "<b>El balanceo de clases no aporta en este dataset</b> porque el "
    "train ya está balanceado por construcción del umbral. El "
    "experimento controlado del NB 07 lo demuestra empíricamente.",
    "<b>La complejidad arquitectónica no compra ventaja:</b> HVRF, pese a "
    "su clasificador interno con AUC 0.968, no supera al promedio "
    "trivial de Ridge + XGBoost-Optuna.",
]))

story.append(h2("Estado del proyecto"))
story.append(bullets([
    "<b>15 capítulos</b> en Jupyter Book reproducible (1 portada + 13 capítulos + bibliografía)",
    "<b>13 tests pytest</b> de anti-leakage pasando",
    "<b>39 figuras, 21 tablas, 22 modelos entrenados</b> persistidos",
    "<b>Cero warnings</b> en el build final",
    "<b>Sitio web público</b> publicado en GitHub Pages",
]))

story.append(page_break())


# ====================== ÍNDICE DEL DOCUMENTO ======================
story.append(h1("Cómo usar este documento"))
story.append(p(
    "Este PDF está organizado para que puedas estudiar el proyecto a "
    "fondo y luego rendir bien en la sustentación. Léelo en orden la "
    "primera vez. En las siguientes pasadas, salta directamente a las "
    "secciones que necesites repasar."
))

story.append(h3("Estructura"))
story.append(bullets([
    "<b>Parte 1 — Sección por sección del plan de presentación.</b> Las "
    "7 secciones de la diapositiva del plan, cada una con qué decir, "
    "qué visual mostrar y qué números recordar.",
    "<b>Parte 2 — Banco de preguntas anticipadas.</b> 27 preguntas "
    "probables del jurado, con respuesta corta y, si profundizan, una "
    "expansión más técnica.",
    "<b>Parte 3 — Cheatsheet de números críticos.</b> Los 20 valores que "
    "debes saber de memoria.",
    "<b>Parte 4 — Glosario técnico.</b> Definiciones rápidas de los "
    "términos que podrían preguntar.",
    "<b>Parte 5 — Errores comunes y cómo evitarlos.</b> Lo que NO debes "
    "hacer durante la sustentación.",
]))

story.append(h3("Recomendación de estudio"))
story.append(p(
    "Lee este documento dos veces antes de la sustentación. La primera "
    "vez con calma, marcando con resaltador los números que no recuerdas. "
    "La segunda vez solo para asegurarte de que dominas los números y "
    "puedes responder las preguntas sin leer."
))

story.append(page_break())


# ====================== PARTE 1 — SECCIÓN POR SECCIÓN ======================
story.append(h1("Parte 1 — Plan de presentación: qué decir en cada sección"))

story.append(p(
    "La sustentación dura 15 minutos. Se divide en 7 secciones según el "
    "plan oficial. Aquí desglosamos cada una con: qué decir, qué visual "
    "mostrar del libro, qué números recordar y posibles preguntas "
    "específicas de cada sección."
))


# ---------- SECCIÓN 1 — Introducción (1 min, 7%) ----------
story.append(h2("Sección 1 — Introducción (1 min, 7%)"))

story.append(h3("Qué decir"))
story.append(quote(
    "Buenos días. Vamos a presentar la predicción de volatilidad realizada "
    "de Intel Corporation con horizonte de un día, usando técnicas de "
    "Machine Learning y benchmarks econométricos. El proyecto integra los "
    "tres entregables del curso con énfasis en reproducibilidad, "
    "validación anti-leakage estricta y comparación estadística rigurosa "
    "entre modelos. Trabajamos sobre 7 022 días de cotización entre 1990 "
    "y 2017."
))

story.append(h3("Qué visual mostrar"))
story.append(bullets([
    "Diapositiva de portada con título, equipo, y logo Uninorte",
    "Opcional: gráfico de la serie de precios INTC (figura del NB 02) "
    "para situar visualmente al jurado",
]))

story.append(h3("Datos clave a mencionar"))
story.append(fact_table([
    ["Dato", "Valor"],
    ["Activo", "Intel Corporation (INTC), NASDAQ"],
    ["Período", "1990 – 2017 (27 años)"],
    ["Frecuencia", "Diaria (OHLCV)"],
    ["Días de datos", "7 022"],
    ["Tareas", "Regresión + clasificación binaria"],
    ["Modelos evaluados", "Más de 40 configuraciones"],
]))

story.append(h3("Tono"))
story.append(p(
    "Formal y seguro. Es el primer minuto, donde se establece autoridad. "
    "Hablen claro, sin titubeos, con velocidad moderada. No es momento "
    "para detalles técnicos."
))

story.append(page_break())


# ---------- SECCIÓN 2 — Definición del problema (1 min, 7%) ----------
story.append(h2("Sección 2 — Definición del problema (1 min, 7%)"))

story.append(h3("Qué decir"))
story.append(quote(
    "Para cada día t, queremos predecir dos magnitudes complementarias "
    "para el día siguiente: primero, la volatilidad realizada sigma "
    "t+1, definida como la desviación estándar de los log-retornos en "
    "una ventana causal de 5 días. Esto es una regresión. Segundo, el "
    "régimen de volatilidad — bajo o alto — separado por la mediana de "
    "la volatilidad calculada únicamente sobre el conjunto de "
    "entrenamiento. Esto es una clasificación binaria. La decisión "
    "metodológica más importante del proyecto fue fijar el horizonte en "
    "un día. El proyecto académico anterior usó horizontes de 7 a 21 "
    "días y obtuvo R² negativo universal. Con horizonte de un día "
    "recuperamos el efecto de clustering de corto plazo, donde la "
    "volatilidad sí es predecible."
))

story.append(h3("Qué visual mostrar"))
story.append(bullets([
    "Tabla de comparación: R² del proyecto anterior (negativo) vs R² "
    "del actual (+0.74)",
    "Fórmula formal del target en una diapositiva: σ<sub>t+1</sub> = "
    "std(retornos en ventana de 5 días)",
    "Esquema del split temporal 70/15/15 con fechas",
]))

story.append(h3("Datos clave a mencionar"))
story.append(fact_table([
    ["Dato", "Valor"],
    ["Horizonte", "1 día (corrección metodológica clave)"],
    ["Ventana del target", "5 días (volatilidad realizada rolling)"],
    ["Target regresión", "σ<sub>t+1</sub>"],
    ["Target clasificación", "ρ<sub>t+1</sub> ∈ {0, 1}"],
    ["Umbral del régimen", "Mediana de σ en train solo"],
    ["Split train/val/test", "70% / 15% / 15% cronológico"],
    ["Train", "1990 - 2009 (~5 000 días)"],
    ["Validation", "2009 - 2013 (~1 000 días)"],
    ["Test", "2013 - 2017 (~1 050 días, Gran Calma)"],
]))

story.append(h3("Punto crítico a defender"))
story.append(info_box(
    "¿Por qué cambiar el horizonte? No es huir de la dificultad",
    "El horizonte se eligió metodológicamente, no por evitar el reto. La "
    "autocorrelación de la volatilidad realizada decae rápidamente con "
    "el horizonte. A 1 día capturamos la fracción genuinamente predecible. "
    "A 7-21 días esa señal ya no existe en los datos y por eso ningún "
    "modelo lo lograba. Esta decisión es defensa, no excusa."
))

story.append(page_break())


# ---------- SECCIÓN 3 — EDA y preprocesamiento (3 min, 20%) ----------
story.append(h2("Sección 3 — EDA y preprocesamiento (3 min, 20%)"))

story.append(h3("Qué decir"))
story.append(quote(
    "El análisis exploratorio confirmó los hechos estilizados de la "
    "volatilidad financiera. Los retornos no presentan autocorrelación "
    "significativa, lo cual es consistente con eficiencia débil de "
    "mercado. Pero los retornos al cuadrado, que son un proxy de la "
    "magnitud, sí presentan autocorrelación fuerte. Esto es el "
    "clustering de volatilidad, lo que justifica usar modelos como "
    "GARCH y HAR-RV como benchmarks. Construimos 31 features causales, "
    "todas calculadas con shift de un día para evitar leakage de "
    "información futura. Cada decisión de preprocesamiento está dentro "
    "de un Pipeline de scikit-learn — no escalamos ni imputamos antes "
    "del split."
))

story.append(h3("Qué visuales mostrar"))
story.append(bullets([
    "ACF de retornos vs ACF de retornos al cuadrado (NB 02): el "
    "clustering queda gráficamente evidente",
    "Distribución del target_vol por subconjunto (NB 03): ver si train, "
    "val y test tienen distribuciones similares",
    "Lista de las 31 features agrupadas por familia (lags, rolling, "
    "HAR-RV, momentum, ATR, RSI, volumen, calendario)",
]))

story.append(h3("Datos clave a mencionar"))
story.append(fact_table([
    ["Dato", "Valor"],
    ["Total features construidas", "31 features causales"],
    ["Familias de features", "8 (lags retornos, vol rolling, HAR-RV, momentum, ATR, RSI, volumen, calendario)"],
    ["Anti-leakage", "Shift(1) en todas las features, fit en train solo"],
    ["Tests pytest", "13 tests verificando ausencia de leakage"],
    ["% positivos en train (régimen alto)", "50% por construcción"],
    ["% positivos en test", "10% (distribution shift)"],
    ["Período del distribution shift", "2013-2017 fue período de calma anómala"],
]))

story.append(h3("Punto técnico fuerte a destacar"))
story.append(info_box(
    "Anti-leakage es estructural, no decorativo",
    "Todo el preprocesamiento (imputación, escalado, balanceo con SMOTE/"
    "ADASYN) está dentro de pipelines de scikit-learn o imbalanced-learn. "
    "Los splits son cronológicos. La validación cruzada usa TimeSeriesSplit. "
    "El umbral del régimen se calcula con la mediana del train, NO de todos "
    "los datos. Tenemos 13 tests pytest automatizados que verifican esto.",
    color=PRIMARY,
))

story.append(page_break())


# ---------- SECCIÓN 4 — Metodología y modelos (4 min, 27%) ----------
story.append(h2("Sección 4 — Metodología y modelos (4 min, 27%)"))

story.append(h3("Qué decir"))
story.append(quote(
    "Implementamos modelos en cuatro categorías. Primero, seis benchmarks "
    "econométricos clásicos: Naive, Rolling mean, EWMA, ARIMA, GARCH "
    "y HAR-RV. Naive es el más difícil de batir porque la "
    "autocorrelación de orden uno es alta. Segundo, siete regresores "
    "supervisados: Ridge, Lasso, KNN, Decision Tree, Random Forest, SVR "
    "y XGBoost. Cada uno dentro de pipeline con imputador y escalador. "
    "Tercero, ocho clasificadores para el régimen: KNN, Naive Bayes, "
    "regresión logística L1 y L2, árbol, Random Forest, SVM y XGBoost. "
    "Y cuarto, optimización del XGBoost regresor por cuatro métodos: "
    "Grid Search, Random Search, Optuna y un algoritmo genético con DEAP."
))

story.append(h3("Modelos por familia"))
story.append(fact_table([
    ["Familia", "Modelos"],
    ["Benchmarks econométricos", "Naive, Rolling mean, EWMA, ARIMA, GARCH(1,1), HAR-RV"],
    ["Regresores ML", "Ridge, Lasso, KNN, Decision Tree, Random Forest, SVR, XGBoost"],
    ["Clasificadores ML", "KNN, NB, LogReg L1, LogReg L2, Tree, RF, SVM, XGB"],
    ["Métodos de optimización", "Grid, Random, Optuna (Bayes), DEAP (genético)"],
    ["Modelo original", "HVRF — Hybrid Volatility Regime Forecaster"],
]))

story.append(h3("Qué decir del modelo original (HVRF)"))
story.append(quote(
    "Como modelo original diseñamos HVRF, que combina tres componentes. "
    "Un clasificador XGBoost que estima la probabilidad p de régimen "
    "alto para el día siguiente. Un regresor entrenado solo con días de "
    "régimen alto. Y un regresor entrenado solo con días de régimen "
    "bajo. La predicción final es la mezcla suave ponderada por la "
    "probabilidad: p multiplicado por el regresor alto más uno menos p "
    "por el regresor bajo. Es una mezcla de expertos sin meta-learner. "
    "Lo elegimos porque pensamos que un modelo único general no "
    "capturaba bien los dos regímenes."
))

story.append(h3("Visualizaciones a mostrar"))
story.append(bullets([
    "Diagrama conceptual del HVRF con su fórmula matemática en una diapositiva",
    "Esquema del pipeline anti-leakage: imputer → scaler → modelo",
    "Comparativa de los 4 métodos de optimización (curva de convergencia "
    "del NB 08)",
]))

story.append(h3("Hiperparámetros clave a saber"))
story.append(fact_table([
    ["Modelo", "Hiperparámetro principal", "Valor"],
    ["Ridge", "alpha", "Optimizado por CV"],
    ["XGBoost-Optuna", "n_estimators", "1000 con early stopping"],
    ["XGBoost-Optuna", "tree_method", "hist (rápido)"],
    ["HVRF clasificador", "early_stopping_rounds", "30 con val"],
    ["TimeSeriesSplit", "n_splits", "5"],
], col_widths=[1.7 * inch, 2.0 * inch, 2.5 * inch]))

story.append(page_break())


# ---------- SECCIÓN 5 — Evaluación de modelos (4 min, 27%) ----------
story.append(h2("Sección 5 — Evaluación de modelos (4 min, 27%)"))

story.append(h3("Qué decir"))
story.append(quote(
    "El mejor modelo del proyecto es el ensamble simple — el promedio "
    "sin pesos entrenables de Ridge y XGBoost-Optuna. RMSE test 0.00370, "
    "R² test 0.74, supera a Naive en 13% con diferencia estadísticamente "
    "significativa por Diebold-Mariano. El hallazgo más importante es "
    "que XGBoost-Optuna y Ridge no son estadísticamente distinguibles "
    "entre sí. El test de Diebold-Mariano arroja p igual a 0.39, los "
    "intervalos de confianza bootstrap se solapan. Esto significa que el "
    "ranking puntual no implica significancia. Para clasificación, "
    "Random Forest y XGBoost obtienen AUC test de 0.9559, también "
    "indistinguibles entre sí según el test de DeLong. La interpretabilidad "
    "explica por qué el modelo lineal logra lo mismo que el no lineal: "
    "una sola feature, vol_5, explica el 55% del modelo. Cuando un modelo "
    "depende casi todo de una feature, la regresión lineal en esa feature "
    "captura prácticamente lo mismo."
))

story.append(h3("Resultados de regresión (RMSE test)"))
story.append(fact_table([
    ["Modelo", "RMSE test"],
    ["1. Ensamble simple (½ Ridge + ½ XGB-Optuna)", "<b>0.00370</b> ⭐"],
    ["2. XGBoost-Optuna", "0.00382"],
    ["3. Ridge", "0.00384"],
    ["4. HVRF (modelo original)", "0.00390"],
    ["5. XGBoost (NB 05 base)", "0.00396"],
    ["6. Naive (benchmark)", "0.00425"],
    ["7. Lasso", "0.00426"],
    ["Peor: SVR sin tuning", "0.02286"],
]))

story.append(h3("Resultados de clasificación (AUC test)"))
story.append(fact_table([
    ["Modelo", "AUC test", "F1 test"],
    ["Random Forest", "0.9559", "0.8061"],
    ["XGBoost", "0.9557", "0.8125"],
    ["LogReg L1 / L2", "0.9443 / 0.9442", "0.7708 / 0.7708"],
    ["Decision Tree", "0.9251", "0.7053"],
    ["SVM (RBF)", "0.9136", "0.6532"],
    ["Gaussian NB", "0.8703", "0.5455"],
    ["KNN", "0.8393", "0.4384"],
], col_widths=[2.0 * inch, 1.5 * inch, 1.0 * inch]))

story.append(h3("Resultados estadísticos formales"))
story.append(fact_table([
    ["Test", "Resultado"],
    ["Diebold-Mariano regresión", "117/136 pares significativos α=0.05<br/>99/136 sobreviven Bonferroni"],
    ["DeLong clasificación", "18/28 pares significativos α=0.05<br/>12/28 sobreviven Bonferroni"],
    ["DM: HVRF vs Ridge", "p = 0.39 (NO significativo)"],
    ["DM: HVRF vs XGB-Optuna", "p = 0.10 (NO significativo)"],
    ["DM: HVRF vs Ensamble simple", "p < 0.001 (Ensamble gana)"],
    ["Bootstrap CI 95% Ridge", "[0.00345, 0.00427]"],
    ["Bootstrap CI 95% XGB-Optuna", "[0.00351, 0.00415]"],
]))

story.append(h3("Visualizaciones críticas a mostrar"))
story.append(bullets([
    "Forest plot del NB 11 — comparación de RMSE con IC bootstrap",
    "Heatmap DM — ver el verde dominante (significativo) y el rojo (no sig)",
    "Heatmap importancia combinada del NB 12 — vol_5 dominando con 0.55",
]))

story.append(h3("Lo más importante a defender"))
story.append(info_box(
    "Honestidad estadística como rigor académico",
    "Reportar honestamente que XGBoost-Optuna no supera a Ridge "
    "significativamente, y que el modelo original HVRF no supera al "
    "ensamble simple, es lo que distingue un proyecto académico maduro "
    "de uno que vende humo. La conclusión no es una falla — es el "
    "hallazgo principal del proyecto.",
    color=PRIMARY,
))

story.append(page_break())


# ---------- SECCIÓN 6 — Utilidad en toma de decisiones (1 min, 7%) ----------
story.append(h2("Sección 6 — Utilidad en toma de decisiones (1 min, 7%)"))

story.append(h3("Qué decir"))
story.append(quote(
    "El modelo entrega valor en dos dimensiones distintas. Primero, para "
    "decisiones direccionales del régimen, el AUC de 0.96 permite "
    "clasificar con alta confianza si mañana será un día de alta o baja "
    "volatilidad. Esto es operacionalmente útil para dimensionar "
    "posiciones, decidir coberturas, ajustar primas de opciones o "
    "calcular Value-at-Risk dinámico. Segundo, para predecir el nivel "
    "exacto, el RMSE de 0.0037 representa aproximadamente un 25% de "
    "error relativo sobre la volatilidad media. Útil pero no preciso. "
    "Un trader real combinaría este modelo con análisis fundamental — "
    "noticias, eventos macro — que no están en las features."
))

story.append(h3("Usos prácticos del modelo"))
story.append(fact_table([
    ["Aplicación", "Métrica relevante", "Calidad"],
    ["Clasificar régimen mañana", "AUC clasificación", "AUC 0.96 — alta"],
    ["Estimar nivel exacto", "RMSE regresión", "0.0037 — moderada"],
    ["Detectar pico volatilidad", "Recall clase positiva", "0.72 — aceptable"],
    ["Evitar falso pico", "Precision clase positiva", "0.91 — alta"],
], col_widths=[2.0 * inch, 1.8 * inch, 1.8 * inch]))

story.append(h3("Limitaciones operacionales a reconocer"))
story.append(bullets([
    "El modelo NO captura shocks fundamentales (noticias, eventos macro) "
    "que no están en las features",
    "El test cubre 2013-2017, período de baja volatilidad. No validamos "
    "contra COVID-19 ni el rally de IA de 2023-2024",
    "El horizonte de un día es operativamente corto. Decisiones de "
    "mediano plazo necesitan otra metodología",
]))

story.append(page_break())


# ---------- SECCIÓN 7 — Conclusión y trabajos futuros (1 min, 5%) ----------
story.append(h2("Sección 7 — Conclusión y trabajos futuros (1 min, 5%)"))

story.append(h3("Qué decir"))
story.append(quote(
    "En resumen, el proyecto integra las técnicas pedidas por la rúbrica "
    "con anti-leakage estricto, comparación estadística rigurosa, "
    "interpretabilidad con SHAP y LIME, y un modelo original con "
    "ablación completa. El mejor modelo es el ensamble simple Ridge + "
    "XGBoost-Optuna, con RMSE test 0.00370 y R² 0.74. El modelo "
    "original HVRF está implementado correctamente, aunque no supera "
    "estadísticamente al ensamble simple. Como trabajo futuro proponemos "
    "tres líneas: primero, walk-forward backtesting con ventanas "
    "deslizantes; segundo, incorporar datos exógenos como sentimiento de "
    "noticias o el VIX; tercero, modelos de cambio de régimen tipo "
    "Markov-switching para capturar transiciones explícitas. Gracias."
))

story.append(h3("Trabajos futuros priorizados"))
story.append(fact_table([
    ["Línea", "Descripción", "Plazo"],
    ["Walk-forward backtesting", "Múltiples ventanas de evaluación en lugar de un test único", "1-2 semanas"],
    ["Recalibración de probabilidades", "Platt scaling o isotónica sobre el clasificador HVRF", "1 semana"],
    ["Optuna con presupuesto mayor", "200+ trials con pruning", "3 días"],
    ["Markov-switching", "Modelar transiciones de régimen explícitamente", "1-2 meses"],
    ["LSTM / Transformer", "Modelos secuenciales sobre lags de retornos", "1-2 meses"],
    ["Datos exógenos", "Sentimiento de noticias FinBERT, VIX, datos macro", "3 meses"],
], col_widths=[1.8 * inch, 3.4 * inch, 0.9 * inch]))

story.append(h3("Cierre"))
story.append(p(
    "El cierre es breve, agradece y deja espacio para preguntas. "
    "Mantengan tono firme, no se disculpen por limitaciones reconocidas. "
    "Las limitaciones honestas son fortaleza académica, no debilidad."
))

story.append(page_break())


# ====================== PARTE 2 — BANCO DE PREGUNTAS ======================
story.append(h1("Parte 2 — Banco de preguntas anticipadas"))
story.append(p(
    "27 preguntas probables del jurado, organizadas por temática. Cada "
    "una con respuesta corta (memorizable) y, si profundizan más, una "
    "expansión técnica. Memoricen el sentido, no el texto literal."
))


# --- SOBRE EL PROBLEMA Y HORIZONTE ---
story.append(h2("Sobre el problema y la elección del horizonte"))

story.append(qa_block(
    "¿Por qué horizonte de 1 día y no 7 como en el proyecto anterior?",
    "Decisión metodológica, no para evitar dificultad. En el proyecto "
    "anterior con horizonte 7 días el R² fue negativo universal — ningún "
    "modelo superaba la media histórica. Con horizonte 1 día capturamos "
    "el efecto de clustering de corto plazo donde la autocorrelación de "
    "la volatilidad realizada es fuerte y predecible.",
    "La autocorrelación de la volatilidad realizada decae rápidamente "
    "con el horizonte. A 1 día la dependencia serial es alta (alrededor "
    "de 0.6). A 7 días cae a 0.2 y a 21 días casi se anula. Por eso "
    "horizontes largos producen R² negativo: no hay señal predecible "
    "en los datos."
))

story.append(qa_block(
    "¿Qué pasaría con un horizonte de 5 o 10 días?",
    "Esperamos que R² baje sustancialmente. A 5 días HAR-RV y XGBoost "
    "con features de larga escala podrían mantener R² positivo entre "
    "0.3 y 0.5. A 10 días la mayoría de modelos probablemente caerían "
    "por debajo de Naive. No lo validamos empíricamente.",
    "Es una extensión natural del proyecto. Bastaría con redefinir "
    "target_vol y target_regime con shift mayor y reentrenar. La "
    "infraestructura está lista para esto."
))

story.append(qa_block(
    "¿Por qué eligieron Intel Corporation y no otro stock?",
    "Tres razones. Primero, histórico OHLCV largo (más de 25 años), "
    "ofrece regímenes diversos. Segundo, liquidez alta — la volatilidad "
    "observada es genuina, no microestructura. Tercero, sector "
    "tecnológico con episodios de estrés claramente identificables, "
    "útiles para evaluar robustez.",
    "También consideramos AAPL y MSFT pero INTC tenía la cobertura "
    "temporal más larga sin gaps y un dataset limpio disponible en "
    "Kaggle."
))

story.append(qa_block(
    "¿El dataset cubre la pandemia COVID-19?",
    "No. El dataset llega hasta 2017. Volatilidad realizada de INTC "
    "durante COVID-19 (marzo 2020) no está en este estudio.",
    "Esto es una limitación honesta del proyecto. Para evaluar contra "
    "COVID o el rally de IA de 2023-2024 deberíamos re-bajar el dataset "
    "y re-validar. El pipeline está diseñado para ser re-aplicado a "
    "cualquier período."
))


# --- SOBRE MODELOS Y MÉTODOS ---
story.append(h2("Sobre los modelos y métodos"))

story.append(qa_block(
    "Dicen que Ridge y XGBoost-Optuna son indistinguibles. Entonces, "
    "¿por qué hicieron toda la optimización del NB 8?",
    "Porque sin haberla hecho no podríamos demostrar el empate. Aplicar "
    "los cuatro métodos de optimización con presupuestos comparables nos "
    "permite reportar con evidencia que la diferencia entre Ridge y "
    "XGBoost-Optuna no es significativa. Si solo hubiéramos entrenado "
    "Ridge baseline, habríamos asumido incorrectamente que XGBoost es "
    "mejor.",
    "Además, el ejercicio de optimización es valioso pedagógicamente — "
    "permite comparar empíricamente las características de cada método "
    "(Grid es exhaustivo pero rígido, Random es eficiente, Optuna usa "
    "información previa, DEAP explora con diversidad genética)."
))

story.append(qa_block(
    "¿Qué es exactamente HVRF y por qué lo eligieron como modelo original?",
    "HVRF significa Hybrid Volatility Regime Forecaster. Combina un "
    "clasificador XGBoost que estima la probabilidad p del régimen alto, "
    "y dos regresores especializados — uno entrenado solo con días "
    "altos, otro solo con días bajos. La predicción final es la mezcla "
    "ponderada por p. Es una mezcla de expertos sin meta-learner. Lo "
    "elegimos porque la interpretabilidad sugería que los dos regímenes "
    "tienen dinámicas distintas.",
    "La justificación teórica es triple. Primero, la especialización "
    "reduce el sesgo de regresión hacia la media. Segundo, la "
    "ponderación por probabilidad permite transición suave entre "
    "regímenes. Tercero, captura no-linealidades cruzadas que un modelo "
    "único no puede modelar explícitamente."
))

story.append(qa_block(
    "Si el ensamble simple es mejor que HVRF, ¿por qué no lo declaran como "
    "modelo original?",
    "Por dos razones. Primero, el ensamble simple no es un diseño "
    "original — es un combinador trivial. Declararlo como original "
    "sería poco apropiado para la rúbrica que pide creatividad "
    "metodológica. Segundo, sería oportunismo retroactivo. Reportamos "
    "honestamente que HVRF no aporta ventaja y que el ensamble simple "
    "emerge como mejor — respetando el método pero reconociendo el "
    "resultado.",
    "Esta es la conclusión académica más madura del proyecto. La "
    "complejidad arquitectónica no compra ventaja automáticamente. La "
    "diversificación de sesgos (que el ensamble logra naturalmente) "
    "puede ser más valiosa que la sofisticación de mezcla de expertos "
    "condicional."
))

story.append(qa_block(
    "¿Por qué no usaron deep learning como LSTM o Transformer?",
    "Porque la rúbrica del curso pidió específicamente las técnicas que "
    "aplicamos: KNN, NB, LogReg, árboles, ensambles, SVM y XGBoost. Las "
    "técnicas secuenciales como LSTM y Transformer son material del "
    "siguiente curso. Las mencionamos en trabajo futuro porque sería "
    "una extensión natural.",
    "Además, en este dataset de tamaño moderado (5000 muestras de "
    "entrenamiento), los métodos clásicos son competitivos con deep "
    "learning. Los modelos profundos brillan en regímenes de big data y "
    "alta dimensionalidad, no necesariamente aquí."
))

story.append(qa_block(
    "¿No es trampa usar los hiperparámetros del Optuna del NB 8 para "
    "entrenar los componentes del HVRF?",
    "No es trampa. Esos hiperparámetros se obtuvieron usando solo train "
    "y validación cruzada interna; no tocaron el test. Usarlos en HVRF "
    "es transferir conocimiento legítimo del proyecto. Lo que sería "
    "leakage es haber optimizado los hiperparámetros de HVRF mirando el "
    "test, y eso no lo hicimos.",
    "El principio general es: el test se usa una sola vez al final con "
    "la arquitectura completamente congelada. Todo lo demás (selección "
    "de modelo, hiperparámetros, ablación) usa solo train y val."
))


# --- SOBRE EVALUACIÓN Y ESTADÍSTICA ---
story.append(h2("Sobre evaluación y estadística"))

story.append(qa_block(
    "¿Por qué Diebold-Mariano y no un t-test pareado?",
    "Porque los residuos de los modelos no son normales (Jarque-Bera "
    "rechaza para todos) y presentan autocorrelación serial (Ljung-Box "
    "rechaza). Estas dos violaciones invalidan los t-tests paramétricos. "
    "Diebold-Mariano con corrección Newey-West tolera autocorrelación y "
    "no requiere normalidad.",
    "Newey-West es una corrección de la varianza que ajusta por la "
    "autocorrelación de la serie de diferencias de pérdida. Usamos "
    "bandwidth igual al horizonte (1 en este caso). La estadística "
    "resultante converge a N(0,1) bajo H₀."
))

story.append(qa_block(
    "¿Qué es DeLong y por qué lo usaron?",
    "DeLong es un test para comparar dos AUCs sobre las mismas "
    "etiquetas. Aprovecha la teoría de U-estadísticas para construir "
    "una varianza estructural sin asumir distribuciones específicas. "
    "Usamos la implementación de Sun y Xu de 2014, que es la más "
    "robusta numéricamente.",
    "El test es esencialmente: para cada par de clasificadores, calcula "
    "la diferencia de AUC y la divide por su error estándar (que toma en "
    "cuenta que ambos clasificadores se evalúan sobre las MISMAS "
    "instancias). Es por eso mucho más potente que comparar AUCs como "
    "estimadores independientes."
))

story.append(qa_block(
    "Aplican Bonferroni. ¿Por qué no Holm o Benjamini-Hochberg que son "
    "menos conservadores?",
    "Por simplicidad y porque la rúbrica acepta Bonferroni sin requerir "
    "métodos más sofisticados. Bonferroni es más conservador pero más "
    "fácil de explicar y defender en una sustentación. Holm y BH "
    "controlarían mejor el FDR pero darían resultados cualitativamente "
    "similares en este caso.",
    "Con 136 comparaciones a alpha = 0.05/136 = 0.00037, sobreviven 99 "
    "pares. Si usáramos Holm o BH sobrevivirían quizá 105-110. La "
    "conclusión central no cambia: las diferencias entre los modelos "
    "top no son significativas."
))

story.append(qa_block(
    "¿Cómo manejaron el desbalance del test?",
    "Con tres decisiones explícitas. Primero, usamos AUC como métrica "
    "principal, que es robusta al desbalance. Segundo, evaluamos SMOTE, "
    "ADASYN y class_weight, y reportamos honestamente que no aportan "
    "porque el desbalance del test es distribution shift, no class "
    "imbalance tradicional. Tercero, controlamos el bootstrap CI por el "
    "número de positivos.",
    "El distribution shift se debe a que el período de test (2013-2017) "
    "fue una era de baja volatilidad anómala — la Gran Calma — donde "
    "los días de régimen alto fueron escasos. No es un problema del "
    "train sino del cambio de distribución temporal."
))

story.append(qa_block(
    "¿Qué significa que los IC bootstrap se solapen?",
    "Que estadísticamente no podemos distinguir esos dos modelos con "
    "95% de confianza. Es una regla informal: si los IC se solapan, la "
    "diferencia entre las métricas puntuales no es robusta a la "
    "variación muestral.",
    "Es más conservador que un test formal de diferencia de medias, "
    "pero da una intuición visual muy poderosa para sustentaciones. En "
    "el forest plot del NB 11 se ve claramente que los IC de Ridge y "
    "XGBoost-Optuna se solapan completamente."
))


# --- SOBRE INTERPRETABILIDAD ---
story.append(h2("Sobre interpretabilidad"))

story.append(qa_block(
    "¿Cuál es la diferencia entre SHAP y LIME, y por qué usaron ambos?",
    "SHAP descompone cada predicción en contribuciones por feature usando "
    "valores de Shapley de teoría de juegos. Es matemáticamente sólido y "
    "globalmente consistente. LIME entrena un modelo lineal local en una "
    "vecindad de la instancia y reporta sus coeficientes. Es más "
    "intuitivo pero menos riguroso. Usamos ambos porque la rúbrica "
    "exigía LIME y SHAP es complementario.",
    "SHAP tiene la propiedad de aditividad — la suma de los valores SHAP "
    "más el valor base reproduce exactamente la predicción del modelo. "
    "LIME no garantiza esto. Por eso SHAP es preferible para interpretación "
    "global, mientras LIME es útil para casos específicos."
))

story.append(qa_block(
    "Encontraron que vol_5 domina el modelo. ¿Eso significa que las otras "
    "30 features no sirven?",
    "No exactamente. Significa que vol_5 captura la mayor parte de la "
    "señal predecible, pero las otras features aportan contribuciones "
    "menores que sí mejoran el desempeño. Quitar vol_5 degradaría el "
    "modelo dramáticamente; quitar features individuales del resto "
    "tendría efecto marginal.",
    "El proyecto incluye un análisis de permutation importance que "
    "cuantifica esto: vol_5 contribuye con 0.59 normalizado, las "
    "siguientes features (retornos rezagados, componentes HAR-RV) "
    "contribuyen entre 0.05 y 0.07 cada una."
))

story.append(qa_block(
    "El caso de enero 2016 con LIME muestra FN, HIT y FP en 8 días. ¿Eso es "
    "una limitación del modelo?",
    "Más bien es una característica esperable de cualquier modelo basado "
    "en rolling windows. El cluster de volatilidad inicia abruptamente, "
    "el modelo no lo detecta (FN); luego se ajusta y predice bien (HIT); "
    "y al final del cluster sigue prediciendo alto cuando ya bajó (FP). "
    "Es el efecto de retraso natural de cualquier predictor basado en "
    "promedios móviles.",
    "Mitigarlo requeriría modelos con memoria explícita del cambio de "
    "régimen, como Markov-switching o filtros de Kalman. Lo mencionamos "
    "en trabajo futuro."
))


# --- SOBRE REPRODUCIBILIDAD Y CÓDIGO ---
story.append(h2("Sobre reproducibilidad y código"))

story.append(qa_block(
    "¿Cómo garantizan que no hay data leakage?",
    "Con cuatro mecanismos. Primero, todo el preprocesamiento está dentro "
    "de Pipelines de scikit-learn — el imputador y el escalador se "
    "ajustan solo en train. Segundo, la validación cruzada usa "
    "TimeSeriesSplit que respeta el orden temporal. Tercero, el umbral "
    "del régimen se calcula con la mediana del train solo. Cuarto, "
    "tenemos 13 tests pytest que verifican explícitamente que las "
    "features no usen información futura y que los targets no aparezcan "
    "en X.",
    "Los tests son automáticos: cualquier cambio futuro al código que "
    "rompa anti-leakage hace fallar los tests. Es defensa estructural, "
    "no solo cuidado puntual al escribir el código."
))

story.append(qa_block(
    "¿Es reproducible el proyecto?",
    "Sí, completamente. RANDOM_STATE = 42 está centralizado en "
    "src/config.py y se propaga a todos los modelos. Las versiones de "
    "las librerías están fijadas en requirements.txt. Los outputs están "
    "persistidos en formato parquet y JSON con esquemas estables. "
    "Cualquier persona puede clonar el repo y ejecutar jupyter-book "
    "build para obtener exactamente el mismo libro.",
    "El proyecto sigue el estándar de reproducibilidad académica: "
    "código abierto, semillas fijas, datos disponibles públicamente "
    "(Kaggle), versiones explícitas. Es replicable en cualquier máquina "
    "con Python 3.11+."
))


# --- SOBRE LIMITACIONES Y FUTURO ---
story.append(h2("Sobre limitaciones y trabajo futuro"))

story.append(qa_block(
    "¿Cuál es la limitación más importante del proyecto?",
    "Tres son importantes. Primero, no incorporamos información exógena "
    "— solo OHLCV, sin noticias ni datos macro. Segundo, el test es "
    "único, no walk-forward, lo cual limita la robustez del análisis. "
    "Tercero, el dataset llega hasta 2017 y no incluye períodos "
    "recientes de volatilidad como COVID-19 ni el rally de IA.",
    "Las tres son limitaciones de diseño del proyecto que reconocemos "
    "explícitamente. Cada una tiene una propuesta de mitigación en la "
    "sección de trabajo futuro: FinBERT para sentiment, walk-forward "
    "para evaluación robusta, re-bajar dataset para período moderno."
))

story.append(qa_block(
    "El test va de 2013 a 2017, período de baja volatilidad post-crisis. "
    "¿Cómo defienden la generalización?",
    "No defendemos generalización a 2020-2025 con este proyecto, sería "
    "deshonesto. Lo que sí defendemos es que el pipeline está diseñado "
    "para ser re-aplicado. Las features son causales, los targets están "
    "bien definidos, los procedimientos anti-leakage funcionan para "
    "cualquier período. Si tuviéramos que evaluar contra COVID, "
    "deberíamos re-bajar el dataset y re-validar — el código está listo "
    "para eso.",
    "Una mejor evaluación de generalización sería un walk-forward "
    "backtesting con múltiples ventanas (por ejemplo, 5 años de train + "
    "1 año de test, deslizando un año en cada paso). Lo mencionamos en "
    "trabajo futuro como la primera prioridad."
))

story.append(qa_block(
    "Su mejor RMSE es 0.00370. ¿Eso es operacionalmente útil?",
    "Depende del uso. Para predecir el nivel exacto, 0.0037 representa "
    "aproximadamente un 25% de error relativo sobre la volatilidad media "
    "— útil pero no preciso. Para decisiones direccionales (régimen "
    "alto/bajo), la clasificación con AUC 0.96 sí es operacionalmente "
    "útil. Un trader real combinaría este modelo con análisis "
    "fundamental, no lo usaría como única señal.",
    "Por contexto: un broker que decide cobertura diaria necesita más "
    "precisión que un gestor que decide rebalanceos mensuales. Para el "
    "primer caso nuestro modelo es insuficiente; para el segundo es "
    "competitivo."
))


# --- PREGUNTAS DIFÍCILES / TRAMPA ---
story.append(h2("Preguntas potencialmente difíciles"))

story.append(qa_block(
    "Si HVRF no supera al ensamble simple, ¿por qué consideran que el "
    "modelo original fue exitoso?",
    "Hay dos definiciones de éxito en este caso. Técnicamente HVRF no "
    "fue el mejor modelo del proyecto. Pero metodológicamente lo "
    "consideramos exitoso por tres razones. Primero, cumple todos los "
    "requisitos de la rúbrica: nombre propio, justificación conceptual, "
    "ablación, comparación con baselines, conclusión honesta. Segundo, "
    "su clasificador interno (AUC 0.968) es un componente individual de "
    "alta calidad. Tercero, su evaluación reveló el hallazgo más valioso "
    "del proyecto: que la complejidad no compra ventaja.",
    "Es como diseñar un experimento científico: el experimento puede "
    "rechazar la hipótesis original y aun así ser exitoso si nos "
    "enseña algo valioso. HVRF nos enseñó que el ensamble simple "
    "domina, lo cual es académicamente valioso."
))

story.append(qa_block(
    "Veo muchos modelos con resultados muy parecidos en el top. ¿No están "
    "sobreajustando a la métrica al evaluar tantos modelos?",
    "Es una preocupación legítima de multiple-testing. Por eso aplicamos "
    "corrección Bonferroni en todas las comparaciones por pares. Con 136 "
    "comparaciones y alpha original 0.05, requerimos p menor a 0.00037 "
    "para significancia. Los 99 pares que sobreviven Bonferroni son "
    "robustos a esta corrección.",
    "Adicionalmente, los IC bootstrap del 95% son una segunda línea de "
    "defensa contra sobre-interpretación. Si dos modelos tienen IC que "
    "se solapan, no podemos concluir diferencia aunque la métrica "
    "puntual sugiera lo contrario."
))

story.append(qa_block(
    "¿Cuántas horas de trabajo en total tomó el proyecto?",
    "Aproximadamente 80-100 horas distribuidas entre los tres miembros "
    "del equipo, sin contar el tiempo de cómputo de los modelos. La "
    "mayor parte del tiempo se invirtió en garantizar reproducibilidad y "
    "anti-leakage estricto, que son fundamentos sin los cuales los "
    "resultados no serían confiables.",
    "El diseño anti-leakage tomó aproximadamente 20 horas: separación "
    "estricta de pipelines, tests pytest, validación temporal, "
    "verificación de que no haya información futura en features."
))


story.append(page_break())


# ====================== PARTE 3 — CHEATSHEET ======================
story.append(h1("Parte 3 — Cheatsheet de números críticos"))
story.append(p(
    "Estos son los 20 valores que <b>SÍ o SÍ</b> debes saber de memoria. "
    "Si te preguntan cualquiera de estos números y dudas, baja tu nota. "
    "Si los respondes con seguridad, demuestras dominio del proyecto."
))


story.append(h2("Datos del problema"))
story.append(fact_table([
    ["Concepto", "Valor"],
    ["Días de datos OHLCV", "7 022"],
    ["Período", "1990 – 2017"],
    ["Activo", "INTC (Intel Corporation)"],
    ["Features causales construidas", "31"],
    ["Tests pytest anti-leakage", "13"],
    ["Split train/val/test", "70% / 15% / 15%"],
]))

story.append(h2("Modelos top — regresión"))
story.append(fact_table([
    ["Modelo", "RMSE test", "R² test"],
    ["Ensamble simple ⭐", "0.00370", "+0.74"],
    ["XGBoost-Optuna", "0.00382", "+0.72"],
    ["Ridge", "0.00384", "+0.715"],
    ["HVRF (modelo original)", "0.00390", "+0.71"],
    ["Naive (benchmark)", "0.00425", "+0.65"],
], col_widths=[2.5 * inch, 1.3 * inch, 1.3 * inch]))

story.append(h2("Modelos top — clasificación"))
story.append(fact_table([
    ["Modelo", "AUC test", "F1 test"],
    ["Random Forest", "0.9559", "0.8061"],
    ["XGBoost", "0.9557", "0.8125"],
    ["LogReg L1", "0.9443", "0.7708"],
], col_widths=[2.5 * inch, 1.3 * inch, 1.3 * inch]))

story.append(h2("Tests estadísticos"))
story.append(fact_table([
    ["Test", "Resultado clave"],
    ["DM regresión: # pares sig α=0.05 / Bonferroni", "117/136 / 99/136"],
    ["DeLong clasif: # pares sig α=0.05 / Bonferroni", "18/28 / 12/28"],
    ["DM: HVRF vs Ridge (p)", "0.39 (NO significativo)"],
    ["DM: HVRF vs XGB-Optuna (p)", "0.10 (NO significativo)"],
    ["DM: HVRF vs Ensamble (p)", "< 0.001 (ensamble gana)"],
]))

story.append(h2("Interpretabilidad"))
story.append(fact_table([
    ["Feature", "Importancia combinada (gain + perm + SHAP)"],
    ["vol_5 (DOMINANTE)", "0.55 (más del 50% del modelo)"],
    ["ret_lag_3", "0.05"],
    ["ret_lag_2", "0.05"],
    ["ret_lag_1", "0.04"],
    ["vol_d (HAR diario)", "0.04"],
]))

story.append(h2("HVRF — componentes individuales"))
story.append(fact_table([
    ["Componente", "Métrica"],
    ["Clasificador HVRF AUC val", "0.93"],
    ["Clasificador HVRF AUC test", "0.968"],
    ["Solo R_bajo aplicado a todo el test", "RMSE 0.00501"],
    ["Solo R_alto aplicado a todo el test", "RMSE 0.01750 (mal porque test es 90% bajo)"],
]))

story.append(page_break())


# ====================== PARTE 4 — GLOSARIO ======================
story.append(h1("Parte 4 — Glosario técnico"))
story.append(p(
    "Definiciones rápidas (una frase cada una) de los términos que "
    "podrían surgir en preguntas. Memoriza al menos la definición corta."
))


story.append(h2("Conceptos financieros"))
story.append(fact_table([
    ["Término", "Definición rápida"],
    ["Volatilidad realizada", "Desviación estándar de los retornos en una ventana, medida ex-post"],
    ["Clustering de volatilidad", "Tendencia de los períodos turbulentos a agruparse en el tiempo"],
    ["Distribution shift", "Cambio en la distribución de los datos entre train y test"],
    ["Régimen", "Estado del mercado caracterizado por una dinámica particular (alta vs baja vol)"],
    ["Value-at-Risk (VaR)", "Pérdida máxima esperada con cierta probabilidad sobre un horizonte"],
    ["Cobertura (hedging)", "Estrategia para reducir la exposición al riesgo de precio"],
], col_widths=[1.8 * inch, 4.5 * inch]))

story.append(h2("Modelos econométricos"))
story.append(fact_table([
    ["Término", "Definición rápida"],
    ["Naive", "Predictor de persistencia: ŷ<sub>t+1</sub> = y<sub>t</sub>"],
    ["EWMA", "Promedio móvil con pesos exponencialmente decrecientes"],
    ["ARIMA", "Modelo autoregresivo integrado de media móvil"],
    ["GARCH(1,1)", "Modelo de varianza condicional autoregresiva — Bollerslev 1986"],
    ["HAR-RV", "Heterogeneous Autoregressive — agrega vol diaria, semanal, mensual (Corsi 2009)"],
], col_widths=[1.8 * inch, 4.5 * inch]))

story.append(h2("Modelos de Machine Learning"))
story.append(fact_table([
    ["Término", "Definición rápida"],
    ["Ridge", "Regresión lineal con penalización L2 (suma de coeficientes al cuadrado)"],
    ["Lasso", "Regresión lineal con penalización L1 (suma de |coeficientes|), induce sparsity"],
    ["XGBoost", "Gradient boosting de árboles, con regularización y paralelización"],
    ["Random Forest", "Bagging de árboles de decisión con muestreo aleatorio de features"],
    ["SVR", "Support Vector Regression — busca hiperplano con tubo epsilon-insensitive"],
    ["KNN", "K-nearest neighbors: predice promedio de los K vecinos más cercanos"],
], col_widths=[1.8 * inch, 4.5 * inch]))

story.append(h2("Técnicas de optimización"))
story.append(fact_table([
    ["Término", "Definición rápida"],
    ["Grid Search", "Búsqueda exhaustiva sobre todas las combinaciones de una grilla"],
    ["Random Search", "Muestreo aleatorio del espacio de hiperparámetros"],
    ["Optuna (TPE)", "Optimización bayesiana con árbol de estimadores de Parzen"],
    ["DEAP", "Framework de algoritmos genéticos en Python"],
], col_widths=[1.8 * inch, 4.5 * inch]))

story.append(h2("Tests estadísticos"))
story.append(fact_table([
    ["Término", "Definición rápida"],
    ["Diebold-Mariano", "Test para comparar precisión predictiva entre dos modelos"],
    ["Newey-West", "Corrección de varianza robusta a autocorrelación y heteroscedasticidad"],
    ["DeLong (Sun-Xu 2014)", "Test para comparar AUCs sobre las mismas etiquetas"],
    ["Bootstrap percentil", "IC no paramétrico basado en remuestreo con reemplazo"],
    ["Bonferroni", "Corrección de múltiples comparaciones: alpha / k"],
    ["White test", "Test de heteroscedasticidad sin asumir forma funcional"],
    ["Breusch-Pagan", "Test de heteroscedasticidad asumiendo forma lineal"],
    ["BDS test", "Test de independencia para detectar no-linealidades ocultas"],
    ["Ljung-Box", "Test de autocorrelación serial hasta cierto lag"],
    ["Jarque-Bera", "Test de normalidad basado en asimetría y kurtosis"],
], col_widths=[1.8 * inch, 4.5 * inch]))

story.append(h2("Balanceo de clases"))
story.append(fact_table([
    ["Término", "Definición rápida"],
    ["SMOTE", "Synthetic Minority Oversampling Technique — interpola muestras minoritarias"],
    ["ADASYN", "Adaptive Synthetic Sampling — variante de SMOTE adaptativa a la dificultad"],
    ["class_weight", "Ponderación inversa a la frecuencia de cada clase en la función de pérdida"],
], col_widths=[1.8 * inch, 4.5 * inch]))

story.append(h2("Interpretabilidad"))
story.append(fact_table([
    ["Término", "Definición rápida"],
    ["SHAP", "Valores de Shapley aplicados a ML, descomponen cada predicción"],
    ["LIME", "Local Interpretable Model-agnostic Explanations — modelo local lineal"],
    ["Permutation importance", "Mide degradación de la métrica al permutar una feature"],
    ["Gain (XGBoost)", "Mejora media en la pérdida cuando un feature se usa para dividir un nodo"],
], col_widths=[1.8 * inch, 4.5 * inch]))

story.append(page_break())


# ====================== PARTE 5 — ERRORES COMUNES ======================
story.append(h1("Parte 5 — Errores comunes y cómo evitarlos"))
story.append(p(
    "Los errores típicos en sustentaciones de ML que pueden costar "
    "puntos. Revisa esta sección la noche antes."
))


story.append(h2("Errores de comunicación"))

story.append(h3("1. Leer del guion"))
story.append(p(
    "<b>El error</b>: tener notas y leerlas. El jurado lo detecta de "
    "inmediato y baja la nota porque parece que no dominas el contenido."
))
story.append(p(
    "<b>Cómo evitarlo</b>: ensaya en voz alta antes. Memoriza los puntos "
    "clave en lenguaje natural, no frases literales. Si te trabas, "
    "respira, mira la diapositiva y continúa."
))

story.append(h3("2. Hablar demasiado rápido por nervios"))
story.append(p(
    "<b>El error</b>: acelerar para sentirte productivo. Termina con "
    "sustentación de 8 minutos cuando son 15 y se pierde claridad."
))
story.append(p(
    "<b>Cómo evitarlo</b>: práctica con cronómetro. Cada bloque tiene "
    "tiempo asignado. Pon una alarma silenciosa cada 5 minutos para no "
    "perderte."
))

story.append(h3("3. Disculparse por las limitaciones"))
story.append(p(
    "<b>El error</b>: decir frases como \"lamentablemente no logramos...\" "
    "o \"pedimos disculpas por...\". Hace que el jurado dude del proyecto."
))
story.append(p(
    "<b>Cómo evitarlo</b>: presenta las limitaciones como decisiones "
    "metodológicas o trabajo futuro. \"Por restricciones de alcance no "
    "incluimos X, lo proponemos como extensión natural\"."
))


story.append(h2("Errores de contenido"))

story.append(h3("4. Vender humo sobre los resultados"))
story.append(p(
    "<b>El error</b>: decir \"nuestro modelo es excelente\" o \"superamos "
    "a todos los benchmarks por amplio margen\" cuando no es cierto. El "
    "jurado abre el libro y ve los números."
))
story.append(p(
    "<b>Cómo evitarlo</b>: usen los números reales. \"Mejor RMSE 0.00370, "
    "13% sobre Naive, con DM p < 0.001\". La honestidad numérica genera "
    "confianza."
))

story.append(h3("5. No reconocer el ranking puntual vs significancia"))
story.append(p(
    "<b>El error</b>: decir \"XGBoost-Optuna es el mejor modelo\" cuando "
    "su diferencia con Ridge no es significativa. El jurado pregunta y "
    "no sabes responder."
))
story.append(p(
    "<b>Cómo evitarlo</b>: di siempre \"el mejor modelo según métrica "
    "puntual fue X, pero estadísticamente no es distinguible de Y\". "
    "Es el lenguaje correcto."
))

story.append(h3("6. Confundir SHAP con LIME al explicar"))
story.append(p(
    "<b>El error</b>: usar los términos intercambiablemente o no saber "
    "explicar la diferencia."
))
story.append(p(
    "<b>Cómo evitarlo</b>: SHAP es global, basado en Shapley, "
    "aditivamente consistente. LIME es local, aproximación lineal en "
    "vecindad. Eso basta para responder."
))


story.append(h2("Errores técnicos"))

story.append(h3("7. No saber por qué TimeSeriesSplit y no KFold"))
story.append(p(
    "<b>El error</b>: si te preguntan, responder \"porque sí\" o \"porque "
    "era serie de tiempo\" sin profundizar."
))
story.append(p(
    "<b>Cómo evitarlo</b>: KFold mezcla observaciones aleatoriamente, lo "
    "cual sería leakage porque permitiría usar información futura para "
    "predecir el pasado. TimeSeriesSplit respeta el orden cronológico — "
    "el train siempre precede al val."
))

story.append(h3("8. Confundir RMSE con MAE en la presentación"))
story.append(p(
    "<b>El error</b>: decir RMSE cuando es MAE o viceversa. Demuestra "
    "falta de cuidado."
))
story.append(p(
    "<b>Cómo evitarlo</b>: RMSE penaliza más los errores grandes (cuadra "
    "los errores antes de promediar). MAE penaliza linealmente. RMSE es "
    "la métrica principal del proyecto."
))


story.append(h2("Errores de presentación visual"))

story.append(h3("9. Diapositivas con demasiado texto"))
story.append(p(
    "<b>El error</b>: leer lo mismo que está escrito en la diapositiva. "
    "El jurado se aburre y revisa el celular."
))
story.append(p(
    "<b>Cómo evitarlo</b>: máximo 5 viñetas por diapositiva, cada una de "
    "5-7 palabras. La diapositiva guía, ustedes explican. No leerla "
    "literalmente."
))

story.append(h3("10. Mostrar gráficos sin explicar el eje"))
story.append(p(
    "<b>El error</b>: cambiar de slide a un gráfico complejo y empezar "
    "a hablar sin decir qué muestra."
))
story.append(p(
    "<b>Cómo evitarlo</b>: la primera frase de cada gráfico es siempre "
    "\"Aquí vemos el eje X que representa Y, comparado contra Z\". "
    "Después de eso, interpreta."
))


story.append(h2("Errores en preguntas y respuestas"))

story.append(h3("11. Inventar respuestas que no saben"))
story.append(p(
    "<b>El error</b>: si te preguntan algo que no sabes, improvisar. El "
    "jurado nota la inseguridad."
))
story.append(p(
    "<b>Cómo evitarlo</b>: \"No exploramos eso en el proyecto, pero "
    "considero que la dirección sería X\". Decir \"no sé\" con honestidad "
    "es mejor que inventar. El jurado lo valora."
))

story.append(h3("12. Defenderse a la defensiva"))
story.append(p(
    "<b>El error</b>: cuando el jurado critica algo, ponerse a la "
    "defensiva o discutir."
))
story.append(p(
    "<b>Cómo evitarlo</b>: agradece la observación, reconoce el punto si "
    "es válido, y conecta con el trabajo futuro. \"Excelente "
    "observación, esa es una limitación que reconocemos y proponemos "
    "abordar con X\"."
))

story.append(h3("13. Repartirse las preguntas sin coordinar"))
story.append(p(
    "<b>El error</b>: cuando hacen una pregunta, los tres responden a "
    "la vez o ninguno responde, esperando que el otro hable."
))
story.append(p(
    "<b>Cómo evitarlo</b>: acuerden de antemano. Quien presentó el "
    "bloque relacionado con la pregunta responde primero. Los otros "
    "complementan si es necesario."
))


story.append(page_break())


# ====================== ANEXO — FRASES CLAVE ======================
story.append(h1("Anexo — Frases clave para memorizar"))
story.append(p(
    "Estas son 10 frases que, dichas literalmente, demuestran madurez "
    "académica al jurado. Memorízalas y úsalas en momentos pertinentes."
))


def frase(num, contexto, texto):
    return KeepTogether([
        Paragraph(
            f"<font color='#C9A227'><b>Frase {num}</b></font> "
            f"<font color='#666666'>({contexto})</font>",
            ParagraphStyle("Fhead", parent=body_style, fontSize=11,
                          fontName="Helvetica-Bold", spaceBefore=10,
                          spaceAfter=2)
        ),
        quote(texto)
    ])


story.append(frase(1, "Al presentar el horizonte",
    '"Esta decisión es metodológica, no para evitar la dificultad. La '
    'autocorrelación de la volatilidad realizada decae rápidamente con el '
    'horizonte; a un día capturamos la fracción genuinamente predecible."'
))

story.append(frase(2, "Sobre Ridge vs XGBoost",
    '"Si bien XGBoost-Optuna tiene el RMSE puntual más bajo, el test de '
    'Diebold-Mariano y los intervalos de confianza bootstrap muestran que '
    'estadísticamente Ridge y XGBoost-Optuna son indistinguibles. El ranking '
    'puntual no implica significancia."'
))

story.append(frase(3, "Sobre el modelo original",
    '"HVRF no superó al ensamble simple. Reportar esto honestamente es '
    'parte del rigor metodológico del proyecto y revela una lección '
    'académica importante: la complejidad arquitectónica no compra ventaja '
    'automáticamente."'
))

story.append(frase(4, "Cuando preguntan por anti-leakage",
    '"Todo el preprocesamiento vive dentro de pipelines, la validación '
    'cruzada usa TimeSeriesSplit, el umbral del régimen se calcula con la '
    'mediana del train solamente, y tenemos 13 tests pytest automatizados '
    'que verifican esto."'
))

story.append(frase(5, "Sobre interpretabilidad",
    '"Las tres técnicas — gain nativo, permutation importance y SHAP — '
    'convergen en identificar vol_5 como la feature dominante con un peso '
    'combinado superior al 50%. Esta convergencia da alta confianza en la '
    'jerarquía identificada."'
))

story.append(frase(6, "Sobre balanceo",
    '"Demostramos empíricamente con un experimento controlado que las '
    'técnicas de balanceo corrigen desbalance del train, no distribution '
    'shift del test. En nuestro train balanceado por construcción, SMOTE y '
    'ADASYN no aportan."'
))

story.append(frase(7, "Cuando preguntan por limitaciones",
    '"Reconocemos tres limitaciones principales: no incorporamos datos '
    'exógenos, el test es único en lugar de walk-forward, y el dataset '
    'llega hasta 2017 sin cubrir COVID-19 ni la era reciente."'
))

story.append(frase(8, "Sobre reproducibilidad",
    '"El proyecto es completamente reproducible: RANDOM_STATE centralizado, '
    'versiones fijadas, outputs persistidos en formato parquet y JSON, '
    'tests pytest automatizados. Cualquier persona puede clonar el repo y '
    'obtener exactamente los mismos resultados."'
))

story.append(frase(9, "Al hablar del caso enero 2016",
    '"El cluster de volatilidad de enero 2016 ilustra empíricamente cómo '
    'los modelos basados en rolling windows responden con retraso a los '
    'cambios de régimen — un fenómeno conocido en la literatura de '
    'volatilidad financiera reproducido aquí con LIME."'
))

story.append(frase(10, "Al cerrar la sustentación",
    '"El proyecto integra todas las técnicas pedidas con rigor metodológico '
    'y reporta resultados estadísticamente honestos. El mejor modelo es el '
    'ensamble simple Ridge + XGBoost-Optuna. El modelo original HVRF está '
    'correctamente implementado y revela una lección sobre los límites de '
    'la complejidad arquitectónica en este problema. Gracias."'
))


story.append(page_break())


# ====================== ÚLTIMA PÁGINA — PLAN DE ESTUDIO ======================
story.append(h1("Plan de estudio recomendado"))
story.append(p(
    "Cómo organizar las últimas horas antes de la sustentación. Asume "
    "que la sustentación es en X días — ajusten los plazos hacia atrás."
))


story.append(h2("3 días antes"))
story.append(bullets([
    "Lectura completa de este documento (2 horas)",
    "Revisar el sitio público en mateochang82.github.io/PROYECTO-FINAL "
    "navegando capítulo por capítulo (1 hora)",
    "Marcar con resaltador los números que NO recuerdan",
]))

story.append(h2("2 días antes"))
story.append(bullets([
    "Memorizar el cheatsheet (Parte 3) hasta poder responder cada "
    "número sin mirar",
    "Ensayar las primeras 10 preguntas del banco (Parte 2) en voz alta",
    "Revisar las 10 frases clave del anexo",
]))

story.append(h2("1 día antes"))
story.append(bullets([
    "Ensayo completo de la sustentación con cronómetro (15 minutos)",
    "Revisar errores comunes (Parte 5)",
    "Acordar entre los tres miembros del equipo qué sección presentará "
    "cada uno",
]))

story.append(h2("La mañana del día"))
story.append(bullets([
    "Releer solo el resumen ejecutivo y el cheatsheet",
    "Verificar que el sitio público funciona desde el celular",
    "Llegar 15 minutos antes al lugar de sustentación",
    "Respirar profundo. El proyecto está bien hecho. La defensa es "
    "ahora cuestión de presentación.",
]))


story.append(spacer(0.5 * inch))
story.append(p(
    "<b>Equipo:</b> Juan Camilo Conrado · Sergio Cadavid · Mateo Chang<br/>"
    "<b>Curso:</b> Machine Learning — Pregrado en Ciencia de Datos<br/>"
    "<b>Universidad del Norte</b> · Dr. Lihki Rubio · Semestre 2026-1",
    ParagraphStyle("Closing", parent=body_style, alignment=TA_CENTER,
                   fontSize=10, textColor=MUTED, leading=14)
))


# =====================================================================
#  CREAR PDF
# =====================================================================
def header_footer(canvas, doc):
    """Header y footer para todas las páginas."""
    canvas.saveState()
    
    # Footer
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        72, 30,
        "Documento privado del equipo · Sustentación INTC-Vol-ML"
    )
    canvas.drawRightString(
        letter[0] - 72, 30, f"Página {doc.page}"
    )
    
    # Línea decorativa
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(1)
    canvas.line(72, 45, letter[0] - 72, 45)
    
    canvas.restoreState()


# Construir el PDF
import sys
output_path = sys.argv[1] if len(sys.argv) > 1 else "estudio_sustentacion.pdf"

doc = SimpleDocTemplate(
    output_path, pagesize=letter,
    leftMargin=72, rightMargin=72,
    topMargin=72, bottomMargin=60,
    title="Estudio Sustentación — INTC-Vol-ML",
    author="Equipo: Conrado, Cadavid, Chang",
    subject="Material de estudio para sustentación de proyecto",
)

doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print(f"PDF generado: {output_path}")
