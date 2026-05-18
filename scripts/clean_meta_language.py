"""
Limpieza sistemática del lenguaje meta-sustentación, coaching y
auto-elogio en los notebooks del proyecto INTC-Vol-ML.

Reglas:
- Eliminar parentéticos chiquitos "(mencionable en sustentación)" etc.
- Reformular "Hallazgo defendible" → "Hallazgo"
- Reformular "honestidad estadística" → según contexto
- Eliminar coaching directo al lector ("vas a defender", "te paso", etc.)
- Eliminar auto-elogio ("pocos proyectos de pregrado", "demuestra dominio")
- Suavizar referencias obsesivas a "la rúbrica"
- Eliminar bloques meta enteros ("Reportable en sustentación: ...")
"""
import json
import re
from pathlib import Path


# Diccionario de reemplazos directos.
# Orden importante: las más específicas primero.
REEMPLAZOS = [
    # === Parentéticos de sustentación en títulos y prosa ===
    (r"\s*\(mencionable en sustentación\)", ""),
    (r"\s*\(reportable en sustentación\)", ""),
    (r"\s*\(reportable\)", ""),
    (r"\s*\(defendible\)", ""),
    (r"\s*\(para sustentación\)", ""),
    
    # === Frases meta-sustentación completas ===
    (r"Reportable en sustentación[:.]?", ""),
    (r"Reportable[:.]?\s+", ""),
    (r"Defendible[:.]?\s+", ""),
    (r"para la sustentación", "para el análisis"),
    (r"en la sustentación", "en el análisis"),
    (r"tu sustentación", "el análisis"),
    
    # === "Hallazgo defendible" y variantes ===
    (r"[Hh]allazgo defendible", "hallazgo"),
    (r"[Hh]allazgo más valioso", "hallazgo principal"),
    (r"hallazgo metodológico clave", "hallazgo metodológico"),
    
    # === "Honestidad" usada como tic ===
    (r"honestidad estadística crítica:?\s*", ""),
    (r"honestidad estadística:?\s*", ""),
    (r"honestidad metodológica:?\s*", ""),
    (r"\*\*Honestidad estadística\.\*\*", "**Reproducibilidad y rigor.**"),
    
    # === Coaching al lector ===
    (r"Te paso ", "Se presenta "),
    (r"te paso ", "se presenta "),
    (r"Te explico ", "Se explica "),
    (r"te explico ", "se explica "),
    (r"vas a defender", "se documenta"),
    (r"vas a destacar", "se destaca"),
    (r"vas a usar", "se usará"),
    (r"Para tu sustentación", "Para el análisis"),
    
    # === Auto-elogio promocional ===
    (r"un argumento estadístico \+ interpretativo coherente que pocos proyectos de pregrado logran articular", "un argumento coherente entre el análisis estadístico y el interpretativo"),
    (r"pocos proyectos de pregrado logran articular", ""),
    (r"que muchos proyectos de pregrado no logran articular", ""),
    (r"Esto es una conclusión académicamente honesta que la rúbrica valora explícitamente\.", ""),
    (r"Esta es una lección que muchos proyectos de pregrado no logran articular\.", ""),
    (r"demuestra dominio de", "aplica"),
    (r"demuestra dominio", "aplica las técnicas requeridas"),
    
    # === Referencias obsesivas a rúbrica ===
    (r"que pide la rúbrica", "del proyecto"),
    (r"exigidos por la rúbrica", "considerados en el proyecto"),
    (r"exigido por la rúbrica", "considerado en el proyecto"),
    (r"que exige la rúbrica", "del proyecto"),
    (r"requisito explícito de la rúbrica", "componente clave del análisis"),
    (r"REQUISITO EXPLÍCITO DE LA RÚBRICA", "COMPONENTE CLAVE DEL ANÁLISIS"),
    (r"La rúbrica pide", "El proyecto considera"),
    (r"La rúbrica exige", "El proyecto incluye"),
    (r"cumplido al 100% de la rúbrica", "aplicado en este capítulo"),
    
    # === "Mi recomendación / sugerencia / propuesta" ===
    (r"Mi recomendación:?\s*", ""),
    (r"Mi sugerencia:?\s*", ""),
    (r"Mi propuesta:?\s*", ""),
    
    # === Frases motivacionales ===
    (r"para llevarse:?\s*", ""),
    (r"Lo que esto significa académicamente\.\s*", ""),
    (r"Reportable como decisión informada", "Decisión documentada"),
    (r"Reportable como hallazgo extra", "Hallazgo adicional"),
    (r"Reportable como honestidad metodológica", ""),
    
    # === "Implicación práctica" y similares (tic) ===
    (r"\*\*Implicación práctica\.\*\*", "**Implicación.**"),
    (r"\*\*Implicación práctica para[^.]*\.\*\*", "**Conexión con el siguiente capítulo.**"),
    (r"\*\*Lectura crítica:?\*\*", "**Observación:**"),
    (r"\*\*Lectura crítica\.\*\*", "**Observación.**"),
    (r"\*\*Hallazgo:?\*\*\s*", ""),
    (r"\*\*Lección académica:?\*\*", "**Lección:**"),
    (r"\*\*Lección metodológica:?\*\*", "**Lección:**"),
    
    # === Espacios duplicados que quedan después de eliminar ===
    (r"  +", " "),
    (r"\n\n\n+", "\n\n"),
]


def aplicar_reemplazos(texto: str) -> str:
    """Aplica todos los reemplazos en orden."""
    for patron, reemplazo in REEMPLAZOS:
        texto = re.sub(patron, reemplazo, texto)
    return texto


def procesar_notebook(path: Path) -> int:
    """Procesa un notebook .ipynb. Devuelve número de celdas modificadas."""
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)
    
    n_modif = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'markdown':
            continue
        original = "".join(cell['source'])
        nuevo = aplicar_reemplazos(original)
        if nuevo != original:
            n_modif += 1
            # Reconstruir como lista de líneas para mantener formato JSON
            cell['source'] = nuevo
    
    if n_modif > 0:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
    
    return n_modif


def procesar_markdown(path: Path) -> bool:
    """Procesa un .md. Devuelve True si hubo cambios."""
    texto = path.read_text(encoding='utf-8')
    nuevo = aplicar_reemplazos(texto)
    if nuevo != texto:
        path.write_text(nuevo, encoding='utf-8')
        return True
    return False


def main():
    base = Path(".")
    
    print("=" * 72)
    print("BARRIDO DE LENGUAJE META-SUSTENTACIÓN")
    print("=" * 72)
    
    # Notebooks
    for nb_path in sorted(base.glob("notebooks/*.ipynb")):
        n = procesar_notebook(nb_path)
        marca = "✓" if n > 0 else " "
        print(f"  {marca} {str(nb_path):<55} {n} celdas modificadas")
    
    # Markdown files en raíz
    for md_path in [base / "intro.md", base / "README.md"]:
        if md_path.exists():
            changed = procesar_markdown(md_path)
            marca = "✓" if changed else " "
            print(f"  {marca} {str(md_path):<55} {'modificado' if changed else 'sin cambios'}")
    
    print("=" * 72)
    print("Barrido completo.")


if __name__ == "__main__":
    main()
