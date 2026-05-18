"""
Utilidades de visualización con paleta y estilos consistentes.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns

# --- Paleta canónica del proyecto ---
PALETTE = {
    "primary":   "#1f4e79",   # azul Intel-esque
    "secondary": "#c00000",   # rojo destacar
    "accent":    "#2e7d32",   # verde train
    "neutral":   "#595959",
    "train":     "#4c72b0",
    "val":       "#dd8452",
    "test":      "#937860",
}

DEFAULT_FIGSIZE_TS = (12, 4)        # series temporales
DEFAULT_FIGSIZE_DIST = (10, 6)      # distribuciones


def set_style() -> None:
    """Aplica el estilo canónico del proyecto a matplotlib/seaborn."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def savefig(path, fig=None, dpi: int = 300) -> None:
    """Guarda figura con tight_layout y dpi alto, creando carpetas si hace falta."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if fig is None:
        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
    else:
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
