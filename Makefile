# =====================================================================
#  INTC-Vol-ML — Makefile
#  Tareas de instalación, test, ejecución, build y limpieza.
# =====================================================================
.PHONY: help install test data notebooks build clean all

help:
	@echo "Comandos disponibles:"
	@echo "  make install     - Instala dependencias desde requirements.txt"
	@echo "  make test        - Corre pytest sobre tests/"
	@echo "  make data        - Descarga/regenera el dataset crudo si falta"
	@echo "  make notebooks   - Ejecuta los notebooks en orden"
	@echo "  make build       - Construye el Jupyter Book (HTML)"
	@echo "  make clean       - Borra _build, __pycache__, outputs generados"
	@echo "  make all         - install + test + build"

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest tests/ -v

data:
	python -c "from src.data_loader import load_intc; df = load_intc(); print(df.head()); print(f'{len(df)} filas cargadas')"

notebooks:
	@for nb in notebooks/*.ipynb; do \
	  echo "Ejecutando $$nb ..."; \
	  jupyter nbconvert --to notebook --execute --inplace "$$nb" --ExecutePreprocessor.timeout=600 ; \
	done

build:
	jupyter-book build .

clean:
	rm -rf _build/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf .pytest_cache/

all: install test build
