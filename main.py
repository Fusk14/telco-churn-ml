"""
Orquestador reproducible del proyecto Telco Customer Churn.

Ejecutar desde la raíz:
    .\\.venv\\Scripts\\python.exe main.py

El proceso se detiene ante el primer error para evitar continuar con artefactos
incompletos o desactualizados.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Ajusta el nombre del notebook de clustering si en tu carpeta es diferente.
NOTEBOOKS = [
    "notebooks/00_separacion_train_test.ipynb",
    "notebooks/01_EDA.ipynb",
]

SCRIPTS = [
    "src/data_processing.py",
    "src/model_optimization.py",
]

FINAL_NOTEBOOKS = [
    "notebooks/02_Modelos_Supervisados.ipynb",
    "notebooks/03_Modelo_No_Supervisado.ipynb",
]


def run_command(command: list[str], description: str) -> None:
    """Ejecuta un comando y detiene el flujo si falla."""
    print("\n" + "=" * 78)
    print(description)
    print("Comando:", " ".join(str(part) for part in command))
    print("=" * 78, flush=True)

    subprocess.run(command, cwd=ROOT, check=True)


def check_required_files() -> None:
    """Comprueba que existan los archivos base para iniciar."""
    required = [
        ROOT / "data" / "raw" / "Telco_Customer_Churn_Dataset.csv",
        ROOT / "notebooks" / "00_separacion_train_test.ipynb",
        ROOT / "notebooks" / "01_EDA.ipynb",
        ROOT / "notebooks" / "02_Modelos_Supervisados.ipynb",
        ROOT / "src" / "data_processing.py",
        ROOT / "src" / "model_optimization.py",
    ]

    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f" - {path.relative_to(ROOT)}" for path in missing)
        raise FileNotFoundError(
            "Faltan archivos requeridos para ejecutar el proyecto:\n"
            f"{formatted}\n\n"
            "Comprueba los nombres y la estructura de carpetas."
        )

    clustering_notebook = ROOT / "notebooks" / "04_Modelo_No_Supervisado.ipynb"
    alternate_clustering_notebook = ROOT / "notebooks" / "03_Modelo_No_Supervisado.ipynb"
    if not clustering_notebook.exists() and alternate_clustering_notebook.exists():
        FINAL_NOTEBOOKS[-1] = "notebooks/03_Modelo_No_Supervisado.ipynb"
    elif not clustering_notebook.exists():
        raise FileNotFoundError(
            "No se encontró el notebook de clustering. Se esperaba "
            "'notebooks/04_Modelo_No_Supervisado.ipynb' o "
            "'notebooks/03_Modelo_No_Supervisado.ipynb'."
        )


def execute_notebook(relative_path: str) -> None:
    """Ejecuta un notebook completo con nbconvert y guarda sus salidas."""
    notebook_path = ROOT / relative_path
    run_command(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            f"--ExecutePreprocessor.cwd={ROOT / 'notebooks'}",
            str(notebook_path),
        ],
        f"Ejecutando notebook: {relative_path}",
    )


def execute_script(relative_path: str) -> None:
    """Ejecuta un script Python desde la raíz del proyecto."""
    script_path = ROOT / relative_path
    run_command(
        [sys.executable, str(script_path)],
        f"Ejecutando script: {relative_path}",
    )


def main() -> int:
    print("TELCO CUSTOMER CHURN — EJECUCIÓN REPRODUCIBLE")
    print(f"Raíz del proyecto: {ROOT}")
    print(f"Intérprete Python: {sys.executable}")

    try:
        check_required_files()

        # 1. Partición inicial y 2. EDA solo sobre Train.
        for notebook in NOTEBOOKS:
            execute_notebook(notebook)

        # 3. Preparación de datos y 4. optimización con Optuna.
        for script in SCRIPTS:
            execute_script(script)

        # 5. Evaluación supervisada y 6. análisis no supervisado.
        for notebook in FINAL_NOTEBOOKS:
            execute_notebook(notebook)

    except subprocess.CalledProcessError as exc:
        print(
            "\nERROR: una etapa terminó con código "
            f"{exc.returncode}. El flujo se detuvo.",
            file=sys.stderr,
        )
        print(
            "Revisa el error anterior, corrige esa etapa y vuelve a ejecutar "
            "main.py. Las etapas completadas pueden haber guardado resultados.",
            file=sys.stderr,
        )
        return exc.returncode or 1
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("\n" + "=" * 78)
    print("EJECUCIÓN COMPLETADA")
    print("Revisa los CSV de reports/ y los modelos guardados en models/.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
