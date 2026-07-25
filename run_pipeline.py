import subprocess
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run_step(
    step_number: int,
    step_name: str,
    command: list[str],
) -> None:
    """
    Ejecuta una etapa del pipeline.

    Si una etapa falla, el pipeline se detiene automáticamente.
    """

    print("\n" + "=" * 70)
    print(f"ETAPA {step_number}: {step_name}")
    print("=" * 70)

    start_time = time.time()

    result = subprocess.run(
        command,
        cwd=BASE_DIR,
        check=False,
    )

    elapsed_time = time.time() - start_time

    if result.returncode != 0:
        print("\n" + "!" * 70)
        print(f"ERROR EN LA ETAPA: {step_name}")
        print(
            "El pipeline fue detenido para evitar continuar "
            "con datos o resultados no válidos."
        )
        print("!" * 70)

        sys.exit(result.returncode)

    print(
        f"Etapa completada correctamente "
        f"en {elapsed_time:.2f} segundos."
    )


def main() -> None:
    """
    Ejecuta el pipeline MLOps completo.
    """

    python_executable = sys.executable

    print("\n" + "=" * 70)
    print("PIPELINE MLOPS OPEN SOURCE - SANTOTO")
    print("Estandarización y control de calidad de datos")
    print("=" * 70)

    run_step(
        step_number=1,
        step_name="Preparación de datos",
        command=[
            python_executable,
            "src/prepare.py",
        ],
    )

    run_step(
        step_number=2,
        step_name="Validación y control de calidad",
        command=[
            python_executable,
            "src/validate_data.py",
        ],
    )

    run_step(
        step_number=3,
        step_name="Entrenamiento y registro con MLflow",
        command=[
            python_executable,
            "src/train.py",
        ],
    )

    run_step(
        step_number=4,
        step_name="Pruebas automatizadas",
        command=[
            python_executable,
            "-m",
            "pytest",
            "tests/",
            "-v",
        ],
    )

    print("\n" + "=" * 70)
    print("PIPELINE FINALIZADO CORRECTAMENTE")
    print("=" * 70)
    print("Los datos fueron preparados y validados.")
    print("El modelo fue entrenado y almacenado.")
    print("Las pruebas automatizadas fueron superadas.")
    print("=" * 70)


if __name__ == "__main__":
    main()