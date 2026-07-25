from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# Ruta principal del proyecto.
BASE_DIR = Path(__file__).resolve().parent.parent

# Carpeta donde se almacenarán los datos procesados.
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Archivos de salida.
TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"


def prepare_data() -> None:
    """
    Carga el dataset Iris, estandariza los nombres de las columnas,
    divide los datos y genera los archivos de entrenamiento y prueba.
    """

    print("Cargando dataset Iris...")

    iris = load_iris()

    # Se construye un DataFrame con nombres estandarizados.
    data = pd.DataFrame(
        iris.data,
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ],
    )

    # Variable objetivo.
    data["species"] = iris.target

    # Crear la carpeta si todavía no existe.
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # División reproducible y estratificada.
    train_data, test_data = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data["species"],
    )

    # Guardar los archivos procesados.
    train_data.to_csv(TRAIN_FILE, index=False)
    test_data.to_csv(TEST_FILE, index=False)

    print("Preparación de datos finalizada.")
    print(f"Archivo de entrenamiento: {TRAIN_FILE}")
    print(f"Archivo de prueba: {TEST_FILE}")
    print(f"Registros de entrenamiento: {len(train_data)}")
    print(f"Registros de prueba: {len(test_data)}")


if __name__ == "__main__":
    prepare_data()