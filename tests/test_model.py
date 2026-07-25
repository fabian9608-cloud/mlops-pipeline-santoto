from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.csv"
TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"
MODEL_FILE = BASE_DIR / "models" / "model.pkl"

EXPECTED_COLUMNS = {
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
}


def test_processed_files_exist() -> None:
    """
    Comprueba que los conjuntos procesados existan.
    """

    assert TRAIN_FILE.exists(), (
        "No se encontró train.csv."
    )

    assert TEST_FILE.exists(), (
        "No se encontró test.csv."
    )


def test_datasets_are_not_empty() -> None:
    """
    Comprueba que los archivos contengan registros.
    """

    train_data = pd.read_csv(TRAIN_FILE)
    test_data = pd.read_csv(TEST_FILE)

    assert not train_data.empty
    assert not test_data.empty


def test_dataset_columns() -> None:
    """
    Comprueba que las columnas coincidan con el estándar.
    """

    train_data = pd.read_csv(TRAIN_FILE)
    test_data = pd.read_csv(TEST_FILE)

    assert set(train_data.columns) == EXPECTED_COLUMNS
    assert set(test_data.columns) == EXPECTED_COLUMNS


def test_no_null_values() -> None:
    """
    Comprueba que no existan valores nulos.
    """

    train_data = pd.read_csv(TRAIN_FILE)
    test_data = pd.read_csv(TEST_FILE)

    assert not train_data.isnull().any().any()
    assert not test_data.isnull().any().any()


def test_target_domain() -> None:
    """
    Comprueba que species solo contenga clases válidas.
    """

    train_data = pd.read_csv(TRAIN_FILE)
    test_data = pd.read_csv(TEST_FILE)

    allowed_classes = {0, 1, 2}

    assert set(train_data["species"].unique()).issubset(
        allowed_classes
    )

    assert set(test_data["species"].unique()).issubset(
        allowed_classes
    )


def test_model_file_exists() -> None:
    """
    Comprueba que el modelo haya sido generado.
    """

    assert MODEL_FILE.exists(), (
        "No se encontró models/model.pkl."
    )


def test_model_can_be_loaded() -> None:
    """
    Comprueba que el modelo pueda cargarse.
    """

    model = joblib.load(MODEL_FILE)

    assert model is not None
    assert hasattr(model, "predict")


def test_model_accuracy() -> None:
    """
    Comprueba que el modelo supere la precisión mínima.
    """

    model = joblib.load(MODEL_FILE)
    test_data = pd.read_csv(TEST_FILE)

    X_test = test_data.drop(columns=["species"])
    y_test = test_data["species"]

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.80, (
        f"Accuracy insuficiente: {accuracy:.4f}"
    )