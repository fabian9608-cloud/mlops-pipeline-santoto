from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.csv"
TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"

MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "model.pkl"


def load_datasets() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Carga los conjuntos de entrenamiento y prueba.
    """

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {TRAIN_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {TEST_FILE}"
        )

    train_data = pd.read_csv(TRAIN_FILE)
    test_data = pd.read_csv(TEST_FILE)

    X_train = train_data.drop(columns=["species"])
    y_train = train_data["species"]

    X_test = test_data.drop(columns=["species"])
    y_test = test_data["species"]

    return X_train, X_test, y_train, y_test


def train_model() -> None:
    """
    Entrena el modelo Random Forest, registra el experimento
    en MLflow y almacena el modelo localmente.
    """

    print("Cargando datos procesados...")

    X_train, X_test, y_train, y_test = load_datasets()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    n_estimators = 100
    random_state = 42
    mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR / 'mlflow.db'}")
    mlflow.set_experiment("Proyecto_Final_Grado")

    print("Iniciando entrenamiento...")

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
        )

        # Registrar parámetros.
        mlflow.log_param(
            "algorithm",
            "RandomForestClassifier",
        )
        mlflow.log_param(
            "n_estimators",
            n_estimators,
        )
        mlflow.log_param(
            "random_state",
            random_state,
        )
        mlflow.log_param(
            "test_size",
            0.20,
        )

        # Registrar métricas.
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Registrar el modelo en MLflow.
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris_model",
        )

        # Guardar copia local.
        joblib.dump(model, MODEL_FILE)

        run_id = mlflow.active_run().info.run_id

        print("Entrenamiento finalizado.")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Modelo guardado en: {MODEL_FILE}")
        print(f"MLflow Run ID: {run_id}")


if __name__ == "__main__":
    train_model()