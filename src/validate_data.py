import csv
import math
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.csv"
TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"


# Contrato interno utilizado para validar los archivos CSV.
DATA_RULES = {
    "sepal_length": {
        "type": "float",
        "nullable": False,
        "minimum": 4.0,
        "maximum": 8.0,
    },
    "sepal_width": {
        "type": "float",
        "nullable": False,
        "minimum": 2.0,
        "maximum": 4.5,
    },
    "petal_length": {
        "type": "float",
        "nullable": False,
        "minimum": 1.0,
        "maximum": 7.0,
    },
    "petal_width": {
        "type": "float",
        "nullable": False,
        "minimum": 0.1,
        "maximum": 2.5,
    },
    "species": {
        "type": "integer",
        "nullable": False,
        "allowed_values": [0, 1, 2],
    },
}


class DataValidationError(Exception):
    """Excepción personalizada para errores de calidad de datos."""


def is_null(value: Any) -> bool:
    """
    Determina si un valor debe interpretarse como nulo.
    """

    if value is None:
        return True

    normalized_value = str(value).strip().lower()

    return normalized_value in {
        "",
        "null",
        "none",
        "nan",
        "na",
        "n/a",
    }


def convert_value(value: str, expected_type: str) -> float | int:
    """
    Convierte un valor al tipo establecido en el contrato.
    """

    if expected_type == "float":
        converted_value = float(value)

        if not math.isfinite(converted_value):
            raise ValueError("El valor no es un número finito.")

        return converted_value

    if expected_type == "integer":
        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            raise ValueError("El valor no es un número finito.")

        if not numeric_value.is_integer():
            raise ValueError("El valor debe ser entero.")

        return int(numeric_value)

    raise ValueError(
        f"Tipo de dato no soportado: {expected_type}"
    )


def validate_headers(fieldnames: list[str] | None) -> list[str]:
    """
    Comprueba que el archivo tenga exactamente las columnas esperadas.
    """

    errors: list[str] = []

    if fieldnames is None:
        return ["El archivo no contiene encabezados."]

    expected_columns = set(DATA_RULES.keys())
    received_columns = set(fieldnames)

    missing_columns = expected_columns - received_columns
    unexpected_columns = received_columns - expected_columns

    if missing_columns:
        errors.append(
            "Faltan las siguientes columnas obligatorias: "
            f"{sorted(missing_columns)}."
        )

    if unexpected_columns:
        errors.append(
            "Se encontraron columnas no definidas en el contrato: "
            f"{sorted(unexpected_columns)}."
        )

    return errors


def validate_row(
    row: dict[str, str],
    row_number: int,
) -> list[str]:
    """
    Valida tipos, valores nulos, rangos y dominios de una fila.
    """

    errors: list[str] = []

    for column, rules in DATA_RULES.items():
        value = row.get(column)

        # Validación de valores nulos.
        if is_null(value):
            if not rules["nullable"]:
                errors.append(
                    f"Fila {row_number}, columna '{column}': "
                    "el valor no puede ser nulo."
                )

            continue

        # Validación del tipo de dato.
        try:
            converted_value = convert_value(
                value=value,
                expected_type=rules["type"],
            )
        except (ValueError, TypeError):
            errors.append(
                f"Fila {row_number}, columna '{column}': "
                f"el valor '{value}' no corresponde al tipo "
                f"{rules['type']}."
            )

            continue

        # Validación del valor mínimo.
        minimum = rules.get("minimum")

        if minimum is not None and converted_value < minimum:
            errors.append(
                f"Fila {row_number}, columna '{column}': "
                f"el valor {converted_value} es inferior al mínimo "
                f"permitido ({minimum})."
            )

        # Validación del valor máximo.
        maximum = rules.get("maximum")

        if maximum is not None and converted_value > maximum:
            errors.append(
                f"Fila {row_number}, columna '{column}': "
                f"el valor {converted_value} supera el máximo "
                f"permitido ({maximum})."
            )

        # Validación del conjunto de valores permitidos.
        allowed_values = rules.get("allowed_values")

        if (
            allowed_values is not None
            and converted_value not in allowed_values
        ):
            errors.append(
                f"Fila {row_number}, columna '{column}': "
                f"el valor {converted_value} no pertenece al dominio "
                f"permitido {allowed_values}."
            )

    return errors


def validate_csv(file_path: Path) -> None:
    """
    Ejecuta todas las validaciones sobre un archivo CSV.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo requerido: {file_path}"
        )

    errors: list[str] = []
    row_count = 0

    with file_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        errors.extend(validate_headers(reader.fieldnames))

        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            errors.extend(validate_row(row, row_number))

    if row_count == 0:
        errors.append("El archivo no contiene registros.")

    if errors:
        error_message = "\n".join(
            f"  - {error}" for error in errors
        )

        raise DataValidationError(
            f"\nValidación fallida para {file_path.name}:\n"
            f"{error_message}"
        )

    print(
        f"Validación exitosa: {file_path.name} "
        f"({row_count} registros revisados)."
    )


def main() -> None:
    """
    Valida los archivos de entrenamiento y prueba.
    """

    print("Iniciando control de calidad de datos...")

    validate_csv(TRAIN_FILE)
    validate_csv(TEST_FILE)

    print("Todos los archivos cumplen el contrato de calidad.")


if __name__ == "__main__":
    main()