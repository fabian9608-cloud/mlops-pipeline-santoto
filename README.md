# MLOps Open Source - Santoto
## Estandarización de Pipelines y Control de Calidad de Datos

**Especialización en Ciencia de Datos**  
**Seminario de Grado II**  
**Universidad Santo Tomás**

---

# Descripción

Este proyecto implementa un pipeline reproducible para el entrenamiento de un modelo de Machine Learning utilizando el conjunto de datos **Iris**.

El objetivo es aplicar conceptos de **MLOps**, automatizando el flujo de preparación de datos, validación de calidad, entrenamiento del modelo, registro de experimentos mediante MLflow y pruebas automatizadas.

El pipeline fue desarrollado siguiendo las buenas prácticas de reproducibilidad propuestas en el laboratorio **MLOps Open Source - Santoto**.

---

# Objetivos

- Automatizar el pipeline de entrenamiento mediante un único comando.
- Validar la calidad de los datos antes del entrenamiento.
- Garantizar la reproducibilidad del experimento.
- Registrar métricas y parámetros con MLflow.
- Documentar completamente el proceso para que pueda ser reproducido por un tercero.

---

# Tecnologías utilizadas

- Python 3.13
- MLflow
- Scikit-Learn
- Pandas
- Pytest
- Joblib
- Visual Studio Code

---

# Estructura del proyecto

```text
MLOPS-IRIS-SANTOTO
│
├── data
│   └── processed
│       ├── train.csv
│       └── test.csv
│
├── mlruns
│
├── models
│   └── model.pkl
│
├── src
│   ├── prepare.py
│   ├── train.py
│   └── validate_data.py
│
├── tests
│   └── test_model.py
│
├── data_contract.json
├── README.md
├── requirements.txt
├── run_pipeline.py
└── mlflow.db
```

---
---

# Descripción de los componentes del proyecto

A continuación, se describe el propósito de cada uno de los archivos principales que conforman el pipeline. Cada componente cumple una responsabilidad específica dentro del flujo de trabajo, permitiendo que el proceso sea organizado, reproducible y fácil de comprender.

## run_pipeline.py

Es el script de orquestación del proyecto. Su función es ejecutar automáticamente todas las etapas del pipeline en el orden correcto: preparación de datos, validación, entrenamiento del modelo y ejecución de pruebas automatizadas. Gracias a este archivo, todo el proceso puede ejecutarse mediante un único comando.

---

## src/prepare.py

Este archivo realiza la preparación del conjunto de datos Iris. Carga el dataset original, realiza la división entre datos de entrenamiento y prueba y genera los archivos **train.csv** y **test.csv**, los cuales serán utilizados en las siguientes etapas del pipeline.

---

## src/validate_data.py

Implementa el control de calidad de los datos sin utilizar librerías especializadas. Verifica que los tipos de datos sean correctos, que no existan valores nulos, que los datos se encuentren dentro de los rangos permitidos y que la estructura del conjunto de datos sea consistente antes de permitir el entrenamiento del modelo.

---

## src/train.py

Entrena un modelo de clasificación Random Forest utilizando los datos previamente validados. Durante esta etapa se calculan las métricas de desempeño, se almacena el modelo entrenado y se registra automáticamente el experimento mediante MLflow para garantizar la trazabilidad del proceso.

---

## tests/test_model.py

Contiene las pruebas automatizadas del proyecto. Estas verifican el correcto funcionamiento del modelo entrenado y del pipeline, permitiendo comprobar que los resultados obtenidos cumplen las condiciones esperadas antes de finalizar la ejecución.

---

## data_contract.json

Define el contrato oficial de datos del proyecto. En este archivo se especifican las variables esperadas, los tipos de datos, los rangos permitidos, los valores válidos y ejemplos de entradas correctas e incorrectas que servirán como referencia para futuras inferencias del modelo.

---

## requirements.txt

Relaciona todas las dependencias necesarias para ejecutar correctamente el proyecto. Su utilización garantiza que cualquier usuario pueda instalar las mismas versiones de las librerías y reproducir el pipeline en otro equipo.

---

## README.md

Contiene la documentación técnica del proyecto. Describe la estructura del pipeline, las instrucciones de instalación, la ejecución de cada etapa y el flujo general de trabajo, permitiendo que cualquier usuario pueda reproducir el proyecto desde cero.

---

# Creación del entorno

## 1. Crear el entorno virtual

- Abrir una terminal en Visual Studio Code.
```powershell
python -m venv .venv
```

- Activar el entorno.
```powershell
.venv\Scripts\activate
```

- La terminal debe mostrar algo similar a:
```text
(.venv) PS C:\MLOPS-IRIS-SANTOTO>
```

---

## 2. Instalar las dependencias

```powershell
pip install -r requirements.txt
```

---

# Ejecución del proyecto

## Paso 1. Preparación de datos

- Ejecutar:
```powershell
python src/prepare.py
```

- Resultado esperado:
```text
Cargando dataset Iris...
Preparación de datos finalizada.
Registros entrenamiento : 120
Registros prueba        : 30
```

- Se generan:
```text
data/processed/train.csv
data/processed/test.csv
```

---

## Paso 2. Validación de datos

- Ejecutar:
```powershell
python src/validate_data.py
```

El sistema verifica:

- Tipos de datos.
- Valores nulos.
- Rangos permitidos.
- Clases válidas.

- Salida esperada:
```text
Validación exitosa: train.csv
Validación exitosa: test.csv

Todos los datos cumplen las reglas de calidad.
```

---

## Paso 3. Entrenamiento

- Ejecutar:
```powershell
python src/train.py
```

- Salida esperada:
```text
Entrenamiento finalizado.
Accuracy : 0.9000
F1 Score : 0.8997
Modelo almacenado en:
models/model.pkl
Run ID:
3959cabac784ef897adb59156ed353d
```
---

# Visualización de MLflow

- Abrir una segunda terminal.

Activar nuevamente el entorno.

```powershell
.venv\Scripts\activate
```

Ejecutar MLflow.

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

La terminal mostrará:

```text
INFO: Uvicorn running on

http://127.0.0.1:5000
```

Abrir el navegador.

```text
http://localhost:5000
```

Debe visualizarse el experimento:

```text
Proyecto_Final_Grado
```

Dentro del experimento se registran:

- Accuracy
- F1 Score
- Parámetros
- Modelo entrenado
- Run ID

---

# Flujo general del Pipeline

```text
prepare.py
      │
      ▼
validate_data.py
      │
      ▼
train.py
      │
      ▼
Registro en MLflow
      │
      ▼
Pruebas automatizadas (pytest)
      │
      ▼
Pipeline finalizado correctamente
```

El flujo inicia con la preparación del conjunto de datos, continúa con el control de calidad, posteriormente realiza el entrenamiento del modelo y registra el experimento en MLflow. Finalmente, ejecuta las pruebas automatizadas para verificar el correcto funcionamiento del pipeline antes de finalizar la ejecución.

# Ejecución del pipeline completo

El proyecto incluye un script de automatización.

Ejecutar:

```powershell
python run_pipeline.py
```

Este comando ejecuta automáticamente:

1. Preparación de datos.
2. Validación.
3. Entrenamiento.
4. Registro en MLflow.
5. Pruebas automatizadas.

---

# Ejecución de pruebas

```powershell
python -m pytest tests/ -v
```

Resultado esperado:

```text
======================

8 passed

======================
```

---

# Validaciones implementadas

El archivo

```text
validate_data.py
```

realiza las siguientes verificaciones.

| Validación    | Descripción                                           |
|---------------|-------------------------------------------------------|
| Tipos         | Verifica que los datos sean numéricos.                |
| Valores nulos | Comprueba ausencia de datos faltantes.                |
| Rangos        | Verifica valores mínimos y máximos permitidos.        |
| Dominio       | La variable species solo admite 0, 1 y 2.             |
| Columnas      | Comprueba que existan todas las columnas esperadas.   |

---

# Contrato de Datos

El archivo

```text
data_contract.json
```

define el contrato oficial para futuras inferencias.

Incluye:

- Tipos de datos.
- Variables requeridas.
- Rangos permitidos.
- Ejemplo válido.
- Ejemplo inválido.

---

# Resultados obtenidos

Modelo utilizado:

```text
RandomForestClassifier
```

Resultados registrados en MLflow.

| Métrica       | Valor  |
|---------------|--------|
| Accuracy      | 0.9000 |
| F1 Score      | 0.8997 |

---

# Evidencias de ejecución

Durante el desarrollo del laboratorio se verificó correctamente:

1.) Preparación del conjunto de datos.
2.) Validación de calidad.
3.) Entrenamiento del modelo.
4.) Registro del experimento en MLflow.
5.) Almacenamiento del modelo.
6.) Ejecución satisfactoria de las pruebas automatizadas.

---

# Autor

**Eduar Fabian Castelblanco Galindo**
Especialización en Ciencia de Datos
Universidad Santo Tomás
Seminario de Grado II