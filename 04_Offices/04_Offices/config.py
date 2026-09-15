"""Parámetros configurables del pipeline de detección de anomalías.

Todas las rutas, hiperparámetros y opciones se centralizan aquí
para que el programa sea fácilmente reutilizable con otros datasets.
"""

import os

# Directorio raíz del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta al archivo CSV de entrada
DATA_PATH = os.path.join(BASE_DIR, "00_Dataset", "8_OFICINAS", "satcomm_milltown-office_Customers.csv")



# Correspondencia entre los identificadores originales de las columnas
# y los nombres utilizados en el TFM. Se emplea únicamente el sufijo
# final de cada columna para evitar conservar la nomenclatura anterior.
OFFICE_SUFFIX_MAP = {
    "1A": "oficina 1",
    "1C": "oficina 2",
    "1D": "oficina 3",
    "1E": "oficina 4",
    "1F": "oficina 5",
    "1H": "oficina 6",
    "1J": "oficina 7",
    "1K": "oficina 8",
    "1L": "oficina 9",
    "1": "oficina 1",
    "2": "oficina 2",
    "3": "oficina 3",
    "4": "oficina 4",
    "5": "oficina 5",
    "6": "oficina 6",
    "7": "oficina 7",
    "8": "oficina 8",
    "9": "oficina 9",
}

# Nombre de la columna que contiene los timestamps en el CSV
TIMESTAMP_COL = "timestamp (5 min)"

# Percentiles para el método de percentiles empíricos
PERCENTILE_LOWER = 5
PERCENTILE_UPPER = 100

# Multiplicador k para el método media ± k·desviación típica
STD_MULTIPLIER = 5.0

# Factor de cobertura: ensancha la banda del umbral para dar margen
# límite_superior *= COVERAGE_FACTOR
# límite_inferior /= COVERAGE_FACTOR  (sujeto a límite >= 0)
COVERAGE_FACTOR = 1.4

# Semilla para el generador aleatorio (reproducibilidad)
SEED = 42
# Fracción del conjunto de test que se contaminará con anomalías artificiales
ANOMALY_FRACTION = 0.05

# Directorios base de salida
THRESHOLDS_DIR = os.path.join(BASE_DIR, "thresholds")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Proporción de datos destinada a entrenamiento (el resto es test)
TRAIN_SPLIT = 0.8


def get_thresholds_path(office_name: str) -> str:
    """Devuelve la ruta del JSON de umbrales para una oficina/columna.

    Args:
        office_name: Nombre de la columna/oficina (ej. 'oficina 1').

    Returns:
        Ruta completa al archivo JSON.
    """
    os.makedirs(THRESHOLDS_DIR, exist_ok=True)
    return os.path.join(THRESHOLDS_DIR, f"{office_name}_thresholds.json")


def get_office_results_dir(office_name: str) -> str:
    """Devuelve el directorio de resultados para una oficina/columna.

    Args:
        office_name: Nombre de la columna/oficina.

    Returns:
        Ruta al directorio de resultados de la oficina.
    """
    path = os.path.join(RESULTS_DIR, office_name)
    os.makedirs(path, exist_ok=True)
    return path


def get_office_plots_dir(office_name: str) -> str:
    """Devuelve el directorio base de gráficas para una oficina/columna.

    Args:
        office_name: Nombre de la columna/oficina.

    Returns:
        Ruta al directorio de gráficas de la oficina.
    """
    path = os.path.join(PLOTS_DIR, office_name)
    os.makedirs(path, exist_ok=True)
    return path


def get_office_confusion_dir(office_name: str) -> str:
    """Devuelve el directorio de matrices de confusión para una oficina/columna.

    Args:
        office_name: Nombre de la columna/oficina.

    Returns:
        Ruta al directorio de matrices de confusión.
    """
    path = os.path.join(get_office_plots_dir(office_name), "confusion_matrices")
    os.makedirs(path, exist_ok=True)
    return path


def get_office_threshold_plots_dir(office_name: str) -> str:
    """Devuelve el directorio de gráficas de umbrales para una oficina/columna.

    Args:
        office_name: Nombre de la columna/oficina.

    Returns:
        Ruta al directorio de gráficas de umbrales.
    """
    path = os.path.join(get_office_plots_dir(office_name), "threshold_plots")
    os.makedirs(path, exist_ok=True)
    return path


def get_evaluation_csv_path(office_name: str) -> str:
    """Devuelve la ruta del CSV de evaluación para una oficina/columna.

    Args:
        office_name: Nombre de la columna/oficina.

    Returns:
        Ruta completa al archivo CSV.
    """
    return os.path.join(get_office_results_dir(office_name), "evaluation_results.csv")


def get_metrics_csv_path(office_name: str) -> str:
    """Devuelve la ruta del CSV de métricas para una oficina/columna.

    Args:
        office_name: Nombre de la columna/oficina.

    Returns:
        Ruta completa al archivo CSV.
    """
    return os.path.join(get_office_results_dir(office_name), "metrics.csv")
