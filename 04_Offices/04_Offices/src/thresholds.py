"""Cálculo de umbrales de anomalía mediante cuatro métodos.

Cada método recibe un array de valores y devuelve (limite_inferior,
limite_superior). La regla de negocio fuerza límite_inferior >= 0.
"""

import numpy as np
import json
from typing import Any


def percentile_method(values: np.ndarray, lower: float = 5, upper: float = 99) -> tuple[float, float]:
    """Umbrales basados en percentiles empíricos.

    Args:
        values: Array de valores de entrenamiento.
        lower: Percentil inferior (por defecto 5).
        upper: Percentil superior (por defecto 99).

    Returns:
        (limite_inferior, limite_superior).
    """
    lo = np.percentile(values, lower)
    hi = np.percentile(values, upper)
    lo = max(0.0, lo)
    return float(lo), float(hi)


def mean_std_method(values: np.ndarray, k: float = 2.0) -> tuple[float, float]:
    """Umbrales basados en media ± k·desviación típica.

    Args:
        values: Array de valores de entrenamiento.
        k: Multiplicador de la desviación típica.

    Returns:
        (limite_inferior, limite_superior).
    """
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    lo = mean - k * std
    hi = mean + k * std
    lo = max(0.0, lo)
    return float(lo), float(hi)


def iqr_method(values: np.ndarray) -> tuple[float, float]:
    """Umbrales basados en el método de Tukey (IQR).

    Límites: Q1 - 1.5·IQR y Q3 + 1.5·IQR.

    Args:
        values: Array de valores de entrenamiento.

    Returns:
        (limite_inferior, limite_superior).
    """
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 95)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    lo = max(0.0, lo)
    return float(lo), float(hi)


def mad_method(values: np.ndarray) -> tuple[float, float]:
    """Umbrales basados en la desviación absoluta mediana (MAD).

    Más robusto frente a outliers que la media/desviación típica.
    Límites: mediana ± 3·MAD.

    Args:
        values: Array de valores de entrenamiento.

    Returns:
        (limite_inferior, limite_superior).
    """
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    lo = median - 3 * mad
    hi = median + 3 * mad
    lo = max(0.0, lo)
    return float(lo), float(hi)


# Registro de métodos disponibles para iterar sobre ellos
METHODS: dict[str, callable] = {
    "percentile": percentile_method,
    "mean_std": mean_std_method,
    "iqr": iqr_method,
    "mad": mad_method,
}


def compute_all_thresholds(profiles: dict, config: dict[str, Any]) -> dict[str, Any]:
    """Calcula los cuatro umbrales para cada (columna, día, instante).

    Aplica el factor de cobertura para ensanchar la banda:
    límite_superior *= coverage_factor
    límite_inferior /= coverage_factor  (con mínimo 0).

    Args:
        profiles: Diccionario devuelto por build_profiles.
        config: Diccionario con parámetros (percentile_lower,
                percentile_upper, std_multiplier, coverage_factor).

    Returns:
        Diccionario anidado:
        columna -> día_semana -> instante -> método -> {limites}.
    """
    cov = config.get("coverage_factor", 1.0)
    thresholds = {}
    for col, col_profiles in profiles.items():
        thresholds[col] = {}
        for (dow, tod), values in col_profiles.items():
            dow_str = str(dow)
            tod_str = tod.strftime("%H:%M")
            if dow_str not in thresholds[col]:
                thresholds[col][dow_str] = {}
            if tod_str not in thresholds[col][dow_str]:
                thresholds[col][dow_str][tod_str] = {}
            arr = values.values
            for method_name, method_fn in METHODS.items():
                if method_name == "percentile":
                    lo, hi = method_fn(arr, config["percentile_lower"], config["percentile_upper"])
                elif method_name == "mean_std":
                    lo, hi = method_fn(arr, config["std_multiplier"])
                else:
                    lo, hi = method_fn(arr)
                # Aplica factor de cobertura para dar margen
                lo = max(0.0, lo / cov)
                hi = hi * cov
                thresholds[col][dow_str][tod_str][method_name] = {
                    "limite_inferior": float(lo),
                    "limite_superior": float(hi),
                }
    return thresholds


def save_thresholds(thresholds: dict, path: str) -> None:
    """Guarda el diccionario de umbrales en un archivo JSON.

    Args:
        thresholds: Diccionario de umbrales de una sola columna.
        path: Ruta de salida del JSON.
    """
    with open(path, "w") as f:
        json.dump(thresholds, f, indent=2)


def load_thresholds(path: str) -> dict:
    """Carga umbrales desde un archivo JSON.
    Args:
        path: Ruta al archivo JSON.
    Returns:
        Diccionario de umbrales.
    """
    with open(path, "r") as f:
        return json.load(f)


def save_thresholds_per_office(all_thresholds: dict[str, Any], get_path_fn: callable) -> None:
    """Guarda un JSON de umbrales por cada columna/oficina.

    Args:
        all_thresholds: Diccionario completo con todas las columnas.
        get_path_fn: Función que recibe nombre de columna y devuelve ruta.
    """
    for col, col_thresholds in all_thresholds.items():
        path = get_path_fn(col)
        save_thresholds(col_thresholds, path)
