"""Inyección artificial de anomalías en el conjunto de test.

Introduce drops (multiplicar por -1.0 o -0.5) y spikes (multiplicar
por 3, 4 o 4.5) en aproximadamente el 5% de las filas de test,
de forma independiente por cada columna y con semilla fija.
"""

import numpy as np
import pandas as pd


def inject_anomalies(
    df_test: pd.DataFrame,
    numeric_cols: list[str],
    fraction: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Inyecta anomalías artificiales en el conjunto de test.

    Para cada columna numérica selecciona aleatoriamente un `fraction`
    de las filas y las modifica con un drop o un spike. Cada combinación
    (timestamp, columna) se modifica como máximo una vez.

    Args:
        df_test: DataFrame de test (se modifica una copia).
        numeric_cols: Lista de columnas numéricas.
        fraction: Proporción de filas a contaminar por columna.
        seed: Semilla del generador aleatorio para reproducibilidad.

    Returns:
        DataFrame con las anomalías inyectadas.
    """
    rng = np.random.default_rng(seed)
    df_out = df_test.copy()
    for col in numeric_cols:
        n = len(df_out)
        n_anom = max(2, int(n * fraction))
        if n_anom % 2 != 0:
            n_anom += 1
        indices = rng.choice(n, size=n_anom, replace=False)
        n_half = n_anom // 2
        drop_indices = indices[:n_half]
        spike_indices = indices[n_half:]
        for idx in drop_indices:
            factor = rng.choice([-1.0, -0.5])
            df_out.iloc[idx, df_out.columns.get_loc(col)] *= factor
        for idx in spike_indices:
            factor = rng.choice([3.0, 4.0, 4.5])
            df_out.iloc[idx, df_out.columns.get_loc(col)] *= factor
    return df_out
