"""Lectura y preprocesado genérico del dataset.

Lee un CSV con timestamp y columnas numéricas, filtra lecturas
cada 15 minutos y sustituye valores negativos por 0.
"""
import re
import pandas as pd
import numpy as np
from config import TIMESTAMP_COL, OFFICE_SUFFIX_MAP


def read_and_preprocess(
    data_path: str, timestamp_col: str = TIMESTAMP_COL
) -> tuple[pd.DataFrame, list[str]]:
    """Lee el CSV, parsea el timestamp, filtra a 15 min y corrige negativos.

    Args:
        data_path: Ruta al archivo CSV.
        timestamp_col: Nombre de la columna de timestamps.

    Returns:
        Tupla (df, numeric_cols) con el DataFrame preprocesado y la lista
        de columnas numéricas detectadas automáticamente.
    """
    df = pd.read_csv(data_path)

    # Renombra automáticamente las columnas de consumo como
    # oficina 1 ... oficina 9 a partir de su identificador final.
    # Admite tanto nombres separados por "_" como por espacios.
    rename_map = {}
    for col in df.columns:
        normalized = str(col).strip()
        suffix = re.split(r"[_\\s]+", normalized)[-1].upper()
        if suffix in OFFICE_SUFFIX_MAP:
            rename_map[col] = OFFICE_SUFFIX_MAP[suffix]
    if rename_map:
        df = df.rename(columns=rename_map)

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col).sort_index()
    # Conserva solo lecturas en minutos múltiplo de 15 (00, 15, 30, 45)
    df = df[df.index.minute % 15 == 0].copy()
    # Detecta columnas numéricas automáticamente
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Sustituye valores negativos por 0
    df[numeric_cols] = df[numeric_cols].clip(lower=0)
    return df, numeric_cols
