"""División temporal y construcción de perfiles por día-hora.

Separa el dataset en entrenamiento (80%) y test (20%) cronológicamente.
Construye perfiles agrupando por día de la semana e instante de 15 min.
"""
import pandas as pd


def temporal_split(
    df: pd.DataFrame, train_frac: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el DataFrame cronológicamente en entrenamiento y test.

    El 80% más antiguo se usa como entrenamiento; el 20% más reciente
    como test.

    Args:
        df: DataFrame preprocesado con índice temporal.
        train_frac: Fracción de datos para entrenamiento.

    Returns:
        Tupla (df_train, df_test).
    """
    n = len(df)
    split_idx = int(n * train_frac)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    return df_train, df_test


def build_profiles(
    df_train: pd.DataFrame, numeric_cols: list[str]
) -> dict[str, dict[tuple[int, object], pd.Series]]:
    """Agrupa el entrenamiento por (día de la semana, instante de 15 min)
    para cada columna numérica.

    Args:
        df_train: DataFrame de entrenamiento.
        numeric_cols: Lista de columnas numéricas.

    Returns:
        Diccionario anidado: columna -> (dayofweek, time) -> Series.
    """
    profiles = {}
    for col in numeric_cols:
        series = df_train[col]
        day_of_week = series.index.dayofweek
        time_of_day = series.index.time
        groups = series.groupby([day_of_week, time_of_day])
        col_profiles = {}
        for (dow, tod), values in groups:
            col_profiles[(dow, tod)] = values
        profiles[col] = col_profiles
    return profiles
