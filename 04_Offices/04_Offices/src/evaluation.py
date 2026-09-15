"""Evaluación del test contra los umbrales y cálculo de métricas.

Para cada (columna, método) se detectan anomalías comparando cada
lectura contra su umbral correspondiente. Luego se calculan la matriz
de confusión, precisión, recall y F1 usando como ground truth las
anomalías inyectadas artificialmente.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def evaluate_test(
    df_test: pd.DataFrame, thresholds: dict, numeric_cols: list[str]
) -> pd.DataFrame:
    """Evalúa cada lectura del test contra los umbrales de cada método.

    Para cada (columna, timestamp) se obtiene el umbral correspondiente
    según el día de la semana y la hora, y se marca como anomalía si el
    valor queda fuera del rango [limite_inferior, limite_superior].

    Args:
        df_test: DataFrame de test (con anomalías inyectadas).
        thresholds: Diccionario de umbrales (cargado desde JSON).
        numeric_cols: Lista de columnas numéricas.

    Returns:
        DataFrame con una fila por (timestamp, columna, método) y la
        columna 'anomaly_detected' (0 = normal, 1 = anómala).
    """
    records = []
    for col in numeric_cols:
        col_th = thresholds.get(col, {})
        for ts, row in df_test.iterrows():
            dow = str(ts.dayofweek)
            tod = ts.strftime("%H:%M")
            value = row[col]
            instant_th = col_th.get(dow, {}).get(tod, {})
            if not instant_th:
                continue
            for method_name, limits in instant_th.items():
                lo = limits["limite_inferior"]
                hi = limits["limite_superior"]
                detected = int(value < lo or value > hi)
                records.append({
                    "timestamp": ts,
                    "columna": col,
                    "metodo": method_name,
                    "valor": value,
                    "limite_inferior": lo,
                    "limite_superior": hi,
                    "anomaly_detected": detected,
                })
    return pd.DataFrame(records)


def compute_metrics(
    df_eval: pd.DataFrame,
    df_test_clean: pd.DataFrame,
    df_test_anomalous: pd.DataFrame,
    numeric_cols: list[str],
) -> pd.DataFrame:
    """Calcula matriz de confusión, precisión, recall y F1 por método.

    El ground truth se obtiene comparando el test limpio con el test
    con anomalías: donde los valores difieren hay una anomalía real.
    Las predicciones se toman del DataFrame de evaluación.

    Args:
        df_eval: Resultado de evaluate_test.
        df_test_clean: Test original sin anomalías.
        df_test_anomalous: Test con anomalías inyectadas.
        numeric_cols: Lista de columnas numéricas.

    Returns:
        DataFrame con una fila por (columna, método) y las columnas
        tn, fp, fn, tp, precision, recall, f1.
    """
    records = []
    for col in numeric_cols:
        clean_series = df_test_clean[col]
        anom_series = df_test_anomalous[col]
        for method_name in ["percentile", "mean_std", "iqr", "mad"]:
            sub = df_eval[
                (df_eval["columna"] == col) & (df_eval["metodo"] == method_name)
            ].copy()
            if sub.empty:
                continue
            eval_timestamps = sub["timestamp"].values
            y_pred = sub["anomaly_detected"].values

            # Alinea el ground truth con los mismos timestamps evaluados
            clean_aligned = clean_series.loc[eval_timestamps].values
            anom_aligned = anom_series.loc[eval_timestamps].values
            y_true = (clean_aligned != anom_aligned).astype(int)

            # Maneja el caso en que solo haya una clase
            if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
                tn = int(np.sum((y_true == 0) & (y_pred == 0)))
                fp = int(np.sum((y_true == 0) & (y_pred == 1)))
                fn = int(np.sum((y_true == 1) & (y_pred == 0)))
                tp = int(np.sum((y_true == 1) & (y_pred == 1)))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )
            else:
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, labels=[0, 1], average="binary"
                )
            records.append({
                "columna": col,
                "metodo": method_name,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })
    return pd.DataFrame(records)
