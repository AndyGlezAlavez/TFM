"""Generación de gráficas: matrices de confusión y umbrales.

Produce un PNG por cada (columna, método) tanto para la matriz de
confusión como para la serie temporal con la banda de umbral.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def plot_confusion_matrices(metrics_df: pd.DataFrame, output_dir: str) -> None:
    """Genera y guarda una matriz de confusión por (columna, método).

    Args:
        metrics_df: DataFrame con las columnas tn, fp, fn, tp,
                    columna y metodo.
        output_dir: Directorio de salida para los PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    for _, row in metrics_df.iterrows():
        col = row["columna"]
        method = row["metodo"]
        cm = np.array([
            [row["tn"], row["fp"]],
            [row["fn"], row["tp"]],
        ])
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.matshow(cm, cmap="Blues", alpha=0.7)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Anomaly"])
        ax.set_yticklabels(["Normal", "Anomaly"])
        ax.set_title(f"{col} - {method}")
        plt.tight_layout()
        safe_name = f"{col}_{method}".replace(".", "_")
        plt.savefig(os.path.join(output_dir, f"cm_{safe_name}.png"), dpi=100)
        plt.close(fig)


def plot_thresholds(
    df_test: pd.DataFrame,
    thresholds: dict,
    numeric_cols: list[str],
    df_eval: pd.DataFrame,
    output_dir: str,
    df_test_clean: pd.DataFrame | None = None,
) -> None:
    """Genera gráficas de serie temporal con la banda de umbral.

    Para cada (columna, método) dibuja las lecturas, la banda sombreada
    del umbral y marca las anomalías detectadas diferenciando:
      - TP (verdaderos positivos): anomalía real inyectada → círculo rojo
      - FP (falsos positivos): detectada pero no era anomalía → triángulo naranja

    Args:
        df_test: DataFrame de test con anomalías.
        thresholds: Diccionario de umbrales.
        numeric_cols: Lista de columnas numéricas.
        df_eval: Resultado de evaluate_test con las detecciones.
        output_dir: Directorio de salida para los PNG.
        df_test_clean: Test original sin anomalías (para distinguir TP/FP).
    """
    os.makedirs(output_dir, exist_ok=True)
    for col in numeric_cols:
        col_th = thresholds.get(col, {})
        series = df_test[col]
        for method_name in ["percentile", "mean_std", "iqr", "mad"]:
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(
                series.index,
                series.values,
                label="Readings",
                color="blue",
                linewidth=0.6,
            )
            lo_list = []
            hi_list = []
            for ts in series.index:
                dow = str(ts.dayofweek)
                tod = ts.strftime("%H:%M")
                limits = col_th.get(dow, {}).get(tod, {}).get(method_name)
                if limits:
                    lo_list.append(limits["limite_inferior"])
                    hi_list.append(limits["limite_superior"])
                else:
                    lo_list.append(np.nan)
                    hi_list.append(np.nan)
            ax.fill_between(
                series.index,
                lo_list,
                hi_list,
                alpha=0.2,
                color="green",
                label=f"{method_name} threshold band",
            )
            detected = df_eval[
                (df_eval["columna"] == col)
                & (df_eval["metodo"] == method_name)
                & (df_eval["anomaly_detected"] == 1)
            ]
            if not detected.empty:
                tss = pd.to_datetime(detected["timestamp"])
                vals = detected["valor"].values
                if df_test_clean is not None:
                    clean_vals = df_test_clean[col].loc[tss].values
                    is_tp = vals != clean_vals
                    tp_tss = tss[is_tp]
                    tp_vals = vals[is_tp]
                    fp_tss = tss[~is_tp]
                    fp_vals = vals[~is_tp]
                    if len(tp_tss) > 0:
                        ax.scatter(
                            tp_tss, tp_vals,
                            color="red", s=30, marker="o",
                            label=f"TP (real anomaly) n={len(tp_tss)}",
                        )
                    if len(fp_tss) > 0:
                        ax.scatter(
                            fp_tss, fp_vals,
                            color="orange", s=30, marker="^",
                            label=f"FP (false positive) n={len(fp_tss)}",
                        )
                else:
                    ax.scatter(
                        tss, vals,
                        color="red", s=20, marker="x",
                        label="Detected anomalies",
                    )
            ax.set_title(f"{col} - {method_name}")
            ax.legend(loc="best")
            plt.tight_layout()
            safe_name = f"{col}_{method_name}".replace(".", "_")
            plt.savefig(
                os.path.join(output_dir, f"threshold_{safe_name}.png"), dpi=100
            )
            plt.close(fig)
