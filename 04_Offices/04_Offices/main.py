"""Pipeline principal de detección de anomalías en series temporales.

Orquesta los siete pasos del proceso:
  1. Lectura y preprocesado
  2. División temporal (80/20)
  3. Construcción de perfiles
  4. Cálculo y guardado de umbrales (un JSON por columna/oficina)
  5. Inyección de anomalías en test
  6. Evaluación contra umbrales
  7. Métricas y gráficas (por columna/oficina)
"""

import os
import config
from src.preprocessing import read_and_preprocess
from src.profiling import temporal_split, build_profiles
from src.thresholds import compute_all_thresholds, save_thresholds_per_office
from src.anomaly_injection import inject_anomalies
from src.evaluation import evaluate_test, compute_metrics
from src.plotting import plot_confusion_matrices, plot_thresholds


def main() -> None:
    """Ejecuta el pipeline completo de detección de anomalías."""
    print("[1/7] Leyendo y preprocesando...")
    df, numeric_cols = read_and_preprocess(config.DATA_PATH)
    print(f"  Filas tras filtrado 15 min: {len(df)}, columnas: {len(numeric_cols)}")
    print(f"  Oficinas/columnas detectadas: {numeric_cols}")

    print("[2/7] Dividiendo temporalmente (80/20)...")
    df_train, df_test = temporal_split(df, config.TRAIN_SPLIT)
    print(f"  Train: {len(df_train)}, Test: {len(df_test)}")

    print("[3/7] Construyendo perfiles...")
    profiles = build_profiles(df_train, numeric_cols)

    print("[4/7] Calculando umbrales...")
    thresh_config = {
        "percentile_lower": config.PERCENTILE_LOWER,
        "percentile_upper": config.PERCENTILE_UPPER,
        "std_multiplier": config.STD_MULTIPLIER,
        "coverage_factor": config.COVERAGE_FACTOR,
    }
    thresholds = compute_all_thresholds(profiles, thresh_config)
    # Guarda un JSON por cada columna/oficina
    save_thresholds_per_office(thresholds, config.get_thresholds_path)
    print(f"  Umbrales guardados en {config.THRESHOLDS_DIR}/")

    print("[5/7] Inyectando anomalías en test...")
    df_test_clean = df_test.copy()
    df_test_anom = inject_anomalies(df_test, numeric_cols, config.ANOMALY_FRACTION, config.SEED)
    n_changed = (df_test_clean[numeric_cols].values != df_test_anom[numeric_cols].values).sum()
    print(f"  Valores modificados: {n_changed}")

    print("[6/7] Evaluando contra umbrales...")
    df_eval = evaluate_test(df_test_anom, thresholds, numeric_cols)

    print("[7/7] Calculando métricas y generando gráficas...")
    df_metrics = compute_metrics(df_eval, df_test_clean, df_test_anom, numeric_cols)

    # Guarda resultados y gráficas por cada columna/oficina
    for col in numeric_cols:
        office = col
        eval_path = config.get_evaluation_csv_path(office)
        metrics_path = config.get_metrics_csv_path(office)
        confusion_dir = config.get_office_confusion_dir(office)
        thresh_plots_dir = config.get_office_threshold_plots_dir(office)

        # Filtra evaluación y métricas para esta columna
        df_eval_col = df_eval[df_eval["columna"] == col]
        df_metrics_col = df_metrics[df_metrics["columna"] == col]

        df_eval_col.to_csv(eval_path, index=False)
        df_metrics_col.to_csv(metrics_path, index=False)

        # Genera gráficas solo para esta columna
        col_thresholds = {col: thresholds[col]}
        plot_confusion_matrices(df_metrics_col, confusion_dir)
        plot_thresholds(df_test_anom, col_thresholds, [col], df_eval, thresh_plots_dir, df_test_clean)

        print(f"  {office}: resultados en {config.get_office_results_dir(office)}")

    print("\n¡Pipeline completado con éxito!")


if __name__ == "__main__":
    main()
