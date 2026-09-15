import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor as DT


class Preprocess:
    def columns(self, data):

        df_filtrado = data.copy()
        # Convertir columna de fecha a datetime
        df_filtrado["data"] = pd.to_datetime(df_filtrado["data"], format="%d/%m/%Y %H:%M")
        df_filtrado = df_filtrado.set_index("data")
        """ --- RELLENAR DATOS VACÍOS --- """
        df_filtrado = df_filtrado.fillna(0)
        df_fill = df_filtrado.copy()
        return df_fill

    def anomaly_clean(self, data):
        df_fill = data.copy()
        """ --- FILTRAR DÍAS CON ANOMALÍA --- """
        # Identificar días con alguna anomalía
        df_fill["data"] = df_fill.index.date
        dias_con_anomalia = df_fill[df_fill["anomaly_type"] != 0]["data"].unique()
        # Imprimir los días que serán eliminados
        # print("Días eliminados por tener anomalías:")
        # for dia in dias_con_anomalia:
        #     print(dia)
        # Filtrar eliminando esos días
        df_final = df_fill[~df_fill["data"].isin(dias_con_anomalia)].drop(columns=["data"])
        df_final["type"] = "normal"
        df_final = df_final.drop(columns=["anomaly_type", "hour"])
        return df_final


class AnomalyApply:

    def inject_pico(self, df, time, factor=3.0):
        df.loc[time, "W"] *= factor
        df.loc[time, "type"] = "pico"

    def inject_persistente(self, df, start, end, factor=1.2):
        mask = (df.index >= start) & (df.index < end)
        df.loc[mask, "W"] *= factor
        df.loc[mask, "type"] = "persistente"

    def inject_sensor_congelado(self, df, start, end):
        mask = (df.index >= start) & (df.index < end)
        df.loc[mask, "SENSOR"] = 0  # df.loc[mask, 'W'].iloc[0]
        df.loc[mask, "W"] = 0
        df.loc[mask, "type"] = "congelado"


class RuleBasedPVAnomalyDetector:

    def predict(self, df, max, fechas):
        df = df.copy()
        df["anomaly_type"] = 0
        outliers = np.zeros(len(df), dtype=bool)
        # =====================
        # Regla 1: Ausencia de producción con radiación > 49
        r1 = (df["SENSOR"] > 49) & (df["W"] <= 0)
        outliers |= r1
        df.loc[r1 & (df["anomaly_type"] == 0), "anomaly_type"] = 1
        # =====================
        # Regla 2: Producción de potencia, sensor nulo
        umbral_pot = 5000  # Empresa C1
        umbral_pot = 1000
        r2 = (df["W"] > umbral_pot) & (df["SENSOR"] <= 49)
        outliers |= r2
        df.loc[r2 & (df["anomaly_type"] == 0), "anomaly_type"] = 2
        # =====================
        # Regla 3: Potencia total por encima de límite
        r3 = df["W"] > max
        outliers |= r3
        df.loc[r3 & (df["anomaly_type"] == 0), "anomaly_type"] = 3
        # =====================
        # Regla 4: Sin cambio durante 30min
        r4 = pd.Series(False, index=df.index)
        rad_old = df["SENSOR"].shift(1).rolling(3).sum() > 800  # 100 para cada 5 min, 200 para cada 15min
        r4 = (df["SENSOR"] <= 0) & rad_old
        outliers |= r4
        df.loc[r4 & (df["anomaly_type"] == 0), "anomaly_type"] = 4
        # =====================
        # Regla 5: Se distancia el valor predicho del valor real un 40%
        porcentaje = 60
        df_pred = pd.read_excel("data/raw/data_processed_Empresa C3.xlsx")
        df_pred = preprocess.columns(df_pred)
        # df_pred['type'] = 'normal'
        # Introducir anomalías
        for fecha in fechas:
            if (fecha in df_pred.index) and (fecha in df.index):
                anomaly.inject_pico(df_pred, fecha, factor=3.0)
            else:
                print(f"No hay dato disponible para la fecha: {fecha}")

        X = df_pred[["SENSOR"]]
        w_real = df_pred["W"]
        modelo = joblib.load("knn-n_neighbors_2-weights_uniform_fold_0.joblib")  # Empresa C3
        # modelo = joblib.load("knn-n_neighbors_4-weights_distance_fold_0.joblib") #Empresa C1
        # modelo = joblib.load("decision_tree-criterion_squared_error-max_depth_8_fold_0.joblib")# Empresa C2
        # params = modelo.get_params()
        # modelo = DT(**params)
        # modelo.fit(X, w_real)
        w_pred = modelo.predict(X)
        error_pct = np.abs((w_real - w_pred) / w_real) * 100
        r5 = error_pct > porcentaje
        outliers |= r5
        idx_r5 = df_pred.index[r5]
        df.loc[(df.index.isin(idx_r5)) & (df["anomaly_type"] == 0), "anomaly_type"] = 5
        return outliers, df


"""     MAIN PROGRAM     """
data = pd.read_excel("data/processed/Empresa C3/analisis_Empresa C3_30min.xlsx")
preprocess = Preprocess()
df_processed = preprocess.columns(data)
# df_processed = df_processed[df_processed.index.minute.isin([0, 15, 30, 45])]
df_processed = preprocess.anomaly_clean(df_processed)

anomaly = AnomalyApply()
fechas_pico_modelo = [
    pd.Timestamp("2023-01-09 05:30"),
    pd.Timestamp("2023-02-20 08:30"),
    pd.Timestamp("2023-03-10 10:30"),
    pd.Timestamp("2023-08-11 08:00"),
    pd.Timestamp("2023-10-20 08:30"),
    pd.Timestamp("2023-07-30 18:00"),
]

fechas_pico = [pd.Timestamp("2023-05-09 05:30"), pd.Timestamp("2023-10-20 08:30"), pd.Timestamp("2023-09-10 10:30")]

for fecha in fechas_pico:
    if fecha in df_processed.index:
        anomaly.inject_pico(df_processed, fecha, factor=3.0)
    else:
        print(f"No hay dato disponible en df_processed para la fecha: {fecha}")

anomaly.inject_sensor_congelado(df_processed, pd.Timestamp("2023-09-19 12:00"), pd.Timestamp("2023-09-19 13:30"))

model = RuleBasedPVAnomalyDetector()
# max = 16000 #Empresa C2
# max = 177000 #Empresa C1
max = 60000
outliers, df_final = model.predict(df_processed, max, fechas_pico_modelo)

df_final["hour"] = df_final.index.hour
df_anomalias = df_final[outliers]
# Nº de anomalías por tipo
num_outliers = outliers.sum()
print("Nº de anomalías detectadas: ", num_outliers)

# Diccionario para renombrar
nombres_anomalias = {
    0: "Datos no anómalos",
    1: "Anomalía tipo 1",
    2: "Anomalía tipo 2",
    3: "Anomalía tipo 3",
    4: "Anomalía tipo 4",
    5: "Anomalía tipo 5",
}
conteo = df_final["anomaly_type"].value_counts()
conteo = conteo.sort_index()
conteo.index = conteo.index.map(nombres_anomalias)
for tipo, cantidad in conteo.items():
    print(f"{tipo}: {cantidad}")

# Guardar resultados
print("Guardando datos ...")
# df_anomalias.to_excel("data/processed/Empresa C2/TEST_anomalias_Empresa C2_15min.xlsx")
# df_final.to_excel("data/processed/Empresa C2/TEST_analisis_Empresa C2_15min.xlsx")

# df_anomalias.to_excel("data/processed/Empresa C1/TEST_anomalias_Empresa C1_15min.xlsx")
# df_final.to_excel("data/processed/Empresa C1/TEST_analisis_Empresa C1_15min.xlsx")

df_anomalias.to_excel("data/processed/Empresa C3/TEST_anomalias_Empresa C3_30min.xlsx")
df_final.to_excel("data/processed/Empresa C3/TEST_analisis_Empresa C3_30min.xlsx")

print(df_anomalias[df_anomalias["anomaly_type"] == 0])
print(df_anomalias[df_anomalias["anomaly_type"] == 1])
print(df_anomalias[df_anomalias["anomaly_type"] == 2])
print(df_anomalias[df_anomalias["anomaly_type"] == 3])
print(df_anomalias[df_anomalias["anomaly_type"] == 4])
print(df_anomalias[df_anomalias["anomaly_type"] == 5])

print("Finalizado ...")
