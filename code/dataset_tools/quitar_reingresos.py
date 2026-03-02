import pandas as pd

# ruta al CSV (ajusta si cambia de ubicación)
csv_path = r"C:\Users\Usuario\Documents\Workspace\Mirage\TFG\dataset\diagnosticos_limpios\diagnosticos_unidos.csv"

# leer el dataset
df = pd.read_csv(csv_path)

print(f"filas originales: {len(df)}")

# eliminar duplicados exactos sobre todas las columnas
df_clean = df.drop_duplicates(keep='first')

print(f"filas tras limpiar: {len(df_clean)}")

# si quieres conservar sólo el primer/último registro de cada paciente,
# especifica las columnas que identifican al paciente, por ejemplo:
# df_clean = df.drop_duplicates(subset=['patient_id'], keep='first')

# escribir de nuevo el CSV (sobreescribe o en otro fichero)
df_clean.to_csv(csv_path.replace(".csv", "_limpio.csv"), index=False)