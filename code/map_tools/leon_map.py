import pandas as pd

# Carga el dataset procesado
df = pd.read_csv("datasets/conteo_diagnosticos_leon.csv")

df["DIAG PSQ SPLIT"] = df["DIAG PSQ SPLIT"].str.strip()
diagnosticos = df["DIAG PSQ SPLIT"].unique()

for diag in diagnosticos:
    # Filtra por diagnóstico
    df_diag = df[df["DIAG PSQ SPLIT"] == diag]
    # Agrupa por código postal y cuenta frecuencia
    conteo = df_diag.groupby("Código Postal").size().reset_index(name="frecuencia")
    # Guarda el CSV
    conteo.to_csv(f"datasets/mapas_leon/conteo_{diag}.csv", index=False)
    print(f"Guardado: datasets/mapas_leon/conteo_{diag}.csv")