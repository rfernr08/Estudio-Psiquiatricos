import polars as pl
import pandas as pd

dataset = "datasets/PSQ_F20_F29_Ext.xlsx"
diccionario = "datasets/Conversor_Definitivo.csv"

# Leer el dataset principal
df = pl.read_excel(dataset, sheet_name="Sheet1")

# Leer el diccionario de conversión
dicc = pl.read_csv(diccionario, truncate_ragged_lines=True)

# Suponiendo que la columna de diagnóstico se llama 'diagnostico'


diag_colms = [col for col in df.columns if col.startswith("Diag")]


# Obtener subconjuntos
diag_list = df.select(diag_colms)
diag_psq = df.select(["DIAG PSQ"])

# Unir partes como hacías antes
diagnosticos_unidos = pl.concat([diag_list, diag_psq], how="horizontal")

diagnosticos_unidos.write_csv('diagnosticos_unidos.csv', separator="|")

# Obtener todos los diagnósticos únicos de las columnas seleccionadas
diagnosticos = []
for col in diagnosticos_unidos.columns:
    # Dropna para evitar errores con valores nulos
    diagnosticos += (
        str(x).strip()
        for val in df[col].drop_nans().to_list()
        for x in str(val).split(",")
    )

# Quita duplicados y ordena
diagnosticos_unicos = sorted(set([d for d in diagnosticos if d != ""]))

# Guarda el resultado en un CSV
pd.DataFrame({"diagnostico": diagnosticos_unicos}).to_csv("lista_diagnosticos_unicos.csv", index=False)

"""
diagnosticos_unicos = (
    pl.concat(
        [diagnosticos_unidos[col].drop_nulls().str.split(",") for col in diagnosticos_unidos.columns]
    )
    .arr.strip_chars()
    .explode()
    .unique()
    .sort()
    .to_list()
)

pl.DataFrame({"diagnostico": diagnosticos_unicos}).write_csv("lista_diagnosticos_unicos.csv")
"""


