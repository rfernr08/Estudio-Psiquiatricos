import polars as pl

# Rutas de tus archivos
dataset = "datasets/PSQ_F20_F29_Ext.csv"  # Tu dataset principal
diagnosticos_unicos = "datasets/Full_Datos_Diag_Men.csv"  # CSV con columnas: ICD10, termino_es

# Lee el dataset principal
df = pl.read_csv(dataset, separator="|", schema_overrides={"DIAG PSQ ICD-9": pl.Utf8})

# Lee el CSV de diagnósticos únicos con nombre en español
df_diag = pl.read_csv(diagnosticos_unicos, separator="|")

# Selecciona las columnas de diagnóstico (Diag 01, Diag 02, ..., Diag 20)
diag_colms = [col for col in df.columns if col.startswith("Diag")]
diag_colms.append("DIAG PSQ")  # Añade la columna de diagnóstico psiquiátrico

# Extrae todas las relaciones: (diagnóstico, DIAG PSQ)
relaciones = []
for row in df.iter_rows(named=True):
    diag_psq = row["DIAG PSQ"]
    for col in diag_colms[:-1]:  # Excluye DIAG PSQ de los diagnósticos generales
        diag = row[col]
        if diag and diag_psq and diag != "" and diag_psq != "":
            relaciones.append({"diagnostico": diag, "diagnostico_psiquiatrico": diag_psq})

# Convierte a DataFrame
relaciones_df = pl.DataFrame(relaciones)

# Añade el nombre en español del diagnóstico general
relaciones_df = relaciones_df.join(
    df_diag.select(["ICD10", "termino_es"]),
    left_on="diagnostico",
    right_on="ICD10",
    how="left"
)

# Añade el nombre en español del diagnóstico psiquiátrico
relaciones_df = relaciones_df.join(
    df_diag.select([pl.col("ICD10").alias("ICD10_PSQ"), pl.col("termino_es").alias("termino_es_psq")]),
    left_on="diagnostico_psiquiatrico",
    right_on="ICD10_PSQ",
    how="left"
)

# Selecciona y renombra columnas para el CSV final
resultado = relaciones_df.select([
    pl.col("termino_es").alias("nombre_diagnostico"),
    pl.col("termino_es_psq").alias("nombre_diagnostico_psiquiatrico")
])

# Filtra filas donde ambos campos no estén vacíos ni sean nulos
resultado = resultado.filter(
    pl.col("nombre_diagnostico").is_not_null() & (pl.col("nombre_diagnostico") != "") &
    pl.col("nombre_diagnostico_psiquiatrico").is_not_null() & (pl.col("nombre_diagnostico_psiquiatrico") != "")
)

# Guarda el resultado
resultado.write_csv("relaciones_diagnosticos_psiquiatricos.csv", separator="|")