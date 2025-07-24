import polars as pl
import csv

dataset = "recursos/otros/BERT/diagnosticos_F20_F20.89_sin_dups.csv"
conversor = "datasets/Conversor_Definitivo.csv"

df = pl.read_csv(dataset, separator="|")
df_conv = pl.read_csv(conversor, separator="|")

diag_colms = [col for col in df.columns if col.startswith("Diag")]
diag_list = df.select(diag_colms)
diag_psq = df.select(["DIAG PSQ"])


# Unir los tres DataFrames en uno solo, columna a columna
diagnosticos_unidos = pl.concat([diag_list, diag_psq], how="horizontal")

def estandarizar_diagnosticos_icd10(dataset: pl.DataFrame) -> tuple[pl.DataFrame, list]:
    """
    Convierte los códigos ICD-10 a descripciones según el conversor.
    Maneja valores nulos y códigos no encontrados.
    
    Returns:
        tuple: (DataFrame procesado, lista de códigos no encontrados)
    """
    diag_colms = [col for col in dataset.columns if col.startswith("Diag")]
    codigos_no_encontrados = []  # Lista para registrar códigos faltantes

    # Construir columnas convertidas una a una
    columnas_convertidas = {}
    for col in diag_colms:
        nueva_columna = []
        for i, val in enumerate(dataset[col]):
            # Handle null values
            # Replace the problematic line:
            if val is None or val == "" or str(val) == "null":
                nueva_columna.append(None)
                continue
            
            # Search for description
            resultado = df_conv.filter(pl.col("ICD10") == val)["Description"].to_list()
            if len(resultado) > 0:
                nuevo = resultado[0]
            else:
                if val not in codigos_no_encontrados:
                    codigos_no_encontrados.append(val)
                
                nuevo = None
                # Registrar código no encontrado (evitar duplicados)
                
            nueva_columna.append(nuevo)
        columnas_convertidas[col] = nueva_columna

    # Sobrescribir columnas en el dataset
    for col, valores in columnas_convertidas.items():
        dataset = dataset.with_columns(pl.Series(name=col, values=valores))

    return dataset, codigos_no_encontrados

diag_prueba = diagnosticos_unidos["DIAG PSQ"].to_list()
diag_psq_descripcion_espanol = []

diagnosticos_finales, codigos_perdidos = estandarizar_diagnosticos_icd10(df)

with open("recursos/otros/codigos_perdidos.csv", "w", newline="") as f:
    writer = csv.writer(f, delimiter="|")
    writer.writerow(["codigo_perdido"])  # Header
    for codigo in codigos_perdidos:
        writer.writerow([codigo])

for i, diag in enumerate(diag_prueba):
    diag_final = diagnosticos_unidos[i]["DIAG PSQ"][0]
    if diag_final in df_conv["ICD10"].to_list():
        descripcion = df_conv.filter(pl.col("ICD10") == diag_final)["Description"].to_list()[0]
    else:
        descripcion = "No encontrado"
    diag_psq_descripcion_espanol.append(descripcion)



diagnosticos_finales = diagnosticos_finales.with_columns(
    pl.Series("DIAG PSQ", diag_psq_descripcion_espanol)
)

diagnosticos_finales.write_csv("recursos/otros/BERT/diagnosticos_F20_F20.89_con_descripcion_sin_dups.csv", separator="|")
