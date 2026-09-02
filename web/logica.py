import polars as pl
import requests
import time

def buscar_codigo(df, termino_busqueda, tipo="ICD10", exacto=False):
    """
    Busca coincidencias parciales o exactas en el DataFrame de conversión.
    'tipo' puede ser: 'ICD10', 'ICD9', o 'Descripción'.
    """
    if tipo == "Descripción":
        columna = "Description"
    else:
        columna = tipo

    termino = termino_busqueda.lower()
    
    if exacto:
        resultado = df.filter(pl.col(columna).str.to_lowercase() == termino)
    else:
        resultado = df.filter(pl.col(columna).str.to_lowercase().str.contains(termino))
    
    return resultado

def cargar_diccionario(ruta_csv="maestro.csv"):
    """
    Carga el CSV maestro en un DataFrame de Polars.
    """
    df = pl.read_csv(ruta_csv, separator="|")
    return df

def obtener_ingles_api(icd10_code):
    """
    Consulta la API de NLM para obtener la descripción en inglés.
    """
    url = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
    params = {"terms": icd10_code, "df": "name"}
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data[0] > 0: 
            return data[3][0][0]
        return "No se ha encontrado en la API"
    except Exception as e:
        return "Error de conexión con la API"

def procesar_ingles_lotes(df_final, col_icd10="ICD10", progress_callback=None):
    """
    Consulta la API de forma optimizada solo para códigos ICD-10 únicos.
    """

    primer_codigo = pl.col(col_icd10).str.split(", ").list.get(0)

    codigos_unicos = df_final.select(primer_codigo.alias("codigo")).filter(
        (pl.col("codigo") != "No encontrado") & (pl.col("codigo").is_not_null())
    ).unique().get_column("codigo").to_list()
    
    diccionario_ingles = {}
    total = len(codigos_unicos)

    if total == 0:
        return df_final.with_columns(pl.lit("No encontrado en API").alias("Descripcion (Ingles)"))
    
    for i, codigo in enumerate(codigos_unicos):
        diccionario_ingles[codigo] = obtener_ingles_api(codigo)
        
        if progress_callback:
            progress_callback((i + 1) / total)
            
        time.sleep(0.05) 
        
    df_traducciones = pl.DataFrame({
        "codigo_temp": pl.Series("codigo_temp", list(diccionario_ingles.keys()), dtype=pl.Utf8),
        "Descripcion (Ingles)": pl.Series("Descripcion (Ingles)", list(diccionario_ingles.values()), dtype=pl.Utf8)
    })
    
    df_resultado = df_final.with_columns(primer_codigo.alias("codigo_temp"))
    df_resultado = df_resultado.join(df_traducciones, on="codigo_temp", how="left").drop("codigo_temp")
    df_resultado = df_resultado.fill_null("No encontrado en API")
    
    return df_resultado

def convertir_lotes(df_maestro, df_usuario, columna_origen, tipo_origen, tipos_destino, progress_callback=None):
    """
    Cruza el CSV de conversión y maneja la lógica de la API.
    """
    necesita_ingles = "Inglés (API)" in tipos_destino
    destinos_internos = [t for t in tipos_destino if t != "Inglés (API)"]
    
    icd10_temporal = False
    if necesita_ingles and tipo_origen != "ICD10" and "ICD10" not in destinos_internos:
        destinos_internos.append("ICD10")
        icd10_temporal = True


    if destinos_internos:
        df_maestro_agrupado = df_maestro.group_by(tipo_origen).agg([
            pl.col(destino).drop_nulls().unique().str.join(", ").alias(destino)
            for destino in destinos_internos
        ])
        
        col_cruce = "_cruce_temp"
        df_usuario_temp = df_usuario.with_columns(pl.col(columna_origen).cast(pl.Utf8).str.to_uppercase().alias(col_cruce))
        df_maestro_temp = df_maestro_agrupado.with_columns(pl.col(tipo_origen).cast(pl.Utf8).str.to_uppercase().alias(col_cruce))
        
        df_final = df_usuario_temp.join(df_maestro_temp, on=col_cruce, how="left").drop(col_cruce)
        df_final = df_final.fill_null("No encontrado")
    else:
        df_final = df_usuario

    if necesita_ingles:
        df_final = procesar_ingles_lotes(df_final, col_icd10="ICD10", progress_callback=progress_callback)
        if icd10_temporal:
            df_final = df_final.drop("ICD10")

    return df_final