import polars as pl
import requests
import time

def buscar_codigo(df, termino_busqueda, tipo="ICD10", exacto=False):
    """
    Busca coincidencias parciales o exactas en el DataFrame.
    'tipo' puede ser: 'ICD10', 'ICD9', o 'Descripción'.
    """
    # Mapear el tipo a la columna real del CSV
    if tipo == "Descripción":
        columna = "Description"
    else:
        columna = tipo

    # Pasamos todo a minúsculas para que la búsqueda no sea sensible a mayúsculas
    termino = termino_busqueda.lower()
    
    if exacto:
        # Búsqueda exacta (para cuando el usuario ya ha seleccionado una opción)
        resultado = df.filter(pl.col(columna).str.to_lowercase() == termino)
    else:
        # Búsqueda parcial (para las sugerencias mientras escribe)
        resultado = df.filter(pl.col(columna).str.to_lowercase().str.contains(termino))
    
    return resultado

def cargar_diccionario(ruta_csv="maestro.csv"):
    """Carga el CSV maestro en un DataFrame de Polars."""
    # Usamos separator="|" según tu ejemplo
    df = pl.read_csv(ruta_csv, separator="|")
    return df

def obtener_ingles_api(icd10_code):
    """Consulta la API de NLM para obtener la descripción en inglés."""
    url = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
    params = {"terms": icd10_code, "df": "name"} # Ajustaremos esto según cómo devuelva el JSON la API
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # La API de NLM suele devolver: [numero_resultados, [codigos], None, [[descripciones]]]
        # Hay que extraer el texto correcto. Aquí un ejemplo genérico:
        if data[0] > 0: 
            return data[3][0][0] # Primera descripción del primer resultado
        return "No encontrado en API"
    except Exception as e:
        return "Error de conexión"

def procesar_ingles_lotes(df_final, col_icd10="ICD10", progress_callback=None):
    """Consulta la API de forma optimizada solo para códigos ICD-10 únicos."""
    # Extraemos solo el PRIMER código de la lista si hay varios separados por comas
    primer_codigo = pl.col(col_icd10).str.split(", ").list.get(0)
    
    # Obtenemos los códigos únicos para minimizar las peticiones a la API
    codigos_unicos = df_final.select(primer_codigo.alias("codigo")).filter(
        (pl.col("codigo") != "No encontrado") & (pl.col("codigo").is_not_null())
    ).unique().get_column("codigo").to_list()
    
    diccionario_ingles = {}
    total = len(codigos_unicos)
    
    # Bucle de consultas a la API
    for i, codigo in enumerate(codigos_unicos):
        diccionario_ingles[codigo] = obtener_ingles_api(codigo)
        
        # Actualizamos la barra de progreso de Streamlit
        if progress_callback:
            progress_callback((i + 1) / total)
            
        # Retraso de seguridad para no saturar el servidor de la NLM
        time.sleep(0.05) 
        
    # Mapeamos los resultados de vuelta a un DataFrame temporal
    df_traducciones = pl.DataFrame({
        "codigo_temp": list(diccionario_ingles.keys()),
        "Descripcion (Ingles)": list(diccionario_ingles.values())
    })
    
    # Cruzamos las traducciones con el DataFrame principal
    df_resultado = df_final.with_columns(primer_codigo.alias("codigo_temp"))
    df_resultado = df_resultado.join(df_traducciones, on="codigo_temp", how="left").drop("codigo_temp")
    df_resultado = df_resultado.fill_null("No encontrado en API")
    
    return df_resultado

def convertir_lotes(df_maestro, df_usuario, columna_origen, tipo_origen, tipos_destino, progress_callback=None):
    """Cruza el CSV del usuario con el maestro y maneja la lógica de la API."""
    # Separamos si el usuario pidió inglés o no
    necesita_ingles = "Inglés (API)" in tipos_destino
    destinos_internos = [t for t in tipos_destino if t != "Inglés (API)"]
    
    # REGLA OBLIGATORIA: Para la API necesitamos ICD-10. 
    # Si no es el origen y el usuario no lo pidió como destino, lo añadimos ocultamente.
    icd10_temporal = False
    if necesita_ingles and tipo_origen != "ICD10" and "ICD10" not in destinos_internos:
        destinos_internos.append("ICD10")
        icd10_temporal = True

    # 1. Agrupamos el maestro (Solo si hay destinos internos que cruzar)
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
        df_final = df_usuario # Si solo pidió inglés y partía de ICD10

    # 2. Procesamos el inglés si se solicitó
    if necesita_ingles:
        df_final = procesar_ingles_lotes(df_final, col_icd10="ICD10", progress_callback=progress_callback)
        # Limpiamos el ICD10 si lo añadimos solo temporalmente para la API
        if icd10_temporal:
            df_final = df_final.drop("ICD10")

    return df_final