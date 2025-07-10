import polars as pl
import csv
import re
import requests
import json
import os
from pathlib import Path

data_dir = Path("/home/raul/Workspace/ALBA/datasets/PSQ_F20_F29 Original.xls")
csv_file = Path("/home/raul/Workspace/ALBA/datasets/Conversor_Definitivo.csv")
icd_9_con_letras = Path("datasets/diagnosticos_icd9_con_letras.csv")
CACHE_FILE = "cache_icd10.json"

try:
    dataset = pl.read_excel(data_dir, sheet_name="PSQ F20 a F29-1")
    df_dict = pl.read_csv(
        csv_file, 
        separator='|', 
        has_header=False, 
        new_columns=["ICD9", "ICD10", "Description"],
        schema_overrides={"ICD9": pl.Utf8, "ICD10": pl.Utf8, "Diagnose": pl.Utf8})
    icd_9_list = pl.read_csv(
        icd_9_con_letras,
        separator='|', 
        has_header=False, 
        new_columns=["ICD9", "ICD10", "Description"],
        schema_overrides={"ICD9": pl.Utf8, "ICD10": pl.Utf8, "Diagnose": pl.Utf8})
except FileNotFoundError:
    print(f"File {data_dir} not found.")
    exit(1)

diag_colms = [col for col in dataset.columns if col.startswith("Diag")]
diag_list = dataset.select(diag_colms)
diag_psq = dataset.select(["DIAG PSQ"])
conjunto_dx = dataset.select(["Conjunto Dx"])

# Unir los tres DataFrames en uno solo, columna a columna
diagnosticos_unidos = pl.concat([diag_list, diag_psq, conjunto_dx], how="horizontal")

codigos_perdidos = []

def cargar_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def guardar_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def comprobar_conversion(fila) -> bool:
    """
    Recorrer los campos de diagnosticos de una fila del dataset para establecer si esta hecha en codigo ICD-9 o ICD-10.
    Devuelve True si la fila esta hecha en ICD-10, False si esta hecha en ICD-9.
    """
    di = diagnosticos_unidos[fila]["DIAG PSQ"]
    print(f"@@@@@@@@@@Comprobando fila {fila}: {di}")
    res = icd_9_list.filter(pl.col("ICD9") == diagnosticos_unidos[fila]["DIAG PSQ"])
    print(f"Resultado de la busqueda en ICD-9: {res}")
    print(f"Primer caracter de DIAG PSQ: {di[0]}")
    if di[0][0].isalpha() and di[0][0].isupper() and res.is_empty():
        # Si el primer caracter es una letra mayuscula, es ICD-10
        print(f"Fila {fila} es ICD-10: {diagnosticos_unidos[fila]['DIAG PSQ'][0]}")
        return False
    else:
        # Si el primer caracter no es una letra mayuscula, es ICD-9
        print(f"Fila {fila} es ICD-9: {diagnosticos_unidos[fila]['DIAG PSQ'][0]}")
        return True

def buscar_codigo_icd9(codigo) -> str:
    """
    Busca en el csv un codigo ICD-9 (columna 0) y devuelve el codigo ICD-10 (columna 1).
    """
    codigo_ICD_10 = ""
    # Primera vuelta al csv para buscar el codigo exacto
    res = df_dict.filter(pl.col("ICD9") == codigo)
    if not res.is_empty():
        codigo_ICD_10 = res["ICD10"][0]
        return codigo_ICD_10
    
    print(f"Codigo original {codigo} no encontrado en el CSV.")
    variantes = []
    if "." not in codigo:
        variantes.append(f"{codigo}.")
        for i in range(10):
            variantes.append(f"{codigo}.{i}")
        for i in range(100):
            variantes.append(f"{codigo}.{i:02d}")
    else:
        base = codigo
        for i in range(10):
            variantes.append(f"{base}{i}")
        for i in range(100):
            variantes.append(f"{base}{i:02d}")

    for variante in variantes:
        res_var = df_dict.filter(pl.col("ICD9") == variante)
        if not res_var.is_empty():
            print(f"Encontrada variante {variante} para el codigo {codigo} : {res_var["ICD10"][0]}.")
            return res_var["ICD10"][0]
        
    print(f"Codigo {codigo} no encontrado en el CSV de respaldo.")
    codigos_perdidos.append(codigo)
    return ""

def obtener_descripcion_icd10_con_cache(codigo: str, cache: dict) -> str:
    """
    Buscar la descripción del código ICD-10 en la API de Clinical Tables.
    Utiliza un caché para evitar consultas repetidas.
    Si el código ya está en el caché, devuelve la descripción desde allí.
    """
    if codigo in cache:
        return cache[codigo]

    url = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
    params = {
        "sf": "code",
        "terms": codigo
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list) and len(data) >= 4 and isinstance(data[3], list):
            for item in data[3]:
                if item[0] == codigo:
                    cache[codigo] = item[1]
                    return item[1]
            if data[3]:
                cache[codigo] = data[3][0][1]
                return data[3][0][1]

    except Exception as e:
        print(f"Error consultando {codigo}: {e}")

    cache[codigo] = "Descripción no disponible"
    return "Descripción no disponible"

def obtener_descripciones_codigos(codigos: list[str]) -> str:
    """
    Metdo para obtener las descripciones de los codigos ICD-10.
    Utiliza un caché para evitar consultas repetidas a la API.
    Devuelve una cadena con las descripciones de los códigos separados por comas.
    """
    cache = cargar_cache()
    resultado = {}

    for codigo in codigos:
        codigo = codigo.strip()
        descripcion = obtener_descripcion_icd10_con_cache(codigo, cache)
        resultado[codigo] = descripcion

    guardar_cache(cache)
    resultado_str = ", ".join(f"{desc}" for codigo, desc in resultado.items())
    return resultado_str

def añadir_descripciones(codigos: list) -> str:
    """
    Buscar segun el codigo ICD-10 del diagnotico la enfermedad psicotica y añadirla a una nueva columna del dataset.
    Se le puede pasar un codigo o varios codigos, para añadir todos los nombres de las enfermedades psicoticas a la columna resultado.
    """
    descripcion_final = ""
    for codigo in codigos:
        codigo = codigo.strip()
        res = df_dict.filter(pl.col("ICD10") == codigo)
        if not res.is_empty():
            descripcion_final += res["Description"][0] + ", "
            print(f"Descripcion encontrada para {codigo}: {res['Description'][0]}")
            continue

        # Si no se encuentra, buscar variantes
        variantes = []
        if "." not in codigo:
            variantes.append(f"{codigo}.")
            for i in range(10):
                variantes.append(f"{codigo}.{i}")
            for i in range(100):
                variantes.append(f"{codigo}.{i:02d}")
        else:
            base = codigo
            for i in range(10):
                variantes.append(f"{base}{i}")
            for i in range(100):
                variantes.append(f"{base}{i:02d}")

        descripcion = ""
        for variante in variantes:
            res_var = df_dict.filter(pl.col("ICD10") == variante)
            if not res_var.is_empty() and res_var["Description"][0] is not None:
                descripcion = res_var["Description"][0]
                descripcion_final += descripcion + ", "
                print(f"Descripcion encontrada para variante {variante}: {descripcion}")
                break

        if descripcion == "":
            print(f"Descripcion no encontrada para el codigo {codigo}.")
            descripcion_final += "Descripcion no encontrada #, "
            codigos_perdidos.append(codigo)

    descripcion_final = descripcion_final.rstrip(", ")  # Eliminar la ultima coma y espacio
    return descripcion_final

def convertir_a_icd10(codigo):
    res = icd_9_list.filter(pl.col("ICD9") == codigo)
    if codigo is None or codigo == "" or codigo == "null":
        return ""
    # Si el código ya es ICD-10 (empieza por letra), lo dejamos igual
    if codigo[0].isalpha() and codigo[0].isupper() and res.is_empty():
        return codigo
    # Si es ICD-9, lo convertimos
    return buscar_codigo_icd9(codigo)

def convertir_a_icd9(codigo):
    """
    Convierte un código ICD-10 a ICD-9 utilizando el CSV de respaldo.
    Si no se encuentra el código, devuelve una cadena vacía.
    """
    if codigo is None or codigo == "" or codigo == "null":
        return ""
    
    # Buscar en el CSV de respaldo
    res = df_dict.filter(pl.col("ICD10") == codigo)
    print(f"================== Respuesta de búsqueda en ICD-10: {res}")
    if not res.is_empty() and res["ICD9"][0] != "NN":
        print(f"Encontrado mapeo ICD-9 para el código ICD-10 '{codigo}': {res['ICD9'][0]}")
        return res["ICD9"][0]
    
    # Si no se encuentra, buscar variantes
    variantes = []
    if "." not in codigo:
        variantes.append(f"{codigo}.")
        for i in range(10):
            variantes.append(f"{codigo}.{i}")
        for i in range(100):
            variantes.append(f"{codigo}.{i:02d}")
    else:
        base = codigo
        for i in range(10):
            variantes.append(f"{base}{i}")
        for i in range(100):
            variantes.append(f"{base}{i:02d}")

    for variante in variantes:
        res_var = df_dict.filter(pl.col("ICD10") == variante)
        if not res_var.is_empty() and res_var["ICD9"][0] != "NN":
            print(f"Encontrado mapeo ICD-9 para el código ICD-10 '{codigo}' con variante '{variante}': {res_var['ICD9'][0]}")
            print(f"Codigo encontrada: {res_var['ICD9'][0]}")
            return res_var["ICD9"][0]

    print(f"Advertencia: No se encontró mapeo ICD-9 para el código ICD-10 '{codigo}'.")
    codigos_perdidos.append(codigo)
    return ""

def estandarizar_diagnosticos_icd10(dataset: pl.DataFrame) -> pl.DataFrame:
    """
    Convierte los códigos ICD-9 a ICD-10 en columnas 'Diag*' según la columna 'DIAG PSQ'.
    Sobrescribe directamente las columnas originales.
    """
    diag_colms = [col for col in dataset.columns if col.startswith("Diag")]

    # Detectar qué filas están en ICD-9 (DIAG PSQ que no empieza por letra)
    es_icd9_flags = [
        not (str(val)[0].isalpha()) if isinstance(val, str) and val else True
        for val in dataset["DIAG PSQ"]
    ]

    # Construir columnas convertidas una a una
    columnas_convertidas = {}
    for col in diag_colms:
        nueva_columna = []
        for i, val in enumerate(dataset[col]):
            if es_icd9_flags[i]:
                nuevo = convertir_a_icd10(val) if isinstance(val, str) else val
                nueva_columna.append(nuevo)
            else:
                nueva_columna.append(val)
        columnas_convertidas[col] = nueva_columna

    # Sobrescribir columnas en el dataset
    for col, valores in columnas_convertidas.items():
        dataset = dataset.with_columns(pl.Series(name=col, values=valores))

    return dataset

def procesar_psiquiatria(fila: int, lista_icd10, lista_icd9, lista_descripciones_espanol, lista_descripciones_ingles, version_icd_9: bool = True) -> None:
    """
    Procesa la columna 'DIAG PSQ' del dataset, que contiene varios codigos separados por coma.
    Devuelve una lista de codigos procesados , todos en icd-10 ,y otra lista con los terminos de cada uno.
    """
    diag_final = diagnosticos_unidos[fila]["DIAG PSQ"][0]
    print(f"-......Fila {fila} diagnostico: {diag_final}")
    print(f"------> Version ICD-9: {version_icd_9}")
    if version_icd_9:
        codigos_icd10 = []
        lista_icd9.append(diag_final)
        for codigo in diag_final.split(","):
            codigo = codigo.strip()
            codigo_icd10 = buscar_codigo_icd9(codigo)
            codigos_icd10.append(codigo_icd10)
            
        lista_icd10.append(", ".join(filter(None, codigos_icd10)))
        lista_descripciones_espanol.append(añadir_descripciones(codigos_icd10))
        lista_descripciones_ingles.append(obtener_descripciones_codigos(codigos_icd10))
    else:
        lista_icd10.append(diag_final)
        codigos = diag_final.split(",")
        nuevos_codigos = []
        for i, codigo in enumerate(codigos):
            codigos[i] = codigo.strip()
            codigo_icd9 = convertir_a_icd9(codigo)
            if not codigo_icd9:
                print(f"Advertencia: No se encontró mapeo ICD-9 para el código ICD-10 '{codigo}' en la fila {fila}.")
                nuevos_codigos.append("")
            else:
                nuevos_codigos.append(codigo_icd9)
        lista_icd9.append(", ".join(filter(None, nuevos_codigos)).rstrip(", "))
        lista_descripciones_espanol.append(añadir_descripciones(codigos))  
        lista_descripciones_ingles.append(obtener_descripciones_codigos(codigos))      
        

def procesar_conjunto(fila: int, lista_icd10, lista_icd9, lista_descripciones_espanol, lista_descripciones_ingles, version_icd_9: bool = True) -> None:
    """
    Procesa la columna 'Conjunto Dx' del dataset, que contiene varios codigos separados por corchetes [].
    Devuelve una lista de codigos procesados , todos en icd-10 ,y otra lista con los terminos de cada uno.
    """
    diag_conjunto = diagnosticos_unidos[fila]["Conjunto Dx"][0]
    if version_icd_9:
        print(f"Fila {fila} es ICD-9: {diag_conjunto} -Vamos a buscar el codigo ICD-10")
        codigos_icd10 = []
        codigos_descripcion = []
        lista_icd9.append(diag_conjunto)
        for codigo in re.findall(r"\[([^\[\]]+)\]", diag_conjunto):
            codigo = codigo.strip()
            codigo_icd10 = buscar_codigo_icd9(codigo)
            if codigo_icd10:
                codigos_icd10.append("[" + codigo_icd10 + "]")
                codigos_descripcion.append(codigo_icd10)
            else:
                print(f"Advertencia: No se encontró mapeo ICD-10 para el código ICD-9 '{codigo}' en la fila {fila}.")

        print(f"Codigos encontrados: {codigos_icd10}")
        
        lista_icd10.append(", ".join(filter(None, codigos_icd10)))
        lista_descripciones_espanol.append(añadir_descripciones(codigos_descripcion))
        lista_descripciones_ingles.append(obtener_descripciones_codigos(codigos_descripcion))
    else:
        print(f"Fila conjunto {fila} es ICD-10: {diag_conjunto}")
        codigos_icd9 = []
        for codigo in re.findall(r"\[([^\[\]]+)\]", diag_conjunto):
            codigo = codigo.strip()
            codigo_icd9 = convertir_a_icd9(codigo)
            if codigo_icd9:
                codigos_icd9.append("[" + codigo_icd9 + "]")
            else:
                codigos_icd9.append("[NN]")
                print(f"Advertencia: No se encontró mapeo ICD-9 para el código ICD-10 '{codigo}' en la fila {fila}.")
        lista_icd10.append(diag_conjunto)
        lista_icd9.append(", ".join(filter(None, codigos_icd9)))
        codigos = re.findall(r"\[([^\[\]]+)\]", diag_conjunto)
        lista_descripciones_espanol.append(añadir_descripciones(codigos))
        lista_descripciones_ingles.append(obtener_descripciones_codigos(codigos))
        

if __name__ == "__main__":
    print("Iniciando el proceso de prueba con 100 filas aleatorias...")

    # Seleccionar aleatoriamente 100 filas del dataset
    dataset_muestra = dataset.sample(n=50, seed=42)
    diag_colms = [col for col in dataset_muestra.columns if col.startswith("Diag")]

    # Obtener subconjuntos
    diag_list = dataset_muestra.select(diag_colms)
    diag_psq = dataset_muestra.select(["DIAG PSQ"])
    conjunto_dx = dataset_muestra.select(["Conjunto Dx"])

    # Unir partes como hacías antes
    diagnosticos_unidos = pl.concat([diag_list, diag_psq, conjunto_dx], how="horizontal")

    # Variables auxiliares
    diag_psq_icd10 = []
    diag_psq_icd9 = []
    diag_psq_descripcion_espanol = []
    diag_psq_descripcion_ingles = []

    diag_conjunto_icd10 = []
    diag_conjunto_icd9 = []
    diag_conjunto_descripcion_espanol = []
    diag_conjunto_descripcion_ingles = []

    codigos_perdidos.clear()

    # Procesamiento de muestra
    diag_prueba = diagnosticos_unidos["DIAG PSQ"].to_list()
    for i, diag in enumerate(diag_prueba):
        if isinstance(diag, str):
            if comprobar_conversion(i):
                print(f"Fila {i} es ICD-9: {diag}")
                procesar_psiquiatria(i, diag_psq_icd10, diag_psq_icd9, diag_psq_descripcion_espanol, diag_psq_descripcion_ingles, True)
                procesar_conjunto(i, diag_conjunto_icd10, diag_conjunto_icd9, diag_conjunto_descripcion_espanol, diag_conjunto_descripcion_ingles, True)
            else:
                print(f"Fila {i} es ICD-10: {diag}")
                procesar_psiquiatria(i, diag_psq_icd10, diag_psq_icd9, diag_psq_descripcion_espanol, diag_psq_descripcion_ingles, False)
                procesar_conjunto(i, diag_conjunto_icd10, diag_conjunto_icd9, diag_conjunto_descripcion_espanol, diag_conjunto_descripcion_ingles, False)
        else:
            diag_psq_icd10.append(diag)

    # Guardar los resultados de esta muestra
    diagnosticos_finales = estandarizar_diagnosticos_icd10(dataset_muestra)

    diagnosticos_finales = diagnosticos_finales.with_columns(
        pl.Series("DIAG PSQ", diag_psq_icd10),
        pl.Series("DIAG PSQ ICD-9", diag_psq_icd9),
        pl.Series("DIAG PSQ Descripcion Español", diag_psq_descripcion_espanol),
        pl.Series("DIAG PSQ Descripcion Ingles", diag_psq_descripcion_ingles),
        pl.Series("Conjunto Dx", diag_conjunto_icd10),
        pl.Series("Conjunto Dx ICD-9", diag_conjunto_icd9),
        pl.Series("Conjunto Dx Descripcion Español", diag_conjunto_descripcion_espanol),
        pl.Series("Conjunto Dx Descripcion Ingles", diag_conjunto_descripcion_ingles)
    )

    diagnosticos_finales.write_excel("Muestra_100_Prueba_Conversion_ICD.xlsx")

    print("Diagnósticos procesados en muestra de 100 filas.")
    print("Codigos no encontrados:")
    print(codigos_perdidos)

    
"""
Mision: 
X-Recorer cada fila del csv y identificar cuales estan escritas en ICD-9 y cuales en ICD-10.
X-Si un codigo ICD-9 le faltan letras al final, rellenar con ceros hasta completar 5 caracteres o encontrar un codigo apropiado.
X-Casos de codigo como 298.8 que encuentran codigo, pero 295 que es un caso general y habria que añadir ceros hasta encontrar un codigo apropiado.
X-Si se detecta codigos sin letra al inicio,considerar esa linea entera como ICD-9,por si econtramos algun codigo ICD-9 que contengan letras como V o E.
Crear 2 nuevas columnas al final del dataset, con el nombre en ingles del diagnostico psicotico y el resumen de los demas diagnosticos psicoticos.
Al añadir las descripciones, cuidado con que se junta el diagnostica final con el conjunto Dx,que a la vez contiene varios codigos separados por [].
Guardar el dataset modificado en un nuevo archivo Excel.

Usar una API para buscar las descripciones para el conunto Dx, si no se encuentra en el csv de respaldo.

def procesar_diagnosticos(fila: int, lista_psq, version_icd_9: bool = True) -> None:
    diagnosticos = diagnosticos_unidos[fila][diag_colms]
    print(f"Fila {fila} diagnosticos: {diagnosticos}")
    if version_icd_9:
        codigos_icd10 = []
        for codigo in diagnosticos:
            if isinstance(codigo, str) and codigo is not "null":
                codigo = codigo.strip()
                codigo_icd10 = buscar_codigo_icd9(codigo)
                if codigo_icd10:
                    codigos_icd10.append(codigo_icd10)
                else:
                    print(f"Advertencia: No se encontró mapeo ICD-10 para el código ICD-9 '{codigo}' en la fila {fila}.")
        
        lista_psq.append(", ".join(filter(None, codigos_icd10)))
    else:
        lista_psq.append(diagnosticos)
"""
