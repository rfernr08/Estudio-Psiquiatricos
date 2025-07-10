import polars as pl
import os
import json
import requests
from sentence_transformers import SentenceTransformer

dataset = "datasets/lista_diagnosticos_unicos.csv"
diccionario = "datasets/Conversor_Definitivo.csv"

df = pl.read_csv(dataset, separator="|")

# Leer el diccionario de conversión
dicc = pl.read_csv(diccionario, separator="|", truncate_ragged_lines=True)

CACHE_FILE = "cache_diag.json"
    
def api_diag(codigo: str, cache: dict) -> str:
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

def cargar_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def guardar_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

resultados = []

model = SentenceTransformer("all-MiniLM-L6-v2")

diagnosticos_unicos = pl.read_csv(dataset, separator="|")['diagnostico'].to_list()

for diag in diagnosticos_unicos:
    cache = cargar_cache()
    # Buscar en el diccionario el término en español, ICD-9 e ICD-10
    res = dicc.filter(pl.col("ICD10") == diag)
    if not res.is_empty():
        termino_es = res[0]['Description']
        icd9 = res[0]['ICD9']
        termino_in = api_diag(diag, cache)
        embedding = model.encode(termino_in, convert_to_tensor=True)
    else:
        termino_es = None
        icd9 = None
        icd10 = None
        termino_in = None
        embedding = None

    resultados.append({
        'termino_es': termino_es[0] if termino_es else "",
        'ICD10': diag,
        'ICD9': icd9[0],
        'termino_in': termino_in,
        'embedding': embedding.tolist() if embedding is not None else None,
    })

    guardar_cache(cache)

# Clean up resultados before creating the DataFrame
for r in resultados:
    for k, v in r.items():
        if v is None:
            r[k] = ""
        elif not isinstance(v, (str, int, float)):
            r[k] = str(v)

pl.DataFrame(resultados).write_csv('diagnosticos_datos.csv', separator="|")

# Guardar resultados en un nuevo CSV

