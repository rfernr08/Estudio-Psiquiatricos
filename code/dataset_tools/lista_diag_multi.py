import concurrent.futures
from sentence_transformers import SentenceTransformer
import polars as pl
import os
import json

dataset = "datasets/lista_diagnosticos_unicos.csv"
diccionario = "datasets/Conversor_Definitivo.csv"

df = pl.read_csv(dataset, separator="|")
dicc = pl.read_csv(diccionario, separator="|", truncate_ragged_lines=True)

CACHE_FILE = "cache_diag.json"

def cargar_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

cache = cargar_cache()
model = SentenceTransformer("all-MiniLM-L6-v2")
diagnosticos_unicos = pl.read_csv(dataset, separator="|")['diagnostico'].to_list()

def procesar_diag(diag):
    # Buscar en el diccionario el término en español, ICD-9 e ICD-10
    res = dicc.filter(pl.col("ICD10") == diag)
    if not res.is_empty():
        desc_list = res["Description"].to_list()
        icd9_list = res["ICD9"].to_list()
        termino_es = desc_list[0] if desc_list else ""
        icd9 = icd9_list[0] if icd9_list else ""
        termino_in = cache.get(diag, "")
        embedding = model.encode(termino_in, convert_to_tensor=True) if termino_in else None
    else:
        termino_es = None
        icd9 = None
        termino_in = None
        embedding = None

    return {
        'termino_es': termino_es[0],
        'ICD10': diag,
        'ICD9': icd9[0],
        'termino_in': termino_in,
        'embedding': embedding.tolist() if embedding is not None else None,
    }

# Usa ThreadPoolExecutor para paralelizar el cálculo de embeddings
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    resultados = list(executor.map(procesar_diag, diagnosticos_unicos))

# Limpieza final para asegurar que todo es serializable
for r in resultados:
    for k, v in r.items():
        if v is None:
            r[k] = ""
        elif not isinstance(v, (str, int, float)):
            r[k] = str(v)

pl.DataFrame(resultados).write_csv('diagnosticos_datos.csv', separator="|")