import requests
import json
import os

CACHE_FILE = "datasets/cache_icd10.json"

def cargar_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def guardar_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def obtener_descripcion_icd10_con_cache(codigo: str, cache: dict) -> str:
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


def obtener_descripciones_codigos(codigos: list[str]) -> dict:
    cache = cargar_cache()
    resultado = {}

    for codigo in codigos:
        codigo = codigo.strip()
        descripcion = obtener_descripcion_icd10_con_cache(codigo, cache)
        resultado[codigo] = descripcion

    guardar_cache(cache)
    return resultado


codigos = ["R78.81", "B96.20", "C25.9", "C78.7", "Z74.3", "F25.0"]
descripciones = obtener_descripciones_codigos(codigos)

for cod, desc in descripciones.items():
    print(f"{cod}: {desc}")
