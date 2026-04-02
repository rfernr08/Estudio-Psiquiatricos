import pandas as pd
from pathlib import Path
import glob
import re

def extraer_diagnostico_de_nombre(path: Path) -> str:
    # Ajusta según tu patrón: e.g. "conteo_F22.csv" -> "F22"
    name = path.stem  # "conteo_F22"
    # En el caso de nombres complejos (e.g. conteo_F20_F20,89_NEO4J), separa por delimitadores.
    token = name.split("_", 1)[1] if "_" in name else name
    token = token.upper()
    token = token.replace(" ", "")

    # Normaliza F20 y F20.89 en un mismo diagnóstico
    partes = re.split(r"[_,\-]+", token)
    if any(p in ("F20", "F20.89") for p in partes):
        return "F20"

    # Si no es alguno de los casos especiales, devuelve tal cual (o token base si hay separadores)
    return partes[0] if partes else token

def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    # Normaliza nombres y codifica
    df = df.rename(columns={
        "Código Postal": "Codigo Postal",
        "CódigoPostal": "Codigo Postal",
        "codigo postal": "Codigo Postal",
        "codigo_postal": "Codigo Postal",
        "frecuencia": "Frecuencia",
        "Frecuencia": "Frecuencia",
        "freq": "Frecuencia"
    })
    # Asegurar columnas obligatorias
    if "Codigo Postal" not in df.columns or "Frecuencia" not in df.columns:
        raise ValueError(f"Falta columna esperada: {df.columns.tolist()}")
    return df[["Codigo Postal", "Frecuencia"]]

def juntar_csvs(ruta_glob: str, salida: str):
    todos = []
    for file in sorted(glob.glob(ruta_glob)):
        p = Path(file)
        diag = extraer_diagnostico_de_nombre(p)
        df = pd.read_csv(p)
        df = normalizar_columnas(df)

        df["Frecuencia"] = pd.to_numeric(df["Frecuencia"], errors="coerce").fillna(0).astype(int)
        df["Diagnostico"] = diag
        todos.append(df[["Código Postal", "Diagnóstico", "Frecuencia"]])

    if not todos:
        raise FileNotFoundError(f"No se encontraron archivos con patrón {ruta_glob}")

    result = pd.concat(todos, ignore_index=True)

    # Agrupa en caso de filas duplicadas de (Codigo Postal, Diagnostico)
    result = result.groupby(["Codigo Postal", "Diagnostico"], as_index=False)["Frecuencia"].sum()
    result.to_csv(salida, index=False, encoding="utf-8")
    print(f"Creado: {salida} con {len(result)} filas (después de agrupar).")

if __name__ == "__main__":
    # Ajusta la ruta de entrada y salida según tu proyecto:
    juntar_csvs("TFG/mapas_frecuencias/mapas_leon/conteo_*.csv", "conteo_global.csv")