import dotenv
import os
from neo4j import GraphDatabase
import pandas as pd
import ast

load_status = dotenv.load_dotenv("code/neo4j/Neo4j-b0adcf45-Created-2025-07-09.txt")

diagnosticos = "datasets/Full_Datos_Diag_Men.csv"
relaciones = "datasets/relaciones_diagnosticos_psiquiatricos.csv"

if load_status is False:
    raise RuntimeError('Environment variables not loaded.')

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

# === Crea el driver y la sesión ===
driver = GraphDatabase.driver(URI, auth=AUTH)

session = driver.session()


# 1. Contar total de diagnósticos cargados
def contar_diagnosticos():
    query = "MATCH (d:Diagnostico) RETURN count(d) AS total"
    result = session.run(query).data()
    print(f"Total de diagnósticos en la base de datos: {result[0]['total']}")

# 2. Mostrar un diagnóstico específico por nombre
def buscar_diagnostico(nombre):
    query = """
    MATCH (d:Diagnostico {terminoEN: $nombre})
    RETURN d.terminoEN AS termino, d.ICD10 AS ICD10, d.ICD9 AS ICD9
    """
    result = session.run(query, nombre=nombre).data()
    if result:
        print("Diagnóstico encontrado:")
        print(result[0])
    else:
        print("Diagnóstico no encontrado.")

# 3. Contar relaciones desde un diagnóstico específico
def contar_relaciones(diagnostico_nombre):
    query = """
    MATCH (d:Diagnostico {terminoEN: $nombre})-[:RELACIONADO_CON]->(d2)
    RETURN count(d2) AS relaciones
    """
    result = session.run(query, nombre=diagnostico_nombre).data()
    print(f"Número de diagnósticos relacionados con '{diagnostico_nombre}': {result[0]['relaciones']}")

# 4. Listar todos los diagnósticos relacionados a uno específico
def listar_relaciones(diagnostico_nombre):
    query = """
    MATCH (d:Diagnostico {terminoEN: $nombre})-[:RELACIONADO_CON]->(rel)
    RETURN rel.terminoEN AS relacionado
    """
    results = session.run(query, nombre=diagnostico_nombre).data()
    print(f"Diagnósticos relacionados con '{diagnostico_nombre}':")
    for r in results:
        print("-", r['relacionado'])

# ----- PRUEBAS -----
contar_diagnosticos()
buscar_diagnostico("Bacteriemia")
contar_relaciones("Bacteriemia")
listar_relaciones("Bacteriemia")
session.close()
