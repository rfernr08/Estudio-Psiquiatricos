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

diagnosticos_df = pd.read_csv(diagnosticos, sep="|")

# Limpieza de embeddings (de texto a lista de floats)
diagnosticos_df['embedding'] = diagnosticos_df['embedding'].apply(ast.literal_eval)

# Cargar relaciones
relaciones_df = pd.read_csv(relaciones, sep="|")

def crear_diagnosticos(tx, row):
    query = """
    MERGE (d:Diagnostico {terminoEN: $terminoEN})
    SET d.ICD10 = $ICD10,
        d.ICD9 = $ICD9,
        d.terminoIN = $terminoIN,
        d.embedding = $embedding
    """
    tx.run(query, **row)

def crear_relacion(tx, diag, diag_psi):
    query = """
    MATCH (d1:Diagnostico {terminoEN: $nombre})
    MATCH (d2:Diagnostico {terminoEN: $nombre_psi})
    MERGE (d1)-[:RELACIONADO_CON]->(d2)
    """
    tx.run(query, nombre=diag, nombre_psi=diag_psi)

if __name__ == "__main__":
    try:
       with driver.session() as session:
            print("Subiendo diagnósticos...")
            for _, row in diagnosticos_df.iterrows():
                session.execute_write(crear_diagnosticos, row.to_dict())

            print("Creando relaciones...")
            for _, row in relaciones_df.iterrows():
                session.execute_write(crear_relacion, row['nombre_diagnostico'], row['nombre_diagnostico_psiquiatrico'])
    finally:
        driver.close()
        print("Connection closed.")
        print("Datos subidos correctamente a Neo4j.")