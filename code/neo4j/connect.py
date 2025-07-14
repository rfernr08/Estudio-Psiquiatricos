import dotenv
import os
from neo4j import GraphDatabase

load_status = dotenv.load_dotenv("code/neo4j/Neo4j-b0adcf45-Created-2025-07-09.txt")
if load_status is False:
    raise RuntimeError('Environment variables not loaded.')

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

# === Crea el driver y la sesión ===
driver = GraphDatabase.driver(URI, auth=AUTH)

def test_conexion():
    with driver.session() as session:
        result = session.run("RETURN 'Conexión exitosa con Neo4j' AS mensaje")
        print(result.single()["mensaje"])

# Ejecutamos la prueba
if __name__ == "__main__":
    try:
        driver.verify_connectivity()
        test_conexion()
    finally:
        driver.close()



with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
    print("Connection established.")