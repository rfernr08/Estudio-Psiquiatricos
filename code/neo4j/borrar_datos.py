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

with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
