import dotenv
import os
from neo4j import GraphDatabase


URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

# === Crea el driver y la sesión ===
driver = GraphDatabase.driver(URI, auth=AUTH)

with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
