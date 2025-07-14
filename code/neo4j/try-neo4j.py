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

def subir_nodo():
    summary = driver.execute_query(""" 
        CREATE (a:Person {name: $name})
        CREATE (b:Person {name: $friendName})
        CREATE (a)-[:KNOWS]->(b)
        """,
        name="Chema", friendName="Juan",  
        database_="neo4j",  
    ).summary
    print("Created {nodes_created} nodes in {time} ms.".format(
        nodes_created=summary.counters.nodes_created,
        time=summary.result_available_after
    ))

def mirar_nodo():
    records, summary, keys = driver.execute_query("""
        MATCH (p:Person)-[:KNOWS]->(:Person)
        RETURN p.name AS name
        """,
        database_="neo4j",
    )

    # Loop through results and do something with them
    for record in records:
        print(record.data())  # obtain record as dict

    # Summary information
    print("The query `{query}` returned {records_count} records in {time} ms.".format(
        query=summary.query, 
        records_count=len(records),
        time=summary.result_available_after
    ))

def eliminar_nodo():
    # This does not delete _only_ p, but also all its relationships!
    records, summary, keys = driver.execute_query("""
        MATCH (p:Person {name: $name})
        DETACH DELETE p
        """, name="Alice",
        database_="neo4j",
    )
    print(f"Query counters: {summary.counters}.")

if __name__ == "__main__":
    try:
        driver.verify_connectivity()
        print("Connection established.")
        subir_nodo()
        #mirar_nodo()
        #eliminar_nodo()
    finally:
        driver.close()