from neo4j import GraphDatabase
import dotenv
import os
import ast
import pandas as pd

# Conexión a Neo4j
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

# Obtener nodos y algunas propiedades
query_nodos = """
MATCH (d:Diagnostico)
RETURN d.terminoEN AS termino, d.ICD10 AS icd10, d.ICD9 AS icd9, d.terminoIN AS terminoIN, d.embedding AS embedding
"""
nodos = pd.DataFrame(session.run(query_nodos).data())
nodos.to_csv("nodos_diagnosticos.csv", index=False, sep="|")
# Obtener relaciones entre diagnósticos
query_relaciones = """
MATCH (a:Diagnostico)-[:RELACIONADO_CON]->(b:Diagnostico)
RETURN a.terminoEN AS source, b.terminoEN AS target
"""
relaciones = pd.DataFrame(session.run(query_relaciones).data())
relaciones.to_csv("relaciones_diagnosticos.csv", index=False, sep="|")

# Rellenar valores nulos por string vacío para evitar errores
nodos["icd10"] = nodos["icd10"].fillna("")

# Crear una etiqueta binaria: 1 si empieza por A o B (infectious), 0 si no
nodos["target_infeccion"] = nodos["icd10"].str.upper().str[0].apply(lambda x: 1 if x in ["A", "B"] else 0)

# Algunas features simples
nodos["icd10_len"] = nodos["icd10"].apply(len)
nodos["icd10_prefix"] = nodos["icd10"].str[:1]  # primera letra del código

# Variables categóricas a dummies
X = pd.get_dummies(nodos[["icd10_len", "icd10_prefix"]])
y = nodos["target_infeccion"]


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar clasificador
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluar
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


