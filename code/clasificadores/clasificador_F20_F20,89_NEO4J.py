from neo4j import GraphDatabase
import dotenv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Conexión a Neo4j
load_status = dotenv.load_dotenv("code/neo4j/Neo4j-b0adcf45-Created-2025-07-09.txt")
dataset = "datasets/PSQ_F20_F29_Ext.csv"
diagnosticos = "datasets/Full_Datos_Diag_Men.csv"
relaciones = "datasets/relaciones_diagnosticos_psiquiatricos.csv"

if load_status is False:
    raise RuntimeError('Environment variables not loaded.')

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

# === Crea el driver y la sesión ===
driver = GraphDatabase.driver(URI, auth=AUTH)

session = driver.session()

query_nodos = """
MATCH (d:Diagnostico)
WHERE d.ICD10 IN ['F20', 'F20.89']
RETURN d.terminoEN AS termino, d.ICD10 AS icd10, d.ICD9 AS icd9, d.terminoIN AS terminoIN, d.embedding AS embedding
"""
df = pd.DataFrame(session.run(query_nodos).data())

query_relaciones = """
MATCH (a:Diagnostico)-[:RELACIONADO_CON]->(b:Diagnostico)
RETURN a.terminoEN AS source, b.terminoEN AS target
"""
relaciones = pd.DataFrame(session.run(query_relaciones).data())

F20 = "Esquizofrenia"
F20_89 = "Otros tipos de esquizofrenia"

# Etiquetas: F20 → 0, F20.89 → 1
label_encoder = LabelEncoder()
print("Etiquetas únicas:", df["icd10"].unique())
df["target"] = label_encoder.fit_transform(df["icd10"])

# Texto en minúscula, limpieza simple
df["termino"] = df["termino"].fillna("").str.lower()

# TF-IDF del texto del diagnóstico
vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1,2))
X = vectorizer.fit_transform(df["termino"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluación
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

