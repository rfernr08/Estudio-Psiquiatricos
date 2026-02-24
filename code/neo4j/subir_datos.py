import dotenv
import os
from neo4j import GraphDatabase
import pandas as pd
import ast

load_status = dotenv.load_dotenv()

diagnosticos = "datasets/Full_Datos_Diag_Men.csv"
pacientes = "datasets/neo4j/info_pacientes.csv"
relacion_diag = "datasets/neo4j/diag_por_paciente.csv"
relacion_psi = "datasets/neo4j/diag_psiquiatricos_por_paciente.csv"


URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

# === Crea el driver y la sesión ===
driver = GraphDatabase.driver(URI, auth=AUTH)

diagnosticos_df = pd.read_csv(diagnosticos, sep="|")
# CAMBIO CLAVE: Usar coma como separador para el CSV de pacientes
pacientes_df = pd.read_csv(pacientes, sep="|")

# Debug: Verificar la estructura del DataFrame
print("=== DEBUG INFO ===")
print(f"Columnas en pacientes_df: {pacientes_df.columns.tolist()}")
print(f"Número de columnas: {len(pacientes_df.columns)}")
print(f"Primera fila: {pacientes_df.iloc[0].to_dict()}")
print("==================")

# Limpieza de embeddings (de texto a lista de floats)
diagnosticos_df['embedding'] = diagnosticos_df['embedding'].apply(ast.literal_eval)

# Cargar relaciones - también usar coma como separador
relaciones_diag_df = pd.read_csv(relacion_diag, sep="|")
relaciones_psi_df = pd.read_csv(relacion_psi, sep="|")

def crear_diagnosticos(tx, row):
    query = """
    MERGE (d:Diagnostico {terminoEN: $terminoEN})
    SET d.ICD10 = $ICD10,
        d.ICD9 = $ICD9,
        d.terminoIN = $terminoIN,
        d.embedding = $embedding
    """
    tx.run(query, **row)

def crear_paciente(tx, row):
    try:
        # Normalizar nombres de parámetros
        clean_row = {
            'numero_historia': row['Nº Historia'],
            'fecha_nacimiento': row['Fecha Nacimiento'],
            'edad': row['Edad en años'],
            'sexo': row['Sexo  (Desc)'],
            'pais_nacimiento': row['País de Nacimiento'],
            'provincia': row['Provincia'],
            'municipio_cod': row['Municipio Residencia   (Cód)'],
            'municipio_desc': row['Municipio Residencia   (Des)'],
            'codigo_postal': row['Código Postal'],
            'fecha_ingreso': row['Fecha Ingreso'],
            'año': row['Año'],
            'fecha': row['Fecha'],
            'servicio_alta': row['Servicio Alta (Código)'],
            'seccion_alta': row['Sección Alta (Código)']
        }
        
        query = """
        MERGE (p:Paciente {numero_historia: $numero_historia})
        SET p.fecha_nacimiento = $fecha_nacimiento,
            p.edad = $edad,
            p.sexo = $sexo,
            p.pais_nacimiento = $pais_nacimiento,
            p.provincia = $provincia,
            p.municipio_cod = $municipio_cod,
            p.municipio_desc = $municipio_desc,
            p.codigo_postal = $codigo_postal,
            p.fecha_ingreso = $fecha_ingreso,
            p.año = $año,
            p.fecha = $fecha,
            p.servicio_alta = $servicio_alta,
            p.seccion_alta = $seccion_alta
        """
        
        tx.run(query, **clean_row)
        print(f"✅ Paciente creado: {clean_row['numero_historia']}")
        
    except Exception as e:
        print(f"Error creando paciente: {e}")
        print(f"Columnas disponibles: {list(row.keys())}")
        raise

def crear_relacion_diagnosticos(tx, n_paciente, diag):
    query = """
    MATCH (p:Paciente {numero_historia: $n_paciente})
    MATCH (d:Diagnostico {terminoEN: $nombre})
    MERGE (p)-[:DIAGNOSTICO_ASOCIADO]->(d)
    """
    tx.run(query, nombre=diag, n_paciente=n_paciente)

def crear_relacion_psiquiatrico(tx, n_paciente, diag_psi):
    query = """
    MATCH (p:Paciente {numero_historia: $n_paciente})
    MATCH (d:Diagnostico {terminoEN: $nombre_psi})
    MERGE (p)-[:DIAGNOSTICO_PSIQUIATRICO]->(d)
    """
    tx.run(query, nombre_psi=diag_psi, n_paciente=n_paciente)

if __name__ == "__main__":
    try:
       with driver.session() as session:
            print("Subiendo diagnósticos...")
            for _, row in diagnosticos_df.iterrows():
                session.execute_write(crear_diagnosticos, row.to_dict())
            
            print("Subiendo pacientes...")
            for _, row in pacientes_df.iterrows():
                session.execute_write(crear_paciente, row.to_dict())

            print("Creando relaciones de diagnósticos...")
            for _, row in relaciones_diag_df.iterrows():
                session.execute_write(crear_relacion_diagnosticos, row['Numero_Historia'], row['Diagnostico'])
            
            print("Creando relaciones psiquiátricas...")
            for _, row in relaciones_psi_df.iterrows():
                session.execute_write(crear_relacion_psiquiatrico, row['Numero_Historia'], row['Diagnostico'])
                
    except Exception as e:
        print(f"Error general: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()
        print("Connection closed.")
        print("Datos subidos correctamente a Neo4j.")