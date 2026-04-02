import pandas as pd
import numpy as np

# 1. CARGA DE DATOS
# Asumo que tu archivo se llama 'dataset_psq.csv'. Cámbialo por el tuyo.
df = pd.read_csv('C:\\Users\\Usuario\\Documents\\Workspace\\Mirage\\TFG\\dataset\\PSQ_F20_F29_Ext.csv', encoding='utf-8', sep='|')

# 2. IDENTIFICAR COLUMNAS DE DIAGNÓSTICOS SECUNDARIOS
# Filtramos todas las columnas que contengan la palabra 'Secundario' o sigan el patrón 'Diag XX'
# Según tu ejemplo, van del Diag 02 al Diag 20.
cols_secundarias = [col for col in df.columns if 'Secundario' in col or 'Diag 0' in col or 'Diag 1' in col or 'Diag 20' in col]

# Quitamos el 'Diag 01 Principal' si se coló en la lista anterior
if 'Diag 01 Principal    (cod)' in cols_secundarias:
    cols_secundarias.remove('Diag 01 Principal    (cod)')

print(f"Analizando comorbilidad en {len(cols_secundarias)} columnas secundarias...")

# 3. CÁLCULO DE LA COMPLEJIDAD POR PACIENTE
# Creamos una función que cuente cuántos diagnósticos secundarios no están vacíos
def calcular_carga(fila):
    # Contamos valores que no sean NaN, ni nulos, ni espacios en blanco
    count = fila[cols_secundarias].dropna().astype(str).str.strip().replace('', np.nan).dropna().count()
    return count

# Aplicamos la función a cada fila (paciente)
df['Indice_Comorbilidad'] = df.apply(calcular_carga, axis=1)

# 4. AGREGACIÓN POR CÓDIGO POSTAL (Métrica para el Mapa)
# Calculamos la media de diagnósticos secundarios por cada CP
df_mapa_comorbilidad = df.groupby('Código Postal').agg({
    'Indice_Comorbilidad': 'mean',  # Media de complejidad
    'Nº Historia': 'count'         # Número de pacientes (para saber si la muestra es representativa)
}).reset_index()

# Renombramos para que sea más claro
df_mapa_comorbilidad.columns = ['CP', 'Media_Comorbilidad', 'Num_Pacientes']

# 5. ORDENAR Y MOSTRAR RESULTADOS
# Ordenamos de mayor a menor complejidad
df_mapa_comorbilidad = df_mapa_comorbilidad.sort_values(by='Media_Comorbilidad', ascending=False)

print("\nTop 5 Códigos Postales con mayor complejidad media (Comorbilidad):")
print(df_mapa_comorbilidad.head())

# 6. EXPORTAR PARA FOLIUM
# Este CSV es el que unirás a tu GeoJSON para pintar el mapa de coropletas
df_mapa_comorbilidad.to_csv('resultado_comorbilidad_mapa.csv', index=False)