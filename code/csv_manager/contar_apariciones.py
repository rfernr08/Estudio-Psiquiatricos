import pandas as pd

# Leer el dataset
df = pd.read_csv('C:\\Users\\Usuario\\Documents\\Workspace\\Mirage\\TFG\\dataset\\Diagnosticos_codigos_sin_duplicar_simplificado.csv', sep='|')

# Contar la frecuencia de cada diagnóstico final (ICD10)
frecuencias = df['DIAG PSQ'].value_counts().reset_index()
frecuencias.columns = ['Diagnostico', 'Frecuencia']

# Guardar el resultado en un nuevo CSV
frecuencias.to_csv('diagnosticos_frecuencia.csv', index=False)

print("Archivo 'diagnosticos_frecuencia.csv' generado con éxito.")