import pandas as pd

file_path = 'datasets/importante/relaciones_diagnosticos_psiquiatricos.csv'  # Cambia esto por el path de tu CSV

# Lee el CSV
df = pd.read_csv(file_path, sep='|')

# Elimina filas duplicadas
df = df.drop_duplicates()

df = df[df['nombre_diagnostico'] != df['nombre_diagnostico_psiquiatrico']]

# Guarda el CSV sin duplicados, sobrescribiendo el archivo original
df.to_csv(file_path, index=False, sep='|')