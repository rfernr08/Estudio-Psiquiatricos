import pandas as pd

# Rutas de los archivos CSV de entrada
csv1 = 'diagnosticos_datos_mitad.csv'
csv2 = 'diagnosticos_datos_otra_mitad.csv'

# Leer los CSVs
df1 = pd.read_csv(csv1, sep='|')
df2 = pd.read_csv(csv2, sep='|')

# Concatenar los DataFrames
df_merged = pd.concat([df1, df2], ignore_index=True)

# Guardar el resultado en un nuevo CSV
df_merged.to_csv('csv_merged.csv', index=False, sep='|')

print("CSV fusionados exitosamente en 'csv_merged.csv'")