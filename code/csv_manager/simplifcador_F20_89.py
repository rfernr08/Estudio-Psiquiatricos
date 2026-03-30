import pandas as pd

# Ruta del archivo
file = r'C:\Users\Usuario\Documents\Workspace\Mirage\TFG\dataset\Diagnosticos_codigos_sin_duplicar'
file_path = file + '.csv'  # Agregar la extensión .csv a la ruta

# Leer el CSV con separador '|'
df = pd.read_csv(file_path, sep='|')

# Reemplazar 'F20.89' por 'F20' en la columna 'DIAG PSQ'
df['DIAG PSQ'] = df['DIAG PSQ'].replace('F20.89', 'F20')

# Guardar el archivo modificado
df.to_csv(file + "_simplificado.csv", sep='|', index=False)

print("Cambios realizados y archivo guardado.")