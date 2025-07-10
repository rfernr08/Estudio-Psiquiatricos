import pandas as pd

csv_input_file = 'icd9toicd10cmgem.csv'
csv_output_file = 'backup_conversor.csv'

# Lee el archivo CSV
df = pd.read_csv(csv_input_file, usecols=[0, 1], dtype=str)

def insertar_punto(codigo):
    if len(codigo) > 3:
        return codigo[:3] + '.' + codigo[3:]
    elif len(codigo) == 3:
        return codigo + '.'
    else:
        return codigo

# Aplica la función a ambas columnas
df.iloc[:, 0] = df.iloc[:, 0].apply(insertar_punto)
df.iloc[:, 1] = df.iloc[:, 1].apply(insertar_punto)

# Guarda las dos primeras columnas en un CSV separado por '|'
df.to_csv(csv_output_file, sep='|', index=False, header=False)