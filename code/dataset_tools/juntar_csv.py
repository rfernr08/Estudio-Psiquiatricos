import pandas as pd

# Rutas de tus archivos
csv_grande = "ICD_10_Español.csv"   # El grande, sin columna icd-9
csv_pequeno = "datasets/ICD10_Completo.csv" # El pequeño, con columna icd-9
csv_salida = "csv_unido.csv"

# Lee los archivos
df1 = pd.read_csv(csv_grande, sep="|", on_bad_lines='warn')
df2 = pd.read_csv(csv_pequeno, sep="|")

# Asegúrate de que los nombres de columna coinciden para el merge
col_icd10 = 'ICD10'
col_icd9 = 'ICD9'

# Haz el merge por el código ICD-10
df_merged = df1.merge(df2[[col_icd10, col_icd9]], on=col_icd10, how='left')

# Rellena los valores NaN de icd-9 con 'NN'
df_merged[col_icd9] = df_merged[col_icd9].fillna('NN')

# Reordena las columnas para que ICD9 sea la primera
cols = [col_icd9] + [col for col in df_merged.columns if col != col_icd9]
df_merged = df_merged[cols]

# Guarda el resultado con delimitador '|'
df_merged.to_csv(csv_salida, index=False, sep="|")