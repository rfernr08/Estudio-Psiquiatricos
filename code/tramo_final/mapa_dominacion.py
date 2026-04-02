import pandas as pd

# Cargar tus datos
df = pd.read_csv('C:\\Users\\Usuario\\Documents\\Workspace\\Mirage\\code\\tramo_final\\diagnosticos_por_cp.csv')

# Pivotar la tabla: CPs en filas, Diagnósticos en columnas
# Llenamos con 0 los diagnósticos que no existan en ciertos CPs
df_pivot = df.pivot(index='Código Postal', columns='Diagnóstico', values='Frecuencia').fillna(0)
# 1. Partimos de tu tabla pivotada (CPs en filas, Diagnósticos en columnas)
# df_pivot ya debería tener las frecuencias por cada diagnóstico
df_dominancia = df_pivot.copy()

# 2. Encontramos el nombre de la columna (Diagnóstico) con el valor máximo para cada fila
df_dominancia['Diagnostico_Predominante'] = df_pivot.idxmax(axis=1)

# 3. Calculamos también el % de dominancia (qué peso tiene el principal sobre el total del CP)
df_dominancia['Total_CP'] = df_pivot.sum(axis=1)
df_dominancia['Porcentaje_Dominante'] = (df_pivot.max(axis=1) / df_dominancia['Total_CP']) * 100

# 4. Resultado para Folium
resultado_dominancia = df_dominancia.reset_index()[['Código Postal', 'Diagnostico_Predominante', 'Porcentaje_Dominante']]
resultado_dominancia.to_csv('resultado_dominancia_cp.csv', index=False)
print("Ejemplo de Dominancia:")
print(resultado_dominancia.head())