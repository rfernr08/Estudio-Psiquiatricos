import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cargar tus datos
df = pd.read_csv('diagnosticos_por_cp.csv')

# Pivotar la tabla: CPs en filas, Diagnósticos en columnas
# Llenamos con 0 los diagnósticos que no existan en ciertos CPs
df_pivot = df.pivot(index='Código Postal', columns='Diagnóstico', values='Frecuencia').fillna(0)

# Escalado de datos (FUNDAMENTAL para K-Means)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_pivot)

# Método del Codo (Elbow Method)
inercias = []
rango_k = range(1, 10) # Probamos de 1 a 10 clústeres

for k in rango_k:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(df_scaled)
    inercias.append(model.inertia_)

# Visualización del codo
plt.figure(figsize=(8, 5))
plt.plot(rango_k, inercias, 'bx-')
plt.xlabel('Número de clústeres (K)')
plt.ylabel('Inercia (Varianza intra-clúster)')
plt.title('Método del Codo para determinar K óptima')
plt.show()

#Aplicar K-Means con la K elegida (ejemplo: K=3)
k_elegida = 4
kmeans = KMeans(n_clusters=k_elegida, random_state=42, n_init=10)
clusters = kmeans.fit_predict(df_scaled)

# Añadir el resultado al DataFrame original pivotado
df_pivot['Cluster_ID'] = clusters

# Guardar el resultado para usar en Folium
df_pivot.reset_index()[['Código Postal', 'Cluster_ID']].to_csv('resultado_clusters_cp.csv', index=False)

print("¡Hecho! Se ha generado 'resultado_clusters_cp.csv' con la asignación de cada CP a su grupo.")

# Opcional: Ver qué caracteriza a cada clúster (las medias de cada diagnóstico)
print("\nPerfil promedio de cada clúster:")
print(df_pivot.groupby('Cluster_ID').mean())