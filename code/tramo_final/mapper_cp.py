import pandas as pd
import folium
import json

# Cargamos fichero de clusters por código postal
clusters_path = "resultado_clusters_cp.csv"
df_clusters = pd.read_csv(clusters_path)

# Normalizamos el formato de Código Postal (sin decimales)
if "Código Postal" in df_clusters.columns:
    df_clusters["Código Postal"] = df_clusters["Código Postal"].astype(str).str.replace("\.0$", "", regex=True)
else:
    raise ValueError("archivo de clusters debe tener columna 'Código Postal'")

if "Cluster_ID" not in df_clusters.columns:
    raise ValueError("archivo de clusters debe tener columna 'Cluster_ID'")

df_clusters["Cluster_ID"] = pd.to_numeric(df_clusters["Cluster_ID"], errors="coerce").fillna(-1).astype(int)

geojson_path = "recursos/mapas_leon/LEÓN.geojson"
with open(geojson_path, "r", encoding="utf-8") as f:
    geo_data = json.load(f)

m = folium.Map(location=[42.6, -5.57], zoom_start=8, tiles="CartoDB positron")

folium.Choropleth(
    geo_data=geo_data,
    name='Clusters por CP',
    data=df_clusters,
    columns=["Código Postal", "Cluster_ID"],
    key_on="feature.properties.COD_POSTAL",
    fill_color='Spectral',
    nan_fill_color="white",
    nan_fill_opacity=0.2,
    fill_opacity=0.8,
    line_opacity=0.2,
    legend_name="Cluster ID",
    show=True,
    threshold_scale=None
).add_to(m)

folium.GeoJson(
    geo_data,
    name="Etiquetas Códigos Postales",
    style_function=lambda feature: {
        "fillOpacity": 0,
        "color": "transparent"
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["COD_POSTAL"],
        aliases=["Código Postal:"],
        labels=True,
        sticky=True
    )
).add_to(m)


folium.LayerControl().add_to(m)
m.save("mapa_clusters_leon.html")