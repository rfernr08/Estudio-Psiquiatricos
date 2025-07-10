import pandas as pd
import folium
import json
from shapely.geometry import shape

color_palette = ["BuPu", "GnBu", "OrRd", "YlGn", "PuBuGn", "YlOrBr", "RdPu", "Blues"]

df = pd.read_csv("datasets/conteo_diagnosticos_leon.csv")

geojson_path = f"codigos-postales/data/LEÓN.geojson"
with open(geojson_path, "r", encoding="utf-8") as f:
    geo_data = json.load(f)

df["DIAG PSQ SPLIT"] = df["DIAG PSQ SPLIT"].str.strip()
diagnosticos = df["DIAG PSQ SPLIT"].unique()
m = folium.Map(location=[42.6, -5.57], zoom_start=8, tiles="CartoDB positron")

for diag in diagnosticos:
    diag_clean = str(diag).strip()
    conteo_path = f"datasets/mapas_leon/conteo_{diag_clean}.csv"

    df_diag = pd.read_csv(conteo_path)
    
    folium.Choropleth(
        geo_data=geo_data,
        name=f'Distribución {diag}',
        data=df_diag,
        columns=["Código Postal", "frecuencia"],
        key_on="feature.properties.COD_POSTAL",
        fill_color=color_palette[diagnosticos.tolist().index(diag) % len(color_palette)],
        nan_fill_color="white",
        nan_fill_opacity=0.0,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"Frecuencia diagnóstico {diag}",
        highlight=True,
        show=True,
        threshold_scale=[0, 1, 2, 5, 10, 25, 50, 75, 100, 150]
    ).add_to(m)

folium.GeoJson(
    geo_data,
    name=f"Etiquetas Códigos Postales",
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
m.save("mapa_diagnosticos_entero_leon.html")