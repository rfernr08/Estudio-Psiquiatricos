import pandas as pd
import folium
import json
import os

# Carga el conteo por código postal y diagnóstico
dataset = pd.read_excel("datasets/PSQ_F20_F29.xlsx", sheet_name="Sheet1")
df = pd.read_csv("datasets/conteo_Leon.csv")


dfT = dataset[["Provincia","Código Postal", "DIAG PSQ"]]
print(f"Total registros: {dfT}")
dfT = dfT[dfT["Provincia"].str.upper() == "LEÓN"]
print(f"Total registros en León: {dfT}")
dfT["DIAG PSQ SPLIT"] = dfT["DIAG PSQ"].str.split(",")
dfT = dfT.explode("DIAG PSQ SPLIT")


# Carga el geojson de León
geojson_path = "codigos-postales/data/LEÓN.geojson"
with open(geojson_path, "r", encoding="utf-8") as f:
    geo_data = json.load(f)

# Lista de diagnósticos únicos
diagnosticos = df["DIAG PSQ SPLIT"].unique()

print(f"Total diagnósticos únicos: {diagnosticos}")
print(dfT)
dfT.to_csv("datasets/conteo_diagnosticos_leon.csv")
"""
for diag in diagnosticos:
    df_diag = dfT[dfT["DIAG PSQ SPLIT"] == diag]
    m = folium.Map(location=[42.6, -5.57], zoom_start=8, tiles="CartoDB positron")

    folium.Choropleth(
        geo_data=geo_data,
        name=f'Distribución {diag}',
        data=df_diag,
        columns=["Código Postal", "frecuencia"],
        key_on="feature.properties.COD_POSTAL",  # Ajusta si tu propiedad es diferente
        fill_color="PuRd",
        nan_fill_color="white",
        nan_fill_opacity=0.0,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"Frecuencia diagnóstico {diag}",
        highlight=True,
        show=True,
        threshold_scale=[0, 50, 100, 200, 500, 1000, 2000],
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(f"dataset/mapas_leon/mapa_leon_{diag}.html")
    print(f"Mapa guardado: mapas_leon/mapa_leon_{diag}.html")

folium.LayerControl().add_to(m)
m.save("mapa_diagnosticos_entero_leon.html")
"""
