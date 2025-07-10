import polars as pl
import pandas as pd
import folium
import json
import os

df = pl.read_excel("datasets/PSQ_F20_F29.xlsx", sheet_name="Sheet1")
df = df.with_columns(pl.col("Provincia").str.to_uppercase())

df = df.with_columns(
    pl.col("DIAG PSQ")
    .str.split(", ")
    .alias("DIAG PSQ SPLIT")
).explode("DIAG PSQ SPLIT")


conteo = df.group_by("Provincia", "DIAG PSQ SPLIT").len().rename({"len": "frecuencia"})
conteo = conteo.filter(pl.col("Provincia") == "LEÓN")
df_conteo = conteo.to_pandas()
#df_conteo["Código Postal"] = df_conteo["Código Postal"].astype(str).str.zfill(5)
df_conteo["Provincia"] = df_conteo["Provincia"].str.upper()


print(df_conteo)
df_conteo.to_csv("datasets/conteo_Leon.csv", index=False)

m = folium.Map(location=[41.65, -4.72], zoom_start=7, tiles="CartoDB positron")


"""
for provincia in provincias:
    geojson_path = f"codigos-postales/data/{provincia}.geojson"

    if not os.path.exists(geojson_path):
        print(f"GeoJSON no encontrado para {provincia}, saltando...")
        continue

    with open(geojson_path, "r", encoding="utf-8") as f:
        geo_data = json.load(f)

    df_provincia = df_conteo[df_conteo["Provincia"] == provincia]
    df_provincia.to_csv(f"datasets/conteos/conteo_municipios_{provincia}.csv", index=False)

    folium.Choropleth(
        geo_data=geo_data,
        name=f'Densidad Diagnósticos {provincia}',
        data=df_provincia,
        columns=["Código Postal", "frecuencia"],
        key_on="feature.properties.COD_POSTAL",
        fill_color="PuRd",
        nan_fill_color="white",
        nan_fill_opacity=0.0,
        fill_opacity=0.7,
        line_opacity=0.2,
        #legend_name=f"Frecuencia de diagnósticos {provincia}",
        highlight=True,
        show=True,
        threshold_scale=[0, 2, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300],
    ).add_to(m)

# Agregar tooltip con municipio
folium.LayerControl().add_to(m)

# Guardar mapa
m.save("mapa_diagnosticos_leon.html")

"""


"""
Mision: 
Conectar los codigos postales que tenemos en el dataset, ya que estan todos, con los varios geojson que tenemos de provincias 
Los geojson estan en codigos-postales/data.
Sacar las geosjon correspondientes segun la provincia o municipio que ponga en el dataset.
Juntando estos campos, intentar pintarun mapa de españa con la fecuencia de diagnósticos por codigo postal.
"""
