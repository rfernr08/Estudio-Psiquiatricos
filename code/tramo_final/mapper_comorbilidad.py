import pandas as pd
import folium
import json

# Cargamos fichero por código postal (media de comorbilidad)
input_path = "C:\\Users\\Usuario\\Documents\\Workspace\\Mirage\\resultado_comorbilidad_mapa.csv"
df = pd.read_csv(input_path)

# Normalizamos el formato de Código Postal (sin decimales)
if "CP" in df.columns:
    df["CP"] = df["CP"].astype(str).str.replace("\.0$", "", regex=True)
else:
    raise ValueError("El CSV debe tener la columna 'CP'")

required_columns = ["Media_Comorbilidad", "Num_Pacientes"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"El CSV debe tener la columna '{col}'")

# Cargar GeoJSON de las provincias
geojson_path = "recursos/mapas_leon/LEÓN.geojson"
with open(geojson_path, "r", encoding="utf-8") as f:
    geo_data = json.load(f)

# Crear mapa base
m = folium.Map(location=[42.6, -5.57], zoom_start=8, tiles="CartoDB positron")

# Choropleth por Media_Comorbilidad
folium.Choropleth(
    geo_data=geo_data,
    name='Media de Comorbilidad por CP',
    data=df,
    columns=["CP", "Media_Comorbilidad"],
    key_on="feature.properties.COD_POSTAL",
    fill_color='YlGnBu',
    nan_fill_color="white",
    nan_fill_opacity=0.2,
    fill_opacity=0.8,
    line_opacity=0.2,
    legend_name="Media de comorbilidad",
    threshold_scale=[0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18],
    reset=True
).add_to(m)

# Enriquecer polígono con tooltip (CP, media, nº pacientes)
cp_info = df.set_index("CP")["Media_Comorbilidad"].to_dict()
cp_num = df.set_index("CP")["Num_Pacientes"].to_dict()

for feature in geo_data.get("features", []):
    cp = str(feature["properties"].get("COD_POSTAL", "")).strip()
    feature["properties"]["CP"] = cp
    feature["properties"]["Media_Comorbilidad"] = cp_info.get(cp, "N/A")
    feature["properties"]["Num_Pacientes"] = cp_num.get(cp, "N/A")

folium.GeoJson(
    geo_data,
    name="Info CP",
    style_function=lambda x: {"fillOpacity": 0, "color": "transparent"},
    tooltip=folium.GeoJsonTooltip(
        fields=["CP", "Media_Comorbilidad", "Num_Pacientes"],
        aliases=["CP:", "Media comorbilidad:", "Num pacientes:"],
        localize=True,
        sticky=True
    )
).add_to(m)

folium.LayerControl().add_to(m)

# Guardar mapa
m.save("mapa_comorbilidad_cp.html")
