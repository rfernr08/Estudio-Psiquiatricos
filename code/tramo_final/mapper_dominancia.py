import pandas as pd
import folium
import json

# Cargamos fichero por código postal (nueva estructura)
input_path = "C:\\Users\\Usuario\\Documents\\Workspace\\Mirage\\code\\tramo_final\\resultado_dominancia_cp.csv"
df = pd.read_csv(input_path)

# Normalizamos el formato de Código Postal (sin decimales)
if "Código Postal" in df.columns:
    df["Código Postal"] = df["Código Postal"].astype(str).str.replace("\.0$", "", regex=True)
else:
    raise ValueError("El CSV debe tener la columna 'Código Postal'")

# Columnas esperadas en la nueva fuente
required_columns = ["Diagnostico_Predominante", "Porcentaje_Dominante"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"El CSV debe tener la columna '{col}'")

from branca.colormap import linear

# Convertimos la categoría a numérica para la capa Choropleth;
# mantener también texto para popup
df["Diagnostico_Cat"] = df["Diagnostico_Predominante"].astype(str)
cat_to_num = {v: i + 1 for i, v in enumerate(sorted(df["Diagnostico_Cat"].unique()))}
df["Diagnostico_Code"] = df["Diagnostico_Cat"].map(cat_to_num).fillna(0).astype(int)

# Colores categóricos (Set2 de branca) y mapeo diagnóstico=>color
max_cat = max(cat_to_num.values()) if cat_to_num else 1
palette = linear.Set1_09.scale(1, max_cat)
cat_to_color = {cat: palette(idx) for cat, idx in cat_to_num.items()}

# Cargamos geojson de León
geojson_path = "recursos/mapas_leon/LEÓN.geojson"
with open(geojson_path, "r", encoding="utf-8") as f:
    geo_data = json.load(f)

m = folium.Map(location=[42.6, -5.57], zoom_start=8, tiles="CartoDB positron")

# Mapa de diagnóstico por CP con colores sincronizados con la leyenda
cp_to_diag = df.set_index(df["Código Postal"].astype(str))["Diagnostico_Cat"].to_dict()
cp_to_pct = df.set_index(df["Código Postal"].astype(str))["Porcentaje_Dominante"].to_dict()

# Enriquecer propiedades geojson para tooltip directo
for feature in geo_data.get("features", []):
    cp = str(feature["properties"].get("COD_POSTAL", "")).strip()
    feature["properties"]["Diagnostico_Predominante"] = cp_to_diag.get(cp, "Sin dato")
    feature["properties"]["Porcentaje_Dominante"] = cp_to_pct.get(cp, "N/A")


def style_function(feature):
    diag = feature["properties"].get("Diagnostico_Predominante", "Sin dato")
    color = cat_to_color.get(diag, "#dddddd")
    return {
        "fillColor": color,
        "color": "#444444",
        "weight": 0.5,
        "fillOpacity": 0.75,
        "opacity": 0.7,
    }

# GeoJson con tooltip y popup a partir de propiedades ya especificadas
g = folium.GeoJson(
    geo_data,
    name="Diagnóstico predominante por CP",
    style_function=style_function,
    highlight_function=lambda feat: {"weight": 2, "color": "#000000"},
    tooltip=folium.GeoJsonTooltip(
        fields=["COD_POSTAL", "Diagnostico_Predominante", "Porcentaje_Dominante"],
        aliases=["Código Postal:", "Diagnóstico:", "Porcentaje:"],
        localize=True,
        sticky=True,
        labels=True
    ),
    popup=folium.GeoJsonPopup(
        fields=["COD_POSTAL", "Diagnostico_Predominante", "Porcentaje_Dominante"],
        aliases=["Código Postal:", "Diagnóstico:", "Porcentaje:"],
        localize=True
    )
)

g.add_to(m)

# Añadir leyenda con diagnóstico y color
legend_html = '''
 <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; max-height: 300px; overflow: auto; 
             border:2px solid grey; z-index:9999; font-size:14px; background-color: white; opacity: 0.9; padding: 10px;">
 &nbsp;<b>Diagnóstico predominante</b><br>
 '''
for diag, color in sorted(cat_to_color.items(), key=lambda x: cat_to_num[x[0]]):
    legend_html += f"<div style='display:flex; align-items:center; margin-bottom:4px;'><span style='background:{color}; width:16px; height:16px; display:inline-block; border:1px solid #666; margin-right:8px;'></span>{diag}</div>"
legend_html += "</div>"

m.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl().add_to(m)
m.save("mapa_dominancia_cp.html")