import streamlit as st
import polars as pl
import logica 
import os 
import streamlit.components.v1 as components 

st.set_page_config(page_title="Conversor ICD", layout="wide")

@st.cache_data
def cargar_datos():
    return logica.cargar_diccionario(r"TFG\dataset\Conversor_Definitivo.csv")

df_maestro = cargar_datos()

st.title("Conversor y Buscador de Diagnósticos (ICD-9 / ICD-10)")

tab_buscador, tab_lotes, tab_mapas = st.tabs(["Buscador Individual", "Conversor por Lotes", "Mapas de Investigacion"])

with tab_buscador:
    st.header("Búsqueda y Traducción de Diagnósticos")
    
    tipo_busqueda = st.radio("Buscar a partir de:", ["ICD10", "ICD9", "Descripción"], horizontal=True)
    
    termino_input = st.text_input(f"Escribe el {tipo_busqueda} (o una parte) para buscar:")
    
    if termino_input:

        sugerencias_df = logica.buscar_codigo(df_maestro, termino_input, tipo_busqueda, exacto=False)
        
        if len(sugerencias_df) == 0:
            st.warning("No se encontraron coincidencias. Prueba con otra palabra o código.")
        else:
            columna_real = "Description" if tipo_busqueda == "Descripción" else tipo_busqueda
            opciones = sugerencias_df.get_column(columna_real).to_list()
            if len(opciones) > 100:
                st.info(f"Se encontraron {len(opciones)} coincidencias. Mostrando las 100 primeras. Sé más específico si no encuentras la tuya.")
                opciones = opciones[:100]
            
            seleccion_exacta = st.selectbox("Sugerencias encontradas (Selecciona una):", opciones)
            
            if seleccion_exacta:
                st.divider()
                
                resultado_final = logica.buscar_codigo(df_maestro, seleccion_exacta, tipo_busqueda, exacto=True)
                
                icd10_vals = resultado_final.get_column("ICD10").unique().to_list()
                icd9_vals = resultado_final.get_column("ICD9").unique().to_list()
                desc_vals = resultado_final.get_column("Description").unique().to_list()
                
                icd9_vals_limpios = [val for val in icd9_vals if val != "NN"]
                
                str_icd10 = "\n".join([f"• {val}" for val in icd10_vals])
                str_desc = "\n".join([f"• {val}" for val in desc_vals])
                
                if icd9_vals_limpios:
                    str_icd9 = "\n".join([f"• {val}" for val in icd9_vals_limpios])
                else:
                    str_icd9 = "No disponible en ICD-9"
                
                st.subheader("Datos del Diagnóstico")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.info(f"**ICD-9 Asociado(s):**\n\n{str_icd9}")
                with col2:
                    st.success(f"**ICD-10 Asociado(s):**\n\n{str_icd10}")
                with col3:
                    st.warning(f"**Descripción (Español):**\n\n{str_desc}")
                
                st.write("") 
                if st.button("Obtener descripción en Inglés (API NLM)"):
                    with st.spinner('Consultando base de datos internacional...'):
                        texto_ingles = logica.obtener_ingles_api(icd10_vals[0])
                        if texto_ingles not in ["Error de conexión", "No encontrado en API"]:
                            st.success(f"**Inglés:** {texto_ingles}")
                        else:
                            st.error(texto_ingles)
                

with tab_lotes:
    st.header("Procesador de archivos CSV")
    st.markdown("Sube un archivo con tus códigos o descripciones y añade las columnas equivalentes automáticamente.")
    
    archivo_subido = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    
    if archivo_subido is not None:
            df_usuario = pl.read_csv(archivo_subido)
            st.write("Vista previa de tu archivo (Primeras 5 filas):")
            st.dataframe(df_usuario.head().to_pandas(), use_container_width=True)
            
            st.divider()
            st.subheader("Configuración de la conversión")
            
            col1, col2 = st.columns(2)
            
            formatos_disponibles = ["ICD9", "ICD10", "Description"]
            
            with col1:
                columna_objetivo = st.selectbox("1. ¿Qué columna contiene los datos a convertir?", df_usuario.columns)
                tipo_origen = st.selectbox("2. ¿Qué formato tienen los datos de entrada?", formatos_disponibles)
                
            with col2:
                opciones_destino = [fmt for fmt in formatos_disponibles if fmt != tipo_origen]
                opciones_destino.append("Inglés (API)")
                
                tipos_destino = st.multiselect(
                    "3. ¿Qué columnas nuevas quieres añadir?", 
                    options=opciones_destino, 
                    default=opciones_destino
                )
                
                if "Inglés (API)" in tipos_destino:
                    st.caption("*Nota: Solicitar el inglés requiere conexión externa y aumentará el tiempo de procesamiento.*")
            
            if st.button("Procesar Archivo", type="primary"):
                if not tipos_destino:
                    st.error("Debes seleccionar al menos un formato de destino.")
                else:
                    with st.spinner("Preparando archivo..."):
                        barra_progreso = st.progress(0.0)
                        texto_progreso = st.empty()
                        
                        def actualizar_progreso(porcentaje):
                            barra_progreso.progress(porcentaje)
                            texto_progreso.text(f"Consultando traducciones en API... {int(porcentaje * 100)}%")
                            
                        callback = actualizar_progreso if "Inglés (API)" in tipos_destino else None
                        
                        df_final = logica.convertir_lotes(df_maestro, df_usuario, columna_objetivo, tipo_origen, tipos_destino, callback)
                        
                        barra_progreso.empty()
                        texto_progreso.empty()
                        
                        st.success("¡Conversión completada con éxito!")
                        st.write("Vista previa del resultado:")
                        st.dataframe(df_final.head(10).to_pandas(), use_container_width=True)
                        
                        st.download_button(
                            label="Descargar Archivo Convertido",
                            data=df_final.write_csv(),
                            file_name="diagnosticos_estandarizados.csv",
                            mime="text/csv"
                        )

with tab_mapas:
    st.header("Mapas de Investigacion")
    st.markdown("Selecciona uno de los mapas interactivos generados durante el estudio para explorarlo.")
    
    if 'mapa_seleccionado' not in st.session_state:
        st.session_state.mapa_seleccionado = None

    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    carpeta_mapas = os.path.join(directorio_actual, "mapas")

    if not os.path.exists(carpeta_mapas):
        os.makedirs(carpeta_mapas)
        st.warning(f"He creado la carpeta '{carpeta_mapas}/'. Por favor, mete ahí tus archivos .html.")
    else:
        archivos_html = [f for f in os.listdir(carpeta_mapas) if f.endswith('.html')]
        
        if not archivos_html:
            st.info(f"La carpeta '{carpeta_mapas}' está vacía. Añade archivos .html para verlos aquí.")
        else:
            columnas_por_fila = 3
            
            for i in range(0, len(archivos_html), columnas_por_fila):
                cols = st.columns(columnas_por_fila) # Creamos 3 columnas vacías
                
                for j, col in enumerate(cols):
                    indice_actual = i + j
                    if indice_actual < len(archivos_html):
                        archivo_actual = archivos_html[indice_actual]
                        
                        nombre_bonito = archivo_actual.replace(".html", "").replace("_", " ").title()
                        
                        with col:
                            if st.button(f"{nombre_bonito}", key=archivo_actual, use_container_width=True):
                                st.session_state.mapa_seleccionado = os.path.join(carpeta_mapas, archivo_actual)

    st.divider()
    if st.session_state.mapa_seleccionado:
        ruta_mapa = st.session_state.mapa_seleccionado
        if os.path.exists(ruta_mapa):
            with st.spinner("Cargando mapa interactivo..."):
                with open(ruta_mapa, 'r', encoding='utf-8') as archivo:
                    codigo_html = archivo.read()
                components.html(codigo_html, height=750, scrolling=True)
        else:
            st.error("El archivo seleccionado ya no existe.")