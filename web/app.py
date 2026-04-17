import streamlit as st
import polars as pl
import logica  # Importamos tu motor

# 1. Configuración de página y caché de datos
st.set_page_config(page_title="Conversor ICD", layout="wide")

# Caché para que el CSV de 9MB se cargue solo una vez en la RAM
@st.cache_data
def cargar_datos():
    return logica.cargar_diccionario(r"TFG\dataset\Conversor_Definitivo.csv") # Cambia esto por tu ruta real

df_maestro = cargar_datos()

st.title("⚕️ Conversor y Buscador de Diagnósticos (ICD-9 / ICD-10)")

# 2. Creación de las Pestañas
tab_buscador, tab_lotes = st.tabs(["🔍 Buscador Individual", "📁 Conversor por Lotes"])

# --- PESTAÑA 1: BUSCADOR INDIVIDUAL ---
# --- PESTAÑA 1: BUSCADOR INDIVIDUAL ---
with tab_buscador:
    st.header("🔍 Búsqueda y Traducción de Diagnósticos")
    
    # Añadimos "Descripción" a las opciones
    tipo_busqueda = st.radio("Buscar a partir de:", ["ICD10", "ICD9", "Descripción"], horizontal=True)
    
    # 1. El usuario escribe una parte del código o texto
    termino_input = st.text_input(f"Escribe el {tipo_busqueda} (o una parte) para buscar:")
    
    if termino_input:
        # Buscamos coincidencias parciales con Polars (¡es rapidísimo!)
        sugerencias_df = logica.buscar_codigo(df_maestro, termino_input, tipo_busqueda, exacto=False)
        
        if len(sugerencias_df) == 0:
            st.warning("No se encontraron coincidencias. Prueba con otra palabra o código.")
        else:
            # Extraemos la lista de opciones para el selector
            columna_real = "Description" if tipo_busqueda == "Descripción" else tipo_busqueda
            opciones = sugerencias_df.get_column(columna_real).to_list()
            
            # Si hay demasiadas sugerencias, avisamos para que refine la búsqueda
            if len(opciones) > 100:
                st.info(f"Se encontraron {len(opciones)} coincidencias. Mostrando las 100 primeras. Sé más específico si no encuentras la tuya.")
                opciones = opciones[:100]
            
            # 2. El usuario selecciona la coincidencia exacta de la lista de sugerencias
            seleccion_exacta = st.selectbox("🎯 Sugerencias encontradas (Selecciona una):", opciones)
            
            if seleccion_exacta:
                st.divider()
                
                # Buscamos TODAS las filas que coinciden con la selección exacta
                resultado_final = logica.buscar_codigo(df_maestro, seleccion_exacta, tipo_busqueda, exacto=True)
                
                # Extraemos listas de valores ÚNICOS para cada columna
                icd10_vals = resultado_final.get_column("ICD10").unique().to_list()
                icd9_vals = resultado_final.get_column("ICD9").unique().to_list()
                desc_vals = resultado_final.get_column("Description").unique().to_list()
                
                # Filtramos los "NN" de la lista de ICD-9
                icd9_vals_limpios = [val for val in icd9_vals if val != "NN"]
                
                # Formateamos las listas como texto con viñetas (Markdown) para la UI
                str_icd10 = "\n".join([f"• {val}" for val in icd10_vals])
                str_desc = "\n".join([f"• {val}" for val in desc_vals])
                
                if icd9_vals_limpios:
                    str_icd9 = "\n".join([f"• {val}" for val in icd9_vals_limpios])
                else:
                    str_icd9 = "❌ No disponible en ICD-9"
                
                # 3. Presentación visual (UI) en formato "Ficha" para múltiples resultados
                st.subheader("Datos del Diagnóstico")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.info(f"**ICD-9 Asociado(s):**\n\n{str_icd9}")
                with col2:
                    st.success(f"**ICD-10 Asociado(s):**\n\n{str_icd10}")
                with col3:
                    st.warning(f"**Descripción (Español):**\n\n{str_desc}")
                
                # 4. Botón de la API en inglés
                st.write("") 
                if st.button("🌐 Obtener descripción en Inglés (API NLM)"):
                    with st.spinner('Consultando base de datos internacional...'):
                        # Usamos el primer ICD-10 de la lista para buscar en la API
                        texto_ingles = logica.obtener_ingles_api(icd10_vals[0])
                        if texto_ingles not in ["Error de conexión", "No encontrado en API"]:
                            st.success(f"**Inglés:** {texto_ingles}")
                        else:
                            st.error(texto_ingles)
                

# --- PESTAÑA 2: CONVERSOR POR LOTES ---
with tab_lotes:
    st.header("📁 Procesador de archivos CSV")
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
                # Añadimos la opción del Inglés
                opciones_destino = [fmt for fmt in formatos_disponibles if fmt != tipo_origen]
                opciones_destino.append("Inglés (API)")
                
                tipos_destino = st.multiselect(
                    "3. ¿Qué columnas nuevas quieres añadir?", 
                    options=opciones_destino, 
                    default=opciones_destino
                )
                
                if "Inglés (API)" in tipos_destino:
                    st.caption("⚠️ *Nota: Solicitar el inglés requiere conexión externa y aumentará el tiempo de procesamiento.*")
            
            # Botón de ejecución
            if st.button("🚀 Procesar Archivo", type="primary"):
                if not tipos_destino:
                    st.error("Debes seleccionar al menos un formato de destino.")
                else:
                    with st.spinner("Preparando archivo..."):
                        # Creamos la barra de progreso y el callback
                        barra_progreso = st.progress(0.0)
                        texto_progreso = st.empty()
                        
                        def actualizar_progreso(porcentaje):
                            barra_progreso.progress(porcentaje)
                            texto_progreso.text(f"Consultando traducciones en API... {int(porcentaje * 100)}%")
                            
                        # Si no hay inglés, pasamos None al callback
                        callback = actualizar_progreso if "Inglés (API)" in tipos_destino else None
                        
                        # Ejecutamos la lógica principal
                        df_final = logica.convertir_lotes(df_maestro, df_usuario, columna_objetivo, tipo_origen, tipos_destino, callback)
                        
                        # Limpiamos los elementos de progreso
                        barra_progreso.empty()
                        texto_progreso.empty()
                        
                        st.success("¡Conversión completada con éxito!")
                        st.write("Vista previa del resultado:")
                        st.dataframe(df_final.head(10).to_pandas(), use_container_width=True)
                        
                        # Descarga
                        st.download_button(
                            label="📥 Descargar Archivo Convertido",
                            data=df_final.write_csv(),
                            file_name="diagnosticos_estandarizados.csv",
                            mime="text/csv"
                        )