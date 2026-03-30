import pandas as pd

def traducir_codigos_a_descripcion(dataset_limpio, archivo_equivalencias, output_file):
    # 1. Cargar el dataset limpio y el diccionario de equivalencias
    # Usamos sep='|' para respetar tu formato
    df = pd.read_csv(dataset_limpio, sep='|')
    df_dict = pd.read_csv(archivo_equivalencias, sep='|')
    
    # 2. Crear un diccionario de búsqueda rápido (Mapping)
    # Usamos ICD10 como llave y Description como valor
    mapping = pd.Series(df_dict.Description.values, index=df_dict.ICD10).to_dict()
    
    # Identificar columnas de diagnósticos (excluyendo el Target si prefieres dejarlo como código)
    cols_diagnosticos = [c for c in df.columns if 'Diag' in c]
    col_target = 'DIAG PSQ'

    # 3. Función para reemplazar código por descripción
    def reemplazar_por_texto(valor):
        if pd.isna(valor) or str(valor).strip() == "" or str(valor).lower() == 'none':
            return "" # Dejar vacío si no hay diagnóstico
        
        # Buscamos el código en nuestro diccionario
        # Si no existe, devolvemos el código original para no perder información
        return mapping.get(str(valor).strip(), valor)

    # Aplicar la traducción a las columnas de entrada
    for col in cols_diagnosticos:
        df[col] = df[col].apply(reemplazar_por_texto)
    
    # Opcional: Traducir también el diagnóstico final (Target)
    # df[col_target] = df[col_target].apply(reemplazar_por_texto)

    # 4. Guardar el nuevo CSV con las descripciones
    df.to_csv(output_file, sep='|', index=False)
    print(f"Traducción completada. Archivo guardado en: {output_file}")


traducir_codigos_a_descripcion('C:\\Users\\Usuario\\Documents\\Workspace\\Mirage\\TFG\\dataset\\dataset_bert_undersampled.csv', 'TFG/dataset/Conversor_Definitivo.csv', 'dataset_descripciones.csv')