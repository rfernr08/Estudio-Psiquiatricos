import pandas as pd

def limpiar_dataset_psiquiatria(input_file, output_file):
    # 1. Cargar el dataset
    df = pd.read_csv(input_file, sep='|')
    
    # Identificar columnas de entrada (Diag 01 a Diag 20) y el target
    cols_entrada = [c for c in df.columns if 'Diag' in c and 'Secundario' in c or 'Principal' in c]
    col_target = 'DIAG PSQ'
    
    # 2. Eliminar registros con diagnóstico único idéntico al final
    # Contamos cuántos diagnósticos no nulos hay por fila en las entradas
    def es_unico_y_duplicado(row):
        diagnosticos_reales = [str(val).strip() for val in row[cols_entrada] if pd.notna(val) and str(val).strip() != ""]
        # Si solo hay uno y ese uno es igual al target (o empieza igual)
        if len(diagnosticos_reales) == 1:
            if diagnosticos_reales[0].startswith(str(row[col_target])[:3]):
                return True
        return False

    df = df[~df.apply(es_unico_y_duplicado, axis=1)]

    # 3. Limpiar las columnas de entrada (Borrar duplicados y familia similar)
    def limpiar_fila_entrada(row):
        target_val = str(row[col_target]).strip()
        familia_target = target_val[:3] # Ejemplo: 'F20' de 'F20.89'
        
        for col in cols_entrada:
            valor_celda = str(row[col]).strip()
            if pd.notna(row[col]) and valor_celda != "":
                # Si el código de la celda pertenece a la misma familia (ej. F2x)
                if valor_celda.startswith(familia_target):
                    row[col] = None # Borramos el sesgo
        return row

    df = df.apply(limpiar_fila_entrada, axis=1)

    # 4. Guardar el dataset limpio
    df.to_csv(output_file, index=False, sep='|')
    print(f"Proceso completado. Dataset guardado en: {output_file}")


limpiar_dataset_psiquiatria('TFG/dataset/diagnosticos_completos/diagnosticos_unidos.csv', 'dataset_limpio_ia.csv')