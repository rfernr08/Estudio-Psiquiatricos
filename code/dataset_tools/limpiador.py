import polars as pl

def limpiar_csv_solo_diag_psq(archivo_entrada, archivo_salida=None):
    """
    Elimina las filas que solo tienen datos en la columna DIAG PSQ
    (todas las demás columnas están vacías o son null)
    """
    # Leer el CSV
    df = pl.read_csv(archivo_entrada, separator="|")
    
    # Obtener todas las columnas excepto DIAG PSQ
    columnas_diagnostico = [col for col in df.columns if col != "DIAG PSQ"]
    
    print(f"Filas originales: {len(df)}")
    print(f"Columnas de diagnóstico a verificar: {len(columnas_diagnostico)}")
    
    # Crear condición: al menos una columna de diagnóstico debe tener datos
    condicion = pl.lit(False)  # Empezar con False
    
    for col in columnas_diagnostico:
        # Verificar si la columna no está vacía, no es null, y no es solo espacios
        condicion_col = (
            df[col].is_not_null() & 
            (df[col] != "") & 
            (df[col].str.strip_chars() != "")
        )
        condicion = condicion | condicion_col
    
    # Filtrar el DataFrame manteniendo solo filas que cumplen la condición
    df_limpio = df.filter(condicion)
    
    print(f"Filas después de limpieza: {len(df_limpio)}")
    print(f"Filas eliminadas: {len(df) - len(df_limpio)}")
    
    # Guardar el resultado
    if archivo_salida is None:
        archivo_salida = archivo_entrada.replace(".csv", "_limpio.csv")
    
    df_limpio.write_csv(archivo_salida, separator="|")
    print(f"Archivo guardado como: {archivo_salida}")
    
    return df_limpio

# Ejemplo de uso
if __name__ == "__main__":
    # Cambiar por tu archivo de entrada
    archivo_csv = "recursos/otros/BERT/diagnosticos_F20_F20.89_con_descripcion_sin_dups.csv"
    
    # Limpiar el archivo
    df_resultado = limpiar_csv_solo_diag_psq(
        archivo_entrada=archivo_csv,
        archivo_salida="recursos/otros/BERT/diagnosticos_F20_F20.89_con_descripcion_sin_dups_limpio.csv"
    )
    
    # Mostrar algunas estadísticas
    print("\n=== Muestra del resultado ===")
    print(df_resultado.head())
    
    # Verificar que no queden filas vacías
    columnas_diag = [col for col in df_resultado.columns if col != "DIAG PSQ"]
    filas_con_datos = 0
    
    for i in range(len(df_resultado)):
        fila = df_resultado[i]
        tiene_datos = False
        for col in columnas_diag:
            valor = fila[col].item()
            if valor is not None and str(valor).strip() != "":
                tiene_datos = True
                break
        if tiene_datos:
            filas_con_datos += 1
    
    print(f"\nVerificación: {filas_con_datos}/{len(df_resultado)} filas tienen datos en columnas de diagnóstico")