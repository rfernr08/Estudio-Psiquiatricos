def encontrar_lineas_inconsistentes(nombre_archivo, separador=','):
    with open(nombre_archivo, 'r', encoding='utf-8') as f:
        lineas = f.readlines()

    num_columnas = len(lineas[0].strip().split(separador))
    lineas_inconsistentes = []

    for i, linea in enumerate(lineas, start=1):
        columnas = linea.strip().split(separador)
        if len(columnas) != num_columnas:
            lineas_inconsistentes.append((i, len(columnas), linea.strip()))

    return lineas_inconsistentes

# Ejemplo de uso:
archivo = 'relaciones_diagnosticos_psiquiatricos.csv'
inconsistentes = encontrar_lineas_inconsistentes(archivo)

if inconsistentes:
    print("Líneas con diferente número de columnas:")
    for num_linea, num_columnas, contenido in inconsistentes:
        print(f"Línea {num_linea}: {num_columnas} columnas -> {contenido}")
else:
    print("Todas las líneas tienen el mismo número de columnas.")