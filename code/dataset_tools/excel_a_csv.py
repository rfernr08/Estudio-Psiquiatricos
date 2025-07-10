import pandas as pd
import sys
import os

def excel_a_csv(ruta_excel, hoja=None, ruta_csv=None):
    # Leer el archivo Excel
    df = pd.read_excel(ruta_excel, sheet_name=hoja)
    # Definir ruta de salida si no se especifica
    if ruta_csv is None:
        base, _ = os.path.splitext(ruta_excel)
        ruta_csv = base + '.csv'
    # Guardar como CSV
    df.to_csv(ruta_csv, index=False, sep='|')
    print(f'Archivo CSV guardado en: {ruta_csv}')

if __name__ == "__main__":
    
    archivo_excel = "datasets/PSQ_F20_F29_Ext.xlsx"
    hoja = "Sheet1"
    archivo_csv = "datasets/PSQ_F20_F29_Ext.csv"
    excel_a_csv(archivo_excel, hoja, archivo_csv)