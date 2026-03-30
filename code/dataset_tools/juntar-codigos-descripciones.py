import pandas as pd

df_cod = pd.read_csv("C:\\Users\\Usuario\\Documents\\Workspace\\Mirage\\TFG\\dataset\\dataset_bert_undersampled.csv", sep="|")
df_desc = pd.read_csv("C:\\Users\\Usuario\\Documents\\Workspace\\Mirage\\dataset_descripciones.csv", sep="|")

diag_cols = [col for col in df_cod.columns if col.startswith("Diag")]

def combinar_codigo_descripcion(cod, desc):
    """Combina código y descripción en formato 'codigo: descripcion'"""
    cod = str(cod).strip() if pd.notna(cod) else ""
    desc = str(desc).strip() if pd.notna(desc) else ""
    
    if cod and desc:
        return f"{cod}: {desc}"
    elif cod:
        return cod
    elif desc:
        return desc
    else:
        return ""

# Sobrescribir cada columna usando pandas apply (más eficiente)
df_combined = df_cod.copy()

for col in diag_cols:
    df_combined[col] = df_cod[col].combine(
        df_desc[col], 
        combinar_codigo_descripcion
    )

# Guardar el resultado
df_combined.to_csv("diagnosticos_combinados.csv", sep="|", index=False)

print("✅ Archivo guardado con columnas sobrescritas")
print(f"📊 Columnas procesadas: {len(diag_cols)}")
print(f"📄 Filas procesadas: {len(df_combined)}")

# Mostrar ejemplo de transformación
print("\n🔍 Ejemplo de transformación:")
for i, col in enumerate(diag_cols[:3]):  # Mostrar solo las primeras 3 columnas
    print(f"\n--- {col} ---")
    print(f"Antes: '{df_cod[col].iloc[0]}'")
    print(f"Después: '{df_combined[col].iloc[0]}'")