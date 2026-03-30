import pandas as pd

def realizar_undersampling(input_file, output_file, max_por_clase=600):
    """
    Reduce las clases mayoritarias a un límite fijo para balancear el dataset.
    """
    # 1. Cargar el dataset (usando el separador | que definimos)
    df = pd.read_csv(input_file, sep='|')
    
    # 2. Ver la distribución actual
    print("Distribución original:")
    print(df['DIAG PSQ'].value_counts())
    
    # 3. Aplicar el recorte (Undersampling)
    df_balanceado = pd.DataFrame()
    
    for clase in df['DIAG PSQ'].unique():
        # Extraemos todos los registros de esta enfermedad
        subset = df[df['DIAG PSQ'] == clase]
        
        # Si la clase tiene más registros que nuestro límite, recortamos
        if len(subset) > max_por_clase:
            print(f"-> Recortando clase {clase}: de {len(subset)} a {max_por_clase}")
            subset = subset.sample(n=max_por_clase, random_state=42)
        else:
            print(f"-> Manteniendo clase {clase}: {len(subset)} registros")
            
        df_balanceado = pd.concat([df_balanceado, subset])
    
    # 4. Mezclar el dataset (importante para que BERT no vea todos los F20 juntos)
    df_balanceado = df_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 5. Guardar el nuevo dataset reducido
    df_balanceado.to_csv(output_file, sep='|', index=False)
    
    print("\n" + "="*30)
    print(f"Dataset reducido guardado en: {output_file}")
    print(f"Total de registros final: {len(df_balanceado)}")
    print("="*30)

# --- EJECUCIÓN ---
# Probemos con un límite de 600 para que el F20 no domine tanto
realizar_undersampling('C:\\Users\\Usuario\\Documents\\Workspace\\Mirage\\TFG\\dataset\\Diagnosticos_codigos_sin_duplicar_simplificado.csv', 'dataset_bert_undersampled.csv', max_por_clase=600)