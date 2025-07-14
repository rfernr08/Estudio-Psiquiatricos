import polars as pl

# Cambia 'archivo.xlsx' y 'NombreColumna' por el nombre real del archivo y la columna de municipios
df = pl.read_excel('datasets/PSQ_F20_F29.xlsx')
municipios_unicos = df['DIAG PSQ'].unique()
municipios_unicos_df = municipios_unicos.to_frame()
municipios_unicos_df.write_csv('diagnosticos_unicos.csv')